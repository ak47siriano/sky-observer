from __future__ import annotations

import torch

from app.ai.datasets import bootstrap_local_sky_dataset, get_dataset_summary
from app.ai.device import get_hardware_status, resolve_device
from app.ai.registry import activate_model_by_id, get_model_cache, rollback_model_family
from app.ai.trainer import get_training_manager
from app.database import get_db_session
from app.models_db import ModelRegistry, TrainingSession


def test_device_detection_and_cpu_fallback():
    dev_cpu = resolve_device("cpu", 0.85)
    assert dev_cpu == "cpu"
    dev_auto = resolve_device("auto", 0.85)
    assert dev_auto in {"cuda", "directml", "mps", "cpu"}
    hw = get_hardware_status("auto", 0.85, 512)
    assert "gpu" in hw
    assert hw["gpu"]["recommended_batch_size"] >= 1


def test_dataset_bootstrap_training_registry_and_rollback(isolated_env):
    with get_db_session() as db:
        ds = bootstrap_local_sky_dataset(db, isolated_env, sequences_per_scenario=2)
        summary = get_dataset_summary(db, ds.id)
        assert summary["item_count"] == 96
        assert summary["integrity_verified"] is True
        assert summary["splits"]["train"] > 0 and summary["splits"]["val"] > 0 and summary["splits"]["test"] > 0
        ds_id = ds.id

    tm = get_training_manager()

    # Train Segmentation v1
    tm.start_training("segmentation", ds_id, epochs=2, batch_size=4, async_run=False)
    # Train Segmentation v2 to test rollback
    tm.start_training("segmentation", ds_id, epochs=2, batch_size=4, async_run=False)

    # Train Classification, Temporal, and Event models
    tm.start_training("classification", ds_id, epochs=2, batch_size=4, async_run=False)
    tm.start_training("temporal", ds_id, epochs=3, batch_size=4, async_run=False)
    tm.start_training("event", ds_id, epochs=3, batch_size=4, async_run=False)

    with get_db_session() as db:
        sess_rows = db.query(TrainingSession).all()
        assert len(sess_rows) >= 5
        assert all(s.status == "completed" for s in sess_rows)

        seg_models = db.query(ModelRegistry).filter(ModelRegistry.model_family == "segmentation").all()
        assert len(seg_models) == 2
        active_seg = [m for m in seg_models if m.is_active]
        assert len(active_seg) == 1

        # Rollback segmentation to previous version
        rolled = rollback_model_family(db, "segmentation")
        assert rolled.model_id != active_seg[0].model_id
        assert rolled.is_active is True

        # Switch back
        re_act = activate_model_by_id(db, active_seg[0].model_id)
        assert re_act.is_active is True

        cache = get_model_cache()
        cache.refresh_from_db(db)
        seg_m, _, _ = cache.get("segmentation")
        cls_m, _, _ = cache.get("classification")
        tmp_m, _, _ = cache.get("temporal")
        assert isinstance(seg_m, torch.nn.Module)
        assert isinstance(cls_m, torch.nn.Module)
        assert isinstance(tmp_m, torch.nn.Module)


def test_auto_label_and_auto_train_local_reinforcement(isolated_env):
    import io
    from pathlib import Path

    import cv2
    from fastapi.testclient import TestClient

    from app.ai.autolearn import AUTO_DATASET_NAME, get_auto_learn_manager
    from app.ai.datasets import _render_synthetic_sky_frame
    from app.main import app
    from app.models_db import DatasetRecord, EventAlert, Observation

    scenarios = ["cumulus", "cirrus", "cumulonimbus_like", "clear_sky"]
    with TestClient(app) as client:
        for i, sc in enumerate(scenarios):
            img, _, _, _, _ = _render_synthetic_sky_frame(sc, t_step=i, seed=200 + i, width=256, height=256)
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            r_up = client.post(
                "/api/camera/upload",
                files={"file": (f"{sc}.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
            )
            assert r_up.status_code == 200

        with get_db_session() as db:
            obs_rows = db.query(Observation).order_by(Observation.id.asc()).all()
            assert len(obs_rows) == 4
            db.add(
                EventAlert(
                    event_type="storm_like_development",
                    severity="warning",
                    title="Confirmed Dark Cloud",
                    description="Confirmed locally",
                    observation_id=obs_rows[0].id,
                    acknowledged=True,
                    user_status="confirmed",
                )
            )
            db.add(
                EventAlert(
                    event_type="camera_movement",
                    severity="warning",
                    title="Denied Movement",
                    description="False alarm",
                    observation_id=obs_rows[1].id,
                    acknowledged=True,
                    user_status="rejected",
                )
            )

        r_cfg = client.put(
            "/api/models/auto-train/config",
            json={
                "enabled": True,
                "epochs": 1,
                "families": ["segmentation", "classification"],
                "min_new_observations": 2,
                "include_baseline_replay": True,
            },
        )
        assert r_cfg.status_code == 200
        assert r_cfg.json()["enabled"] is True
        assert r_cfg.json()["config"]["epochs"] == 1

        mgr = get_auto_learn_manager()
        mgr.trigger_run(families=["segmentation", "classification"], epochs=1, async_run=False)

        r_st = client.get("/api/models/auto-train/status")
        assert r_st.status_code == 200
        st = r_st.json()
        assert st["stage"] == "completed"
        assert st["eligible_local_observations"] == 4
        assert st["last_run_at"] is not None
        assert st["last_summary"]["local_items"] == 4
        assert st["last_summary"]["confirmed_event_items"] == 1
        assert st["last_summary"]["replay_items"] > 0

        with get_db_session() as db:
            auto_ds = db.query(DatasetRecord).filter(DatasetRecord.name == AUTO_DATASET_NAME).first()
            assert auto_ds is not None
            ds_sum = get_dataset_summary(db, auto_ds.id)
            assert ds_sum["integrity_verified"] is True
            assert Path(auto_ds.root_path).exists()
            active_models = db.query(ModelRegistry).filter(ModelRegistry.is_active.is_(True)).all()
            active_families = {m.model_family for m in active_models}
            assert "segmentation" in active_families
            assert "classification" in active_families

        # Train a second version of segmentation & classification so we have old standby models to delete/prune
        mgr.trigger_run(families=["segmentation", "classification"], epochs=1, async_run=False)
        all_models = client.get("/api/models").json()["items"]
        standby_seg = [m for m in all_models if m["model_family"] == "segmentation" and not m["is_active"]]
        active_seg = [m for m in all_models if m["model_family"] == "segmentation" and m["is_active"]]
        assert len(standby_seg) >= 1 and len(active_seg) == 1

        # Active model cannot be deleted without force
        r_del_active = client.delete(f"/api/models/{active_seg[0]['model_id']}")
        assert r_del_active.status_code == 409

        # Standby model can be deleted individually
        r_del_old = client.delete(f"/api/models/{standby_seg[0]['model_id']}")
        assert r_del_old.status_code == 200
        assert r_del_old.json()["deleted"] is True

        # Prune remaining old standby models keeping only 1 active model per family
        r_prune = client.post("/api/models/prune", json={"keep_per_family": 1, "also_prune_sessions": True})
        assert r_prune.status_code == 200
        assert r_prune.json()["deleted_models"] >= 1
        after_models = client.get("/api/models").json()["items"]
        assert all(m["is_active"] for m in after_models)


def test_hardware_profile_presets_and_device_selection(isolated_env):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r_hw = client.get("/api/system/gpu")
        assert r_hw.status_code == 200
        hw = r_hw.json()
        assert hw["detected_tier"] in {"low_end", "balanced", "high_end", "ultra"}
        assert "minimum_requirements" in hw and "recommended_requirements" in hw
        assert isinstance(hw["available_devices"], list) and len(hw["available_devices"]) >= 3

        # Apply low_end CPU profile
        r_low = client.put("/api/system/hardware-profile", json={"profile": "low_end", "device": "cpu"})
        assert r_low.status_code == 200
        d_low = r_low.json()
        assert d_low["applied_profile"] == "low_end"
        assert d_low["device"] == "cpu"
        assert d_low["active_device"] == "cpu"
        assert d_low["inference_resolution"] == 256
        assert d_low["batch_size"] == 1

        # Apply high_end profile (gracefully falls back to cpu if CUDA is absent on test host)
        r_high = client.put("/api/system/hardware-profile", json={"profile": "high_end", "device": "cuda"})
        assert r_high.status_code == 200
        d_high = r_high.json()
        assert d_high["applied_profile"] == "high_end"
        assert d_high["device"] == "cuda"
        assert d_high["inference_resolution"] == 512
        assert d_high["batch_size"] == 4


def test_convnext_aspp_and_focal_dice_loss():
    import torch

    from app.ai.architectures import build_model_for_family
    from app.ai.trainer import _FocalDiceSegLoss

    seg_model, seg_name = build_model_for_family("segmentation", "CloudSegUNet-ConvNeXt-ASPP", pretrained=False)
    assert seg_name == "CloudSegUNet-ConvNeXt-ASPP"
    seg_model.train()
    # Verify batch_size=1 in train mode works (ASPP global pool must not fail BatchNorm on 1x1x1)
    single_logits = seg_model(torch.randn(1, 3, 256, 256))
    assert single_logits.shape == (1, 5, 256, 256)
    x = torch.randn(2, 3, 256, 256)
    logits = seg_model(x)
    assert logits.shape == (2, 5, 256, 256)

    targets = torch.randint(0, 5, (2, 256, 256))
    targets[:, :32, :] = 255  # ignore Not_Sky
    loss_fn = _FocalDiceSegLoss(ignore_index=255, gamma=1.5)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss) and float(loss.item()) > 0.0

    cls_model, cls_name = build_model_for_family("classification", "CloudTypeClassifier-EfficientNetB0", pretrained=False)
    assert cls_name == "CloudTypeClassifier-EfficientNetB0"
    cls_logits = cls_model(torch.randn(2, 3, 224, 224), torch.randn(2, 18))
    assert cls_logits.shape == (2, 11)
