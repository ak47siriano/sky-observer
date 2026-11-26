from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.ai.architectures import DeepLabV3CloudSeg, DINOv3LinearProbeSeg, build_model_for_family
from app.ai.autolearn import FIXED_HOLDOUT_FILENAME, _extract_primary_metric, _load_or_update_fixed_holdout
from app.ai.datasets import _render_synthetic_sky_frame
from app.ai.registry import get_model_cache, register_and_maybe_activate
from app.database import get_db_session
from app.main import app
from app.services.worker import process_and_store_frame
from app.vision.masking import propose_sky_obstacle_mask


def test_deeplabv3_and_dinov3_linear_probe_forward_and_registry(isolated_env):
    dl_model, dl_name = build_model_for_family("segmentation", "DeepLabV3-MobileNetV3", pretrained=False)
    assert isinstance(dl_model, DeepLabV3CloudSeg)
    assert dl_name == "DeepLabV3-MobileNetV3"
    dl_model.eval()
    with torch.no_grad():
        out_dl = dl_model(torch.randn(1, 3, 256, 256))
    assert out_dl.shape == (1, 5, 256, 256)

    dino_model, dino_name = build_model_for_family("segmentation", "DINOv3-LinearProbe", pretrained=False)
    assert isinstance(dino_model, DINOv3LinearProbeSeg)
    assert dino_name == "DINOv3-LinearProbe"
    # Backbone must stay frozen; only linear probe head + norm are trainable
    trainable_names = [n for n, p in dino_model.named_parameters() if p.requires_grad]
    assert all(n.startswith(("head.", "probe_norm.")) for n in trainable_names)
    assert len(trainable_names) >= 2

    out_dino = dino_model(torch.randn(2, 3, 256, 256))
    assert out_dino.shape == (2, 5, 256, 256)

    # Verify round-trip registration & cache loading of DINOv3-LinearProbe checkpoint
    ckpt_path = isolated_env.models_dir / "test_dinov3_probe.pt"
    torch.save({"model_state_dict": dino_model.state_dict()}, ckpt_path)
    with get_db_session() as db:
        register_and_maybe_activate(
            db=db,
            model_id="seg-dinov3-test",
            model_family="segmentation",
            architecture="DINOv3-LinearProbe",
            version="v1.0.0",
            weights_path=ckpt_path,
            metrics={"val_miou": 0.72, "val_pixel_acc": 0.91},
            activate=True,
        )
        cache = get_model_cache()
        cache.refresh_from_db(db)
        loaded_m, meta, _ = cache.get("segmentation")
        assert isinstance(loaded_m, DINOv3LinearProbeSeg)
        assert meta is not None and meta["architecture"] == "DINOv3-LinearProbe"


def test_propose_sky_obstacle_mask_and_api(isolated_env):
    # Create synthetic image with bright blue sky in top half and dark green/brown trees/buildings in bottom half
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:120, :] = (225, 165, 95)  # Blue sky in BGR
    img[120:, :] = (28, 72, 32)  # Dark foliage obstacle
    # Add high-frequency building texture in bottom half
    rng = np.random.default_rng(7)
    noise = rng.integers(-25, 25, size=(120, 320, 3), dtype=np.int16)
    img[120:] = np.clip(img[120:].astype(np.int16) + noise, 0, 255).astype(np.uint8)

    excl, stats = propose_sky_obstacle_mask(img)
    assert excl.shape == (240, 320)
    assert 30.0 <= float(stats["proposed_excluded_pct"]) <= 70.0
    assert float(np.mean(excl[:80, :] == 0)) > 0.85  # Top sky mostly valid
    assert float(np.mean(excl[150:, :] > 127)) > 0.85  # Bottom obstacles mostly excluded

    with get_db_session() as db:
        obs = process_and_store_frame(db, img, isolated_env, "local_import", run_forecast=False)
        obs_id = obs.id

    with TestClient(app) as client:
        r = client.post(f"/api/camera/exclusion-mask/propose?observation_id={obs_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["observation_id"] == obs_id
        assert data["mask_png_base64"].startswith("data:image/png;base64,")
        assert 30.0 <= float(data["proposed_excluded_pct"]) <= 70.0


def test_fixed_holdout_manifest_never_leaks_to_train(tmp_path: Path):
    items_cycle1 = [
        {"observation_id": i, "group_key": f"loc_seq_{i // 3}", "image_path": f"obs_{i}.jpg", "user_labeled": (i == 2)}
        for i in range(1, 11)
    ]
    split1, locked1 = _load_or_update_fixed_holdout(tmp_path, items_cycle1, holdout_fraction=0.20)
    assert len(locked1) >= 2
    assert "obs:2" in locked1  # human-labeled item prioritized
    assert (tmp_path / FIXED_HOLDOUT_FILENAME).exists()
    for it in split1:
        if f"obs:{it['observation_id']}" in locked1:
            assert it["split"] == "val"

    # Second Auto-Train cycle adds more samples; previously locked holdout keys MUST remain 'val'
    items_cycle2 = [
        {"observation_id": i, "group_key": f"loc_seq_{i // 3}", "image_path": f"obs_{i}.jpg", "user_labeled": (i == 2)}
        for i in range(1, 21)
    ]
    split2, locked2 = _load_or_update_fixed_holdout(tmp_path, items_cycle2, holdout_fraction=0.20)
    assert set(locked1).issubset(set(locked2))
    for it in split2:
        if f"obs:{it['observation_id']}" in locked1:
            assert it["split"] == "val"

    assert _extract_primary_metric("segmentation", {"val_miou": 0.68}) == 0.68
    assert _extract_primary_metric("temporal", {"val_mae_pct": 9.0}) == 0.1


def test_train_dinov3_probe_script_end_to_end(tmp_path: Path):
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_dinov3_probe import collect_pairs_from_dir, train_probe

    img_dir = tmp_path / "ds" / "images"
    msk_dir = tmp_path / "ds" / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    for i, sc in enumerate(["clear_sky", "cumulus", "stratus", "cirrus", "cumulonimbus_like", "clear_sky"]):
        img, gt, _, _, _ = _render_synthetic_sky_frame(sc, t_step=i, seed=90 + i, width=128, height=128)
        cv2.imwrite(str(img_dir / f"f_{i}.jpg"), img)
        cv2.imwrite(str(msk_dir / f"f_{i}.png"), gt)

    pairs = collect_pairs_from_dir(tmp_path / "ds")
    assert len(pairs) == 6
    out_ckpt = tmp_path / "probe.pt"
    report = train_probe(pairs, out_ckpt, epochs=2, batch_size=2, resolution=128, device_pref="cpu")
    assert out_ckpt.exists()
    assert 0.0 <= report["val_miou"] <= 1.0
    assert report["trainable_params"] > 0
    # Verify saved JSON-serializable report
    assert json.dumps(report)
