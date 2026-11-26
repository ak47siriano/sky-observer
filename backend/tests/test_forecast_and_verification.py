from __future__ import annotations

from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.ai.datasets import _render_synthetic_sky_frame
from app.database import get_db_session
from app.forecast.advection import (
    advect_cloud_mask_optical_flow,
    compute_forecast_skill,
    render_advected_probability_png,
)
from app.forecast.verification import compute_verification_summary
from app.main import app
from app.models_db import DatasetItem, DatasetRecord, Forecast, safe_json_loads
from app.services.storage import apply_retention_policy
from app.services.worker import process_and_store_frame


def test_advection_mask_and_ensemble_bounds():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:, 16:72] = 2  # dense cloud band on left-center

    adv = advect_cloud_mask_optical_flow(
        seg_mask=mask,
        dx_px_s=0.12,
        dy_px_s=0.0,
        horizon_minutes=5.0,
        orig_width=128,
        orig_height=128,
        resolution=64,
        ensemble_members=5,
        motion_consistency=0.85,
    )
    assert adv["available"] is True
    assert 0.0 <= adv["cover_p10"] <= adv["cover_p50"] <= adv["cover_p90"] <= 100.0
    assert 0.0 <= adv["fov_exit_fraction"] <= 1.0
    assert adv["cloud_probability_map"].shape == (64, 64)

    png_bytes = render_advected_probability_png(adv["cloud_probability_map"], out_width=128)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    assert compute_forecast_skill(5.0, 10.0) == 0.5
    assert compute_forecast_skill(15.0, 10.0) == -0.5
    assert compute_forecast_skill(0.0, 0.0) == 0.0


def test_forecast_generation_and_chronological_verification(isolated_env):
    t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)

    with get_db_session() as db:
        for i in range(4):
            img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=i, seed=300 + i, width=256, height=256)
            process_and_store_frame(
                db=db,
                image_bgr=img,
                config=isolated_env,
                source_type="local_import",
                captured_at=t0 + timedelta(minutes=i * 2),
                run_forecast=True,
            )

        # Now simulate an observation arriving at t0 + 15 minutes to verify the +15m forecast generated at t0
        img_later, _, _, _, _ = _render_synthetic_sky_frame("stratocumulus", t_step=2, seed=400, width=256, height=256)
        process_and_store_frame(
            db=db,
            image_bgr=img_later,
            config=isolated_env,
            source_type="local_import",
            captured_at=t0 + timedelta(minutes=15),
            run_forecast=True,
        )

        ver = compute_verification_summary(db)
        assert ver["total_verified"] >= 1
        assert ver["status"] == "measured"
        assert "mae_pct" in ver["overall"]
        assert "rmse_pct" in ver["overall"]
        assert "brier_score" in ver["overall"]
        assert "persistence_mae_pct" in ver["overall"]
        assert "skill_vs_persistence" in ver["overall"]
        assert "beats_persistence" in ver["overall"]

        # Camera-only offline mode is physically limited to +5, +15, +20 min horizons
        fcs = db.query(Forecast).all()
        horizons = {f.horizon_minutes for f in fcs}
        assert {5, 15, 20}.issubset(horizons)
        assert 30 not in horizons and 60 not in horizons

        # Check Phase 2 baseline metadata persisted on forecasts
        sample_meta = safe_json_loads(fcs[-1].baseline_metrics_json, {})
        assert "persistence_cover_pct" in sample_meta
        assert "cover_p10" in sample_meta and "cover_p90" in sample_meta
        assert sample_meta["cover_p10"] <= sample_meta["cover_p90"]

    with TestClient(app) as client:
        r_png = client.get("/api/forecast/advected-mask.png?horizon=15")
        assert r_png.status_code == 200
        assert r_png.headers["content-type"].startswith("image/png")
        assert r_png.content[:8] == b"\x89PNG\r\n\x1a\n"

        r_latest = client.get("/api/forecast/latest")
        assert r_latest.status_code == 200
        h0 = r_latest.json()["horizons"][0]
        assert "baselines" in h0 and "low_reliability" in h0 and "cover_p10" in h0 and "cover_p90" in h0


def test_storage_retention_protects_training_images(isolated_env):
    old_ts = datetime.now(UTC) - timedelta(days=90)
    with get_db_session() as db:
        img1, _, _, _, _ = _render_synthetic_sky_frame("clear_sky", t_step=0, seed=1, width=128, height=128)
        img2, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=0, seed=2, width=128, height=128)

        obs_old = process_and_store_frame(
            db, img1, isolated_env, "local_import", captured_at=old_ts, run_forecast=False
        )
        obs_train = process_and_store_frame(
            db, img2, isolated_env, "local_import", captured_at=old_ts + timedelta(seconds=5), run_forecast=False
        )
        obs_train.is_training_sample = True

        ds = DatasetRecord(name="protected_ds", root_path=str(isolated_env.datasets_dir / "protected_ds"))
        db.add(ds)
        db.flush()
        db.add(DatasetItem(dataset_id=ds.id, image_path=obs_train.raw_path, group_key="g1"))
        db.commit()

        old_raw = Path(obs_old.raw_path)
        train_raw = Path(obs_train.raw_path)
        assert old_raw.exists() and train_raw.exists()

        res = apply_retention_policy(db, isolated_env)
        assert res["deleted_raw"] >= 1
        assert not old_raw.exists()
        assert train_raw.exists()  # Strictly protected!
