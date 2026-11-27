from __future__ import annotations

from datetime import datetime, timedelta, UTC
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.ai.datasets import _render_synthetic_sky_frame
from app.database import get_db_session
from app.main import app
from app.models_db import Observation
from app.services.events import evaluate_objective_future_events
from app.services.metar import compare_observation_with_metar, fetch_nearest_metar, parse_metar_cloud_layers


def test_objective_future_events_and_missed_event_reporting(isolated_env):
    base = datetime(2026, 6, 15, 11, 0, tzinfo=UTC)
    # Chronological observations: 10% clear -> 55% (+45 pts in 20m = rapid_increase) -> 10% for 20m (clearing)
    obs_seq = [
        Observation(
            id=1, captured_at=base, is_usable=True, cloud_coverage_pct=10.0, avg_luminance=78.0, source_type="demo"
        ),
        Observation(
            id=2,
            captured_at=base + timedelta(minutes=20),
            is_usable=True,
            cloud_coverage_pct=55.0,
            avg_luminance=56.0,
            source_type="demo",
        ),
        Observation(
            id=3,
            captured_at=base + timedelta(minutes=30),
            is_usable=True,
            cloud_coverage_pct=12.0,
            avg_luminance=76.0,
            source_type="demo",
        ),
        Observation(
            id=4,
            captured_at=base + timedelta(minutes=50),
            is_usable=True,
            cloud_coverage_pct=8.0,
            avg_luminance=78.0,
            source_type="demo",
        ),
    ]
    res = evaluate_objective_future_events(obs_seq, isolated_env)
    assert res[0]["rapid_increase"] is True
    assert res[0]["darkening"] is True
    assert res[1]["clearing"] is True

    # Test missed-event API reporting + active-learning uncertainty ranking
    img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=0, seed=11, width=160, height=120)
    raw_p = isolated_env.raw_images_dir / "al_test.jpg"
    raw_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_p), img)
    with get_db_session() as db:
        db.add(
            Observation(
                captured_at=base,
                source_type="demo_simulation",
                raw_path=str(raw_p),
                is_usable=True,
                cloud_coverage_pct=42.0,
                classification_confidence=0.22,
                quality_score=0.55,
            )
        )
        db.commit()

    with TestClient(app) as client:
        r_miss = client.post(
            "/api/events/report-missed",
            json={"event_type": "storm_like_development", "note": "Sudden squall line not alerted"},
        )
        assert r_miss.status_code == 200
        ev_data = r_miss.json()
        assert ev_data["user_status"] == "confirmed"
        assert ev_data["metrics"]["false_negative"] is True

        r_al = client.get("/api/dataset/active-learning?limit=5")
        assert r_al.status_code == 200
        al_items = r_al.json()["items"]
        assert len(al_items) == 1
        assert al_items[0]["uncertainty_score"] > 0.3


def test_metar_parser_comparison_and_offline_guard(isolated_env):
    parsed = parse_metar_cloud_layers("LIRF 271850Z 21010KT 9999 FEW025 SCT045 BKN080CB 18/11 Q1016")
    assert parsed["parsed"] is True
    assert parsed["oktas"] == 6
    assert parsed["cloud_cover_pct"] == 75.0
    assert parsed["ceilometer_base_m"] == round(2500 * 0.3048, 1)
    assert parsed["has_convective_tag"] is True

    cmp = compare_observation_with_metar(camera_cloud_pct=68.0, estimated_base_m=900.0, metar_report=parsed)
    assert cmp["comparable"] is True
    assert cmp["coverage_abs_error_pct"] == 7.0
    assert cmp["base_height_error_m"] is not None

    # Offline lock prevents outbound AviationWeather.gov call
    assert isolated_env.server.offline_mode is True
    off = fetch_nearest_metar(41.9, 12.5, isolated_env)
    assert off["available"] is False and off["offline_mode"] is True


def test_build_golden_set_script_creates_locked_stratified_manifest(isolated_env, tmp_path: Path):
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_golden_set import GOLDEN_LOCK_FILENAME, build_golden_set

    isolated_env.raw_images_dir.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    with get_db_session() as db:
        for i, cov in enumerate([5.0, 45.0, 92.0, 10.0, 60.0, 88.0]):
            p = isolated_env.raw_images_dir / f"g_{i}.jpg"
            cv2.imwrite(str(p), np.full((64, 64, 3), 120, dtype=np.uint8))
            db.add(
                Observation(
                    captured_at=base + timedelta(hours=i * 3),
                    source_type="demo_simulation",
                    raw_path=str(p),
                    is_usable=True,
                    cloud_coverage_pct=cov,
                )
            )
        db.commit()

    out_dir = tmp_path / "golden_eval"
    manifest = build_golden_set(out_dir, target_size=10)
    assert manifest["golden_locked"] is True
    assert manifest["selected_count"] == 6
    assert (out_dir / GOLDEN_LOCK_FILENAME).exists()
    assert (out_dir / "label_studio_tasks.json").exists()
