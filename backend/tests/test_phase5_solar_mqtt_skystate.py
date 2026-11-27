from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.ai.datasets import _render_synthetic_sky_frame
from app.camera.manager import CameraManager
from app.config import CameraConfig
from app.database import get_db_session
from app.main import app
from app.services.mqtt_ha import build_ha_discovery_payloads, get_mqtt_publisher
from app.services.solar_pv import (
    compute_clear_sky_ghi_wm2,
    estimate_cloud_transmittance,
    estimate_solar_pv_and_ramps,
)
from app.services.worker import process_and_store_frame
from app.vision.classification import SKY_STATE_TAGS, WMO_OFFICIAL_GENERA, classify_cloud_type


def test_solar_pv_irradiance_and_ramp_detection():
    ghi_noon = compute_clear_sky_ghi_wm2(60.0)
    ghi_night = compute_clear_sky_ghi_wm2(-10.0)
    assert 700.0 <= ghi_noon <= 1050.0
    assert ghi_night == 0.0

    tau_clear = estimate_cloud_transmittance(0.0, 0.0, 0.0)
    tau_dark = estimate_cloud_transmittance(0.0, 20.0, 80.0)
    assert tau_clear == 1.0
    assert 0.12 <= tau_dark <= 0.30

    pv = estimate_solar_pv_and_ramps(
        lat=45.0,
        lon=9.0,
        thin_cloud_pct=5.0,
        dense_cloud_pct=5.0,
        dark_cloud_pct=0.0,
        cloud_coverage_pct=10.0,
        pv_kwp=4.0,
        panel_tilt_deg=30.0,
        panel_azimuth_deg=180.0,
        ramp_alert_pct=20.0,
        forecast_horizons=[
            {"horizon_minutes": 5, "cloud_cover_prediction": 20.0},
            {"horizon_minutes": 15, "cloud_cover_prediction": 92.0},
        ],
        dt_utc=datetime(2026, 6, 21, 11, 30, 0, tzinfo=UTC),
    )
    assert pv["is_daylight"] is True
    assert pv["estimated_pv_kw"] > 1.5
    assert pv["ramp_alert"] is not None
    assert pv["ramp_alert"]["ramp_type"] == "ramp_down"
    assert pv["ramp_alert"]["horizon_minutes"] == 15


def test_classification_sky_state_tags_and_taxonomy():
    assert len(WMO_OFFICIAL_GENERA) == 10
    assert set(SKY_STATE_TAGS) == {"contrails", "rain", "snow", "fog"}

    img = np.full((128, 128, 3), (150, 150, 150), dtype=np.uint8)
    fog_res = classify_cloud_type(
        img,
        coverage_metrics={
            "cloud_coverage_pct": 95.0,
            "clear_sky_pct": 5.0,
            "thin_cloud_pct": 20.0,
            "dense_cloud_pct": 75.0,
            "dark_cloud_pct": 0.0,
        },
        color_metrics={"avg_luminance": 62.0, "blue_sky_index": 0.0, "saturation": 0.05, "grayness": 0.95},
        texture_metrics={"local_contrast": 3.0, "edge_density": 0.005, "fragmentation_index": 0.05},
        min_confidence=0.35,
    )
    assert "fog" in fog_res["sky_state_tags"]
    assert "is_uncertain" in fog_res


def test_sky_state_solar_pv_api_mqtt_and_allsky_folder(isolated_env, tmp_path: Path):
    # Test local AllSky folder source capture
    allsky_dir = tmp_path / "allsky"
    allsky_dir.mkdir()
    img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=0, seed=55, width=256, height=192)
    cv2.imwrite(str(allsky_dir / "image.jpg"), img)

    cm = CameraManager()
    cam_cfg = CameraConfig(source_type="allsky_folder", source_url=str(allsky_dir))
    cap = cm.capture_once(cfg_override=cam_cfg, bypass_backoff=True)
    assert cap.success is True
    assert cap.image_bgr is not None and cap.image_bgr.shape[:2] == (192, 256)

    # Test MQTT discovery payload builder & disabled-by-default offline safety
    payloads = build_ha_discovery_payloads()
    assert len(payloads) >= 5
    assert payloads[0][0].startswith("homeassistant/sensor/")
    assert get_mqtt_publisher().publish_observation({"state": "clear_sky"}, isolated_env) is False

    # Test /api/sky-state & /api/solar/pv endpoints
    with get_db_session() as db:
        process_and_store_frame(
            db, img, isolated_env, "allsky_folder", captured_at=datetime(2026, 6, 21, 11, 30, 0, tzinfo=UTC)
        )

    with TestClient(app) as client:
        r_sky = client.get("/api/sky-state")
        assert r_sky.status_code == 200
        sky = r_sky.json()
        assert sky["available"] is True
        assert "cloud_coverage_pct" in sky and "sky_regime" in sky and "state" in sky

        r_pv = client.get("/api/solar/pv")
        assert r_pv.status_code == 200
        pv = r_pv.json()
        assert pv["available"] is True
        assert "clear_sky_ghi_wm2" in pv and "estimated_pv_kw" in pv

        r_prom = client.get("/metrics")
        assert r_prom.status_code == 200
        assert "text/plain" in r_prom.headers["content-type"]
        body = r_prom.text
        assert "sky_observer_up 1" in body
        assert "sky_observer_observations_total 1" in body
        assert "sky_observer_cloud_coverage_pct" in body
        assert "sky_observer_storage_used_mb" in body


def test_prometheus_metrics_respects_api_token(isolated_env, monkeypatch):
    monkeypatch.setenv("SKYOBSERVER_API_TOKEN", "prom-secret-99")
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        ok = client.get("/metrics", headers={"Authorization": "Bearer prom-secret-99"})
        assert ok.status_code == 200
        assert "sky_observer_up 1" in ok.text
