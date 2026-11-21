from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.ai.datasets import _render_synthetic_sky_frame
from app.main import app
from app.services.weather_online import get_online_weather_service


def test_offline_lock_blocks_outbound_weather_calls(isolated_env):
    svc = get_online_weather_service()
    svc.clear_cache()

    # In offline mode (default), Open-Meteo/RainViewer must never invoke httpx.Client
    with patch("app.services.weather_online.httpx.Client") as mock_http:
        res = svc.get_weather(41.9, 12.5, isolated_env)
        grid = svc.get_regional_cloud_grid(41.9, 12.5, isolated_env)
        radar = svc.get_radar_layers(isolated_env)
        geo = svc.geocode_search("Rome", "en", isolated_env)
        rev = svc.reverse_geocode(41.9028, 12.4964, isolated_env)

    assert mock_http.call_count == 0
    assert res["offline_mode"] is True and res["available"] is False
    assert grid == []
    assert radar is None
    assert geo == []
    assert rev is None


def test_radar_location_mode_switch_and_hybrid_forecast(isolated_env):
    img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=2, seed=88, width=320, height=240)
    ok, jpeg_buf = cv2.imencode(".jpg", img)
    assert ok

    mock_wx = {
        "offline_mode": False,
        "available": True,
        "provider": "Open-Meteo (Mock)",
        "latitude": 45.46,
        "longitude": 9.19,
        "fetched_at": "2026-11-21T10:30:00+00:00",
        "current": {
            "temperature_c": 16.5,
            "apparent_temperature_c": 15.8,
            "relative_humidity_pct": 85,
            "pressure_msl_hpa": 1004.0,
            "cloud_cover_pct": 78,
            "cloud_cover_low_pct": 60,
            "cloud_cover_mid_pct": 40,
            "cloud_cover_high_pct": 20,
            "precipitation_mm": 0.4,
            "rain_mm": 0.4,
            "wind_speed_kmh": 14.2,
            "wind_direction_deg": 240,
            "weather_code": 3,
        },
        "horizons": {
            5: {"cloud_cover_pct": 80.0, "precip_probability_pct": 45.0},
            15: {"cloud_cover_pct": 85.0, "precip_probability_pct": 65.0},
            30: {"cloud_cover_pct": 90.0, "precip_probability_pct": 75.0},
            60: {"cloud_cover_pct": 92.0, "precip_probability_pct": 80.0},
        },
        "minutely_15_series": [],
    }
    mock_grid = [
        {"latitude": 45.46, "longitude": 9.19, "cloud_cover_pct": 82.0, "cloud_low_pct": 60.0, "cloud_mid_pct": 30.0, "cloud_high_pct": 10.0, "precipitation_mm": 0.2}
    ]
    mock_radar = {
        "provider": "RainViewer",
        "attribution": "Weather data by RainViewer",
        "generated": 1700000000,
        "radar_frames": [{"time": 1700000000, "kind": "radar_past", "tile_url": "https://tilecache.rainviewer.com/v2/radar/1/256/{z}/{x}/{y}/2/1_1.png"}],
        "satellite_frames": [],
    }

    with TestClient(app) as client:
        # Upload a real observation first
        r_up = client.post("/api/camera/upload", files={"file": ("sky.jpg", jpeg_buf.tobytes(), "image/jpeg")})
        assert r_up.status_code == 200

        # Update radar geolocation pin
        r_loc = client.put(
            "/api/radar/location",
            json={"latitude": 45.4642, "longitude": 9.19, "location_description": "Milan Observatory", "radar_radius_km": 20.0},
        )
        assert r_loc.status_code == 200
        assert r_loc.json()["latitude"] == 45.4642

        # Check offline radar state & PNG sky-dome cloud overlay
        r_state_off = client.get("/api/radar/state")
        assert r_state_off.status_code == 200
        assert r_state_off.json()["offline_mode"] is True
        assert r_state_off.json()["online_weather"] is None
        assert len(r_state_off.json()["camera_cloud_grid"]) == 12

        r_png = client.get("/api/radar/cloud-overlay.png")
        assert r_png.status_code == 200
        assert r_png.headers["content-type"] == "image/png"
        rgba = cv2.imdecode(np.frombuffer(r_png.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert rgba is not None and rgba.shape == (384, 384, 4)

        # Switch to Online Hybrid mode with mocked Open-Meteo + RainViewer
        with (
            patch("app.services.weather_online.OnlineWeatherService.get_weather", return_value=mock_wx),
            patch("app.services.weather_online.OnlineWeatherService.get_regional_cloud_grid", return_value=mock_grid),
            patch("app.services.weather_online.OnlineWeatherService.get_radar_layers", return_value=mock_radar),
        ):
            r_mode = client.put("/api/system/mode", json={"offline_mode": False})
            assert r_mode.status_code == 200
            assert r_mode.json()["mode"] == "online_hybrid"

            r_state_on = client.get("/api/radar/state")
            assert r_state_on.status_code == 200
            body = r_state_on.json()
            assert body["offline_mode"] is False
            assert body["online_weather"]["current"]["temperature_c"] == 16.5
            assert len(body["regional_cloud_grid"]) == 1
            assert len(body["rainviewer"]["radar_frames"]) == 1

            # Verify hybrid forecast fused Open-Meteo signals
            r_fc = client.get("/api/forecast/latest")
            assert r_fc.status_code == 200
            fc = r_fc.json()
            assert "open-meteo" in fc["model_version"]
            assert "open_meteo_nwp_fusion" in fc["horizons"][0]["signals"]

        # Switch back to Offline Only
        r_back = client.put("/api/system/mode", json={"offline_mode": True})
        assert r_back.status_code == 200
        assert r_back.json()["offline_mode"] is True


def test_offline_map_tile_cache_parent_upscale_and_offline_lock(isolated_env):
    parent_dir = isolated_env.data_dir / "map_tiles" / "dark_base" / "10" / "546"
    parent_dir.mkdir(parents=True, exist_ok=True)
    parent_img = np.full((256, 256, 3), (220, 40, 20), dtype=np.uint8)  # Blue background in BGR
    parent_img[:128, :128] = (20, 220, 30)  # Top-left quadrant bright green
    ok, parent_jpg = cv2.imencode(".jpg", parent_img)
    assert ok
    (parent_dir / "365.jpg").write_bytes(parent_jpg.tobytes())

    with patch("app.services.map_tiles.httpx.Client", side_effect=AssertionError("Outbound HTTP forbidden while offline")):
        with TestClient(app) as client:
            # 1) Uncached tile in offline mode renders local coordinate grid tile with zero network calls
            r_grid = client.get("/api/radar/tiles/dark_base/10/100/200.png")
            assert r_grid.status_code == 200
            assert r_grid.headers["content-type"] == "image/png"
            dec_grid = cv2.imdecode(np.frombuffer(r_grid.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert dec_grid is not None and dec_grid.shape == (256, 256, 3)

            # 2) Zooming into child tile (11/1092/730) upscales the green top-left quadrant of parent (10/546/365) offline
            r_child = client.get("/api/radar/tiles/dark_base/11/1092/730.png")
            assert r_child.status_code == 200
            assert r_child.headers["content-type"] == "image/png"
            dec_child = cv2.imdecode(np.frombuffer(r_child.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert dec_child is not None and dec_child.shape == (256, 256, 3)
            assert float(np.mean(dec_child[:, :, 1])) > float(np.mean(dec_child[:, :, 0])) + 100.0

            # 3) Prefetch refuses outbound requests in offline mode
            r_pf = client.post("/api/radar/tiles/prefetch")
            assert r_pf.status_code == 403

            # 4) Radar state reports local cached tile count
            r_state = client.get("/api/radar/state")
            assert r_state.status_code == 200
            assert r_state.json()["tile_cache"]["cached_tiles"] >= 1


def test_reverse_geocode_populates_city_name_for_online_weather(isolated_env):
    from unittest.mock import MagicMock

    from app.config import get_config_manager
    from app.services.weather_online import (
        BIGDATACLOUD_REVERSE_URL,
        OPEN_METEO_FORECAST_URL,
        RAINVIEWER_MAPS_URL,
        get_online_weather_service,
    )

    mgr = get_config_manager()
    mgr.update(
        {
            "server": {"offline_mode": False},
            "camera": {
                "latitude": 41.9028,
                "longitude": 12.4964,
                "location_description": "",
            },
        }
    )
    get_online_weather_service().clear_cache()

    def _fake_get(url, params=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if BIGDATACLOUD_REVERSE_URL in url:
            resp.json.return_value = {
                "city": "Rome",
                "principalSubdivision": "Lazio",
                "countryName": "Italia",
            }
        elif OPEN_METEO_FORECAST_URL in url:
            resp.json.return_value = {
                "latitude": 41.90,
                "longitude": 12.50,
                "elevation": 21.0,
                "current": {
                    "temperature_2m": 12.5,
                    "relative_humidity_2m": 75,
                    "pressure_msl": 1016.3,
                    "wind_speed_10m": 3.5,
                    "cloud_cover": 0,
                    "precipitation": 0.0,
                    "is_day": 1,
                },
                "hourly": {},
            }
        elif RAINVIEWER_MAPS_URL in url:
            resp.json.return_value = {"radar": {"past": []}}
        else:
            resp.json.return_value = {}
        return resp

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.get.side_effect = _fake_get

    with (
        patch("app.services.weather_online.httpx.Client", return_value=mock_client),
        patch("app.services.map_tiles.prefetch_tiles_around", return_value={"cached": 0}),
    ):
        with TestClient(app) as client:
            r_state = client.get("/api/radar/state")
            assert r_state.status_code == 200
            data = r_state.json()
            assert data["online_weather"]["city"] == "Rome"
            assert data["online_weather"]["location_name"] == "Rome, Lazio"
            assert data["location"]["location_description"] == "Rome, Lazio"

            r_put = client.put(
                "/api/radar/location",
                json={"latitude": 41.9028, "longitude": 12.4964, "location_description": ""},
            )
            assert r_put.status_code == 200
            assert r_put.json()["location_description"] == "Rome, Lazio"

    get_online_weather_service().clear_cache()


def test_radar_tilt_projection_and_camera_geometry_api(isolated_env):
    from datetime import datetime, UTC

    from app.database import get_db_session
    from app.models_db import Observation

    # Save synthetic segmentation mask (full dense cloud) tied to an Observation
    mask_path = isolated_env.processed_images_dir / "seg_tilt_test.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    seg = np.full((120, 160), 2, dtype=np.uint8)
    cv2.imwrite(str(mask_path), seg)

    with get_db_session() as db:
        obs = Observation(
            captured_at=datetime.now(UTC),
            source_type="demo_simulation",
            width=160,
            height=120,
            seg_mask_path=str(mask_path),
            is_usable=True,
            cloud_coverage_pct=100.0,
            dense_cloud_pct=100.0,
            clear_sky_pct=0.0,
            dominant_cloud_type="cumulus",
            visual_condition="overcast",
        )
        db.add(obs)
        db.commit()

    with TestClient(app) as client:
        # Tilt camera to 35 deg elevation pointing East (90 deg)
        r_geom = client.put(
            "/api/camera/geometry",
            json={
                "orientation_elevation_deg": 35.0,
                "orientation_azimuth_deg": 90.0,
                "orientation_roll_deg": 0.0,
                "fov_horizontal_deg": 72.0,
                "fov_vertical_deg": 45.0,
                "assumed_cloud_altitude_m": 2000.0,
                "tilt_correction_enabled": True,
                "recalculate_latest": True,
            },
        )
        assert r_geom.status_code == 200
        g_data = r_geom.json()
        assert g_data["orientation_elevation_deg"] == 35.0
        assert g_data["orientation_azimuth_deg"] == 90.0
        assert g_data["recalculated_observation_id"] is not None

        r_state = client.get("/api/radar/state")
        assert r_state.status_code == 200
        st = r_state.json()
        proj = st["camera_projection"]
        cam_lon = st["location"]["longitude"]
        assert proj["view_center_latlon"][1] > cam_lon  # Projected sky view shifted East
        assert len(proj["fov_polygon"]) >= 12
        assert proj["center_distance_km"] > 1.5

        # Cloud overlay PNG must render non-zero cloud alpha in the Eastern half of the overlay
        r_png = client.get("/api/radar/cloud-overlay.png")
        assert r_png.status_code == 200
        img_bgra = cv2.imdecode(np.frombuffer(r_png.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert img_bgra is not None and img_bgra.shape[2] == 4
        w_mid = img_bgra.shape[1] // 2
        east_alpha = float(np.mean(img_bgra[:, w_mid:, 3]))
        west_alpha = float(np.mean(img_bgra[:, :w_mid, 3]))
        assert east_alpha > west_alpha * 2.0 + 5.0

        # Point camera straight up (zenith 90 deg) -> center distance collapses to ~0 km
        r_zen = client.put(
            "/api/camera/geometry",
            json={
                "orientation_elevation_deg": 90.0,
                "orientation_azimuth_deg": 0.0,
                "tilt_correction_enabled": True,
                "recalculate_latest": False,
            },
        )
        assert r_zen.status_code == 200
        assert r_zen.json()["camera_projection"]["center_distance_km"] < 0.05


def test_radar_local_cloud_history_timeline_and_obs_id_overlay(isolated_env):
    from datetime import datetime, timedelta, UTC

    from app.database import get_db_session
    from app.models_db import Observation

    isolated_env.processed_images_dir.mkdir(parents=True, exist_ok=True)
    m1_path = isolated_env.processed_images_dir / "hist_mask_1.png"
    m2_path = isolated_env.processed_images_dir / "hist_mask_2.png"
    m3_path = isolated_env.processed_images_dir / "hist_mask_3.png"

    # 1: 25% thin cloud, 2: 100% dense cloud, 3: 0% clear sky
    m1 = np.zeros((120, 160), dtype=np.uint8)
    m1[:30, :] = 1
    m2 = np.full((120, 160), 2, dtype=np.uint8)
    m3 = np.zeros((120, 160), dtype=np.uint8)
    cv2.imwrite(str(m1_path), m1)
    cv2.imwrite(str(m2_path), m2)
    cv2.imwrite(str(m3_path), m3)

    now = datetime.now(UTC)
    with get_db_session() as db:
        o1 = Observation(
            captured_at=now - timedelta(minutes=20),
            source_type="demo_simulation",
            width=160,
            height=120,
            seg_mask_path=str(m1_path),
            is_usable=True,
            cloud_coverage_pct=25.0,
            thin_cloud_pct=25.0,
            clear_sky_pct=75.0,
            dominant_cloud_type="cirrus",
            visual_condition="partly_cloudy",
        )
        o2 = Observation(
            captured_at=now - timedelta(minutes=10),
            source_type="demo_simulation",
            width=160,
            height=120,
            seg_mask_path=str(m2_path),
            is_usable=True,
            cloud_coverage_pct=100.0,
            dense_cloud_pct=100.0,
            clear_sky_pct=0.0,
            dominant_cloud_type="cumulus",
            visual_condition="overcast",
        )
        o3 = Observation(
            captured_at=now,
            source_type="demo_simulation",
            width=160,
            height=120,
            seg_mask_path=str(m3_path),
            is_usable=True,
            cloud_coverage_pct=0.0,
            clear_sky_pct=100.0,
            dominant_cloud_type="clear_sky",
            visual_condition="clear_sky",
        )
        db.add_all([o1, o2, o3])
        db.commit()
        id1, id2, id3 = o1.id, o2.id, o3.id

    with TestClient(app) as client:
        r_state = client.get("/api/radar/state")
        assert r_state.status_code == 200
        hist = r_state.json()["camera_history"]
        assert len(hist) == 3
        assert [h["id"] for h in hist] == [id1, id2, id3]
        assert hist[0]["time"] < hist[1]["time"] < hist[2]["time"]
        assert hist[0]["cloud_coverage_pct"] == 25.0
        assert hist[1]["cloud_coverage_pct"] == 100.0
        assert hist[2]["cloud_coverage_pct"] == 0.0

        # Historical dense cloud frame renders non-zero cloud alpha even when latest frame is clear sky
        r_png2 = client.get(f"/api/radar/cloud-overlay.png?obs_id={id2}")
        assert r_png2.status_code == 200
        assert "max-age=300" in r_png2.headers.get("cache-control", "")
        img2 = cv2.imdecode(np.frombuffer(r_png2.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert img2 is not None and img2.shape[2] == 4
        assert int(np.max(img2[:, :, 3])) >= 180
        assert float(np.mean(img2[:, :, 3])) > 0.5

        # Latest clear frame renders empty transparent alpha
        r_png3 = client.get(f"/api/radar/cloud-overlay.png?obs_id={id3}")
        assert r_png3.status_code == 200
        img3 = cv2.imdecode(np.frombuffer(r_png3.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert float(np.mean(img3[:, :, 3])) == 0.0

        # Missing observation id fails gracefully with empty transparent PNG
        r_missing = client.get("/api/radar/cloud-overlay.png?obs_id=999999")
        assert r_missing.status_code == 200
        img_missing = cv2.imdecode(np.frombuffer(r_missing.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert float(np.mean(img_missing[:, :, 3])) == 0.0

        # Solar telemetry & LCL dewpoint helper verification
        from app.services.weather_online import _estimate_dewpoint_and_lcl

        dew_c, lcl_m = _estimate_dewpoint_and_lcl(20.0, 60.0)
        assert dew_c is not None and 11.0 <= dew_c <= 13.0
        assert lcl_m is not None and 900.0 <= lcl_m <= 1150.0
        assert "solar" in r_state.json()["camera_projection"]
        assert "elevation_deg" in r_state.json()["camera_projection"]["solar"]
