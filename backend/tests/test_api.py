from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from unittest.mock import MagicMock, patch

import cv2
from fastapi.testclient import TestClient

from app.ai.datasets import _render_synthetic_sky_frame
from app.camera.manager import get_camera_manager
from app.main import app


def test_api_endpoints_and_camera_http_snapshot(isolated_env):
    # Spin up local mock Android IP-Webcam server serving a real JPEG on 127.0.0.1
    img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=1, seed=55, width=320, height=240)
    ok, jpeg_buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg_bytes = jpeg_buf.tobytes()

    class _MockCamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg_bytes)))
            self.end_headers()
            self.wfile.write(jpeg_bytes)

        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), _MockCamHandler)
    port = httpd.server_address[1]
    srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv_thread.start()

    try:
        with TestClient(app) as client:
            # Health & Status
            r_health = client.get("/api/system/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "healthy"
            assert r_health.json()["offline_mode"] is True

            r_gpu = client.get("/api/system/gpu")
            assert r_gpu.status_code == 200

            r_storage = client.get("/api/system/storage")
            assert r_storage.status_code == 200

            # Configure HTTP Snapshot camera pointing to local mock Android endpoint
            r_cfg = client.put(
                "/api/camera/config",
                json={"source_type": "http_snapshot", "source_url": f"http://127.0.0.1:{port}/shot.jpg"},
            )
            assert r_cfg.status_code == 200

            r_test = client.post("/api/camera/test")
            assert r_test.status_code == 200
            assert r_test.json()["ok"] is True

            # Trigger live snapshot capture & CV pipeline
            r_snap = client.get("/api/camera/snapshot")
            assert r_snap.status_code == 200
            obs = r_snap.json()["observation"]
            assert obs["is_usable"] is True

            # Verify camera status reflects connected
            r_cam_st = client.get("/api/camera/status")
            assert r_cam_st.status_code == 200
            assert r_cam_st.json()["connected"] is True

            # Upload local image (Method C)
            r_up = client.post(
                "/api/camera/upload",
                files={"file": ("sky.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
            )
            assert r_up.status_code == 200
            assert r_up.json()["imported_frames"] == 1

            # Verify analysis, mask overlay, motion, quality, forecast, events, dataset, models endpoints
            assert client.get("/api/analysis/latest").status_code == 200
            assert client.get("/api/analysis/history").status_code == 200
            assert client.get("/api/analysis/cloud-mask").status_code == 200
            assert client.get("/api/analysis/motion").status_code == 200
            assert client.get("/api/analysis/quality").status_code == 200
            assert client.get("/api/forecast/latest").status_code == 200
            assert client.get("/api/forecast/history").status_code == 200
            assert client.get("/api/forecast/verification").status_code == 200
            assert client.get("/api/events").status_code == 200
            assert client.get("/api/models").status_code == 200
            assert client.get("/api/models/active").status_code == 200
            assert client.get("/api/dataset").status_code == 200
            assert client.get("/api/timelapse/frames").status_code == 200

            # Test reconnection handling when camera URL stops responding
            httpd.shutdown()
            r_fail = client.post(
                "/api/camera/test",
                json={"source_type": "http_snapshot", "source_url": f"http://127.0.0.1:{port}/shot.jpg"},
            )
            assert r_fail.status_code == 200
            assert r_fail.json()["ok"] is False
            assert get_camera_manager().get_status()["connected"] is False
    finally:
        httpd.server_close()


def test_webcam_device_iriun_support(isolated_env):
    img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=2, seed=77, width=640, height=480)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, img)
    mock_cap.getBackendName.return_value = "DSHOW"
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 480.0

    with patch("app.camera.manager.cv2.VideoCapture", return_value=mock_cap):
        with TestClient(app) as client:
            r_cfg = client.put(
                "/api/camera/config",
                json={"source_type": "webcam_device", "device_index": 0, "name": "Iriun Webcam (Android)"},
            )
            assert r_cfg.status_code == 200
            assert r_cfg.json()["source_type"] == "webcam_device"

            r_devs = client.get("/api/camera/devices")
            assert r_devs.status_code == 200
            assert len(r_devs.json()["devices"]) >= 1

            r_test = client.post(
                "/api/camera/test",
                json={"source_type": "webcam_device", "device_index": 0},
            )
            assert r_test.status_code == 200
            assert r_test.json()["ok"] is True
            assert r_test.json()["width"] == 640

            r_snap = client.get("/api/camera/snapshot")
            assert r_snap.status_code == 200
            assert r_snap.json()["observation"]["source_type"] == "webcam_device"
    get_camera_manager().close()


def test_painted_exclusion_mask_api_and_regional_clouds_png(isolated_env):
    import base64

    import numpy as np

    rgba = np.zeros((120, 160, 4), dtype=np.uint8)
    rgba[90:, :, 3] = 255
    rgba[90:, :, 2] = 230
    ok, png_buf = cv2.imencode(".png", rgba)
    assert ok
    b64 = "data:image/png;base64," + base64.b64encode(png_buf.tobytes()).decode("ascii")

    with TestClient(app) as client:
        r_put = client.put("/api/camera/exclusion-mask", json={"mask_png_base64": b64})
        assert r_put.status_code == 200
        assert r_put.json()["saved"] is True
        assert r_put.json()["excluded_pct"] == 25.0

        r_get = client.get("/api/camera/exclusion-mask.png")
        assert r_get.status_code == 200
        assert r_get.headers["content-type"] == "image/png"

        r_reg = client.get("/api/radar/regional-clouds.png")
        assert r_reg.status_code == 200
        assert r_reg.headers["content-type"] == "image/png"

        r_del = client.delete("/api/camera/exclusion-mask")
        assert r_del.status_code == 200
        assert r_del.json()["cleared"] is True


def test_event_confirm_deny_undo_delete_and_sqlite_migration(isolated_env):
    import sqlite3

    from app.database import get_db_session, init_database
    from app.models_db import EventAlert, Observation

    # 1) Verify automatic SQLite schema migration on an existing legacy `events` table
    legacy_db = isolated_env.data_dir / "db" / "legacy.db"
    conn = sqlite3.connect(str(legacy_db))
    conn.execute(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME,
            event_type VARCHAR(64),
            severity VARCHAR(16),
            title VARCHAR(128),
            description TEXT,
            observation_id INTEGER,
            metrics_json TEXT,
            acknowledged BOOLEAN DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO events (event_type, severity, title, description, acknowledged) VALUES ('t1', 'info', 'Unack', '', 0)"
    )
    conn.execute(
        "INSERT INTO events (event_type, severity, title, description, acknowledged) VALUES ('t2', 'warning', 'Acked', '', 1)"
    )
    conn.commit()
    conn.close()

    migrated_engine = init_database(legacy_db)
    with migrated_engine.connect() as mconn:
        cols = {row[1] for row in mconn.exec_driver_sql("PRAGMA table_info(events)").fetchall()}
        assert "user_status" in cols
        assert "user_note" in cols
        statuses = dict(mconn.exec_driver_sql("SELECT title, user_status FROM events").fetchall())
        assert statuses["Unack"] == "pending"
        assert statuses["Acked"] == "confirmed"
    migrated_engine.dispose()

    # Restore engine to isolated_env's main test DB
    init_database(isolated_env.db_path)

    # 2) Verify full confirm / deny / feedback / filter / delete API lifecycle
    with TestClient(app) as client:
        with get_db_session() as db:
            obs = Observation(
                source_type="local_file",
                width=320,
                height=240,
                is_usable=True,
                is_event_snapshot=True,
            )
            db.add(obs)
            db.flush()
            obs_id = obs.id
            ev1 = EventAlert(
                event_type="camera_movement",
                severity="warning",
                title="Camera Movement Detected",
                description="Test shift",
                observation_id=obs_id,
            )
            ev2 = EventAlert(
                event_type="dense_cloud_region",
                severity="info",
                title="Dense Cloud Cover Detected",
                description="Test dense cloud",
            )
            db.add_all([ev1, ev2])
            db.flush()
            ev1_id = ev1.id
            ev2_id = ev2.id

        r_all = client.get("/api/events")
        assert r_all.status_code == 200
        body = r_all.json()
        assert body["summary"]["pending"] == 2
        ev1_item = next(item for item in body["items"] if item["id"] == ev1_id)
        assert ev1_item["user_status"] == "pending"
        assert ev1_item["acknowledged"] is False

        # Reject (deny as false alarm) clears obs.is_event_snapshot
        r_rej = client.post(f"/api/events/{ev1_id}/reject")
        assert r_rej.status_code == 200
        assert r_rej.json()["user_status"] == "rejected"
        assert r_rej.json()["acknowledged"] is True
        with get_db_session() as db:
            obs_row = db.query(Observation).filter(Observation.id == obs_id).first()
            assert obs_row is not None and obs_row.is_event_snapshot is False

        # Confirm via /feedback with note restores obs.is_event_snapshot
        r_fb = client.post(
            f"/api/events/{ev1_id}/feedback",
            json={"status": "confirmed", "note": "Real dark cloud"},
        )
        assert r_fb.status_code == 200
        assert r_fb.json()["user_status"] == "confirmed"
        assert r_fb.json()["user_note"] == "Real dark cloud"
        assert r_fb.json()["acknowledged"] is True
        with get_db_session() as db:
            obs_row = db.query(Observation).filter(Observation.id == obs_id).first()
            assert obs_row is not None and obs_row.is_event_snapshot is True

        # Reset back to pending
        r_rst = client.post(f"/api/events/{ev1_id}/feedback", json={"status": "pending"})
        assert r_rst.status_code == 200
        assert r_rst.json()["user_status"] == "pending"
        assert r_rst.json()["acknowledged"] is False

        # Confirm via legacy /ack endpoint + reject ev2
        r_ack = client.post(f"/api/events/{ev1_id}/ack")
        assert r_ack.status_code == 200
        assert r_ack.json()["user_status"] == "confirmed"
        assert r_ack.json()["acknowledged"] is True
        assert client.post(f"/api/events/{ev2_id}/reject").status_code == 200

        # Filtered list queries
        r_conf = client.get("/api/events?status=confirmed")
        assert r_conf.status_code == 200
        assert [x["id"] for x in r_conf.json()["items"]] == [ev1_id]

        r_denied = client.get("/api/events?status=rejected")
        assert r_denied.status_code == 200
        assert [x["id"] for x in r_denied.json()["items"]] == [ev2_id]

        # Delete ev2
        r_del_ev = client.delete(f"/api/events/{ev2_id}")
        assert r_del_ev.status_code == 200
        assert r_del_ev.json() == {"deleted": True, "id": ev2_id}
        remaining_ids = [x["id"] for x in client.get("/api/events").json()["items"]]
        assert ev2_id not in remaining_ids
        assert ev1_id in remaining_ids


def test_timelapse_clear_and_restart_preserves_training_data(isolated_env):
    from pathlib import Path

    from app.database import get_db_session
    from app.models_db import Observation

    img1, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=1, seed=101, width=320, height=240)
    img2, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=2, seed=102, width=320, height=240)
    img3, _, _, _, _ = _render_synthetic_sky_frame("clear", t_step=3, seed=103, width=320, height=240)
    _, buf1 = cv2.imencode(".jpg", img1)
    _, buf2 = cv2.imencode(".jpg", img2)
    _, buf3 = cv2.imencode(".jpg", img3)

    with TestClient(app) as client:
        r_up1 = client.post("/api/camera/upload", files={"file": ("f1.jpg", io.BytesIO(buf1.tobytes()), "image/jpeg")})
        r_up2 = client.post("/api/camera/upload", files={"file": ("f2.jpg", io.BytesIO(buf2.tobytes()), "image/jpeg")})
        assert r_up1.status_code == 200 and r_up2.status_code == 200

        r_frames = client.get("/api/timelapse/frames")
        assert r_frames.status_code == 200
        assert r_frames.json()["count"] == 2
        obs_ids = [f["observation_id"] for f in r_frames.json()["frames"]]

        with get_db_session() as db:
            first_obs = db.query(Observation).filter(Observation.id == obs_ids[0]).first()
            second_obs = db.query(Observation).filter(Observation.id == obs_ids[1]).first()
            assert first_obs is not None and second_obs is not None
            first_obs.is_training_sample = True
            first_raw = Path(first_obs.raw_path)
            second_raw = Path(second_obs.raw_path)

        dummy_mp4 = isolated_env.timelapse_dir / "timelapse_raw_test.mp4"
        dummy_mp4.write_bytes(b"fake-mp4")

        r_clr = client.post("/api/timelapse/clear", json={"delete_files": True})
        assert r_clr.status_code == 200
        clr_body = r_clr.json()
        assert clr_body["cleared"] is True
        assert clr_body["cleared_frames"] == 2
        assert clr_body["deleted_exports"] == 1
        assert not dummy_mp4.exists()

        # Timelapse frame list is now empty and export rejects with 400
        r_after = client.get("/api/timelapse/frames")
        assert r_after.status_code == 200
        assert r_after.json()["count"] == 0
        assert r_after.json()["frames"] == []
        assert client.post("/api/timelapse/export", json={"layer": "raw", "fps": 6}).status_code == 400

        # Analytics rows, training sample files, and latest live frame remain intact
        with get_db_session() as db:
            assert db.query(Observation).count() == 2
        assert first_raw.exists()
        assert second_raw.exists()

        # New capture restarts the timelapse cleanly with only the fresh observation
        r_up3 = client.post("/api/camera/upload", files={"file": ("f3.jpg", io.BytesIO(buf3.tobytes()), "image/jpeg")})
        assert r_up3.status_code == 200
        r_restarted = client.get("/api/timelapse/frames")
        assert r_restarted.status_code == 200
        restarted_frames = r_restarted.json()["frames"]
        assert len(restarted_frames) == 1
        assert restarted_frames[0]["observation_id"] not in obs_ids


def test_event_cooldown_rejection_backoff_and_bulk_clear(isolated_env):
    from datetime import datetime, timedelta, UTC

    from app.database import get_db_session
    from app.models_db import EventAlert, Observation
    from app.services.events import evaluate_events_for_observation

    t0 = datetime.now(UTC) - timedelta(minutes=25)
    t1 = t0 + timedelta(seconds=30)

    with get_db_session() as db:
        obs0 = Observation(
            captured_at=t0,
            source_type="webcam_device",
            width=640,
            height=480,
            quality_score=0.9,
            is_usable=True,
            dense_cloud_pct=82.0,
            quality_metrics_json='{"flags": ["camera_movement"], "camera_shift_px": 45.0}',
        )
        obs1 = Observation(
            captured_at=t1,
            source_type="webcam_device",
            width=640,
            height=480,
            quality_score=0.9,
            is_usable=True,
            dense_cloud_pct=85.0,
            quality_metrics_json='{"flags": ["camera_movement"], "camera_shift_px": 48.0}',
        )
        db.add_all([obs0, obs1])
        db.flush()

        ev_first = evaluate_events_for_observation(db, obs0, None, isolated_env)
        assert {e.event_type for e in ev_first} == {"camera_movement", "dense_cloud_region"}

        # 30s later within 15m cooldown -> zero duplicate events created
        ev_second = evaluate_events_for_observation(db, obs1, obs0, isolated_env)
        assert ev_second == []

        # Mark camera_movement as rejected 20m ago (>15m normal cooldown, <60m false-alarm backoff)
        cm_ev = db.query(EventAlert).filter(EventAlert.event_type == "camera_movement").first()
        assert cm_ev is not None
        cm_ev.user_status = "rejected"
        cm_ev.acknowledged = True
        cm_ev.created_at = datetime.now(UTC) - timedelta(minutes=20)
        db.commit()

        obs2 = Observation(
            captured_at=datetime.now(UTC),
            source_type="webcam_device",
            width=640,
            height=480,
            quality_score=0.9,
            is_usable=True,
            dense_cloud_pct=10.0,
            quality_metrics_json='{"flags": ["camera_movement"], "camera_shift_px": 50.0}',
        )
        db.add(obs2)
        db.flush()
        ev_third = evaluate_events_for_observation(db, obs2, obs1, isolated_env)
        assert ev_third == []

    # Test bulk clear endpoints (duplicates, rejected, all)
    with TestClient(app) as client:
        with get_db_session() as db:
            now = datetime.now(UTC)
            db.add_all([
                EventAlert(created_at=now - timedelta(minutes=3), event_type="camera_movement", severity="warning", title="Dup 1"),
                EventAlert(created_at=now - timedelta(minutes=2), event_type="camera_movement", severity="warning", title="Dup 2"),
                EventAlert(created_at=now - timedelta(minutes=1), event_type="camera_movement", severity="warning", title="Dup 3"),
            ])

        r_dup = client.post("/api/events/clear", json={"scope": "duplicates"})
        assert r_dup.status_code == 200
        assert r_dup.json()["deleted"] >= 2

        r_rej = client.post("/api/events/clear", json={"scope": "rejected"})
        assert r_rej.status_code == 200
        assert r_rej.json()["deleted"] >= 1

        r_all = client.post("/api/events/clear", json={"scope": "all"})
        assert r_all.status_code == 200
        assert client.get("/api/events").json()["count"] == 0


def test_storage_granular_purge_and_full_factory_reset(isolated_env):
    from pathlib import Path

    from app.database import get_db_session
    from app.models_db import EventAlert, Forecast, Observation

    img1, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=1, seed=201, width=320, height=240)
    img2, _, _, _, _ = _render_synthetic_sky_frame("stratus", t_step=2, seed=202, width=320, height=240)
    _, buf1 = cv2.imencode(".jpg", img1)
    _, buf2 = cv2.imencode(".jpg", img2)

    with TestClient(app) as client:
        r1 = client.post("/api/camera/upload", files={"file": ("p1.jpg", io.BytesIO(buf1.tobytes()), "image/jpeg")})
        r2 = client.post("/api/camera/upload", files={"file": ("p2.jpg", io.BytesIO(buf2.tobytes()), "image/jpeg")})
        assert r1.status_code == 200 and r2.status_code == 200

        st_before = client.get("/api/system/storage").json()
        assert st_before["db_counts"]["observations"] >= 2
        assert st_before["file_counts"]["raw_images"] >= 2

        # 1) Delete a single observation via DELETE /api/analysis/observation/{id}
        with get_db_session() as db:
            first_obs = db.query(Observation).order_by(Observation.id.asc()).first()
            assert first_obs is not None
            first_id = first_obs.id
            first_raw = Path(first_obs.raw_path)

        r_del_one = client.delete(f"/api/analysis/observation/{first_id}")
        assert r_del_one.status_code == 200
        assert r_del_one.json()["deleted"] is True
        assert not first_raw.exists()

        # 2) Purge only image files on disk while keeping Observation DB record
        r_img_only = client.post(
            "/api/system/storage/purge",
            json={"targets": ["raw_images", "processed_images"], "preserve_training_samples": False},
        )
        assert r_img_only.status_code == 200
        assert r_img_only.json()["storage"]["file_counts"]["raw_images"] == 0
        assert r_img_only.json()["storage"]["db_counts"]["observations"] >= 1

        # 3) Full monitoring & config reset while preserving camera location pin
        client.put("/api/radar/location", json={"latitude": 41.9028, "longitude": 12.4964, "location_description": "Rome"})
        r_full = client.post(
            "/api/system/storage/purge",
            json={
                "targets": [
                    "raw_images",
                    "processed_images",
                    "observations",
                    "forecasts",
                    "events",
                    "timelapse",
                    "exclusion_masks",
                    "datasets",
                    "models",
                    "reset_config",
                ],
                "preserve_training_samples": False,
                "preserve_camera_connection": True,
            },
        )
        assert r_full.status_code == 200
        st_after = r_full.json()["storage"]
        assert st_after["db_counts"]["observations"] == 0
        assert st_after["db_counts"]["forecasts"] == 0
        assert st_after["db_counts"]["events"] == 0
        with get_db_session() as db:
            assert db.query(Observation).count() == 0
            assert db.query(Forecast).count() == 0
            assert db.query(EventAlert).count() == 0
        assert client.get("/api/radar/state").json()["location"]["location_description"] == "Rome"


def test_timelapse_hud_daily_digest_analytics_export_and_multicam_360(isolated_env):
    import io
    from app.ai.datasets import _render_synthetic_sky_frame

    img1, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=1, seed=301, width=320, height=240)
    img2, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=2, seed=302, width=320, height=240)
    _, buf1 = cv2.imencode(".jpg", img1)
    _, buf2 = cv2.imencode(".jpg", img2)

    with TestClient(app) as client:
        assert client.post("/api/camera/upload", files={"file": ("d1.jpg", io.BytesIO(buf1.tobytes()), "image/jpeg")}).status_code == 200
        assert client.post("/api/camera/upload", files={"file": ("d2.jpg", io.BytesIO(buf2.tobytes()), "image/jpeg")}).status_code == 200

        # 1) Daily Sky Digest MP4 with Meteo HUD
        r_dg = client.post("/api/timelapse/daily-digest", json={"layer": "raw", "fps": 6, "include_hud": True})
        assert r_dg.status_code == 200
        dg = r_dg.json()["digest"]
        assert dg["filename"].endswith("_daily.mp4")
        r_list = client.get("/api/timelapse/digests")
        assert r_list.status_code == 200 and len(r_list.json()["digests"]) >= 1
        r_dl = client.get(dg["download_url"])
        assert r_dl.status_code == 200 and r_dl.headers["content-type"] == "video/mp4"

        # 2) Analytics CSV & JSON Export
        r_csv = client.get("/api/analytics/export?format=csv")
        assert r_csv.status_code == 200 and "cloud_coverage_pct" in r_csv.text
        r_json = client.get("/api/analytics/export?format=json")
        assert r_json.status_code == 200 and r_json.json()["observations_count"] >= 2

        # 3) Multi-Camera 360° secondary camera config, snapshot fusion & radar projections
        r_put_cam = client.put(
            "/api/cameras/secondary",
            json={
                "cameras": [
                    {
                        "id": "cam_north",
                        "name": "North Sky Cam",
                        "enabled": True,
                        "source_type": "demo_simulation",
                        "source_url": "",
                        "device_index": 1,
                        "orientation_azimuth_deg": 0.0,
                        "orientation_elevation_deg": 55.0,
                        "orientation_roll_deg": 0.0,
                        "fov_horizontal_deg": 74.0,
                        "fov_vertical_deg": 54.0,
                    }
                ]
            },
        )
        assert r_put_cam.status_code == 200
        r_snap = client.post("/api/cameras/secondary/cam_north/snapshot")
        assert r_snap.status_code == 200 and r_snap.json()["captured"] is True
        r_radar = client.get("/api/radar/state")
        assert r_radar.status_code == 200
        sec_projs = r_radar.json()["secondary_camera_projections"]
        assert len(sec_projs) == 1 and sec_projs[0]["id"] == "cam_north" and sec_projs[0]["has_mask"] is True


def test_camera_geometry_auto_detect_endpoint(isolated_env):
    import io
    import numpy as np

    # Upload a synthetic tilted skyline frame (horizon ~58% down -> small positive elevation)
    w, h = 480, 270
    frame = np.full((h, w, 3), (218, 198, 178), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    horizon_y = 0.58 * h - 0.07 * (xx - w / 2.0)
    ground = yy >= horizon_y
    frame[ground] = (52, 72, 58)
    _, buf = cv2.imencode(".jpg", frame)

    with TestClient(app) as client:
        assert client.post("/api/camera/upload", files={"file": ("horiz.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}).status_code == 200
        r_det = client.post("/api/camera/geometry/auto-detect", json={"apply": True, "recalculate_latest": True})
        assert r_det.status_code == 200
        body = r_det.json()
        assert body["detected_horizon"] is True
        assert body["applied"] is True
        assert 0.5 <= body["orientation_elevation_deg"] <= 10.0
        assert body["orientation_roll_deg"] > 1.5
        assert "geometry" in body
        assert body["geometry"]["orientation_elevation_deg"] == body["orientation_elevation_deg"]


def test_dataset_import_blocks_path_traversal(isolated_env):
    import numpy as np

    with TestClient(app) as client:
        # Traversal outside configured data_dir must be rejected
        r_bad = client.post(
            "/api/dataset/import",
            json={"name": "escape_attempt", "directory_path": "../../../etc"},
        )
        assert r_bad.status_code == 400

        # Valid directory inside configured datasets_dir succeeds
        ok_dir = isolated_env.datasets_dir / "valid_cam_ds"
        ok_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ok_dir / "sky_01.jpg"), np.full((64, 64, 3), (200, 150, 90), dtype=np.uint8))
        r_ok = client.post(
            "/api/dataset/import",
            json={"name": "valid_cam_ds", "directory_path": "valid_cam_ds"},
        )
        assert r_ok.status_code == 200
        assert r_ok.json()["item_count"] == 1

        # Absolute path strictly inside configured data_dir also succeeds
        r_abs = client.post(
            "/api/dataset/import",
            json={"name": "valid_cam_ds_abs", "directory_path": str(ok_dir)},
        )
        assert r_abs.status_code == 200
        assert r_abs.json()["item_count"] == 1
