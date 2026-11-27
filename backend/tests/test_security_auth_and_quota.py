from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.camera.manager import validate_camera_source_url
from app.main import app


def test_api_token_auth_http_ws_and_destructive_confirmation(isolated_env, monkeypatch: pytest.MonkeyPatch):
    secret = "test-super-secret-token-42"
    monkeypatch.setenv("SKYOBSERVER_API_TOKEN", secret)

    with TestClient(app) as client:
        # Health & auth status remain public
        assert client.get("/api/health").status_code == 200
        st_unauth = client.get("/api/auth/status").json()
        assert st_unauth == {"auth_required": True, "authenticated": False}

        # Regular API endpoint rejects missing or wrong token
        assert client.get("/api/system/status").status_code == 401
        assert (
            client.get("/api/system/status", headers={"Authorization": "Bearer wrong-token"}).status_code
            == 401
        )

        # Valid bearer token grants read access
        auth_headers = {"Authorization": f"Bearer {secret}"}
        assert client.get("/api/system/status", headers=auth_headers).status_code == 200
        assert client.get("/api/auth/status", headers=auth_headers).json()["authenticated"] is True

        # Destructive endpoint requires explicit confirmation header/query (428 without it)
        r_no_confirm = client.post("/api/system/storage/vacuum", headers=auth_headers)
        assert r_no_confirm.status_code == 428

        # With X-Confirm-Action: confirm -> succeeds
        r_confirmed = client.post(
            "/api/system/storage/vacuum",
            headers={**auth_headers, "X-Confirm-Action": "confirm"},
        )
        assert r_confirmed.status_code == 200
        assert r_confirmed.json()["vacuumed"] is True

        # WebSocket rejects unauthorized client and accepts valid ?token=
        with pytest.raises(WebSocketDisconnect) as ws_exc:
            with client.websocket_connect("/ws/live"):
                pass
        assert ws_exc.value.code == 1008

        with client.websocket_connect(f"/ws/live?token={secret}") as ws:
            assert ws is not None


def test_ssrf_source_allowlist_and_metadata_block(monkeypatch: pytest.MonkeyPatch):
    # Cloud metadata IP must ALWAYS be blocked even if in allowlist
    monkeypatch.setenv("SKYOBSERVER_SOURCE_ALLOWLIST", "169.254.169.254,192.168.1.50")
    with pytest.raises(ValueError, match="blocked"):
        validate_camera_source_url("http://169.254.169.254/latest/meta-data/", source_type="web_stream")

    # Explicitly allowlisted LAN IP is accepted even on web_stream
    ok = validate_camera_source_url("http://192.168.1.50:8080/shot.jpg", source_type="web_stream")
    assert ok == "http://192.168.1.50:8080/shot.jpg"

    # Non-allowlisted IP is rejected when allowlist is active
    with pytest.raises(ValueError, match="ALLOWLIST"):
        validate_camera_source_url("http://192.168.1.99:8080/shot.jpg", source_type="http_snapshot")


def test_rainviewer_past_only_attribution_and_librewxr_provider(isolated_env):
    from unittest.mock import MagicMock, patch

    from app.services.weather_online import get_online_weather_service

    svc = get_online_weather_service()
    svc.clear_cache()
    isolated_env.server.offline_mode = False

    # 1) RainViewer returns past frames only (ignoring discontinued nowcast/satellite) with attribution_url & max_zoom=7
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "host": "https://tilecache.rainviewer.com",
        "generated": 1795727123,
        "radar": {
            "past": [{"time": 1795726800, "path": "/v2/radar/abc123hash"}],
            "nowcast": [{"time": 1795727400, "path": "/v2/radar/legacy_nowcast"}],
        },
        "satellite": {"infrared": [{"time": 1795726800, "path": "/v2/satellite/legacy_ir"}]},
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.get.return_value = mock_resp

    with patch("app.services.weather_online.httpx.Client", return_value=mock_client):
        rv = svc.get_radar_layers(isolated_env)
    assert rv is not None
    assert rv["max_zoom"] == 7
    assert rv["attribution_url"].startswith("https://www.rainviewer.com")
    assert len(rv["radar_frames"]) == 1
    assert rv["radar_frames"][0]["kind"] == "radar_past"
    assert "abc123hash" in rv["radar_frames"][0]["tile_url"]
    assert rv["satellite_frames"] == []

    # 2) Self-hosted LibreWXR provider override works with zero RainViewer requests
    isolated_env.online_weather.radar_provider = "librewxr"
    isolated_env.online_weather.librewxr_tile_url = "https://radar.local/tiles/{z}/{x}/{y}.png"
    lw = svc.get_radar_layers(isolated_env)
    assert lw is not None
    assert "LibreWXR" in lw["provider"]
    assert lw["radar_frames"][0]["tile_url"] == "https://radar.local/tiles/{z}/{x}/{y}.png"


def test_pmtiles_range_server_and_osm_bulk_prefetch_policy_guard(isolated_env):
    from app.services.map_tiles import prefetch_tiles_around

    pm_dir = isolated_env.data_dir / "map_tiles"
    pm_dir.mkdir(parents=True, exist_ok=True)
    sample = b"PMTiles\x03" + b"0123456789ABCDEF" * 64
    (pm_dir / "region.pmtiles").write_bytes(sample)

    # Online prefetch must NOT scrape OSM/Esri and must report bulk_prefetch_disabled
    isolated_env.server.offline_mode = False
    pf = prefetch_tiles_around(isolated_env, 41.9, 12.5)
    assert pf["bulk_prefetch_disabled"] is True
    assert pf["cached"] == 0
    assert len(pf["tile_cache"]["pmtiles_archives"]) == 1

    with TestClient(app) as client:
        r_list = client.get("/api/map/pmtiles")
        assert r_list.status_code == 200
        assert r_list.json()["archives"][0]["filename"] == "region.pmtiles"

        # HTTP Range request returns 206 Partial Content
        r_range = client.get("/api/map/pmtiles/region.pmtiles", headers={"Range": "bytes=0-15"})
        assert r_range.status_code == 206
        assert r_range.content == sample[:16]
        assert r_range.headers["content-range"] == f"bytes 0-15/{len(sample)}"

        # Traversal attempt is blocked
        assert client.get("/api/map/pmtiles/..%2Fsecret.pmtiles").status_code in {400, 404}


def test_storage_retention_enforces_max_storage_gb_quota_preserving_training_samples(isolated_env):
    from datetime import datetime, timedelta, UTC

    from app.database import get_db_session
    from app.models_db import Observation
    from app.services.storage import apply_retention_policy

    isolated_env.raw_images_dir.mkdir(parents=True, exist_ok=True)
    # Set tiny 1 MiB quota (~0.001 GB) and create 3 x 700 KiB raw files (1 protected training sample, 2 normal)
    isolated_env.storage.max_storage_gb = 0.001
    isolated_env.storage.raw_retention_days = 365  # Ensure day-retention does NOT trigger, only GB quota!

    f_old = isolated_env.raw_images_dir / "old_normal.jpg"
    f_train = isolated_env.raw_images_dir / "protected_train.jpg"
    f_new = isolated_env.raw_images_dir / "new_normal.jpg"
    blob = b"\xff\xd8" + b"X" * (700 * 1024) + b"\xff\xd9"
    f_old.write_bytes(blob)
    f_train.write_bytes(blob)
    f_new.write_bytes(blob)

    now = datetime.now(UTC)
    with get_db_session() as db:
        db.add_all([
            Observation(
                captured_at=now - timedelta(hours=3),
                source_type="demo_simulation",
                raw_path=str(f_old),
                is_training_sample=False,
            ),
            Observation(
                captured_at=now - timedelta(hours=2),
                source_type="demo_simulation",
                raw_path=str(f_train),
                is_training_sample=True,
            ),
            Observation(
                captured_at=now - timedelta(hours=1),
                source_type="demo_simulation",
                raw_path=str(f_new),
                is_training_sample=False,
            ),
        ])
        db.commit()

        res = apply_retention_policy(db, isolated_env)
        assert res["quota_enforced"] is True
        assert res["quota_deleted_files"] >= 2
        assert not f_old.exists()
        assert not f_new.exists()
        assert f_train.exists()  # Training sample MUST survive quota enforcement!


def test_safe_http_get_handles_gzip_without_double_decompression():
    import gzip
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    from app.camera.manager import CameraManager

    payload_text = "<html><meta property='og:image' content='https://example.com/live.jpg' /></html>"
    gz_body = gzip.compress(payload_text.encode("utf-8"))

    class GzipHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(gz_body)))
            self.end_headers()
            self.wfile.write(gz_body)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), GzipHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        cm = CameraManager()
        resp = cm._safe_http_get(f"http://127.0.0.1:{port}/webcam.html", timeout=5.0, source_type="http_snapshot")
        assert resp.status_code == 200
        assert "og:image" in resp.text
        assert "content-encoding" not in {k.lower() for k in resp.headers.keys()}
    finally:
        server.shutdown()
        server.server_close()
