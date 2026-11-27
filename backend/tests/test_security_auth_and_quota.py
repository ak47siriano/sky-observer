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
