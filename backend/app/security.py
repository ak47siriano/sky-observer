"""API Token Authentication (HTTP + WebSocket) and Server-Side Destructive Action Confirmation."""

from __future__ import annotations

import hmac
import os
from collections.abc import Mapping

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

PUBLIC_API_PATHS = {
    "/api/health",
    "/api/system/health",
    "/api/auth/status",
}

# Destructive or privileged state-mutating endpoints that require explicit server-side confirmation
# (X-Confirm-Action: confirm header or ?confirm=true) whenever SKYOBSERVER_API_TOKEN is enabled.
DESTRUCTIVE_EXACT_ROUTES: set[tuple[str, str]] = {
    ("POST", "/api/models/rollback"),
    ("POST", "/api/models/prune"),
    ("POST", "/api/models/train"),
    ("POST", "/api/models/auto-train/run"),
    ("DELETE", "/api/models/training/sessions"),
    ("POST", "/api/system/storage/cleanup"),
    ("POST", "/api/system/storage/purge"),
    ("POST", "/api/system/storage/vacuum"),
    ("PUT", "/api/camera/config"),
    ("PUT", "/api/cameras/secondary"),
    ("POST", "/api/events/clear"),
    ("POST", "/api/timelapse/clear"),
}

DESTRUCTIVE_PREFIX_ROUTES: tuple[tuple[str, str], ...] = (
    ("DELETE", "/api/models/"),
    ("DELETE", "/api/analysis/observation/"),
)


def get_configured_api_token() -> str:
    """Return configured SKYOBSERVER_API_TOKEN (empty string disables auth for localhost-only setups)."""
    return (os.environ.get("SKYOBSERVER_API_TOKEN") or "").strip()


def extract_token_from_request_parts(
    headers: Mapping[str, str],
    query_params: Mapping[str, str],
) -> str:
    """Extract bearer/API token from Authorization header, X-API-Key header, or ?token= query parameter."""
    auth_hdr = (headers.get("authorization") or headers.get("Authorization") or "").strip()
    if auth_hdr.lower().startswith("bearer "):
        return auth_hdr[7:].strip()
    x_key = (headers.get("x-api-key") or headers.get("X-API-Key") or "").strip()
    if x_key:
        return x_key
    return (query_params.get("token") or "").strip()


def is_valid_api_token(presented: str, expected: str) -> bool:
    if not expected:
        return True
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def is_destructive_route(method: str, path: str) -> bool:
    m = method.upper()
    if (m, path) in DESTRUCTIVE_EXACT_ROUTES:
        return True
    for dm, prefix in DESTRUCTIVE_PREFIX_ROUTES:
        if m == dm and path.startswith(prefix):
            return True
    return False


def has_destructive_confirmation(
    headers: Mapping[str, str],
    query_params: Mapping[str, str],
) -> bool:
    hdr = (headers.get("x-confirm-action") or headers.get("X-Confirm-Action") or "").strip().lower()
    if hdr in {"confirm", "confirmed", "true", "1", "yes"}:
        return True
    qp = (query_params.get("confirm") or "").strip().lower()
    return qp in {"true", "1", "yes", "confirm"}


class ApiTokenAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces SKYOBSERVER_API_TOKEN on all /api/* routes (except health & auth-status probes)
    and requires server-side confirmation headers on destructive routes when auth is enabled.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        is_protected = path.startswith("/api/") or path == "/metrics"
        if not is_protected or path in PUBLIC_API_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        expected = get_configured_api_token()
        if not expected:
            return await call_next(request)

        presented = extract_token_from_request_parts(request.headers, request.query_params)
        if not is_valid_api_token(presented, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "API token required (set Authorization: Bearer <token> or X-API-Key)."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        if is_destructive_route(request.method, path) and not has_destructive_confirmation(
            request.headers, request.query_params
        ):
            return JSONResponse(
                status_code=428,
                content={
                    "detail": (
                        "Destructive or privileged action requires server-side confirmation "
                        "(header 'X-Confirm-Action: confirm' or query '?confirm=true')."
                    )
                },
            )

        return await call_next(request)
