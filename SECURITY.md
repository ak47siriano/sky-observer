# Security Policy

## Reporting a Vulnerability
If you discover a security vulnerability in **Sky Observer**, please report it privately via **GitHub Security Advisories** (`Security → Advisories → Report a vulnerability`) or open a minimal issue without publishing exploit details. We aim to acknowledge reports within 72 hours.

---

## Network Exposure & Deployment Guidance
Sky Observer is designed as a **local-first, self-hosted** observatory without built-in multi-user authentication:

1. **Localhost by default**: `config/config.yaml`, `.env.example`, `docker-compose.yml`, and `docker-compose.hub.yml` bind to `127.0.0.1` (`http://127.0.0.1:8080`) by default.
2. **LAN access**: If you want to view the PWA from your phone on your home Wi-Fi, set `SKYOBSERVER_HOST=0.0.0.0` in `.env` (or `--host 0.0.0.0` natively) only on a **trusted private LAN** and pair with API token authentication (and HTTPS via `docker-compose.https.yml` / Tailscale for PWA install).
3. **Do NOT expose Sky Observer directly to the public internet**: Always place it behind a VPN (e.g., Tailscale / WireGuard) or an HTTPS reverse proxy (Caddy, Nginx, Traefik) with authentication (e.g., Basic Auth, Authelia, Cloudflare Access) if remote access is required.

---

## Built-In Protections
- **API & WebSocket Token Authentication (`SKYOBSERVER_API_TOKEN`)**:
  - Set `SKYOBSERVER_API_TOKEN` in `.env` before exposing Sky Observer beyond `127.0.0.1`.
  - When set, all `/api/*` endpoints (except `/api/health`, `/api/system/health`, `/api/auth/status`) and `/ws/live` enforce constant-time token verification (`Authorization: Bearer <token>`, `X-API-Key`, or `?token=`).
  - Destructive/privileged endpoints (rollback, prune, delete model, training start, storage cleanup/purge/vacuum, camera source changes, event/timelapse clear) additionally require explicit server-side confirmation (`X-Confirm-Action: confirm` or `?confirm=true`).
- **SSRF Protection (`validate_camera_source_url`)**:
  - Camera and stream URLs allow only `http`, `https`, `rtsp`, and `rtsps` schemes with bounded HTTP response size (`<= 16 MiB`) and clamped timeouts (`<= 30 s`).
  - Hostnames are DNS-resolved before fetching; link-local / cloud metadata (`169.254.0.0/16`, `fe80::/10`), multicast, and unspecified (`0.0.0.0`) addresses are unconditionally blocked.
  - `web_stream` blocks private RFC1918 and loopback addresses by default (`SKYOBS_ALLOW_PRIVATE_SOURCES=false` for `web_stream`, `true` for local LAN camera types `http_snapshot`, `mjpeg_stream`, `rtsp`), with optional explicit `SKYOBSERVER_SOURCE_ALLOWLIST` restriction.
- **Camera Credential Masking**:
  - Any `user:pass@` embedded in RTSP/HTTP camera URLs is automatically masked (`***:***@`) in API responses (`/api/camera/config`, `/api/camera/status`, `/api/system/config`, `/api/cameras/secondary`) and error strings while preserving stored credentials across settings edits.
- **Dataset Path Containment**:
  - `POST /api/dataset/import` canonicalizes paths with `os.path.normpath` + `os.path.realpath` and restricts imports strictly inside the configured `data/` directory.
- **CORS Hardening**:
  - Wildcard `*` CORS origins are rejected; configure allowed origins via `SKYOBS_CORS_ORIGINS` (comma-separated).
