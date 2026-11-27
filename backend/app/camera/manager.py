"""Android Camera Capture Manager with connection testing, reconnection, RTSP, HTTP/MJPEG, and local uploads."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from threading import RLock
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import cv2
import httpx
import numpy as np

from app.ai.datasets import _render_synthetic_sky_frame
from app.config import CameraConfig, get_config_manager

_URL_CREDS_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/@\s]+)@")
_ALLOWED_CAMERA_SCHEMES: dict[str, set[str]] = {
    "http_snapshot": {"http", "https"},
    "mjpeg_stream": {"http", "https"},
    "web_stream": {"http", "https"},
    "rtsp": {"rtsp", "rtsps", "http", "https"},
}
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024  # 16 MiB cap on HTTP camera/web_stream fetches
MAX_HTTP_TIMEOUT_S = 30.0


def _get_source_allowlist() -> list[str]:
    raw = (os.environ.get("SKYOBSERVER_SOURCE_ALLOWLIST") or "").strip()
    if not raw:
        return []
    return [entry.strip().lower() for entry in raw.split(",") if entry.strip()]


def _host_matches_allowlist(
    host: str,
    resolved_ips: set[str],
    allowlist: list[str],
) -> bool:
    for entry in allowlist:
        if entry == host or (entry.startswith(".") and host.endswith(entry)) or host.endswith(f".{entry}"):
            return True
        try:
            net = ipaddress.ip_network(entry, strict=False)
            for ip_s in resolved_ips:
                if ipaddress.ip_address(ip_s) in net:
                    return True
        except ValueError:
            continue
    return False


def mask_url_credentials(url: str) -> str:
    """Replace embedded username:password@ in camera URLs with ***:***@ for safe API/log output."""
    if not url or "@" not in url:
        return url
    try:
        parts = urlsplit(url)
        if parts.username is None and parts.password is None:
            return url
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port_part = f":{parts.port}" if parts.port is not None else ""
        cred_part = "***:***@" if parts.password is not None else "***@"
        return urlunsplit((parts.scheme, f"{cred_part}{host}{port_part}", parts.path, parts.query, parts.fragment))
    except Exception:
        return _URL_CREDS_RE.sub(r"\1***:***@", url)


def sanitize_error_credentials(message: str) -> str:
    """Strip any embedded URL credentials from arbitrary error strings."""
    if not message:
        return message
    return _URL_CREDS_RE.sub(r"\1***:***@", str(message))


def _resolve_allow_private_sources(source_type: str, explicit_override: bool | None = None) -> bool:
    if explicit_override is not None:
        return explicit_override
    env_val = os.environ.get("SKYOBS_ALLOW_PRIVATE_SOURCES")
    if env_val is not None and env_val.strip() != "":
        return env_val.strip().lower() in {"1", "true", "yes"}
    # Local LAN cameras are private by design; public web_stream defaults to public-only (SSRF protection)
    return source_type in {"http_snapshot", "mjpeg_stream", "rtsp"}


def _check_ip_allowed(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, allow_private: bool) -> None:
    # Always block link-local / cloud metadata (169.254.169.254, fe80::), unspecified (0.0.0.0), multicast, reserved
    if ip_obj.is_link_local or ip_obj.is_unspecified or ip_obj.is_multicast or ip_obj.is_reserved:
        raise ValueError("URL points to a blocked link-local, metadata, or reserved IP address.")
    if not allow_private and (ip_obj.is_private or ip_obj.is_loopback):
        raise ValueError(
            "Private or loopback addresses are blocked for this source type "
            "(set SKYOBS_ALLOW_PRIVATE_SOURCES=true to opt in explicitly)."
        )


def validate_camera_source_url(
    url: str,
    source_type: str = "web_stream",
    allow_private: bool | None = None,
) -> str:
    """
    Validate camera/stream URL scheme and resolved IP addresses against SSRF.
    Allows http/https/rtsp/rtsps only; resolves DNS and blocks link-local/metadata (169.254.169.254)
    unconditionally, plus private/loopback IPs on web_stream unless SKYOBS_ALLOW_PRIVATE_SOURCES=true.
    """
    clean_url = (url or "").strip()
    if not clean_url:
        raise ValueError("Camera source_url is empty.")

    parts = urlsplit(clean_url)
    scheme = (parts.scheme or "").lower()
    allowed_schemes = _ALLOWED_CAMERA_SCHEMES.get(source_type, {"http", "https", "rtsp", "rtsps"})
    if scheme not in allowed_schemes:
        raise ValueError(f"Unsupported URL scheme '{scheme or 'none'}' for source type '{source_type}'.")

    host = (parts.hostname or "").strip().lower()
    if not host:
        raise ValueError("Camera source_url is missing a valid hostname.")

    allow_priv = _resolve_allow_private_sources(source_type, allow_private)
    allowlist = _get_source_allowlist()
    if not allow_priv and not allowlist and host in {"localhost", "localhost.localdomain"}:
        raise ValueError("Loopback host 'localhost' is blocked for web_stream sources.")

    try:
        ip_direct = ipaddress.ip_address(host)
    except ValueError:
        ip_direct = None

    if ip_direct is not None:
        resolved_ips = {str(ip_direct)}
        explicitly_allowed = bool(allowlist and _host_matches_allowlist(host, resolved_ips, allowlist))
        if allowlist and not explicitly_allowed:
            raise ValueError(f"Camera host '{host}' is not in SKYOBSERVER_SOURCE_ALLOWLIST.")
        _check_ip_allowed(ip_direct, allow_priv or explicitly_allowed)
        return clean_url

    try:
        port = parts.port or (443 if scheme in {"https", "rtsps"} else 80)
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve camera host '{host}'") from exc

    resolved_ips = set()
    for info in infos:
        sockaddr = info[4]
        if sockaddr and sockaddr[0]:
            resolved_ips.add(str(sockaddr[0]).split("%", 1)[0])
    if not resolved_ips:
        raise ValueError(f"No IP addresses resolved for host '{host}'")

    explicitly_allowed = bool(allowlist and _host_matches_allowlist(host, resolved_ips, allowlist))
    if allowlist and not explicitly_allowed:
        raise ValueError(f"Camera host '{host}' is not in SKYOBSERVER_SOURCE_ALLOWLIST.")
    if not (allow_priv or explicitly_allowed) and host in {"localhost", "localhost.localdomain"}:
        raise ValueError("Loopback host 'localhost' is blocked for web_stream sources.")

    for ip_str in resolved_ips:
        _check_ip_allowed(ipaddress.ip_address(ip_str), allow_priv or explicitly_allowed)
    return clean_url


@dataclass
class CaptureResult:
    success: bool
    image_bgr: np.ndarray | None
    captured_at: datetime
    latency_ms: float
    source_type: str
    error: str = ""


class CameraManager:
    """Manages local Android IP-camera connections without relying on a USB webcam or cloud APIs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._rtsp_cap: cv2.VideoCapture | None = None
        self._rtsp_url: str = ""
        self._device_cap: cv2.VideoCapture | None = None
        self._device_index: int = -1
        self._connected: bool = False
        self._last_capture_at: datetime | None = None
        self._last_error: str = ""
        self._last_latency_ms: float = 0.0
        self._consecutive_failures: int = 0
        self._reconnect_attempts: int = 0
        self._next_retry_ts: float = 0.0
        self._demo_step: int = 0
        self._web_stream_cache: dict[str, tuple[float, str | None, str | None]] = {}

    def get_status(self) -> dict[str, Any]:
        cfg = get_config_manager().get().camera
        with self._lock:
            return {
                "connected": self._connected,
                "source_type": cfg.source_type,
                "source_url": mask_url_credentials(cfg.source_url),
                "device_index": cfg.device_index,
                "camera_name": cfg.name,
                "location_description": cfg.location_description,
                "is_demo_mode": cfg.source_type == "demo_simulation",
                "last_capture_at": self._last_capture_at.isoformat() if self._last_capture_at else None,
                "last_latency_ms": round(self._last_latency_ms, 2),
                "last_error": sanitize_error_credentials(self._last_error),
                "consecutive_failures": self._consecutive_failures,
                "reconnect_attempts": self._reconnect_attempts,
            }

    def close(self) -> None:
        with self._lock:
            if self._rtsp_cap is not None:
                try:
                    self._rtsp_cap.release()
                except Exception:
                    pass
                self._rtsp_cap = None
                self._rtsp_url = ""
            if self._device_cap is not None:
                try:
                    self._device_cap.release()
                except Exception:
                    pass
                self._device_cap = None
                self._device_index = -1

    def _decode_jpeg_bytes(self, content: bytes) -> np.ndarray | None:
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _safe_http_get(
        self,
        url: str,
        timeout: float,
        source_type: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_SNAPSHOT_BYTES,
    ) -> httpx.Response:
        current_url = validate_camera_source_url(url, source_type=source_type)
        clamped_timeout = max(1.0, min(float(timeout), MAX_HTTP_TIMEOUT_S))
        with httpx.Client(timeout=clamped_timeout, follow_redirects=False, headers=headers) as client:
            for _ in range(4):
                with client.stream("GET", current_url) as resp:
                    if resp.status_code in {301, 302, 303, 307, 308} and resp.headers.get("location"):
                        next_url = urljoin(current_url, resp.headers["location"])
                        current_url = validate_camera_source_url(next_url, source_type=source_type)
                        continue
                    resp.raise_for_status()
                    cl_hdr = resp.headers.get("content-length")
                    if cl_hdr and cl_hdr.isdigit() and int(cl_hdr) > max_bytes:
                        raise ValueError(f"Remote response exceeds maximum allowed size ({max_bytes} bytes).")
                    buf = bytearray()
                    for chunk in resp.iter_bytes(chunk_size=32768):
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            raise ValueError(f"Remote response exceeded maximum allowed size ({max_bytes} bytes).")
                    return httpx.Response(
                        status_code=resp.status_code,
                        headers=resp.headers,
                        content=bytes(buf),
                        request=resp.request,
                    )
        raise ConnectionError("Too many HTTP redirects while fetching camera frame.")

    def _capture_http_snapshot(self, url: str, timeout: float, source_type: str = "http_snapshot") -> np.ndarray:
        if not url:
            raise ValueError("Camera source_url is empty. Configure your Android IP Webcam URL (e.g., http://192.168.1.50:8080/shot.jpg).")
        resp = self._safe_http_get(url, timeout=timeout, source_type=source_type)
        ctype = resp.headers.get("content-type", "").lower()
        if "multipart" in ctype or "x-mixed-replace" in ctype or b"--" in resp.content[:64]:
            img = self._extract_first_mjpeg_frame(resp.content)
        else:
            img = self._decode_jpeg_bytes(resp.content)
        if img is None:
            raise ValueError("Failed to decode JPEG image from HTTP camera response")
        return img

    def _extract_first_mjpeg_frame(self, raw: bytes) -> np.ndarray | None:
        start = raw.find(b"\xff\xd8")
        end = raw.find(b"\xff\xd9", start + 2) if start != -1 else -1
        if start != -1 and end != -1:
            return self._decode_jpeg_bytes(raw[start : end + 2])
        return None

    def _capture_mjpeg_stream(self, url: str, timeout: float) -> np.ndarray:
        if not url:
            raise ValueError("MJPEG stream source_url is empty.")
        safe_url = validate_camera_source_url(url, source_type="mjpeg_stream")
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", safe_url) as resp:
                resp.raise_for_status()
                buf = bytearray()
                t_start = time.perf_counter()
                for chunk in resp.iter_bytes(chunk_size=16384):
                    buf.extend(chunk)
                    start = buf.find(b"\xff\xd8")
                    if start != -1:
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end != -1:
                            img = self._decode_jpeg_bytes(bytes(buf[start : end + 2]))
                            if img is not None:
                                return img
                    if time.perf_counter() - t_start > timeout or len(buf) > 8 * 1024 * 1024:
                        break
        raise TimeoutError("Timed out reading JPEG frame from MJPEG stream")

    def _capture_rtsp(self, url: str, frame_sampling: int = 1) -> np.ndarray:
        if not url:
            raise ValueError("RTSP source_url is empty.")
        with self._lock:
            if self._rtsp_cap is None or self._rtsp_url != url or not self._rtsp_cap.isOpened():
                self.close()
                safe_rtsp = validate_camera_source_url(url, source_type="rtsp")
                cap = cv2.VideoCapture(safe_rtsp, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    cap.release()
                    raise ConnectionError(f"Unable to open RTSP stream: {mask_url_credentials(url)}")
                self._rtsp_cap = cap
                self._rtsp_url = url

            frame = None
            for _ in range(max(1, frame_sampling)):
                ok, frm = self._rtsp_cap.read()
                if ok and frm is not None:
                    frame = frm
            if frame is None:
                self.close()
                raise ConnectionError("RTSP stream returned empty frame; connection dropped")
            return frame

    def _is_private_url(self, url: str) -> bool:
        try:
            host = (urlparse(url).hostname or "").strip().lower()
            if host in {"localhost", "127.0.0.1", "::1"}:
                return True
            ip = ipaddress.ip_address(host)
            return ip.is_private or ip.is_loopback
        except ValueError:
            return False

    def _resolve_web_stream_target(self, url: str, timeout: float) -> tuple[str | None, str | None]:
        """
        Resolve a public webcam page URL (e.g., SkylineWebcams) or direct HLS (.m3u8) URL
        into (hls_stream_url, fallback_snapshot_jpeg_url). Caches page resolution for 180s.
        """
        clean_url = validate_camera_source_url(url, source_type="web_stream")

        if ".m3u8" in clean_url.lower():
            return clean_url, None

        now = time.time()
        cached = self._web_stream_cache.get(clean_url)
        if cached and (now - cached[0]) < 180.0:
            return cached[1], cached[2]

        resp = self._safe_http_get(
            clean_url,
            timeout=timeout,
            source_type="web_stream",
            headers={"User-Agent": "SkyObserver/1.0"},
        )
        ctype = resp.headers.get("content-type", "").lower()
        if ctype.startswith("image/"):
            self._web_stream_cache[clean_url] = (now, None, clean_url)
            return None, clean_url
        html = resp.text

        hls_url: str | None = None
        # 1) Direct absolute .m3u8 URL embedded in page HTML
        abs_matches = re.findall(r'https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*', html)
        if abs_matches:
            hls_url = abs_matches[0].replace("\\/", "/")
        else:
            # 2) SkylineWebcams relative HLS playlist pattern: livee.m3u8?a=... or live.m3u8?a=...
            sky_matches = re.findall(r'livee?\.m3u8\?a=[A-Za-z0-9_-]+', html)
            if sky_matches:
                token_path = sky_matches[0].replace("livee.m3u8", "live.m3u8")
                hls_url = f"https://hd-auth.skylinewebcams.com/{token_path}"

        # 3) Live preview JPEG fallback (og:image)
        og_matches = re.findall(r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html)
        snap_url = og_matches[0] if og_matches else None

        if not hls_url and not snap_url:
            raise ValueError("Could not find a public HLS (.m3u8) stream or live JPEG snapshot on this web page.")

        if hls_url:
            hls_url = validate_camera_source_url(hls_url, source_type="web_stream")
        if snap_url:
            snap_url = validate_camera_source_url(snap_url, source_type="web_stream")

        self._web_stream_cache[clean_url] = (now, hls_url, snap_url)
        return hls_url, snap_url

    def _capture_web_stream(self, url: str, timeout: float, frame_sampling: int = 1) -> np.ndarray:
        app_cfg = get_config_manager().get()
        if app_cfg.server.offline_mode and not self._is_private_url(url):
            raise PermissionError(
                "Offline Only mode is active. Switch to Online Hybrid mode to capture from public web streams."
            )

        hls_url, snap_url = self._resolve_web_stream_target(url, timeout)
        if hls_url:
            cap = cv2.VideoCapture(hls_url, cv2.CAP_FFMPEG)
            try:
                if cap.isOpened():
                    frame = None
                    max_reads = max(6, frame_sampling)
                    for i in range(max_reads):
                        ok, frm = cap.read()
                        if ok and frm is not None:
                            frame = frm
                            # Skip initial black H.264 decoder warmup frames when possible
                            if i >= max(1, frame_sampling - 1) and float(frm.mean()) >= 3.0:
                                break
                    if frame is not None:
                        return frame
            finally:
                cap.release()
            # Invalidate cache if HLS token expired so next attempt refreshes
            self._web_stream_cache.pop(url.strip(), None)

        if snap_url:
            return self._capture_http_snapshot(snap_url, timeout, source_type="web_stream")

        raise ConnectionError(f"Unable to read live frame from web stream: {mask_url_credentials(url)}")

    def _open_device_capture(self, device_index: int) -> cv2.VideoCapture:
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_V4L2, cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(device_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        raise ConnectionError(f"Unable to open local video camera device index {device_index}")

    def _capture_webcam_device(
        self,
        device_index: int,
        width: int,
        height: int,
        frame_sampling: int = 1,
    ) -> np.ndarray:
        with self._lock:
            newly_opened = False
            if self._device_cap is None or self._device_index != device_index or not self._device_cap.isOpened():
                if self._device_cap is not None:
                    try:
                        self._device_cap.release()
                    except Exception:
                        pass
                cap = self._open_device_capture(device_index)
                if width > 0 and height > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                self._device_cap = cap
                self._device_index = device_index
                newly_opened = True

            frame = None
            reads = max(6 if newly_opened else 2, frame_sampling)
            for _ in range(reads):
                ok, frm = self._device_cap.read()
                if ok and frm is not None:
                    frame = frm
                if newly_opened:
                    time.sleep(0.03)

            if frame is None:
                self.close()
                raise ConnectionError(f"Local webcam device {device_index} returned empty frame")
            return frame

    def _capture_allsky_folder(self, folder_or_file: str) -> np.ndarray:
        """Read the latest JPEG/PNG frame from a local AllSky (thomasjacquin/allsky or AllSkEye) output path."""
        from pathlib import Path

        raw = (folder_or_file or "").strip()
        if not raw:
            raise ValueError("AllSky folder path is empty.")
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            target = p
        elif p.is_dir():
            cands = [
                f
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if not cands:
                raise FileNotFoundError(f"No JPEG/PNG images found in AllSky folder: {p}")
            target = max(cands, key=lambda f: f.stat().st_mtime)
        else:
            raise FileNotFoundError(f"AllSky path does not exist: {raw}")
        img = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError(f"Unable to decode AllSky image: {target.name}")
        return img

    def list_local_video_devices(self, max_index: int = 5) -> list[dict[str, Any]]:
        """Probe available local webcam / virtual camera devices (e.g., Iriun Webcam)."""
        devices: list[dict[str, Any]] = []
        with self._lock:
            active_idx = self._device_index if (self._device_cap is not None and self._device_cap.isOpened()) else -1
            for idx in range(max_index + 1):
                if idx == active_idx and self._device_cap is not None:
                    w = int(self._device_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(self._device_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    devices.append({
                        "index": idx,
                        "opened": True,
                        "active_in_use": True,
                        "width": w,
                        "height": h,
                        "backend": self._device_cap.getBackendName(),
                    })
                    continue

                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
                if cap.isOpened():
                    ok, frm = cap.read()
                    w = int(frm.shape[1]) if (ok and frm is not None) else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(frm.shape[0]) if (ok and frm is not None) else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    devices.append({
                        "index": idx,
                        "opened": True,
                        "active_in_use": False,
                        "width": w,
                        "height": h,
                        "backend": cap.getBackendName(),
                    })
                cap.release()
        return devices

    def capture_once(self, cfg_override: CameraConfig | None = None, bypass_backoff: bool = False) -> CaptureResult:
        cfg = cfg_override or get_config_manager().get().camera
        now_ts = time.time()

        with self._lock:
            if not bypass_backoff and now_ts < self._next_retry_ts:
                return CaptureResult(
                    success=False,
                    image_bgr=None,
                    captured_at=datetime.now(UTC),
                    latency_ms=0.0,
                    source_type=cfg.source_type,
                    error=self._last_error or "Waiting for reconnect backoff",
                )

        t0 = time.perf_counter()
        try:
            if cfg.source_type == "local_import":
                raise ValueError("Camera is in local_import mode; upload local images or videos via dashboard/API.")
            elif cfg.source_type == "demo_simulation":
                # Explicitly labeled synthetic local test feed (never enabled by default)
                with self._lock:
                    step = self._demo_step
                    self._demo_step += 1
                scenarios = ["cumulus", "stratocumulus", "cumulonimbus_like", "clearing"]
                sc = scenarios[(step // 6) % len(scenarios)]
                img, _, _, _, _ = _render_synthetic_sky_frame(sc, step % 6, seed=900 + step, width=960, height=540)
            elif cfg.source_type == "http_snapshot":
                img = self._capture_http_snapshot(cfg.source_url, cfg.timeout_seconds)
            elif cfg.source_type == "mjpeg_stream":
                img = self._capture_mjpeg_stream(cfg.source_url, cfg.timeout_seconds)
            elif cfg.source_type == "rtsp":
                img = self._capture_rtsp(cfg.source_url, cfg.frame_sampling)
            elif cfg.source_type == "web_stream":
                img = self._capture_web_stream(cfg.source_url, cfg.timeout_seconds, cfg.frame_sampling)
            elif cfg.source_type == "webcam_device":
                img = self._capture_webcam_device(
                    cfg.device_index,
                    cfg.image_width,
                    cfg.image_height,
                    cfg.frame_sampling,
                )
            elif cfg.source_type == "allsky_folder":
                img = self._capture_allsky_folder(cfg.source_url)
            else:
                raise ValueError(f"Unsupported camera source_type: {cfg.source_type}")

            latency_ms = (time.perf_counter() - t0) * 1000.0
            now_dt = datetime.now(UTC)
            with self._lock:
                self._connected = True
                self._last_capture_at = now_dt
                self._last_latency_ms = latency_ms
                self._last_error = ""
                self._consecutive_failures = 0
                self._next_retry_ts = 0.0
            return CaptureResult(
                success=True,
                image_bgr=img,
                captured_at=now_dt,
                latency_ms=latency_ms,
                source_type=cfg.source_type,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            safe_err = sanitize_error_credentials(str(exc))
            with self._lock:
                self._connected = False
                self._last_error = safe_err
                self._last_latency_ms = latency_ms
                self._consecutive_failures += 1
                if cfg.reconnect_enabled:
                    self._reconnect_attempts += 1
                    self._next_retry_ts = time.time() + cfg.reconnect_backoff_seconds
                else:
                    self._next_retry_ts = time.time() + 3600.0
            return CaptureResult(
                success=False,
                image_bgr=None,
                captured_at=datetime.now(UTC),
                latency_ms=latency_ms,
                source_type=cfg.source_type,
                error=safe_err,
            )

    def test_connection(
        self,
        source_type: str | None = None,
        source_url: str | None = None,
        device_index: int | None = None,
    ) -> dict[str, Any]:
        cfg = get_config_manager().get().camera.model_copy(deep=True)
        if source_type:
            cfg.source_type = source_type  # type: ignore[assignment]
        if source_url is not None:
            cfg.source_url = source_url
        if device_index is not None:
            cfg.device_index = device_index
        res = self.capture_once(cfg_override=cfg, bypass_backoff=True)
        if res.success and res.image_bgr is not None:
            h, w = res.image_bgr.shape[:2]
            return {
                "ok": True,
                "source_type": cfg.source_type,
                "source_url": cfg.source_url,
                "device_index": cfg.device_index,
                "latency_ms": round(res.latency_ms, 2),
                "width": w,
                "height": h,
                "message": "Successfully captured test frame from local camera source.",
            }
        return {
            "ok": False,
            "source_type": cfg.source_type,
            "source_url": cfg.source_url,
            "device_index": cfg.device_index,
            "latency_ms": round(res.latency_ms, 2),
            "message": res.error,
        }

    def fetch_ipwebcam_orientation(self, source_url: str, timeout: float = 4.0) -> dict[str, Any]:
        """
        Read optional Android IP Webcam `/sensors.json` endpoint from a LAN camera URL
        and derive elevation (pitch above horizon), roll, and compass azimuth.
        """
        if not source_url or not source_url.strip():
            raise ValueError("Camera source_url is empty")
        parsed = urlparse(source_url.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Android IP Webcam sensors require an http:// or https:// camera URL")
        sensors_url = f"{parsed.scheme}://{parsed.netloc}/sensors.json"
        resp = self._safe_http_get(sensors_url, timeout=timeout, source_type="http_snapshot")
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected /sensors.json payload")

        def _latest_vec(keys: tuple[str, ...]) -> list[float] | None:
            for k in keys:
                entry = data.get(k)
                if isinstance(entry, dict) and isinstance(entry.get("data"), list) and entry["data"]:
                    last = entry["data"][-1]
                    if isinstance(last, list) and len(last) >= 2 and isinstance(last[1], list):
                        return [float(v) for v in last[1]]
            return None

        accel = _latest_vec(("accel", "gravity", "lin_accel"))
        orient = _latest_vec(("orientation", "rot_vector"))
        mag = _latest_vec(("mag",))

        if accel is None and orient is None:
            raise ValueError("No accelerometer or orientation sensor streams found in /sensors.json")

        import math

        elevation_deg = 65.0
        roll_deg = 0.0
        azimuth_deg: float | None = None
        sources_used: list[str] = []

        if accel is not None and len(accel) >= 3:
            ax, ay, az = accel[0], accel[1], accel[2]
            norm = math.sqrt(ax * ax + ay * ay + az * az) or 9.81
            # Back-camera looks along -Z in Android device frame; zenith angle from +Z gravity component
            sin_el = max(-1.0, min(1.0, -az / norm))
            elevation_deg = round(math.degrees(math.asin(sin_el)), 2)
            roll_deg = round(max(-45.0, min(45.0, math.degrees(math.atan2(ax, max(1e-3, abs(ay) + abs(az)))))), 2)
            sources_used.append("accelerometer")

        if orient is not None and len(orient) >= 1:
            azimuth_deg = round(float(orient[0]) % 360.0, 2)
            sources_used.append("orientation")
        elif mag is not None and len(mag) >= 2:
            azimuth_deg = round(math.degrees(math.atan2(mag[0], mag[1])) % 360.0, 2)
            sources_used.append("magnetometer")

        return {
            "ok": True,
            "sensors_url": mask_url_credentials(sensors_url),
            "sources_used": sources_used,
            "orientation_elevation_deg": max(-20.0, min(90.0, elevation_deg)),
            "orientation_roll_deg": roll_deg,
            "orientation_azimuth_deg": azimuth_deg,
        }


_camera_manager = CameraManager()


def get_camera_manager() -> CameraManager:
    return _camera_manager
