"""Local Disk Map Tile Cache, Parent-Tile Upscaling & Offline-Safe Fallback Renderer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from app.config import AppConfig

TILE_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "dark_base": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "image/jpeg",
        "jpg",
    ),
    "dark_ref": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        "image/png",
        "png",
    ),
    "osm": (
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "image/png",
        "png",
    ),
    "sat_base": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "image/jpeg",
        "jpg",
    ),
}

_HTTP_HEADERS = {"User-Agent": "SkyObserver/1.0 (Local Weather Observatory)"}


def tile_dir(config: AppConfig) -> Path:
    return config.data_dir / "map_tiles"


def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    lat_c = max(-85.0511, min(85.0511, float(lat)))
    n = 1 << z
    x = int((float(lon) + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat_c)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_center_latlon(z: int, x: int, y: int) -> tuple[float, float]:
    n = 1 << z
    lon = ((x + 0.5) / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def _find_cached_file(root: Path, style: str, z: int, x: int, y: int) -> tuple[Path, str] | None:
    for ext, mime in (("png", "image/png"), ("jpg", "image/jpeg")):
        cand = root / style / str(z) / str(x) / f"{y}.{ext}"
        if cand.exists() and cand.stat().st_size > 50:
            return cand, mime
    return None


def _upscale_from_parent(root: Path, style: str, z: int, x: int, y: int) -> tuple[bytes, str] | None:
    for dz in range(1, min(6, z) + 1):
        pz = z - dz
        px = x >> dz
        py = y >> dz
        hit = _find_cached_file(root, style, pz, px, py)
        if hit is None:
            continue
        path, _ = hit
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 2 or img.size == 0:
            continue
        factor = 1 << dz
        rx = x % factor
        ry = y % factor
        h, w = img.shape[:2]
        x0 = int(rx * w / factor)
        x1 = max(x0 + 1, int((rx + 1) * w / factor))
        y0 = int(ry * h / factor)
        y1 = max(y0 + 1, int((ry + 1) * h / factor))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        up = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", up)
        if ok:
            return buf.tobytes(), "image/png"
    return None


def _render_offline_fallback_tile(style: str, z: int, x: int, y: int) -> tuple[bytes, str]:
    if style == "dark_ref":
        blank = np.zeros((256, 256, 4), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", blank)
        return (buf.tobytes() if ok else b""), "image/png"

    tile = np.full((256, 256, 3), (22, 16, 13), dtype=np.uint8)
    # Crisp subtle coordinate graticule lines
    tile[0, :] = (34, 29, 25)
    tile[:, 0] = (34, 29, 25)
    tile[128, ::4] = (28, 24, 20)
    tile[::4, 128] = (28, 24, 20)
    lat, lon = tile_center_latlon(z, x, y)
    cv2.putText(
        tile,
        f"{lat:.2f}, {lon:.2f}",
        (8, 244),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (78, 72, 65),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".png", tile)
    return (buf.tobytes() if ok else b""), "image/png"


def get_tile_bytes(config: AppConfig, style: str, z: int, x: int, y: int) -> tuple[bytes, str]:
    if style not in TILE_PROVIDERS:
        raise ValueError(f"Unsupported map tile style: {style}")
    if z < 0 or z > 18:
        raise ValueError(f"Unsupported zoom level: {z}")
    limit = 1 << z
    if x < 0 or x >= limit or y < 0 or y >= limit:
        raise ValueError(f"Tile coordinates out of range for z={z}: ({x}, {y})")

    root = tile_dir(config)
    hit = _find_cached_file(root, style, z, x, y)
    if hit is not None:
        return hit[0].read_bytes(), hit[1]

    if not config.server.offline_mode:
        url_tpl, mime, ext = TILE_PROVIDERS[style]
        url = url_tpl.format(z=z, x=x, y=y)
        try:
            with httpx.Client(timeout=8.0, headers=_HTTP_HEADERS) as client:
                resp = client.get(url)
            if resp.status_code == 200 and len(resp.content) > 100:
                out_file = root / style / str(z) / str(x) / f"{y}.{ext}"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_bytes(resp.content)
                return resp.content, mime
        except Exception:
            pass

    up = _upscale_from_parent(root, style, z, x, y)
    if up is not None:
        return up
    return _render_offline_fallback_tile(style, z, x, y)


def get_tile_cache_stats(config: AppConfig) -> dict[str, Any]:
    root = tile_dir(config)
    if not root.exists():
        return {"cached_tiles": 0, "size_mb": 0.0}
    count = 0
    total_bytes = 0
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".png", ".jpg"}:
            count += 1
            total_bytes += f.stat().st_size
    return {"cached_tiles": count, "size_mb": round(total_bytes / (1024 * 1024), 2)}


def prefetch_tiles_around(
    config: AppConfig,
    lat: float,
    lon: float,
    styles: list[str] | None = None,
    zooms: list[int] | None = None,
    radius_tiles: int = 1,
) -> dict[str, Any]:
    if config.server.offline_mode:
        raise PermissionError("Cannot download map tiles while Offline Only mode is active.")

    target_styles = [s for s in (styles or ["dark_base", "dark_ref", "osm"]) if s in TILE_PROVIDERS]
    target_zooms = sorted({int(z) for z in (zooms or [6, 7, 8, 9, 10, 11, 12]) if 1 <= int(z) <= 16})
    root = tile_dir(config)

    newly_cached = 0
    already_cached = 0
    failed = 0

    with httpx.Client(timeout=8.0, headers=_HTTP_HEADERS) as client:
        for style in target_styles:
            url_tpl, _mime, ext = TILE_PROVIDERS[style]
            for z in target_zooms:
                cx, cy = latlon_to_tile(lat, lon, z)
                limit = 1 << z
                for dx in range(-radius_tiles, radius_tiles + 1):
                    for dy in range(-radius_tiles, radius_tiles + 1):
                        tx, ty = cx + dx, cy + dy
                        if tx < 0 or tx >= limit or ty < 0 or ty >= limit:
                            continue
                        if _find_cached_file(root, style, z, tx, ty) is not None:
                            already_cached += 1
                            continue
                        try:
                            resp = client.get(url_tpl.format(z=z, x=tx, y=ty))
                            if resp.status_code == 200 and len(resp.content) > 100:
                                out_file = root / style / str(z) / str(tx) / f"{ty}.{ext}"
                                out_file.parent.mkdir(parents=True, exist_ok=True)
                                out_file.write_bytes(resp.content)
                                newly_cached += 1
                            else:
                                failed += 1
                        except Exception:
                            failed += 1

    return {
        "cached": newly_cached,
        "already_cached": already_cached,
        "failed": failed,
        "tile_cache": get_tile_cache_stats(config),
    }
