"""3D Camera Tilt, Azimuth, Roll & Planar Sky Projection Geometry Engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, UTC
from threading import RLock
from typing import Any

import cv2
import numpy as np

from app.config import CameraConfig

DEFAULT_PROJECTION_CLOUD_ALTITUDE_M = 2000.0
MIN_HORIZON_ELEVATION_DEG = 2.0
MIN_FLOW_ELEVATION_DEG = 8.0
MAX_RADAR_PROJECTION_KM = 28.0

_CLOUD_CLASS_BGRA: dict[int, tuple[int, int, int, int]] = {
    1: (255, 240, 215, 145),  # thin_cloud (soft icy white)
    2: (252, 250, 245, 215),  # dense_cloud (crisp bright white)
    3: (115, 92, 82, 230),    # dark_cloud (storm slate)
    4: (220, 215, 195, 110),  # haze_fog (pale mist)
}

_MAP_CACHE_LOCK = RLock()
_MAP_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


@dataclass(frozen=True)
class CameraGeometry:
    elevation_deg: float
    azimuth_deg: float
    roll_deg: float
    fov_horizontal_deg: float
    fov_vertical_deg: float
    cloud_altitude_m: float
    tilt_correction_enabled: bool

    @property
    def tilt_from_vertical_deg(self) -> float:
        return round(90.0 - self.elevation_deg, 2)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def geometry_from_camera_config(
    cam: CameraConfig,
    width: int = 1920,
    height: int = 1080,
) -> CameraGeometry:
    """Build validated CameraGeometry from CameraConfig and frame dimensions."""
    el = _clamp(getattr(cam, "orientation_elevation_deg", 65.0), -20.0, 90.0)
    az = float(getattr(cam, "orientation_azimuth_deg", 0.0)) % 360.0
    roll = _clamp(getattr(cam, "orientation_roll_deg", 0.0), -45.0, 45.0)
    fov_h = _clamp(cam.fov_horizontal_deg if cam.fov_horizontal_deg else 72.0, 10.0, 170.0)
    if cam.fov_vertical_deg is not None and cam.fov_vertical_deg > 0:
        fov_v = _clamp(cam.fov_vertical_deg, 10.0, 170.0)
    else:
        aspect = float(max(height, 1)) / float(max(width, 1))
        fov_v = _clamp(2.0 * math.degrees(math.atan(math.tan(math.radians(fov_h) / 2.0) * aspect)), 10.0, 170.0)
    alt = float(cam.assumed_cloud_altitude_m) if cam.assumed_cloud_altitude_m else DEFAULT_PROJECTION_CLOUD_ALTITUDE_M
    alt = max(100.0, min(15000.0, alt))
    enabled = bool(getattr(cam, "tilt_correction_enabled", True))
    return CameraGeometry(
        elevation_deg=el,
        azimuth_deg=az,
        roll_deg=roll,
        fov_horizontal_deg=fov_h,
        fov_vertical_deg=fov_v,
        cloud_altitude_m=alt,
        tilt_correction_enabled=enabled,
    )


def camera_basis_enu(
    elevation_deg: float,
    azimuth_deg: float,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return right-handed camera basis vectors (r_cam, u_cam, f_cam) in ENU (East, North, Up):
      f_cam: optical axis pointing from camera into sky
      r_cam: image right (+u)
      u_cam: image up (-v)
    """
    alpha = math.radians(_clamp(elevation_deg, -20.0, 90.0))
    psi = math.radians(float(azimuth_deg) % 360.0)
    rho = math.radians(_clamp(roll_deg, -45.0, 45.0))

    f_cam = np.array(
        [math.cos(alpha) * math.sin(psi), math.cos(alpha) * math.cos(psi), math.sin(alpha)],
        dtype=np.float64,
    )
    r0 = np.array([math.cos(psi), -math.sin(psi), 0.0], dtype=np.float64)
    u0 = np.array(
        [-math.sin(alpha) * math.sin(psi), -math.sin(alpha) * math.cos(psi), math.cos(alpha)],
        dtype=np.float64,
    )

    cos_r = math.cos(rho)
    sin_r = math.sin(rho)
    r_cam = cos_r * r0 - sin_r * u0
    u_cam = sin_r * r0 + cos_r * u0
    return r_cam, u_cam, f_cam


def focal_lengths_px(width: int, height: int, geom: CameraGeometry) -> tuple[float, float, float, float]:
    """Return pinhole intrinsics (fx, fy, cx, cy) in pixel units."""
    w = max(1, int(width))
    h = max(1, int(height))
    fx = w / (2.0 * math.tan(math.radians(geom.fov_horizontal_deg) / 2.0))
    fy = h / (2.0 * math.tan(math.radians(geom.fov_vertical_deg) / 2.0))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return fx, fy, cx, cy


def compute_pixel_geometry_maps(width: int, height: int, geom: CameraGeometry) -> dict[str, Any]:
    """
    Compute per-pixel elevation sine, horizon visibility mask, and zenith-cosine/solid-angle sky weights.
    Cached for fast repeated frame execution.
    """
    w = max(1, int(width))
    h = max(1, int(height))
    key = (
        w,
        h,
        round(geom.elevation_deg, 3),
        round(geom.azimuth_deg, 3),
        round(geom.roll_deg, 3),
        round(geom.fov_horizontal_deg, 3),
        round(geom.fov_vertical_deg, 3),
    )
    with _MAP_CACHE_LOCK:
        cached = _MAP_CACHE.get(key)
        if cached is not None:
            return cached

    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(w, h, geom)

    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    xc = (uu - cx) / float(fx)
    yc = (cy - vv) / float(fy)
    d_norm = np.sqrt(1.0 + xc * xc + yc * yc)
    d_up = xc * float(r_cam[2]) + yc * float(u_cam[2]) + float(f_cam[2])

    sin_el = (d_up / d_norm).astype(np.float32)
    sin_min = math.sin(math.radians(MIN_HORIZON_ELEVATION_DEG))
    above_horizon = sin_el > sin_min
    solid_angle = 1.0 / (d_norm * d_norm * d_norm)
    sky_weight = np.where(above_horizon, np.maximum(0.0, sin_el) * solid_angle, 0.0).astype(np.float32)
    below_horizon_pct = round(float(np.mean(~above_horizon) * 100.0), 2)

    result = {
        "sin_el": sin_el,
        "above_horizon": above_horizon,
        "sky_weight": sky_weight,
        "below_horizon_pct": below_horizon_pct,
    }
    with _MAP_CACHE_LOCK:
        if len(_MAP_CACHE) >= 16:
            _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
        _MAP_CACHE[key] = result
    return result


def project_pixels_to_cloud_plane(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    geom: CameraGeometry,
    altitude_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project image coordinates (u, v) onto horizontal sky plane Z = altitude_m in ENU coordinates.
    Returns (east_m, north_m, valid_mask).
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(width, height, geom)

    xc = (u_arr - cx) / fx
    yc = (cy - v_arr) / fy
    d_norm = np.sqrt(1.0 + xc * xc + yc * yc)

    de = xc * r_cam[0] + yc * u_cam[0] + f_cam[0]
    dn = xc * r_cam[1] + yc * u_cam[1] + f_cam[1]
    du = xc * r_cam[2] + yc * u_cam[2] + f_cam[2]

    sin_min = math.sin(math.radians(MIN_FLOW_ELEVATION_DEG))
    valid = (du / d_norm) >= sin_min
    safe_du = np.where(valid, du, 1.0)
    east = altitude_m * (de / safe_du)
    north = altitude_m * (dn / safe_du)
    return east, north, valid


def _en_to_latlon(lat: float, lon: float, east_m: float, north_m: float) -> list[float]:
    cos_lat = max(0.15, math.cos(math.radians(lat)))
    d_lat = north_m / 111320.0
    d_lon = east_m / (111320.0 * cos_lat)
    return [round(lat + d_lat, 6), round(lon + d_lon, 6)]


def compute_solar_position(
    lat: float,
    lon: float,
    dt_utc: datetime | None = None,
) -> dict[str, Any]:
    """
    Compute solar elevation and azimuth (100% offline NOAA astronomical equations).
    """
    ts = dt_utc or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)

    day_of_year = ts.timetuple().tm_yday
    frac_hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    gamma = (2.0 * math.pi / 365.0) * (day_of_year - 1 + (frac_hour - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )

    tst = (frac_hour * 60.0 + eqtime + 4.0 * float(lon)) % 1440.0
    ha_deg = tst / 4.0 - 180.0
    ha = math.radians(ha_deg)
    lat_r = math.radians(_clamp(lat, -89.9, 89.9))

    cos_z = _clamp(
        math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha),
        -1.0,
        1.0,
    )
    zenith_deg = math.degrees(math.acos(cos_z))
    elevation_deg = 90.0 - zenith_deg
    sin_z = max(1e-6, math.sin(math.radians(zenith_deg)))
    cos_az = _clamp(
        (math.sin(decl) - math.sin(lat_r) * cos_z) / (math.cos(lat_r) * sin_z),
        -1.0,
        1.0,
    )
    az_base = math.degrees(math.acos(cos_az))
    azimuth_deg = (360.0 - az_base) % 360.0 if ha_deg > 0.0 else az_base % 360.0

    return {
        "elevation_deg": round(elevation_deg, 2),
        "azimuth_deg": round(azimuth_deg, 2),
        "zenith_deg": round(zenith_deg, 2),
        "is_daylight": bool(elevation_deg > 0.0),
    }


def project_sky_angles_to_image(
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    geom: CameraGeometry,
) -> dict[str, Any]:
    """Project a sky direction (azimuth, elevation) into normalized camera coordinates [0..1]."""
    w = max(1, int(width))
    h = max(1, int(height))
    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(w, h, geom)

    el_r = math.radians(float(elevation_deg))
    az_r = math.radians(float(azimuth_deg) % 360.0)
    ray = np.array(
        [math.cos(el_r) * math.sin(az_r), math.cos(el_r) * math.cos(az_r), math.sin(el_r)],
        dtype=np.float64,
    )
    cz = float(np.dot(ray, f_cam))
    if cz <= 0.05:
        return {"in_camera_fov": False, "u_norm": None, "v_norm": None}
    cx_p = float(np.dot(ray, r_cam)) / cz
    cy_p = float(np.dot(ray, u_cam)) / cz
    u_px = cx + fx * cx_p
    v_px = cy - fy * cy_p
    u_norm = u_px / float(max(1, w - 1))
    v_norm = v_px / float(max(1, h - 1))
    in_fov = bool(0.0 <= u_norm <= 1.0 and 0.0 <= v_norm <= 1.0)
    return {
        "in_camera_fov": in_fov,
        "u_norm": round(u_norm, 4),
        "v_norm": round(v_norm, 4),
    }


def compute_sun_glare_mask(
    width: int,
    height: int,
    geom: CameraGeometry,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    max_angle_deg: float = 3.5,
) -> np.ndarray:
    """Return boolean mask (H, W) of pixels within max_angle_deg of the solar disk."""
    w = max(1, int(width))
    h = max(1, int(height))
    if sun_elevation_deg <= 2.0:
        return np.zeros((h, w), dtype=bool)

    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(w, h, geom)
    el_r = math.radians(float(sun_elevation_deg))
    az_r = math.radians(float(sun_azimuth_deg) % 360.0)
    sun_vec = np.array(
        [math.cos(el_r) * math.sin(az_r), math.cos(el_r) * math.cos(az_r), math.sin(el_r)],
        dtype=np.float64,
    )
    # Express sun vector in camera frame (r_cam, u_cam, f_cam)
    sx = float(np.dot(sun_vec, r_cam))
    sy = float(np.dot(sun_vec, u_cam))
    sz = float(np.dot(sun_vec, f_cam))
    if sz <= 0.0:
        return np.zeros((h, w), dtype=bool)

    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    xc = (uu - cx) / float(fx)
    yc = (cy - vv) / float(fy)
    d_norm = np.sqrt(1.0 + xc * xc + yc * yc)
    cos_ang = (xc * sx + yc * sy + sz) / d_norm
    return cos_ang >= math.cos(math.radians(max_angle_deg))


def with_cloud_altitude(geom: CameraGeometry, cloud_altitude_m: float) -> CameraGeometry:
    return replace(geom, cloud_altitude_m=max(100.0, min(15000.0, float(cloud_altitude_m))))


def compute_radar_footprint(
    lat: float,
    lon: float,
    geom: CameraGeometry,
    radar_radius_km: float,
    dt_utc: datetime | None = None,
    cloud_altitude_source: str = "default_2000m",
) -> dict[str, Any]:
    """
    Compute camera sky FOV polygon, view center, distances, and image overlay bounds
    for Leaflet map rendering at assumed cloud altitude.
    """
    w = 100
    h = 100
    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(w, h, geom)
    max_m = MAX_RADAR_PROJECTION_KM * 1000.0
    sin_edge_min = math.sin(math.radians(4.0))
    alt = geom.cloud_altitude_m
    # Physically consistent horizon cap: proportional to cloud altitude (alt / tan(4 deg)),
    # at least radar_radius_km, and capped at MAX_RADAR_PROJECTION_KM.
    horizon_cap_m = min(max_m, max(float(radar_radius_km) * 1000.0, alt / math.tan(math.radians(4.0))))

    def _ray_en(u_px: float, v_px: float) -> tuple[float, float, float]:
        xc = (u_px - cx) / fx
        yc = (cy - v_px) / fy
        d = xc * r_cam + yc * u_cam + f_cam
        d_norm = float(np.linalg.norm(d))
        if d_norm > 0:
            sin_el = float(d[2] / d_norm)
        else:
            sin_el = 0.0
        horiz = math.hypot(float(d[0]), float(d[1]))
        if sin_el < sin_edge_min:
            if horiz < 1e-6:
                return 0.0, 0.0, 0.0
            scale = horizon_cap_m / horiz
            return float(d[0] * scale), float(d[1] * scale), horizon_cap_m / 1000.0
        em = alt * float(d[0]) / float(d[2])
        nm = alt * float(d[1]) / float(d[2])
        dist = math.hypot(em, nm)
        if dist > horizon_cap_m and dist > 1e-6:
            s = horizon_cap_m / dist
            em *= s
            nm *= s
            dist = horizon_cap_m
        return em, nm, dist / 1000.0

    # Sample perimeter of the sensor clockwise
    steps = 10
    perim_uv: list[tuple[float, float]] = (
        [(i * (w - 1) / steps, 0.0) for i in range(steps)]
        + [(float(w - 1), i * (h - 1) / steps) for i in range(steps)]
        + [((w - 1) * (1.0 - i / steps), float(h - 1)) for i in range(steps)]
        + [(0.0, (h - 1) * (1.0 - i / steps)) for i in range(steps)]
    )

    fov_polygon: list[list[float]] = []
    dists_km: list[float] = []
    for u_px, v_px in perim_uv:
        em, nm, dk = _ray_en(u_px, v_px)
        fov_polygon.append(_en_to_latlon(lat, lon, em, nm))
        dists_km.append(dk)

    c_em, c_nm, center_dist_km = _ray_en(cx, cy)
    view_center = _en_to_latlon(lat, lon, c_em, c_nm)
    near_dist_km = min(dists_km) if dists_km else 0.0
    far_dist_km = max(dists_km) if dists_km else float(radar_radius_km)

    if geom.tilt_correction_enabled:
        # Size the cloud overlay tile tightly to the camera's real FOV footprint so zenith/upward-pointing
        # cameras (e.g. 1.5–3 km footprint) use the full 512x512 resolution instead of shrinking into a tiny dot.
        overlay_radius_km = max(0.8, min(MAX_RADAR_PROJECTION_KM, far_dist_km * 1.15))
    else:
        overlay_radius_km = float(radar_radius_km)

    overlay_m = overlay_radius_km * 1000.0
    sw = _en_to_latlon(lat, lon, -overlay_m, -overlay_m)
    ne = _en_to_latlon(lat, lon, overlay_m, overlay_m)
    below_horizon_pct = compute_pixel_geometry_maps(64, 48, geom)["below_horizon_pct"]
    solar_pos = compute_solar_position(lat, lon, dt_utc)
    solar_proj = project_sky_angles_to_image(
        float(solar_pos["azimuth_deg"]),
        float(solar_pos["elevation_deg"]),
        100,
        100,
        geom,
    )

    return {
        "tilt_correction_enabled": geom.tilt_correction_enabled,
        "elevation_deg": round(geom.elevation_deg, 2),
        "tilt_from_vertical_deg": geom.tilt_from_vertical_deg,
        "azimuth_deg": round(geom.azimuth_deg, 2),
        "roll_deg": round(geom.roll_deg, 2),
        "fov_horizontal_deg": round(geom.fov_horizontal_deg, 2),
        "fov_vertical_deg": round(geom.fov_vertical_deg, 2),
        "cloud_altitude_m": round(geom.cloud_altitude_m, 1),
        "cloud_altitude_source": cloud_altitude_source,
        "center_distance_km": round(center_dist_km, 2),
        "near_distance_km": round(near_dist_km, 2),
        "far_distance_km": round(far_dist_km, 2),
        "below_horizon_pct": below_horizon_pct,
        "overlay_radius_km": round(overlay_radius_km, 2),
        "overlay_bounds": [sw, ne],
        "view_center_latlon": view_center,
        "fov_polygon": fov_polygon,
        "solar": {
            **solar_pos,
            **solar_proj,
        },
    }


def _build_smooth_density_stack(
    class_map: np.ndarray,
    raw_bgr: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert discrete segmentation labels (0..4) into a high-detail 4-channel float32 density stack
    [thin, dense, dark, sun_glare] modulated by real sky luminance/contrast relief from raw_bgr
    so cloud ridges, wisps, and dark bases stay crisp and detailed on the radar map.
    """
    h, w = class_map.shape[:2]
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    valid_bool = (valid_mask > 0) if (valid_mask is not None and valid_mask.shape[:2] == (h, w)) else None

    # Extract high-frequency cloud relief from the real sky photo (if provided)
    detail_gain: np.ndarray | None = None
    dark_boost: np.ndarray | None = None
    if raw_bgr is not None and raw_bgr.size > 0:
        if raw_bgr.shape[:2] != (h, w):
            raw_fit = cv2.resize(raw_bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            raw_fit = raw_bgr
        lab = cv2.cvtColor(raw_fit, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32) / 255.0
        # Local multi-scale contrast (reveals cloud puffs & dark bases even in uniform overcast)
        L_local = cv2.GaussianBlur(L, (0, 0), sigmaX=max(6.0, w * 0.04))
        relief = L - L_local  # positive = bright cloud ridge, negative = shadowed cloud base
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(lab[:, :, 0]).astype(np.float32) / 255.0
        detail_gain = np.clip(0.55 + 0.55 * clahe + 1.45 * relief, 0.22, 1.35).astype(np.float32)
        dark_boost = np.clip(-relief * 3.2 + np.maximum(0.0, 0.52 - L) * 0.9, 0.0, 0.85).astype(np.float32)

    stack = np.zeros((h, w, 4), dtype=np.float32)
    for ch_idx, cls_id in enumerate((1, 2, 3, 4)):
        raw_bin = (class_map == cls_id).astype(np.uint8) * 255
        if valid_bool is not None:
            raw_bin[~valid_bool] = 0
        if not np.any(raw_bin):
            continue
        cleaned = cv2.morphologyEx(raw_bin, cv2.MORPH_OPEN, k_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
        if not np.any(cleaned):
            cleaned = raw_bin
        ch_f = cv2.GaussianBlur(cleaned.astype(np.float32) / 255.0, (3, 3), 0.65)
        if detail_gain is not None and ch_idx in (0, 1, 2):
            ch_f = np.clip(ch_f * detail_gain, 0.0, 1.0)
        stack[:, :, ch_idx] = ch_f

    if dark_boost is not None:
        cloud_presence = np.clip(stack[:, :, 0] + stack[:, :, 1] + stack[:, :, 2], 0.0, 1.0)
        stack[:, :, 2] = np.clip(stack[:, :, 2] + dark_boost * cloud_presence * 0.65, 0.0, 1.0)
    return stack


def _render_volumetric_cloud_bgra(
    density_stack: np.ndarray,
    fov_mask: np.ndarray | None = None,
    horizon_fade: np.ndarray | None = None,
) -> np.ndarray:
    """
    Render a 4-channel [thin, dense, dark, sun_glare] float32 radar density field into a crisp,
    high-detail BGRA meteorological cloud layer with internal relief shading and anti-aliased edges.
    """
    thin = cv2.GaussianBlur(density_stack[:, :, 0], (0, 0), sigmaX=0.65)
    dense = cv2.GaussianBlur(density_stack[:, :, 1], (0, 0), sigmaX=0.60)
    dark = cv2.GaussianBlur(density_stack[:, :, 2], (0, 0), sigmaX=0.55)
    glare = cv2.GaussianBlur(density_stack[:, :, 3], (0, 0), sigmaX=0.80)

    total_density = np.clip(thin * 0.86 + dense * 1.05 + dark * 1.18 + glare * 0.35, 0.0, 1.0)
    # Crisp S-curve alpha preserving wispy breaks and cloud-edge detail
    smooth_alpha = np.clip((total_density - 0.08) / 0.78, 0.0, 1.0)
    smooth_alpha = smooth_alpha * smooth_alpha * (3.0 - 2.0 * smooth_alpha)

    # Sharpen class weights so Thin, Dense, and Dark/Storm stay visually distinct (Windy-style)
    w_thin_raw = thin * 0.90
    w_dense_raw = dense * 1.25
    w_dark_raw = dark * 1.65
    w_glare_raw = glare * 0.45

    w_thin_p = np.power(np.maximum(w_thin_raw, 0.0), 1.6)
    w_dense_p = np.power(np.maximum(w_dense_raw, 0.0), 1.6)
    w_dark_p = np.power(np.maximum(w_dark_raw, 0.0), 1.6)
    w_glare_p = np.power(np.maximum(w_glare_raw, 0.0), 1.6)

    w_sum = np.maximum(w_thin_p + w_dense_p + w_dark_p + w_glare_p, 1e-5)
    w_thin = w_thin_p / w_sum
    w_dense = w_dense_p / w_sum
    w_dark = w_dark_p / w_sum
    w_glare = w_glare_p / w_sum

    # High-contrast Windy-style BGR palette:
    # - thin  = Electric Sky-Cyan (#38BDF8 -> BGR 248, 189, 56)
    # - dense = Crisp Pure White  (#F8FAFC -> BGR 252, 250, 248)
    # - dark  = Storm Violet-Magenta (#A855F7 -> BGR 247, 85, 168)
    # - glare = Solar Amber Gold  (#FBBF24 -> BGR 36, 191, 251)
    core_lift = 1.0 + 0.08 * np.clip(total_density - 0.55, 0.0, 0.45)
    b_ch = (w_thin * 248.0 + w_dense * 252.0 + w_dark * 235.0 + w_glare * 36.0) * core_lift
    g_ch = (w_thin * 189.0 + w_dense * 250.0 + w_dark * 75.0 + w_glare * 191.0) * core_lift
    r_ch = (w_thin * 56.0 + w_dense * 248.0 + w_dark * 195.0 + w_glare * 251.0) * core_lift

    max_alpha = np.clip(
        w_thin * 195.0 + w_dense * 242.0 + w_dark * 250.0 + w_glare * 145.0,
        0.0,
        250.0,
    )
    alpha = smooth_alpha * max_alpha

    if fov_mask is not None:
        dist = cv2.distanceTransform(fov_mask.astype(np.uint8), cv2.DIST_L2, 3)
        feather = np.clip(dist / 2.5, 0.0, 1.0)
        alpha *= feather
    if horizon_fade is not None:
        alpha *= horizon_fade

    rgba = np.zeros((density_stack.shape[0], density_stack.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = np.clip(b_ch, 0.0, 255.0).astype(np.uint8)
    rgba[:, :, 1] = np.clip(g_ch, 0.0, 255.0).astype(np.uint8)
    rgba[:, :, 2] = np.clip(r_ch, 0.0, 255.0).astype(np.uint8)
    rgba[:, :, 3] = np.clip(alpha, 0.0, 255.0).astype(np.uint8)
    return rgba


def project_mask_to_radar_bgra(
    class_map: np.ndarray,
    geom: CameraGeometry,
    overlay_radius_km: float,
    size: int = 384,
    raw_bgr: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project segmentation class map into a North-up East-right BGRA radar tile spanning [-R, +R] km.
    Uses continuous denoised density fields with bilinear warping and soft volumetric feathering.
    """
    density_src = _build_smooth_density_stack(class_map, raw_bgr=raw_bgr, valid_mask=valid_mask)

    if not geom.tilt_correction_enabled:
        warped_stack = cv2.resize(density_src, (size, size), interpolation=cv2.INTER_LINEAR)
        if abs(geom.azimuth_deg) > 0.1:
            rot_mat = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), -geom.azimuth_deg, 1.0)
            warped_stack = cv2.warpAffine(
                warped_stack,
                rot_mat,
                (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0, 0.0, 0.0, 0.0),
            )
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        r = np.hypot(xx - (size - 1) / 2.0, yy - (size - 1) / 2.0) / (size / 2.0)
        falloff = np.clip((1.0 - r) / 0.24, 0.0, 1.0)
        return _render_volumetric_cloud_bgra(warped_stack, fov_mask=None, horizon_fade=falloff)

    src_h, src_w = class_map.shape[:2]
    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(src_w, src_h, geom)

    r_m = max(100.0, float(overlay_radius_km) * 1000.0)
    coords_e = np.linspace(-r_m, r_m, size, dtype=np.float64)
    coords_n = np.linspace(r_m, -r_m, size, dtype=np.float64)
    east_grid, north_grid = np.meshgrid(coords_e, coords_n)
    up_grid = np.full_like(east_grid, geom.cloud_altitude_m, dtype=np.float64)

    cx_p = east_grid * r_cam[0] + north_grid * r_cam[1] + up_grid * r_cam[2]
    cy_p = east_grid * u_cam[0] + north_grid * u_cam[1] + up_grid * u_cam[2]
    cz_p = east_grid * f_cam[0] + north_grid * f_cam[1] + up_grid * f_cam[2]

    in_front = cz_p > 1e-3
    safe_z = np.where(in_front, cz_p, 1.0)
    xc = cx_p / safe_z
    yc = cy_p / safe_z

    map_x = np.where(in_front, cx + fx * xc, -1.0).astype(np.float32)
    map_y = np.where(in_front, cy - fy * yc, -1.0).astype(np.float32)

    in_fov = in_front & (map_x >= 0.0) & (map_x <= src_w - 1) & (map_y >= 0.0) & (map_y <= src_h - 1)
    warped_stack = cv2.remap(
        density_src,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )
    warped_stack[~in_fov] = 0.0

    # Gentle radial outer-rim fade (only past 82% of radar radius) so low-elevation cameras keep full visibility
    horiz_dist = np.hypot(east_grid, north_grid).astype(np.float32)
    radial_norm = horiz_dist / max(float(r_m), 1.0)
    horizon_fade = np.clip((1.0 - radial_norm) / 0.18, 0.0, 1.0)

    return _render_volumetric_cloud_bgra(warped_stack, fov_mask=in_fov, horizon_fade=horizon_fade)
