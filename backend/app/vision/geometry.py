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
RADAR_EDGE_MIN_EL_DEG = 3.5
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
    max_projection_zenith_deg: float = 70.0

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
    max_z = _clamp(getattr(cam, "max_projection_zenith_deg", 70.0), 30.0, 88.0)
    return CameraGeometry(
        elevation_deg=el,
        azimuth_deg=az,
        roll_deg=roll,
        fov_horizontal_deg=fov_h,
        fov_vertical_deg=fov_v,
        cloud_altitude_m=alt,
        tilt_correction_enabled=enabled,
        max_projection_zenith_deg=max_z,
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


def undistort_sky_frame(
    image_bgr: np.ndarray,
    geom: CameraGeometry,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> np.ndarray:
    """Apply optional pinhole intrinsic lens undistortion (zero-cost no-op when coefficients are 0)."""
    if abs(k1) < 1e-7 and abs(k2) < 1e-7 and abs(p1) < 1e-7 and abs(p2) < 1e-7:
        return image_bgr
    h, w = image_bgr.shape[:2]
    fx, fy, cx, cy = focal_lengths_px(w, h, geom)
    cam_mtx = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([float(k1), float(k2), float(p1), float(p2), 0.0], dtype=np.float64)
    return cv2.undistort(image_bgr, cam_mtx, dist)


def triangulate_cloud_base_altitude_m(
    cam1_lat: float,
    cam1_lon: float,
    el1_deg: float,
    az1_deg: float,
    cam2_lat: float,
    cam2_lon: float,
    el2_deg: float,
    az2_deg: float,
    cam1_alt_m: float = 0.0,
    cam2_alt_m: float = 0.0,
) -> dict[str, Any]:
    """
    Stereo multi-camera cloud-base height triangulation via 3D ENU skew-ray least-squares intersection.
    Solves for the closest midpoint between the viewing rays of two spatially separated sky cameras.
    """
    cos_lat = max(0.15, math.cos(math.radians(0.5 * (cam1_lat + cam2_lat))))
    p1 = np.array([0.0, 0.0, float(cam1_alt_m)], dtype=np.float64)
    p2 = np.array(
        [
            (float(cam2_lon) - float(cam1_lon)) * 111320.0 * cos_lat,
            (float(cam2_lat) - float(cam1_lat)) * 111320.0,
            float(cam2_alt_m),
        ],
        dtype=np.float64,
    )
    baseline_m = float(np.linalg.norm(p2 - p1))
    if baseline_m < 10.0:
        return {"valid": False, "reason": "Stereo baseline < 10 m is too short for cloud-base triangulation."}

    def _unit_ray(el_d: float, az_d: float) -> np.ndarray:
        el = math.radians(float(el_d))
        az = math.radians(float(az_d))
        return np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)], dtype=np.float64)

    d1 = _unit_ray(el1_deg, az1_deg)
    d2 = _unit_ray(el2_deg, az2_deg)
    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-6:
        return {"valid": False, "baseline_m": round(baseline_m, 1), "reason": "Viewing rays are nearly parallel."}

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    if t1 <= 0.0 or t2 <= 0.0:
        return {
            "valid": False,
            "baseline_m": round(baseline_m, 1),
            "reason": "Rays diverge (intersection behind one or both cameras).",
        }

    q1 = p1 + t1 * d1
    q2 = p2 + t2 * d2
    mid = 0.5 * (q1 + q2)
    miss_m = float(np.linalg.norm(q1 - q2))
    alt_m = float(mid[2])
    if alt_m < 100.0 or alt_m > 18000.0:
        return {
            "valid": False,
            "baseline_m": round(baseline_m, 1),
            "altitude_m": round(alt_m, 1),
            "ray_miss_m": round(miss_m, 1),
            "reason": "Triangulated altitude outside physical troposphere bounds (100-18000 m).",
        }

    parallax_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d1, d2))))))
    # Geometric uncertainty scales with ray miss + 1 deg pointing uncertainty over stereo parallax
    geom_unc_m = max(miss_m * 0.5, (alt_m / max(1.0, parallax_deg)) * 0.5)
    cloud_lat, cloud_lon = _en_to_latlon(cam1_lat, cam1_lon, float(mid[0]), float(mid[1]))
    return {
        "valid": True,
        "altitude_m": round(alt_m, 1),
        "uncertainty_m": round(geom_unc_m, 1),
        "baseline_m": round(baseline_m, 1),
        "parallax_deg": round(parallax_deg, 2),
        "ray_miss_m": round(miss_m, 1),
        "cloud_lat": cloud_lat,
        "cloud_lon": cloud_lon,
    }


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
    prior = compute_sun_aureole_prior(
        width=width,
        height=height,
        geom=geom,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        core_deg=max_angle_deg,
        halo_deg=max_angle_deg,
    )
    return prior >= 0.99


def compute_sun_aureole_prior(
    width: int,
    height: int,
    geom: CameraGeometry,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    core_deg: float = 3.5,
    halo_deg: float = 14.0,
) -> np.ndarray:
    """
    Return smooth float32 map (H, W) in [0..1] encoding angular proximity to the sun:
      - 1.0 inside core_deg
      - smooth cosine/quadratic falloff from core_deg to halo_deg
      - 0.0 beyond halo_deg or when sun is below horizon.
    """
    w = max(1, int(width))
    h = max(1, int(height))
    if sun_elevation_deg <= 0.5:
        return np.zeros((h, w), dtype=np.float32)

    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(w, h, geom)
    el_r = math.radians(float(sun_elevation_deg))
    az_r = math.radians(float(sun_azimuth_deg) % 360.0)
    sun_vec = np.array(
        [math.cos(el_r) * math.sin(az_r), math.cos(el_r) * math.cos(az_r), math.sin(el_r)],
        dtype=np.float64,
    )
    sx = float(np.dot(sun_vec, r_cam))
    sy = float(np.dot(sun_vec, u_cam))
    sz = float(np.dot(sun_vec, f_cam))
    if sz <= 0.0:
        return np.zeros((h, w), dtype=np.float32)

    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    xc = (uu - cx) / float(fx)
    yc = (cy - vv) / float(fy)
    d_norm = np.sqrt(1.0 + xc * xc + yc * yc)
    cos_ang = np.clip((xc * sx + yc * sy + sz) / d_norm, -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cos_ang))

    core = max(0.5, float(core_deg))
    halo = max(core, float(halo_deg))
    if halo <= core + 1e-3:
        return (ang_deg <= core).astype(np.float32)

    t = np.clip((halo - ang_deg) / (halo - core), 0.0, 1.0)
    # 1.0 inside core, smooth squared falloff across Mie scattering halo
    prior = np.where(ang_deg <= core, 1.0, t * t)
    return prior.astype(np.float32)


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
    sin_edge_min = math.sin(math.radians(RADAR_EDGE_MIN_EL_DEG))
    alt = geom.cloud_altitude_m
    # Physically consistent horizon cap: proportional to cloud altitude (alt / tan(RADAR_EDGE_MIN_EL_DEG)),
    # at least radar_radius_km, and capped at MAX_RADAR_PROJECTION_KM.
    horizon_cap_m = min(
        max_m,
        max(float(radar_radius_km) * 1000.0, alt / math.tan(math.radians(RADAR_EDGE_MIN_EL_DEG))),
    )

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
        "max_projection_zenith_deg": round(geom.max_projection_zenith_deg, 1),
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

    # Enforce configurable zenith cutoff (default <= 70 deg -> elevation >= 20 deg) and strictly exclude <= 0 deg
    horiz_dist = np.hypot(east_grid, north_grid).astype(np.float32)
    el_deg_grid = np.degrees(np.arctan2(geom.cloud_altitude_m, np.maximum(horiz_dist, 1.0))).astype(np.float32)
    zenith_deg_grid = 90.0 - el_deg_grid
    max_z = float(np.clip(getattr(geom, "max_projection_zenith_deg", 70.0), 30.0, 88.0))
    # Only enforce steep zenith cutoff if camera elevation is high enough to have sky within max_z;
    # for near-horizon cameras (elevation < 90 - max_z + 5), fall back gracefully to MIN_HORIZON_ELEVATION_DEG
    # while still decaying opacity with zenith angle so low-elevation uncertainty is visually honest.
    min_el_z = max(MIN_HORIZON_ELEVATION_DEG, 90.0 - max_z)
    eff_min_el = min_el_z if geom.elevation_deg >= (90.0 - max_z + 5.0) else MIN_HORIZON_ELEVATION_DEG
    below_cutoff = (el_deg_grid <= 0.0) | (el_deg_grid < eff_min_el)
    warped_stack[below_cutoff] = 0.0
    in_fov = in_fov & (~below_cutoff)

    # Smooth uncertainty opacity decay as zenith angle & distance increase (tan(zenith) error grows near horizon)
    el_t = np.clip((el_deg_grid - eff_min_el) / 10.0, 0.0, 1.0)
    el_fade = el_t * el_t * (3.0 - 2.0 * el_t)
    zenith_unc_fade = np.clip(1.0 - np.maximum(0.0, zenith_deg_grid - 45.0) / 50.0, 0.30, 1.0)
    radial_norm = horiz_dist / max(float(r_m), 1.0)
    radial_fade = np.clip((1.0 - radial_norm) / 0.18, 0.0, 1.0)
    horizon_fade = (el_fade * zenith_unc_fade * radial_fade).astype(np.float32)

    return _render_volumetric_cloud_bgra(warped_stack, fov_mask=in_fov, horizon_fade=horizon_fade)


def estimate_camera_tilt_from_frame(
    image_bgr: np.ndarray,
    fov_horizontal_deg: float = 72.0,
    fov_vertical_deg: float | None = None,
) -> dict[str, Any]:
    """
    Automatically estimate camera elevation (pitch above horizon) and roll angle
    directly from a captured sky/horizon frame using sky-vs-terrain affinity contrast
    and robust outlier-resistant line fitting across vertical image strips.
    """
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim < 3:
        return {
            "detected_horizon": False,
            "mode": "zenith_sky_only",
            "orientation_elevation_deg": 85.0,
            "orientation_roll_deg": 0.0,
            "tilt_from_vertical_deg": 5.0,
            "horizon_y_norm": None,
            "below_horizon_pct": 0.0,
            "confidence": 0.2,
        }

    h, w = image_bgr.shape[:2]
    fov_h = max(10.0, min(170.0, float(fov_horizontal_deg or 72.0)))
    if fov_vertical_deg is None or fov_vertical_deg <= 0:
        tan_v = math.tan(math.radians(fov_h) / 2.0) * (float(h) / max(1.0, float(w)))
        fov_v = math.degrees(2.0 * math.atan(tan_v))
    else:
        fov_v = max(10.0, min(170.0, float(fov_vertical_deg)))

    work_w = min(320, max(96, w))
    work_h = max(72, int(round(work_w * float(h) / max(1.0, float(w)))))
    small = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    L_norm = lab[:, :, 0] / 255.0
    a_norm = lab[:, :, 1] / 255.0
    b_norm = lab[:, :, 2] / 255.0
    S_norm = hsv[:, :, 1] / 255.0

    bgr_f = small.astype(np.float32) / 255.0
    blue_minus_red = bgr_f[:, :, 0] - bgr_f[:, :, 2]

    # Terrain/urban chrominance penalty: terracotta roofs (+a/+b), foliage (-a/+b), strong saturation
    warm_or_green = (
        np.maximum(a_norm - 0.52, 0.0) * 2.2
        + np.maximum(0.48 - a_norm, 0.0) * 1.9
        + np.maximum(b_norm - 0.56, 0.0) * 1.5
    )

    # High-frequency structural texture (buildings/trees have sharp edges compared to cloud/sky regions)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigmaX=2.0)
    tex_norm = np.clip(grad_mag / 0.45, 0.0, 1.0)

    # Vertical top-bright -> bottom-dark edge response reinforces horizon boundaries
    down_edge = np.clip(cv2.Sobel(L_norm, cv2.CV_32F, 0, 1, ksize=3) * -1.0, 0.0, 1.0)
    down_edge = cv2.GaussianBlur(down_edge, (0, 0), sigmaX=1.5)

    sky_affinity = (
        0.58 * L_norm
        + 0.32 * np.clip(blue_minus_red + 0.25, 0.0, 1.0)
        - 0.48 * warm_or_green
        - 0.40 * tex_norm
        - 0.16 * S_norm
    ).astype(np.float32)

    n_cols = 48
    xs_cols = np.linspace(0, work_w, n_cols + 1, dtype=int)
    col_x = np.array([(xs_cols[i] + xs_cols[i + 1]) * 0.5 for i in range(n_cols)], dtype=np.float64)
    S_strip = np.zeros((work_h, n_cols), dtype=np.float64)
    L_strip = np.zeros((work_h, n_cols), dtype=np.float64)
    G_strip = np.zeros((work_h, n_cols), dtype=np.float64)

    for i in range(n_cols):
        x0, x1 = int(xs_cols[i]), max(int(xs_cols[i]) + 1, int(xs_cols[i + 1]))
        S_strip[:, i] = np.mean(sky_affinity[:, x0:x1], axis=1)
        L_strip[:, i] = np.mean(L_norm[:, x0:x1], axis=1)
        G_strip[:, i] = np.mean(down_edge[:, x0:x1], axis=1)

    y_min = max(6, int(round(0.08 * work_h)))
    y_max = min(work_h - 6, int(round(0.95 * work_h)))
    if y_max <= y_min + 2:
        y_min, y_max = 2, max(4, work_h - 2)

    ys = np.arange(y_min, y_max, dtype=np.float64)[:, None]
    cs_S = np.cumsum(S_strip, axis=0)
    cs_L = np.cumsum(L_strip, axis=0)
    tot_S = cs_S[-1:, :]
    tot_L = cs_L[-1:, :]

    above_S = cs_S[y_min:y_max, :] / np.maximum(ys + 1.0, 1.0)
    below_S = (tot_S - cs_S[y_min:y_max, :]) / np.maximum(float(work_h - 1) - ys, 1.0)
    above_L = cs_L[y_min:y_max, :] / np.maximum(ys + 1.0, 1.0)
    below_L = (tot_L - cs_L[y_min:y_max, :]) / np.maximum(float(work_h - 1) - ys, 1.0)

    # Strong positive contrast requires sky above and terrain below
    sep_scores = (above_S - below_S) + 0.35 * (above_L - below_L) + 0.14 * G_strip[y_min:y_max, :]
    best_rel_idx = np.argmax(sep_scores, axis=0)
    best_y = (best_rel_idx + y_min).astype(np.float64)
    best_sep = sep_scores[best_rel_idx, np.arange(n_cols)]

    strong_mask = best_sep >= 0.13
    n_strong = int(np.count_nonzero(strong_mask))
    med_sep = float(np.median(best_sep[strong_mask])) if n_strong > 0 else float(np.median(best_sep))
    med_y_norm = float(np.median(best_y[strong_mask]) / work_h) if n_strong > 0 else 1.0

    # Check if bottom border itself is textured terrain/roof or open sky
    bottom_sky_diff = float(np.mean(S_strip[: max(4, work_h // 6), :]) - np.mean(S_strip[-max(4, work_h // 6) :, :]))

    if n_strong < max(10, int(0.32 * n_cols)) or (med_sep < 0.17 and bottom_sky_diff < 0.16) or med_y_norm > 0.935:
        zenith_geom = CameraGeometry(
            elevation_deg=85.0,
            azimuth_deg=0.0,
            roll_deg=0.0,
            fov_horizontal_deg=fov_h,
            fov_vertical_deg=fov_v,
            cloud_altitude_m=DEFAULT_PROJECTION_CLOUD_ALTITUDE_M,
            tilt_correction_enabled=True,
        )
        zenith_maps = compute_pixel_geometry_maps(64, 48, zenith_geom)
        return {
            "detected_horizon": False,
            "mode": "zenith_sky_only",
            "orientation_elevation_deg": 85.0,
            "orientation_roll_deg": 0.0,
            "tilt_from_vertical_deg": 5.0,
            "horizon_y_norm": None,
            "below_horizon_pct": float(zenith_maps["below_horizon_pct"]),
            "confidence": round(float(np.clip(0.82 - max(0.0, med_sep) * 1.5, 0.45, 0.90)), 2),
        }

    x_pts = col_x[strong_mask]
    y_pts = best_y[strong_mask]
    w_pts = np.clip(best_sep[strong_mask], 0.1, 1.5)

    # Deterministic pair-consensus (RANSAC-like) horizon line fit immune to narrow skyscraper spikes
    tol_px = max(4.0, 0.045 * work_h)
    best_line = (0.0, float(np.percentile(y_pts, 58)))  # (slope, intercept_at_cx)
    best_consensus = -1.0
    cx_w = work_w / 2.0
    m_pts = len(x_pts)
    step_pair = 1 if m_pts <= 24 else 2
    for i in range(0, m_pts, step_pair):
        for j in range(i + 2, m_pts, step_pair):
            dx = x_pts[j] - x_pts[i]
            if abs(dx) < work_w * 0.18:
                continue
            slope_cand = (y_pts[j] - y_pts[i]) / dx
            # Reject extreme implausible camera rolls (> ~42 deg)
            if abs(slope_cand) > 0.90:
                continue
            b_cand = y_pts[i] + slope_cand * (cx_w - x_pts[i])
            pred = b_cand + slope_cand * (x_pts - cx_w)
            res = y_pts - pred
            # Urban skyline preference: tolerate slight upward buildings less than coherent ground base
            inlier = np.abs(res) <= tol_px
            score = float(np.sum(w_pts[inlier] * (1.0 - 0.5 * (np.abs(res[inlier]) / tol_px))))
            if score > best_consensus:
                best_consensus = score
                best_line = (float(slope_cand), float(b_cand))

    slope_fit, b_fit = best_line
    # Refine with weighted least-squares on final inliers (with slight IRLS passes)
    for _ in range(3):
        pred = b_fit + slope_fit * (x_pts - cx_w)
        abs_res = np.abs(y_pts - pred)
        inliers = abs_res <= tol_px * 1.25
        if np.count_nonzero(inliers) < 4:
            break
        dx_in = x_pts[inliers] - cx_w
        y_in = y_pts[inliers]
        huber_w = w_pts[inliers] / np.maximum(1.0, abs_res[inliers] / (0.5 * tol_px))
        # Downweight isolated sharp upward spikes (towers) slightly relative to horizon base
        res_in = y_in - (b_fit + slope_fit * dx_in)
        huber_w = np.where(res_in < -tol_px * 0.45, huber_w * 0.65, huber_w)
        sw = float(np.sum(huber_w))
        if sw <= 1e-6:
            break
        x_mean = float(np.sum(huber_w * dx_in) / sw)
        y_mean = float(np.sum(huber_w * y_in) / sw)
        denom = float(np.sum(huber_w * (dx_in - x_mean) ** 2))
        if denom > 1e-3:
            slope_new = float(np.sum(huber_w * (dx_in - x_mean) * (y_in - y_mean)) / denom)
            if abs(slope_new) <= 0.90:
                slope_fit = slope_new
        b_fit = y_mean - slope_fit * x_mean

    base_geom = CameraGeometry(0.0, 0.0, 0.0, fov_h, fov_v, DEFAULT_PROJECTION_CLOUD_ALTITUDE_M, True)
    fx_w, fy_w, _, cy_w = focal_lengths_px(work_w, work_h, base_geom)

    roll_rad = math.atan(-(fy_w / fx_w) * slope_fit)
    roll_deg = round(max(-45.0, min(45.0, math.degrees(roll_rad))), 1)
    el_rad = math.atan(((b_fit - cy_w) / fy_w) * math.cos(roll_rad))
    el_deg = round(max(-20.0, min(88.0, math.degrees(el_rad))), 1)

    est_geom = CameraGeometry(
        elevation_deg=el_deg,
        azimuth_deg=0.0,
        roll_deg=roll_deg,
        fov_horizontal_deg=fov_h,
        fov_vertical_deg=fov_v,
        cloud_altitude_m=DEFAULT_PROJECTION_CLOUD_ALTITUDE_M,
        tilt_correction_enabled=True,
    )
    est_maps = compute_pixel_geometry_maps(64, 48, est_geom)
    inlier_cnt = np.count_nonzero(np.abs(y_pts - (b_fit + slope_fit * (x_pts - cx_w))) <= tol_px * 1.25)
    inlier_frac = float(inlier_cnt) / max(1, m_pts)
    conf = float(np.clip(0.45 * min(1.0, med_sep / 0.42) + 0.35 * inlier_frac + 0.20 * (n_strong / n_cols), 0.25, 0.97))

    return {
        "detected_horizon": True,
        "mode": "horizon_visible",
        "orientation_elevation_deg": el_deg,
        "orientation_roll_deg": roll_deg,
        "tilt_from_vertical_deg": round(max(0.0, 90.0 - el_deg), 1),
        "horizon_y_norm": round(float(np.clip(b_fit / work_h, 0.0, 1.2)), 4),
        "below_horizon_pct": float(est_maps["below_horizon_pct"]),
        "confidence": round(conf, 2),
    }


def refine_azimuth_from_sun(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    geom: CameraGeometry,
    lat: float,
    lon: float,
    dt_utc: datetime | None = None,
) -> dict[str, Any]:
    """
    Refine camera compass azimuth by matching the brightest solar-disk centroid observed
    inside the valid sky mask against the astronomical NOAA solar azimuth at `dt_utc`.
    Returns proposed azimuth and residual error in degrees for user confirmation before applying.
    """
    solar = compute_solar_position(lat, lon, dt_utc)
    sun_el = float(solar["elevation_deg"])
    sun_az = float(solar["azimuth_deg"])
    if sun_el < 3.0:
        return {
            "ok": False,
            "reason": "Sun is below 3 deg elevation; solar azimuth refinement requires daylight sun in FOV.",
            "solar": solar,
        }

    h, w = image_bgr.shape[:2]
    vm = valid_mask > 0 if (valid_mask is not None and valid_mask.shape[:2] == (h, w)) else np.ones((h, w), dtype=bool)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sky_vals = gray[vm]
    if sky_vals.size < 100 or float(np.max(sky_vals)) < 245.0:
        return {
            "ok": False,
            "reason": "No saturated solar disk detected inside valid sky region.",
            "solar": solar,
        }

    sun_px = (gray >= 248) & vm
    n_sun = int(np.count_nonzero(sun_px))
    if n_sun < 8 or n_sun > int(0.18 * vm.sum()):
        return {
            "ok": False,
            "reason": "Bright clipped area is either too small or too wide (overcast glare) to locate solar disk.",
            "solar": solar,
        }

    ys, xs = np.where(sun_px)
    u_sun = float(np.mean(xs))
    v_sun = float(np.mean(ys))

    # Compute ray azimuth & elevation of observed centroid under current camera orientation
    r_cam, u_cam, f_cam = camera_basis_enu(geom.elevation_deg, geom.azimuth_deg, geom.roll_deg)
    fx, fy, cx, cy = focal_lengths_px(w, h, geom)
    xc = (u_sun - cx) / fx
    yc = (cy - v_sun) / fy
    ray = xc * r_cam + yc * u_cam + f_cam
    ray = ray / max(1e-9, float(np.linalg.norm(ray)))

    obs_el_deg = math.degrees(math.asin(max(-1.0, min(1.0, float(ray[2])))))
    obs_az_deg = math.degrees(math.atan2(float(ray[0]), float(ray[1]))) % 360.0

    # Minimal signed angular difference (sun_az - obs_az) in [-180, +180]
    delta_az = ((sun_az - obs_az_deg + 180.0) % 360.0) - 180.0
    proposed_az = round((geom.azimuth_deg + delta_az) % 360.0, 2)
    el_residual = abs(sun_el - obs_el_deg)

    # Confidence drops if elevation mismatch is large (implies sun-like cloud patch or wrong tilt)
    conf = round(float(np.clip(1.0 - el_residual / 25.0, 0.15, 0.96)), 2)
    return {
        "ok": True,
        "current_azimuth_deg": round(geom.azimuth_deg, 2),
        "proposed_azimuth_deg": proposed_az,
        "delta_azimuth_deg": round(delta_az, 2),
        "residual_elevation_deg": round(el_residual, 2),
        "observed_sun_uv": [round(u_sun / max(1, w - 1), 4), round(v_sun / max(1, h - 1), 4)],
        "confidence": conf,
        "solar": solar,
    }
