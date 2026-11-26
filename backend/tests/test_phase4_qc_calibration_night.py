from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.vision.geometry import (
    CameraGeometry,
    compute_solar_position,
    project_mask_to_radar_bgra,
    project_sky_angles_to_image,
    refine_azimuth_from_sun,
)
from app.vision.night import (
    classify_sky_regime,
    compute_lunar_position_and_phase,
    detect_bright_stars,
    segment_night_sky_adaptive,
)
from app.vision.quality import assess_image_quality


def test_hardware_qc_flags_ae_jump_reflection_and_jpeg_blocking():
    valid = np.full((240, 320), 255, dtype=np.uint8)
    rng = np.random.default_rng(123)

    # 1) AE/AWB jump: bright blue sky -> sudden +55 luminance / warm shift
    prev = np.full((240, 320, 3), (170, 125, 85), dtype=np.uint8)
    curr = np.full((240, 320, 3), (120, 180, 215), dtype=np.uint8)
    q_ae = assess_image_quality(curr, prev_bgr=prev, valid_mask=valid)
    assert "ae_awb_jump" in q_ae.flags

    # 2) Window-glass reflection in evening dim sky: dark background + sharp bright specular patch
    dim = np.full((240, 320, 3), (38, 34, 30), dtype=np.uint8)
    dim[60:110, 90:155] = (235, 235, 240)
    q_ref = assess_image_quality(dim, valid_mask=valid, sky_regime="twilight")
    assert "window_reflection" in q_ref.flags

    # 3) Heavy 8x8 JPEG blockiness (ESP32-CAM style flat 8x8 blocks with sharp 8px boundaries)
    blocks = np.zeros((240, 320, 3), dtype=np.uint8)
    for r in range(0, 240, 8):
        for c in range(0, 320, 8):
            val = int(rng.integers(75, 195))
            blocks[r : r + 8, c : c + 8] = (val, val, val)
    q_blk = assess_image_quality(blocks, valid_mask=valid)
    assert "heavy_jpeg_blocking" in q_blk.flags
    assert q_blk.metrics["blocking_ratio"] > 1.75


def test_zenith_cutoff_and_solar_azimuth_refinement():
    # Zenith-pointing camera with max_projection_zenith_deg = 60 -> elevation < 30 deg must have alpha = 0
    geom = CameraGeometry(
        elevation_deg=85.0,
        azimuth_deg=180.0,
        roll_deg=0.0,
        fov_horizontal_deg=120.0,
        fov_vertical_deg=100.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
        max_projection_zenith_deg=60.0,
    )
    cls_map = np.full((128, 128), 2, dtype=np.uint8)
    bgra = project_mask_to_radar_bgra(cls_map, geom, overlay_radius_km=10.0, size=128)
    # Corner at (-10 km, +10 km) has horiz_dist = 14.14 km -> el = atan2(2, 14.14) ~= 8 deg < 30 deg -> transparent
    assert int(bgra[0, 0, 3]) == 0
    # Center near camera zenith should be opaque cloud
    assert int(bgra[64, 64, 3]) > 150

    # Solar azimuth refinement: place synthetic sun disk at true astronomical azimuth + known 12 deg offset
    dt_noon = datetime(2026, 6, 21, 11, 30, 0, tzinfo=UTC)
    lat, lon = 45.0, 9.0
    solar = compute_solar_position(lat, lon, dt_noon)
    true_az = float(solar["azimuth_deg"])
    true_el = float(solar["elevation_deg"])

    # True camera faces (true_az), but configured azimuth is off by -12 deg
    true_geom = CameraGeometry(
        elevation_deg=true_el,
        azimuth_deg=true_az,
        roll_deg=0.0,
        fov_horizontal_deg=70.0,
        fov_vertical_deg=50.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
    )
    proj = project_sky_angles_to_image(true_az, true_el, 320, 240, true_geom)
    assert proj["in_camera_fov"] is True

    img = np.full((240, 320, 3), (180, 120, 70), dtype=np.uint8)
    cx = int(round(proj["u_norm"] * 319))
    cy = int(round(proj["v_norm"] * 239))
    img[max(0, cy - 6) : min(240, cy + 7), max(0, cx - 6) : min(320, cx + 7)] = (255, 255, 255)

    wrong_geom = CameraGeometry(
        elevation_deg=true_el,
        azimuth_deg=(true_az - 12.0) % 360.0,
        roll_deg=0.0,
        fov_horizontal_deg=70.0,
        fov_vertical_deg=50.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
    )
    res = refine_azimuth_from_sun(img, np.full((240, 320), 255, dtype=np.uint8), wrong_geom, lat, lon, dt_noon)
    assert res["ok"] is True
    assert abs(float(res["proposed_azimuth_deg"]) - true_az) < 2.0
    assert float(res["residual_elevation_deg"]) < 2.0


def test_adaptive_night_mode_stars_moon_and_urban_segmentation():
    assert classify_sky_regime(15.0) == "day"
    assert classify_sky_regime(0.0) == "twilight"
    assert classify_sky_regime(-12.0) == "night"

    lunar = compute_lunar_position_and_phase(45.0, 9.0, datetime(2026, 6, 21, 23, 0, 0, tzinfo=UTC))
    assert 0.0 <= lunar["phase_fraction"] <= 1.0
    assert -90.0 <= lunar["elevation_deg"] <= 90.0

    # 1) Dark-site starry clear night sky -> stars detected, low cloud coverage
    starry = np.full((240, 320, 3), (10, 8, 7), dtype=np.uint8)
    valid = np.full((240, 320), 255, dtype=np.uint8)
    for y in range(25, 215, 38):
        for x in range(30, 290, 48):
            starry[y - 1 : y + 1, x - 1 : x + 1] = (230, 230, 240)
    stars = detect_bright_stars(starry, valid)
    assert stars["star_count"] >= 12
    assert stars["stars_usable"] is True

    seg_dark, meta_dark = segment_night_sky_adaptive(
        starry, valid, urban_light_pollution=False, lunar_info=lunar, star_info=stars
    )
    assert meta_dark["night_site_mode"] == "dark_site"
    assert float(np.mean(seg_dark.cloud_binary_mask > 0)) < 0.15

    # 2) Urban light-polluted night with glowing low clouds on top half and dark clear sky on bottom half
    urban = np.full((240, 320, 3), (22, 20, 20), dtype=np.uint8)
    urban[:120, :] = (58, 74, 110)  # Warm sodium/LED-lit cloud deck
    seg_urb, meta_urb = segment_night_sky_adaptive(
        urban, valid, urban_light_pollution=True, lunar_info=lunar, star_info={"star_count": 0, "stars_usable": False}
    )
    assert meta_urb["night_site_mode"] == "urban_light_pollution"
    assert float(np.mean(seg_urb.cloud_binary_mask[:110, :] > 0)) > 0.70
    assert float(np.mean(seg_urb.cloud_binary_mask[130:, :] == 0)) > 0.70
