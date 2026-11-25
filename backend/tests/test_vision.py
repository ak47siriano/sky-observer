from __future__ import annotations

import cv2
import numpy as np

from app.ai.datasets import _render_synthetic_sky_frame
from app.config import ExclusionPolygon
from app.vision.classification import classify_cloud_type
from app.vision.color import analyze_color
from app.vision.coverage import compute_cloud_coverage
from app.vision.masking import build_valid_sky_mask
from app.vision.motion import estimate_cloud_motion
from app.vision.pipeline import run_vision_pipeline
from app.vision.quality import assess_image_quality
from app.vision.segmentation import run_segmentation
from app.vision.texture import analyze_texture


def test_image_quality_checks():
    sky_img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=0, seed=11, width=320, height=240)
    res_ok = assess_image_quality(sky_img)
    assert res_ok.is_usable is True
    assert res_ok.quality_score > 0.6

    # Excessively dark frame
    dark = np.full((240, 320, 3), 6, dtype=np.uint8)
    res_dark = assess_image_quality(dark)
    assert res_dark.is_usable is False
    assert "excessively_dark" in res_dark.flags

    # Overexposed frame
    bright = np.full((240, 320, 3), 252, dtype=np.uint8)
    res_over = assess_image_quality(bright)
    assert res_over.is_usable is False

    # Identical duplicate frame detection
    res_dup = assess_image_quality(sky_img, prev_bgr=sky_img.copy())
    assert "repeated_identical_frame" in res_dup.flags

    # Sudden camera movement detection (shift non-periodic landmark scene by 35px, 12px)
    scene = sky_img.copy()
    cv2.rectangle(scene, (45, 130), (115, 220), (30, 35, 45), -1)
    cv2.circle(scene, (220, 90), 28, (250, 250, 250), -1)
    M = np.float32([[1, 0, 35], [0, 1, 12]])
    shifted = cv2.warpAffine(scene, M, (320, 240), borderMode=cv2.BORDER_REPLICATE)
    res_move = assess_image_quality(shifted, prev_bgr=scene, camera_move_shift_px=15.0)
    assert "camera_movement" in res_move.flags
    assert res_move.camera_shift_px > 15.0


def test_sky_roi_and_exclusion_masking():
    ex = ExclusionPolygon(id="tree1", label="Tree", category="trees", points=[[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]])
    mask = build_valid_sky_mask(
        height=100,
        width=100,
        sky_roi_polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        exclusion_polygons=[ex],
    )
    assert mask[20, 50] == 255
    assert mask[80, 50] == 0

    # Coverage calculated only on valid sky pixels (top half clear, bottom half excluded white cloud)
    class_map = np.zeros((100, 100), dtype=np.uint8)
    class_map[55:, :] = 2  # dense cloud strictly inside excluded bottom zone
    cov = compute_cloud_coverage(class_map, mask, mean_confidence=0.9, grid_rows=3, grid_cols=3)
    assert cov["cloud_coverage_pct"] == 0.0
    assert cov["clear_sky_pct"] == 100.0
    assert len(cov["grid"]) == 9


def test_segmentation_color_texture_and_classification():
    img, _gt, _genus, _ev, _ = _render_synthetic_sky_frame("cumulus", t_step=2, seed=42, width=320, height=320)
    valid = np.full((320, 320), 255, dtype=np.uint8)
    seg = run_segmentation(img, valid, inference_resolution=256)
    assert seg.class_map.shape == (320, 320)
    assert 0.0 < seg.mean_confidence <= 1.0

    cov = compute_cloud_coverage(seg.class_map, valid, seg.mean_confidence)
    assert 0.0 <= cov["cloud_coverage_pct"] <= 100.0

    col = analyze_color(img, valid)
    assert "rgb_mean" in col and "hsv_mean" in col and "lab_mean" in col
    assert 2000.0 <= col["color_temperature_k"] <= 15000.0

    tex = analyze_texture(img, valid, seg.cloud_binary_mask)
    assert "glcm_contrast" in tex and "fragmentation_index" in tex

    cls = classify_cloud_type(img, cov, col, tex)
    assert abs(sum(cls["probabilities"].values()) - 1.0) < 0.02
    assert cls["dominant_cloud_type"] in cls["probabilities"]


def test_optical_flow_motion_direction():
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame1[:, :] = (190, 130, 50)
    cv2.circle(frame1, (100, 120), 28, (245, 245, 245), -1)
    cv2.circle(frame1, (160, 100), 22, (240, 240, 240), -1)

    # Shift cloud blobs right by +12px over 2.0 seconds -> +6 px/s East (~90 deg)
    M = np.float32([[1, 0, 12], [0, 1, 0]])
    frame2 = cv2.warpAffine(frame1, M, (320, 240), borderMode=cv2.BORDER_REPLICATE)
    valid = np.full((240, 320), 255, dtype=np.uint8)
    cloud_mask = np.where(frame2[:, :, 2] > 180, 255, 0).astype(np.uint8)

    mot, overlay = estimate_cloud_motion(frame1, frame2, valid, cloud_mask, dt_seconds=2.0, fov_horizontal_deg=72.0)
    assert mot["available"] is True
    assert mot["dx_px_s"] > 1.5
    assert mot["direction_cardinal"] in {"E", "NE", "SE"}
    assert mot["angular_speed_deg_s"] is not None
    assert mot["derived_ground_speed_kmh"] is None  # Must NOT invent km/h without altitude
    assert overlay.shape == frame2.shape


def test_full_vision_pipeline(isolated_env):
    img, _, _, _, _ = _render_synthetic_sky_frame("stratocumulus", t_step=1, seed=7, width=320, height=240)
    out = run_vision_pipeline(img, isolated_env)
    assert out.is_usable is True
    assert out.visual_condition != "unusable_frame"
    assert out.inference_duration_ms > 0


def test_painted_exclusion_mask_ignores_logo_and_moving_cars(isolated_env):
    # Create clean clear blue sky frame with moving bright "logo/cars" inside bottom strip (y >= 170)
    base_sky = np.zeros((240, 320, 3), dtype=np.uint8)
    base_sky[:, :] = (205, 130, 50)
    # Subtle grain on sky
    rng = np.random.default_rng(9)
    base_sky = np.clip(base_sky.astype(np.int32) + rng.integers(-3, 4, base_sky.shape), 0, 255).astype(np.uint8)

    frame1 = base_sky.copy()
    cv2.rectangle(frame1, (20, 190), (110, 225), (255, 255, 255), -1)  # Bright logo/car at bottom

    frame2 = base_sky.copy()
    cv2.rectangle(frame2, (160, 190), (260, 225), (255, 255, 255), -1)  # Moved 140px horizontally inside excluded strip

    # Paint bottom strip (y >= 175) in exclusion mask
    mask_file = isolated_env.data_dir / "masks" / "test_excl.png"
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    paint = np.zeros((240, 320), dtype=np.uint8)
    paint[175:, :] = 255
    cv2.imwrite(str(mask_file), paint)
    isolated_env.camera.exclusion_mask_path = str(mask_file)

    out = run_vision_pipeline(frame2, isolated_env, prev_bgr=frame1, dt_seconds=2.0)
    assert out.is_usable is True
    assert "camera_movement" not in out.quality.get("flags", [])
    assert out.coverage["cloud_coverage_pct"] == 0.0
    assert out.motion["magnitude_px_s"] < 0.5


def test_camera_movement_ignores_small_1080p_sky_drift():
    # Two 1080p sky frames with small cloud advection (~18px cloud movement, ~3px on 320x240 grid)
    sky1, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=0, seed=11, width=1920, height=1080)
    sky2, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=1, seed=11, width=1920, height=1080)
    res_sky = assess_image_quality(sky2, prev_bgr=sky1, camera_move_shift_px=18.0)
    assert "camera_movement" not in res_sky.flags

    # Genuine large camera bump on a 640x480 scene with static landmarks shifted (45, 20) px
    scene = sky1[:480, :640].copy()
    cv2.rectangle(scene, (90, 260), (230, 440), (30, 35, 45), -1)
    cv2.circle(scene, (440, 180), 56, (250, 250, 250), -1)
    M = np.float32([[1, 0, 45], [0, 1, 20]])
    shifted = cv2.warpAffine(scene, M, (640, 480), borderMode=cv2.BORDER_REPLICATE)
    res_bump = assess_image_quality(shifted, prev_bgr=scene, camera_move_shift_px=18.0)
    assert "camera_movement" in res_bump.flags


def test_camera_tilt_geometry_and_coverage_weighting():
    from app.vision.geometry import CameraGeometry, camera_basis_enu, compute_pixel_geometry_maps

    # ENU basis must be orthonormal across arbitrary tilt, azimuth, and roll
    for el in (15.0, 45.0, 90.0):
        for az in (0.0, 90.0, 215.0):
            r, u, f = camera_basis_enu(el, az, roll_deg=10.0)
            assert np.allclose([np.linalg.norm(r), np.linalg.norm(u), np.linalg.norm(f)], [1.0, 1.0, 1.0], atol=1e-6)
            assert abs(float(np.dot(r, u))) < 1e-6
            assert abs(float(np.dot(r, f))) < 1e-6
            assert abs(float(np.dot(u, f))) < 1e-6

    # Low elevation + wide vertical FOV -> bottom strip is below horizon
    geom_low = CameraGeometry(
        elevation_deg=15.0,
        azimuth_deg=0.0,
        roll_deg=0.0,
        fov_horizontal_deg=72.0,
        fov_vertical_deg=45.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
    )
    maps_low = compute_pixel_geometry_maps(320, 240, geom_low)
    assert maps_low["below_horizon_pct"] > 5.0
    assert bool(np.all(~maps_low["above_horizon"][-10:, :]))

    # Clouds occupying bottom 50% (lower elevation):
    # - Zenith pointing (90 deg) weights top and bottom symmetrically -> ~50%
    # - Tilted (35 deg) downweights oblique lower-elevation cloud foreshortening -> strictly < 50%
    class_map = np.zeros((240, 320), dtype=np.uint8)
    class_map[120:, :] = 2
    valid = np.full((240, 320), 255, dtype=np.uint8)

    geom_zenith = CameraGeometry(90.0, 0.0, 0.0, 72.0, 45.0, 2000.0, True)
    maps_z = compute_pixel_geometry_maps(320, 240, geom_zenith)
    cov_z = compute_cloud_coverage(class_map, valid, 0.9, pixel_weights=maps_z["sky_weight"])
    assert abs(cov_z["cloud_coverage_pct"] - 50.0) < 1.5

    geom_tilt = CameraGeometry(35.0, 0.0, 0.0, 72.0, 45.0, 2000.0, True)
    maps_t = compute_pixel_geometry_maps(320, 240, geom_tilt)
    cov_t = compute_cloud_coverage(class_map, valid, 0.9, pixel_weights=maps_t["sky_weight"])
    assert cov_t["raw_cloud_coverage_pct"] == 50.0
    assert cov_t["cloud_coverage_pct"] < 42.0


def test_camera_tilt_motion_compass_correction():
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame1[:, :] = (190, 130, 50)
    cv2.circle(frame1, (130, 120), 26, (245, 245, 245), -1)
    cv2.circle(frame1, (190, 95), 22, (240, 240, 240), -1)
    valid = np.full((240, 320), 255, dtype=np.uint8)

    # Rightward pixel drift (dx > 0) rotated by camera compass azimuth
    frame_right = cv2.warpAffine(frame1, np.float32([[1, 0, 12], [0, 1, 0]]), (320, 240), borderMode=cv2.BORDER_REPLICATE)
    mask_right = np.where(frame_right[:, :, 2] > 180, 255, 0).astype(np.uint8)

    m_n, _ = estimate_cloud_motion(
        frame1, frame_right, valid, mask_right, dt_seconds=2.0,
        fov_horizontal_deg=72.0, fov_vertical_deg=45.0,
        orientation_elevation_deg=45.0, orientation_azimuth_deg=0.0, tilt_correction_enabled=True,
    )
    assert m_n["tilt_corrected"] is True
    assert 65.0 <= m_n["direction_deg"] <= 115.0  # East

    m_e, _ = estimate_cloud_motion(
        frame1, frame_right, valid, mask_right, dt_seconds=2.0,
        fov_horizontal_deg=72.0, fov_vertical_deg=45.0,
        orientation_elevation_deg=45.0, orientation_azimuth_deg=90.0, tilt_correction_enabled=True,
    )
    assert 155.0 <= m_e["direction_deg"] <= 205.0  # South when facing East

    m_s, _ = estimate_cloud_motion(
        frame1, frame_right, valid, mask_right, dt_seconds=2.0,
        fov_horizontal_deg=72.0, fov_vertical_deg=45.0,
        orientation_elevation_deg=45.0, orientation_azimuth_deg=180.0, tilt_correction_enabled=True,
    )
    assert 245.0 <= m_s["direction_deg"] <= 295.0  # West when facing South

    # Upward pixel drift (dy < 0) when facing North: cloud moving from northern horizon toward overhead -> South (~180 deg)
    frame_up = cv2.warpAffine(frame1, np.float32([[1, 0, 0], [0, 1, -12]]), (320, 240), borderMode=cv2.BORDER_REPLICATE)
    mask_up = np.where(frame_up[:, :, 2] > 180, 255, 0).astype(np.uint8)
    m_up, _ = estimate_cloud_motion(
        frame1, frame_up, valid, mask_up, dt_seconds=2.0,
        fov_horizontal_deg=72.0, fov_vertical_deg=45.0,
        orientation_elevation_deg=45.0, orientation_azimuth_deg=0.0, tilt_correction_enabled=True,
    )
    assert 155.0 <= m_up["direction_deg"] <= 205.0


def test_astronomical_solar_position_and_sun_glare_mask():
    from datetime import datetime, UTC

    from app.vision.geometry import (
        CameraGeometry,
        compute_solar_position,
        compute_sun_glare_mask,
        project_sky_angles_to_image,
    )

    # Noon UTC in summer over Rome (41.90N, 12.50E) must be high daylight sun in Southern sky
    noon_utc = datetime(2026, 6, 21, 11, 15, 0, tzinfo=UTC)
    sol_day = compute_solar_position(41.9028, 12.4964, noon_utc)
    assert sol_day["is_daylight"] is True
    assert sol_day["elevation_deg"] > 60.0
    assert 150.0 <= sol_day["azimuth_deg"] <= 210.0

    # Midnight UTC must be below horizon
    mid_utc = datetime(2026, 6, 21, 23, 30, 0, tzinfo=UTC)
    sol_night = compute_solar_position(41.9028, 12.4964, mid_utc)
    assert sol_night["is_daylight"] is False
    assert sol_night["elevation_deg"] < 0.0

    # Point camera directly at the daytime sun -> projects to center of image & activates glare mask
    geom_sun = CameraGeometry(
        elevation_deg=float(sol_day["elevation_deg"]),
        azimuth_deg=float(sol_day["azimuth_deg"]),
        roll_deg=0.0,
        fov_horizontal_deg=72.0,
        fov_vertical_deg=45.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
    )
    proj = project_sky_angles_to_image(sol_day["azimuth_deg"], sol_day["elevation_deg"], 320, 240, geom_sun)
    assert proj["in_camera_fov"] is True
    assert abs(proj["u_norm"] - 0.5) < 0.03
    assert abs(proj["v_norm"] - 0.5) < 0.03

    glare = compute_sun_glare_mask(320, 240, geom_sun, sol_day["azimuth_deg"], sol_day["elevation_deg"], max_angle_deg=4.0)
    assert bool(glare[120, 160]) is True
    assert bool(glare[10, 10]) is False


def test_sun_glare_and_aureole_segmentation():
    from app.vision.geometry import CameraGeometry, compute_sun_aureole_prior

    # Clear blue sky with localized bright solar disk + soft Mie aureole in center
    img = np.full((240, 320, 3), (205, 130, 50), dtype=np.uint8)
    # Soft bright bloom around sun center (160, 120)
    cv2.circle(img, (160, 120), 26, (238, 240, 244), -1)
    img = cv2.GaussianBlur(img, (21, 21), 8.0)
    # Clipped solar disk core
    cv2.circle(img, (160, 120), 10, (254, 255, 255), -1)

    valid = np.full((240, 320), 255, dtype=np.uint8)
    geom = CameraGeometry(
        elevation_deg=45.0,
        azimuth_deg=180.0,
        roll_deg=0.0,
        fov_horizontal_deg=72.0,
        fov_vertical_deg=54.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
    )
    sun_prior = compute_sun_aureole_prior(320, 240, geom, 180.0, 45.0, core_deg=3.5, halo_deg=14.0)
    seg = run_segmentation(img, valid, inference_resolution=256, sun_prior_map=sun_prior)
    # Center solar disk must be classified as sun_glare (4), not dense_cloud (2)
    assert int(seg.class_map[120, 160]) == 4
    # Outer corner must remain clear_sky (0)
    assert int(seg.class_map[15, 15]) == 0
    # Sun glare pixels must be excluded from cloud_binary_mask
    assert int(seg.cloud_binary_mask[120, 160]) == 0
    cov = compute_cloud_coverage(seg.class_map, valid, seg.mean_confidence)
    assert cov["sun_glare_pct"] > 0.0
    assert cov["clear_sky_pct"] > 70.0


def test_dark_cloud_underside_not_clear_sky():
    # Overcast scene with bright gray-white cloud tops (L~82, slight cool AWB cast)
    # and darker cool-gray cloud bases (L~54, slight blue-gray cast)
    img = np.full((240, 320, 3), (212, 198, 184), dtype=np.uint8)  # bright cloudy deck
    img[110:210, 40:260] = (156, 136, 118)  # shaded darker cloud underside
    valid = np.full((240, 320), 255, dtype=np.uint8)
    seg = run_segmentation(img, valid, inference_resolution=256)
    cov = compute_cloud_coverage(seg.class_map, valid, seg.mean_confidence)
    # Dark cloud base must be segmented as dark/dense cloud, NOT clear_sky or sun_glare
    assert cov["clear_sky_pct"] < 5.0
    assert cov["sun_glare_pct"] == 0.0
    assert cov["dense_cloud_pct"] + cov["dark_cloud_pct"] > 80.0
    assert int(seg.class_map[160, 150]) in (2, 3)


def test_auto_estimate_camera_tilt_and_horizon():
    from app.vision.geometry import CameraGeometry, estimate_camera_tilt_from_frame, project_mask_to_radar_bgra

    # 1) Tilted webcam with horizon near ~60% height (camera ~ +4..6 deg above horizon) & roll ~ +5 deg
    w, h = 640, 360
    scene = np.full((h, w, 3), (218, 196, 175), dtype=np.uint8)  # bright cloudy sky
    # Paint dark/textured urban terrain + green/warm rooftops below tilted horizon line
    # For roll=+5 deg (tan=+0.0875), fy/fx = tan(fovH/2)/tan(fovV/2) ~ 1 => slope dv/du ~ -0.0875
    cx = w / 2.0
    y_center = 0.58 * h
    slope = -0.09
    yy, xx = np.mgrid[0:h, 0:w]
    horizon_y = y_center + slope * (xx - cx)
    ground_mask = yy >= horizon_y
    scene[ground_mask] = (48, 74, 62)  # dark urban/foliage base
    # Add terracotta roof texture blocks in ground region
    rng = np.random.default_rng(7)
    noise = rng.integers(-18, 19, size=(h, w, 3), dtype=np.int16)
    scene[ground_mask] = np.clip(scene[ground_mask].astype(np.int16) + noise[ground_mask], 0, 255).astype(np.uint8)
    # Tall skyscraper spike jutting above left skyline (should NOT ruin robust RANSAC fit)
    cv2.rectangle(scene, (110, int(y_center - 55)), (145, h), (55, 58, 66), -1)

    est = estimate_camera_tilt_from_frame(scene, fov_horizontal_deg=72.0, fov_vertical_deg=45.0)
    assert est["detected_horizon"] is True
    assert est["mode"] == "horizon_visible"
    assert 1.5 <= est["orientation_elevation_deg"] <= 9.0
    assert 2.0 <= est["orientation_roll_deg"] <= 8.5
    assert est["confidence"] >= 0.45

    # 2) Pure zenith all-sky frame (no terrain) -> zenith_sky_only fallback
    sky_only, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=0, seed=4, width=320, height=240)
    est_zen = estimate_camera_tilt_from_frame(sky_only, fov_horizontal_deg=72.0, fov_vertical_deg=54.0)
    assert est_zen["detected_horizon"] is False
    assert est_zen["mode"] == "zenith_sky_only"
    assert est_zen["orientation_elevation_deg"] >= 80.0

    # 3) Low-elevation radar projection remains smooth & finite
    cls_map = np.full((120, 160), 2, dtype=np.uint8)
    cls_map[80:, :] = 0
    geom_low = CameraGeometry(
        elevation_deg=est["orientation_elevation_deg"],
        azimuth_deg=180.0,
        roll_deg=est["orientation_roll_deg"],
        fov_horizontal_deg=72.0,
        fov_vertical_deg=45.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
    )
    bgra = project_mask_to_radar_bgra(cls_map, geom_low, overlay_radius_km=20.0, size=192)
    assert bgra.shape == (192, 192, 4)
    assert int(np.max(bgra[:, :, 3])) > 0
