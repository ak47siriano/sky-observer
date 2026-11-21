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
