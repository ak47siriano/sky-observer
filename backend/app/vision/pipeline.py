"""Unified Computer Vision Pipeline combining Quality, Masking, Segmentation, Coverage, Motion, Classification, Color & Texture."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.config import AppConfig
from app.vision.classification import classify_cloud_type
from app.vision.color import analyze_color
from app.vision.coverage import compute_cloud_coverage
from app.vision.geometry import (
    compute_pixel_geometry_maps,
    compute_solar_position,
    compute_sun_aureole_prior,
    geometry_from_camera_config,
)
from app.vision.masking import build_valid_sky_mask_from_config, neutralize_excluded_pixels
from app.vision.motion import estimate_cloud_motion
from app.vision.quality import assess_image_quality
from app.vision.segmentation import run_segmentation
from app.vision.texture import analyze_texture


@dataclass
class VisionPipelineOutput:
    quality: dict[str, Any]
    quality_score: float
    is_usable: bool
    frame_hash: str
    coverage: dict[str, Any]
    motion: dict[str, Any]
    classification: dict[str, Any]
    color: dict[str, Any]
    texture: dict[str, Any]
    visual_condition: str
    inference_duration_ms: float
    class_map: np.ndarray | None
    cloud_binary_mask: np.ndarray | None
    seg_overlay_bgr: np.ndarray | None
    seg_color_mask_bgr: np.ndarray | None
    motion_overlay_bgr: np.ndarray | None
    valid_sky_mask: np.ndarray


def _derive_visual_condition(coverage: dict[str, Any], color: dict[str, Any], is_usable: bool) -> str:
    if not is_usable:
        return "unusable_frame"
    cov = float(coverage.get("cloud_coverage_pct", 0.0))
    dark = float(coverage.get("dark_cloud_pct", 0.0))
    haze = float(coverage.get("haze_fog_pct", 0.0))
    lum = float(color.get("avg_luminance", 50.0))
    if dark >= 35.0 and lum < 45.0:
        return "dark_storm_like_clouds"
    if haze >= 45.0 and cov < 50.0:
        return "hazy_sky"
    if cov < 12.0:
        return "clear_sky"
    if cov < 35.0:
        return "mostly_clear"
    if cov < 65.0:
        return "partly_cloudy"
    if cov < 85.0:
        return "mostly_cloudy"
    if dark >= 25.0:
        return "dark_overcast"
    return "overcast"


def run_vision_pipeline(
    image_bgr: np.ndarray,
    config: AppConfig,
    prev_bgr: np.ndarray | None = None,
    prev_cloud_mask: np.ndarray | None = None,
    prev_color_metrics: dict[str, Any] | None = None,
    prev_texture_metrics: dict[str, Any] | None = None,
    recent_coverages: list[tuple[float, float]] | None = None,
    dt_seconds: float = 5.0,
    seg_model: torch.nn.Module | None = None,
    seg_model_name: str = "radiometric-physics-v1",
    seg_nn_weight: float = 0.75,
    cls_model: torch.nn.Module | None = None,
    cls_model_name: str = "physical-heuristic-v1",
    device: str = "cpu",
) -> VisionPipelineOutput:
    """Execute full computer vision suite on a captured BGR frame."""
    t0 = time.perf_counter()
    h, w = image_bgr.shape[:2]
    geom = geometry_from_camera_config(config.camera, width=w, height=h)
    geo_maps = compute_pixel_geometry_maps(w, h, geom)
    valid_mask = build_valid_sky_mask_from_config((h, w), config.camera)
    solar = compute_solar_position(config.camera.latitude, config.camera.longitude)
    sun_prior_map: np.ndarray | None = None
    if geom.tilt_correction_enabled:
        horizon_masked = np.where(geo_maps["above_horizon"], valid_mask, 0).astype(np.uint8)
        if int(np.count_nonzero(horizon_masked)) >= max(100, int(0.05 * h * w)):
            valid_mask = horizon_masked
        if solar["is_daylight"] and float(solar["elevation_deg"]) > 0.5:
            sun_prior_map = compute_sun_aureole_prior(
                w,
                h,
                geom,
                float(solar["azimuth_deg"]),
                float(solar["elevation_deg"]),
                core_deg=3.5,
                halo_deg=14.0,
            )

    # 7.1 Image quality assessment
    if config.vision.image_quality_check_enabled:
        q_res = assess_image_quality(
            image_bgr=image_bgr,
            prev_bgr=prev_bgr,
            valid_mask=valid_mask,
            min_quality_score=config.vision.min_quality_score,
            duplicate_threshold=config.camera.duplicate_threshold,
            camera_move_shift_px=config.events.camera_move_shift_px,
        )
    else:
        from app.vision.quality import QualityResult, compute_dhash

        q_res = QualityResult(
            quality_score=1.0,
            is_usable=True,
            flags=[],
            metrics={"dhash": compute_dhash(image_bgr), "flags": []},
            frame_hash="",
        )

    # 7.7 Color analysis
    color_metrics = analyze_color(
        image_bgr=image_bgr,
        valid_mask=valid_mask,
        prev_color_metrics=prev_color_metrics,
        grid_rows=config.vision.grid_rows,
        grid_cols=config.vision.grid_cols,
    )

    if not q_res.is_usable:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VisionPipelineOutput(
            quality=q_res.metrics,
            quality_score=q_res.quality_score,
            is_usable=False,
            frame_hash=q_res.frame_hash,
            coverage={
                "cloud_coverage_pct": 0.0,
                "clear_sky_pct": 0.0,
                "thin_cloud_pct": 0.0,
                "dense_cloud_pct": 0.0,
                "dark_cloud_pct": 0.0,
                "haze_fog_pct": 0.0,
                "uncertainty_pct": 100.0,
                "confidence": 0.0,
                "trend": "unavailable",
                "rate_of_change_pct_per_min": 0.0,
                "grid": [],
            },
            motion={"available": False, "reason": "Frame flagged unusable by quality check"},
            classification={"probabilities": {"unknown": 1.0}, "dominant_cloud_type": "unknown", "confidence": 0.0},
            color=color_metrics,
            texture={},
            visual_condition="unusable_frame",
            inference_duration_ms=round(elapsed, 2),
            class_map=None,
            cloud_binary_mask=None,
            seg_overlay_bgr=None,
            seg_color_mask_bgr=None,
            motion_overlay_bgr=None,
            valid_sky_mask=valid_mask,
        )

    # 7.3 Cloud segmentation
    seg_out = run_segmentation(
        image_bgr=image_bgr,
        valid_mask=valid_mask,
        inference_resolution=config.vision.inference_resolution,
        uncertainty_entropy_threshold=config.vision.uncertainty_entropy_threshold,
        seg_model=seg_model if config.vision.cloud_segmentation_enabled else None,
        device=device,
        model_name=seg_model_name,
        sun_prior_map=sun_prior_map,
        nn_weight=seg_nn_weight,
    )

    # 7.4 Cloud coverage
    coverage_metrics = compute_cloud_coverage(
        class_map=seg_out.class_map,
        valid_mask=valid_mask,
        mean_confidence=seg_out.mean_confidence,
        recent_coverages=recent_coverages,
        grid_rows=config.vision.grid_rows,
        grid_cols=config.vision.grid_cols,
        pixel_weights=geo_maps["sky_weight"] if geom.tilt_correction_enabled else None,
        geometry_meta={
            "camera_elevation_deg": round(geom.elevation_deg, 2),
            "tilt_from_vertical_deg": geom.tilt_from_vertical_deg,
            "camera_azimuth_deg": round(geom.azimuth_deg, 2),
            "below_horizon_pct": geo_maps["below_horizon_pct"],
        }
        if geom.tilt_correction_enabled
        else None,
    )

    # 7.8 Texture & fragmentation
    texture_metrics = analyze_texture(
        image_bgr=image_bgr,
        valid_mask=valid_mask,
        cloud_binary_mask=seg_out.cloud_binary_mask,
        prev_cloud_binary_mask=prev_cloud_mask,
        prev_texture_metrics=prev_texture_metrics,
    )

    # 7.5 Cloud motion tracking (use configured or dynamically resolved cloud altitude)
    eff_alt_m = config.camera.assumed_cloud_altitude_m
    if eff_alt_m is None:
        thin_p = float(coverage_metrics.get("thin_cloud_pct", 0.0))
        dense_p = float(coverage_metrics.get("dense_cloud_pct", 0.0))
        dark_p = float(coverage_metrics.get("dark_cloud_pct", 0.0))
        tot_p = thin_p + dense_p + dark_p
        eff_alt_m = ((thin_p * 4500.0 + dense_p * 1600.0 + dark_p * 1100.0) / tot_p) if tot_p >= 1.0 else geom.cloud_altitude_m

    if config.vision.optical_flow_enabled:
        motion_metrics, motion_overlay = estimate_cloud_motion(
            prev_bgr=prev_bgr,
            curr_bgr=image_bgr,
            valid_mask=valid_mask,
            cloud_binary_mask=seg_out.cloud_binary_mask,
            dt_seconds=dt_seconds,
            fov_horizontal_deg=geom.fov_horizontal_deg,
            assumed_cloud_altitude_m=eff_alt_m,
            grid_rows=config.vision.grid_rows,
            grid_cols=config.vision.grid_cols,
            fov_vertical_deg=geom.fov_vertical_deg,
            orientation_elevation_deg=geom.elevation_deg,
            orientation_azimuth_deg=geom.azimuth_deg,
            orientation_roll_deg=geom.roll_deg,
            tilt_correction_enabled=geom.tilt_correction_enabled,
        )
    else:
        motion_metrics, motion_overlay = {"available": False, "reason": "Optical flow disabled"}, image_bgr.copy()

    # 7.6 Cloud classification (Stage 2: RGB + 5 Stage-1 class % + solar elevation + radiometric/texture)
    if config.vision.cloud_classification_enabled:
        solar_el_deg: float | None = float(solar.get("elevation_deg", 35.0)) if isinstance(solar, dict) else None
        cls_metrics = classify_cloud_type(
            image_bgr=neutralize_excluded_pixels(image_bgr, valid_mask),
            coverage_metrics=coverage_metrics,
            color_metrics=color_metrics,
            texture_metrics=texture_metrics,
            min_confidence=config.vision.classification_min_confidence,
            cls_model=cls_model,
            device=device,
            model_name=cls_model_name,
            solar_elevation_deg=solar_el_deg,
            group_ambiguous_wmo=bool(getattr(config.ai, "group_ambiguous_wmo", False)),
        )
    else:
        cls_metrics = {"probabilities": {"unknown": 1.0}, "dominant_cloud_type": "unknown", "confidence": 0.0}

    visual_condition = _derive_visual_condition(coverage_metrics, color_metrics, is_usable=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return VisionPipelineOutput(
        quality=q_res.metrics,
        quality_score=q_res.quality_score,
        is_usable=True,
        frame_hash=q_res.frame_hash,
        coverage=coverage_metrics,
        motion=motion_metrics,
        classification=cls_metrics,
        color=color_metrics,
        texture=texture_metrics,
        visual_condition=visual_condition,
        inference_duration_ms=round(elapsed_ms, 2),
        class_map=seg_out.class_map,
        cloud_binary_mask=seg_out.cloud_binary_mask,
        seg_overlay_bgr=seg_out.overlay_bgr,
        seg_color_mask_bgr=seg_out.color_mask_bgr,
        motion_overlay_bgr=motion_overlay,
        valid_sky_mask=valid_mask,
    )
