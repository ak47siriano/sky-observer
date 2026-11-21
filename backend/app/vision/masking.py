"""7.2 Sky Region & Permanent Exclusion Masking module."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from app.config import CameraConfig, ExclusionPolygon


def _norm_poly_to_pixels(points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    pts = np.array([[int(round(p[0] * (width - 1))), int(round(p[1] * (height - 1)))] for p in points], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def build_valid_sky_mask(
    height: int,
    width: int,
    sky_roi_polygon: Sequence[Sequence[float]] | None = None,
    exclusion_polygons: Sequence[ExclusionPolygon] | None = None,
) -> np.ndarray:
    """
    Build a uint8 binary mask (255 = valid sky pixel, 0 = excluded/masked out).

    1. Start with sky ROI polygon if >=3 vertices provided; otherwise full frame is valid.
    2. Subtract all permanent exclusion polygons (trees, buildings, poles, roofs, other).
    """
    if sky_roi_polygon and len(sky_roi_polygon) >= 3:
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = _norm_poly_to_pixels(sky_roi_polygon, width, height)
        cv2.fillPoly(mask, [pts], 255)
    else:
        mask = np.full((height, width), 255, dtype=np.uint8)

    if exclusion_polygons:
        for poly in exclusion_polygons:
            if len(poly.points) >= 3:
                pts = _norm_poly_to_pixels(poly.points, width, height)
                cv2.fillPoly(mask, [pts], 0)

    # Guarantee at least some valid pixels if user masked 100% by accident
    if int(np.count_nonzero(mask)) == 0:
        mask[:, :] = 255
    return mask


def build_valid_sky_mask_from_config(shape: tuple[int, int], camera_cfg: CameraConfig) -> np.ndarray:
    h, w = shape[:2]
    mask = build_valid_sky_mask(
        height=h,
        width=w,
        sky_roi_polygon=camera_cfg.sky_roi_polygon,
        exclusion_polygons=camera_cfg.exclusion_polygons,
    )
    if camera_cfg.exclusion_mask_path:
        mp = Path(camera_cfg.exclusion_mask_path)
        if mp.exists():
            painted = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if painted is not None and painted.size > 0:
                painted_r = cv2.resize(painted, (w, h), interpolation=cv2.INTER_NEAREST)
                mask[painted_r > 127] = 0
                if int(np.count_nonzero(mask)) == 0:
                    mask[:, :] = 255
    return mask


def neutralize_excluded_pixels(image_bgr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Replace excluded pixels (e.g., watermark logos, roads, trees) with valid-sky mean color."""
    valid_bool = valid_mask > 0
    if np.all(valid_bool) or not np.any(valid_bool):
        return image_bgr
    cleaned = image_bgr.copy()
    mean_bgr = np.mean(image_bgr[valid_bool], axis=0).astype(np.uint8)
    cleaned[~valid_bool] = mean_bgr
    return cleaned


def render_mask_overlay(image_bgr: np.ndarray, valid_mask: np.ndarray, camera_cfg: CameraConfig) -> np.ndarray:
    """Render exclusion zones in shaded crimson and sky ROI boundary in cyan."""
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    excluded = valid_mask == 0
    if np.any(excluded):
        tint = overlay.copy()
        tint[excluded] = (40, 35, 190)  # Crimson tint in BGR
        overlay = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0.0)

    if camera_cfg.sky_roi_polygon and len(camera_cfg.sky_roi_polygon) >= 3:
        pts = _norm_poly_to_pixels(camera_cfg.sky_roi_polygon, w, h)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 220, 40), thickness=2)

    for poly in camera_cfg.exclusion_polygons:
        if len(poly.points) >= 3:
            pts = _norm_poly_to_pixels(poly.points, w, h)
            cv2.polylines(overlay, [pts], isClosed=True, color=(70, 90, 255), thickness=2)
            cx = int(np.mean(pts[:, 0, 0]))
            cy = int(np.mean(pts[:, 0, 1]))
            cv2.putText(
                overlay,
                f"{poly.label} ({poly.category})",
                (max(4, cx - 35), max(16, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 255),
                1,
                cv2.LINE_AA,
            )
    return overlay
