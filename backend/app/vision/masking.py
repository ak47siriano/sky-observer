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


def propose_sky_obstacle_mask(
    image_bgr: np.ndarray,
    elevation_deg: float | None = None,
    fov_vertical_deg: float | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """
    Propose a binary exclusion mask (255 = static obstacle/ground to exclude, 0 = valid sky)
    from a single camera frame so the user can review/edit and save instead of painting from scratch.

    Algorithm (100% offline, deterministic, Low-End CPU friendly):
    1. Combine sky likelihood (upper-frame brightness, low green vegetation excess, smooth texture,
       blue/neutral sky chromaticity) into a coarse sky probability map.
    2. Optional geometric horizon prior when camera tilt elevation is calibrated.
    3. Keep only sky components connected to the upper border of the frame (removes bright
       ground roofs/windows/water reflections below the skyline), then morphologically clean.
    """
    h, w = image_bgr.shape[:2]
    work_w = 320
    work_h = max(120, int(round(work_w * float(h) / max(1.0, float(w)))))
    small = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)

    img_f = small.astype(np.float32)
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Local edge/texture energy (trees, buildings, foliage have high local gradient variance)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.GaussianBlur(np.hypot(gx, gy), (9, 9), 2.0)

    # Vegetation index (ExG = 2G - R - B) strongly separates green trees/bushes from sky
    exg = (2.0 * g - r - b) / (r + g + b + 1.0)
    # Blue-sky or bright neutral-cloud signature
    bsi = (b - r) / (b + r + 1.0)
    sat = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) / (
        np.maximum(np.maximum(r, g), b) + 1.0
    )

    # Estimate upper-band sky luminance reference (top 18% of frame)
    top_rows = max(8, int(work_h * 0.18))
    sky_l_ref = float(np.percentile(L[:top_rows, :], 65))
    l_thresh = max(18.0, min(58.0, sky_l_ref * 0.56))

    # Row prior: bottom rows are less likely to be sky unless near-zenith camera
    y_norm = np.linspace(0.0, 1.0, work_h, dtype=np.float32)[:, None]
    zenith_cam = elevation_deg is not None and float(elevation_deg) >= 65.0
    grad_thresh = 42.0 if zenith_cam else 34.0

    is_sky_color = (
        (L >= l_thresh)
        & (exg < 0.08)
        & ((bsi > -0.06) | ((sat < 0.28) & (L >= l_thresh + 6.0)))
    )
    is_low_texture = grad_mag < (grad_thresh + np.where(y_norm < 0.4, 18.0, 0.0))
    sky_cand = (is_sky_color & is_low_texture).astype(np.uint8)

    # Geometric horizon cutoff when camera tilt is calibrated and looks near horizon
    if elevation_deg is not None and fov_vertical_deg is not None and float(elevation_deg) < 60.0:
        half_vfov = max(10.0, float(fov_vertical_deg) * 0.5)
        # Pixel elevation linearly approximated: top=+half_vfov+el, bottom=-half_vfov+el
        pix_el = float(elevation_deg) + half_vfov * (1.0 - 2.0 * y_norm)
        sky_cand[pix_el[:, 0] < -2.0, :] = 0

    # Keep only sky regions connected to top 10% border (removes disconnected ground/water patches)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sky_clean = cv2.morphologyEx(sky_cand * 255, cv2.MORPH_CLOSE, k_close)
    sky_clean = cv2.morphologyEx(sky_clean, cv2.MORPH_OPEN, k_open)

    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(sky_clean, connectivity=8)
    top_band = labels[: max(4, int(work_h * 0.12)), :]
    top_labels = set(np.unique(top_band).tolist()) - {0}

    sky_connected = np.zeros((work_h, work_w), dtype=np.uint8)
    for lbl in top_labels:
        sky_connected[labels == lbl] = 255

    # Safety fallback: if lighting is extreme (e.g. night/overexposed) keep top 65% as sky
    sky_frac = float(np.mean(sky_connected > 0))
    method = "radiometric_horizon_connected"
    if sky_frac < 0.08:
        sky_connected[: int(work_h * 0.65), :] = 255
        sky_frac = float(np.mean(sky_connected > 0))
        method = "upper_frame_fallback"

    excl_small = np.where(sky_connected > 0, 0, 255).astype(np.uint8)
    # Dilate obstacle edge by 3px so tree tips / building rooflines don't bleed into cloud cover
    excl_small = cv2.dilate(excl_small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    excl_full = cv2.resize(excl_small, (w, h), interpolation=cv2.INTER_NEAREST)
    excl_pct = round(float(np.mean(excl_full > 127) * 100.0), 2)
    return excl_full, {
        "method": method,
        "proposed_excluded_pct": excl_pct,
        "sky_connected_pct": round(100.0 - excl_pct, 2),
    }
