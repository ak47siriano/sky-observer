"""Pixel-level Optical-Flow Cloud Mask Advection & Probabilistic Nowcasting Ensemble.

Inspired by:
- LenghuSky-8 nowcasting benchmark (Persistence vs Optical Flow vs learned models; Rui et al., CVPR Findings 2025)
- Probabilistic advection ensembles (SolarSTEPS / dmidk/sunflow; perturbed motion fields -> pixel coverage probability)

Designed to stay well within the Low-End PC (4 GB RAM, CPU-only) budget via configurable
grid resolution (`advection_resolution`, default 96px) and ensemble size (`ensemble_members`, default 5).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def persistence_baseline_cover(current_cover_pct: float) -> float:
    """Persistence baseline: cloud cover at t + horizon equals cloud cover at t."""
    return round(float(np.clip(current_cover_pct, 0.0, 100.0)), 2)


def compute_forecast_skill(model_mae: float, persistence_mae: float) -> float | None:
    """
    Forecast Skill Score relative to persistence:
        skill = 1 - MAE_model / MAE_persistence
    Positive (> 0) means the model beats persistence; <= 0 means persistence is equal or better.
    """
    if not math.isfinite(model_mae) or not math.isfinite(persistence_mae):
        return None
    if persistence_mae <= 1e-6:
        return 0.0 if model_mae <= 1e-6 else -1.0
    return round(float(1.0 - (model_mae / persistence_mae)), 4)


def _advect_field_cv(
    field: np.ndarray,
    shift_x: float,
    shift_y: float,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Semi-Lagrangian backward warp: pixel (x, y) at t + dt comes from (x - shift_x, y - shift_y) at t.
    Returns (warped_field, in_bounds_mask) where in_bounds_mask is True when source pixel lay inside FOV.
    """
    h, w = field.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = xx - np.float32(shift_x)
    src_y = yy - np.float32(shift_y)
    in_bounds = (src_x >= 0.0) & (src_x <= float(w - 1)) & (src_y >= 0.0) & (src_y <= float(h - 1))
    warped = cv2.remap(
        field.astype(np.float32),
        src_x,
        src_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, in_bounds


def advect_cloud_mask_optical_flow(
    seg_mask: np.ndarray,
    dx_px_s: float,
    dy_px_s: float,
    horizon_minutes: float,
    orig_width: int | None = None,
    orig_height: int | None = None,
    valid_mask: np.ndarray | None = None,
    resolution: int = 96,
    ensemble_members: int = 5,
    motion_consistency: float = 0.65,
) -> dict[str, Any]:
    """
    Advect a segmentation class mask forward by `horizon_minutes` using measured optical flow
    (dx_px_s, dy_px_s in original image pixels/sec), plus a lightweight perturbed-flow ensemble
    yielding pixel-level cloud probability and [p10, p50, p90] coverage bounds.
    """
    if seg_mask is None or seg_mask.size == 0:
        return {
            "available": False,
            "advected_cover_pct": None,
            "cover_p10": None,
            "cover_p50": None,
            "cover_p90": None,
            "uncertainty_band_pct": None,
            "fov_exit_fraction": 1.0,
            "cloud_probability_map": None,
            "advected_mask": None,
        }

    src_h, src_w = seg_mask.shape[:2]
    ref_w = max(1, int(orig_width or src_w))
    ref_h = max(1, int(orig_height or src_h))

    grid_w = max(48, min(256, int(resolution)))
    grid_h = max(36, int(round(grid_w * float(src_h) / max(1.0, float(src_w)))))

    mask_small = cv2.resize(seg_mask, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
    cloud_bin = ((mask_small >= 1) & (mask_small <= 3)).astype(np.float32)

    if valid_mask is not None and valid_mask.size > 0:
        v_small = cv2.resize(valid_mask, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST) > 0
        if int(v_small.sum()) < 25:
            v_small = np.ones((grid_h, grid_w), dtype=bool)
    else:
        v_small = np.ones((grid_h, grid_w), dtype=bool)

    dt_s = max(0.0, float(horizon_minutes) * 60.0)
    # Convert native px/s to advection-grid pixels over dt_s
    scale_x = float(grid_w) / float(ref_w)
    scale_y = float(grid_h) / float(ref_h)
    # Smooth kinematic damping for longer horizons (clouds deform & decelerate over 15-60m)
    kinematic_eff = dt_s / (1.0 + (horizon_minutes / 35.0) * 0.45)
    shift_x = float(dx_px_s) * scale_x * kinematic_eff
    shift_y = float(dy_px_s) * scale_y * kinematic_eff

    # Cap displacement to 1.25x grid span so out-of-FOV tracking stays well-conditioned
    max_shift = max(grid_w, grid_h) * 1.25
    mag = math.hypot(shift_x, shift_y)
    if mag > max_shift and mag > 1e-6:
        s = max_shift / mag
        shift_x *= s
        shift_y *= s

    det_cloud, in_bounds = _advect_field_cv(cloud_bin, shift_x, shift_y, interpolation=cv2.INTER_LINEAR)
    det_cls, _ = _advect_field_cv(mask_small.astype(np.float32), shift_x, shift_y, interpolation=cv2.INTER_NEAREST)
    det_cls_u8 = np.clip(np.rint(det_cls), 0, 4).astype(np.uint8)

    valid_count = max(1, int(v_small.sum()))
    in_fov_valid = in_bounds & v_small
    fov_exit_fraction = round(float(1.0 - (int(in_fov_valid.sum()) / float(valid_count))), 3)

    # For pixels advected from outside narrow FOV, blend upwind boundary persistence with current mean prior
    curr_cover_frac = float(np.mean(cloud_bin[v_small]))
    det_blended = np.where(in_bounds, det_cloud, 0.75 * det_cloud + 0.25 * curr_cover_frac)
    det_cover_pct = round(float(np.clip(np.mean(det_blended[v_small]) * 100.0, 0.0, 100.0)), 2)

    # Lightweight deterministic quasi-Monte-Carlo ensemble (reproducible, Low-End CPU friendly)
    n_members = max(1, min(15, int(ensemble_members)))
    cons = float(np.clip(motion_consistency, 0.1, 0.98))
    speed_spread = (1.0 - cons) * 0.30 + min(0.25, horizon_minutes / 90.0)
    angle_spread_rad = math.radians((1.0 - cons) * 26.0 + min(22.0, horizon_minutes * 0.45))
    blur_sigma = max(0.0, min(3.5, math.sqrt(max(horizon_minutes, 1.0)) * 0.35))

    member_covers: list[float] = []
    prob_accum = np.zeros((grid_h, grid_w), dtype=np.float32)

    if n_members == 1:
        perturbations = [(1.0, 0.0)]
    else:
        # Balanced symmetric perturbations around central vector (0, +/-speed, +/-angle)
        perturbations = [(1.0, 0.0)]
        half_n = (n_members - 1) // 2
        for k in range(1, half_n + 1):
            frac = float(k) / float(max(1, half_n))
            perturbations.append((1.0 + speed_spread * frac, angle_spread_rad * frac))
            perturbations.append((max(0.25, 1.0 - speed_spread * frac), -angle_spread_rad * frac))
        while len(perturbations) < n_members:
            perturbations.append((1.0, angle_spread_rad * 0.5))

    for sp_mult, d_ang in perturbations[:n_members]:
        ca, sa = math.cos(d_ang), math.sin(d_ang)
        px = (shift_x * ca - shift_y * sa) * sp_mult
        py = (shift_x * sa + shift_y * ca) * sp_mult
        w_cloud, ib = _advect_field_cv(cloud_bin, px, py, interpolation=cv2.INTER_LINEAR)
        if blur_sigma >= 0.4:
            w_cloud = cv2.GaussianBlur(w_cloud, (0, 0), sigmaX=blur_sigma)
        # Edge uncertainty widens slightly as clouds exit narrow FOV
        w_cloud = np.where(ib, w_cloud, 0.45 * w_cloud + 0.55 * curr_cover_frac)
        w_cloud = np.clip(w_cloud, 0.0, 1.0)
        prob_accum += w_cloud
        member_covers.append(float(np.mean(w_cloud[v_small]) * 100.0))

    prob_map = np.clip(prob_accum / float(len(member_covers)), 0.0, 1.0).astype(np.float32)
    # Add calibrated horizon dispersion floor so uncertainty widens honestly when fov_exit_fraction is high
    extra_unc = round(float(fov_exit_fraction * 14.0 + min(12.0, horizon_minutes * 0.22)), 2)
    p50 = float(np.percentile(member_covers, 50))
    p10 = float(np.clip(min(np.percentile(member_covers, 10), p50 - extra_unc * 0.6), 0.0, 100.0))
    p90 = float(np.clip(max(np.percentile(member_covers, 90), p50 + extra_unc * 0.6), 0.0, 100.0))

    return {
        "available": True,
        "advected_cover_pct": det_cover_pct,
        "cover_p10": round(p10, 1),
        "cover_p50": round(p50, 1),
        "cover_p90": round(p90, 1),
        "uncertainty_band_pct": round((p90 - p10) * 0.5, 1),
        "fov_exit_fraction": fov_exit_fraction,
        "ensemble_members": len(member_covers),
        "cloud_probability_map": prob_map,
        "advected_mask": det_cls_u8,
    }


def render_advected_probability_png(
    prob_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    out_width: int = 384,
) -> bytes:
    """Render per-pixel advected cloud probability map [0..1] as a translucent BGRA PNG."""
    h, w = prob_map.shape[:2]
    target_w = max(64, int(out_width))
    target_h = max(48, int(round(target_w * float(h) / max(1.0, float(w)))))
    p_up = cv2.resize(prob_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    p_up = cv2.GaussianBlur(p_up, (0, 0), sigmaX=1.2)

    bgra = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    # Smooth cyan (#38BDF8) -> crisp white (#F8FAFC) -> storm violet (#A855F7) probability ramp
    low = np.clip(p_up * 2.0, 0.0, 1.0)
    high = np.clip((p_up - 0.5) * 2.0, 0.0, 1.0)
    b = 248.0 * (1.0 - high) + 235.0 * high
    g = 189.0 * (1.0 - low) + 250.0 * low * (1.0 - high) + 120.0 * high
    r = 56.0 * (1.0 - low) + 248.0 * low * (1.0 - high) + 195.0 * high
    alpha = np.clip((p_up - 0.12) / 0.80, 0.0, 1.0) * 215.0

    if valid_mask is not None and valid_mask.size > 0:
        vm = cv2.resize(valid_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
        alpha[~vm] = 0.0

    bgra[:, :, 0] = np.clip(b, 0, 255).astype(np.uint8)
    bgra[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
    bgra[:, :, 2] = np.clip(r, 0, 255).astype(np.uint8)
    bgra[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", bgra)
    return buf.tobytes() if ok else b""
