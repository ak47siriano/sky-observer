"""7.4 Cloud Coverage Calculation & Regional Grid module."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_cloud_coverage(
    class_map: np.ndarray,
    valid_mask: np.ndarray,
    mean_confidence: float,
    recent_coverages: list[tuple[float, float]] | None = None,
    grid_rows: int = 3,
    grid_cols: int = 3,
    pixel_weights: np.ndarray | None = None,
    geometry_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute cloud coverage strictly over valid sky pixels:
        cloud_coverage = cloud_pixels / valid_sky_pixels * 100
    where cloud_pixels = thin_cloud (1) + dense_cloud (2) + dark_cloud (3).
    When pixel_weights is provided, applies camera-tilt zenith & solid-angle weighting.
    """
    valid_bool = valid_mask > 0
    valid_count = int(np.count_nonzero(valid_bool))
    if valid_count == 0:
        valid_bool = np.ones_like(class_map, dtype=bool)
        valid_count = int(class_map.size)

    valid_classes = class_map[valid_bool]
    raw_clear_pct = float(np.count_nonzero(valid_classes == 0) / valid_count * 100.0)
    raw_thin_pct = float(np.count_nonzero(valid_classes == 1) / valid_count * 100.0)
    raw_dense_pct = float(np.count_nonzero(valid_classes == 2) / valid_count * 100.0)
    raw_dark_pct = float(np.count_nonzero(valid_classes == 3) / valid_count * 100.0)
    raw_sun_glare_pct = float(np.count_nonzero(valid_classes == 4) / valid_count * 100.0)
    raw_uncertain_pct = float(np.count_nonzero(valid_classes == 5) / valid_count * 100.0)
    raw_cloud_pct = raw_thin_pct + raw_dense_pct + raw_dark_pct

    use_weights = False
    w_map: np.ndarray | None = None
    if pixel_weights is not None and pixel_weights.shape == class_map.shape:
        candidate = np.maximum(0.0, pixel_weights.astype(np.float64))
        valid_w = candidate[valid_bool]
        total_w = float(np.sum(valid_w))
        if total_w > 1e-6:
            use_weights = True
            w_map = candidate
            clear_pct = float(np.sum(valid_w[valid_classes == 0]) / total_w * 100.0)
            thin_pct = float(np.sum(valid_w[valid_classes == 1]) / total_w * 100.0)
            dense_pct = float(np.sum(valid_w[valid_classes == 2]) / total_w * 100.0)
            dark_pct = float(np.sum(valid_w[valid_classes == 3]) / total_w * 100.0)
            sun_glare_pct = float(np.sum(valid_w[valid_classes == 4]) / total_w * 100.0)
            uncertain_pct = float(np.sum(valid_w[valid_classes == 5]) / total_w * 100.0)
            cloud_pct = thin_pct + dense_pct + dark_pct
        else:
            clear_pct, thin_pct, dense_pct, dark_pct, sun_glare_pct, uncertain_pct, cloud_pct = (
                raw_clear_pct,
                raw_thin_pct,
                raw_dense_pct,
                raw_dark_pct,
                raw_sun_glare_pct,
                raw_uncertain_pct,
                raw_cloud_pct,
            )
    else:
        clear_pct, thin_pct, dense_pct, dark_pct, sun_glare_pct, uncertain_pct, cloud_pct = (
            raw_clear_pct,
            raw_thin_pct,
            raw_dense_pct,
            raw_dark_pct,
            raw_sun_glare_pct,
            raw_uncertain_pct,
            raw_cloud_pct,
        )
    haze_pct = 0.0  # deprecated pixel class (kept at 0.0 for DB/API schema compatibility)

    # Configurable spatial grid coverage
    h, w = class_map.shape[:2]
    rh = max(1, h // grid_rows)
    cw = max(1, w // grid_cols)
    grid: list[dict[str, Any]] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = r * rh, (r + 1) * rh if r < grid_rows - 1 else h
            x0, x1 = c * cw, (c + 1) * cw if c < grid_cols - 1 else w
            cell_valid = valid_bool[y0:y1, x0:x1]
            cell_valid_cnt = int(np.count_nonzero(cell_valid))
            if cell_valid_cnt == 0:
                grid.append({"row": r, "col": c, "valid_pixels": 0, "cloud_pct": None})
            else:
                cell_cls = class_map[y0:y1, x0:x1][cell_valid]
                is_cloud = (cell_cls >= 1) & (cell_cls <= 3)
                if use_weights and w_map is not None:
                    cell_w = w_map[y0:y1, x0:x1][cell_valid]
                    cell_w_sum = float(np.sum(cell_w))
                    if cell_w_sum > 1e-6:
                        cell_cloud = float(np.sum(cell_w[is_cloud]) / cell_w_sum * 100.0)
                    else:
                        cell_cloud = float(np.count_nonzero(is_cloud) / cell_valid_cnt * 100.0)
                else:
                    cell_cloud = float(np.count_nonzero(is_cloud) / cell_valid_cnt * 100.0)
                grid.append({
                    "row": r,
                    "col": c,
                    "valid_pixels": cell_valid_cnt,
                    "cloud_pct": round(cell_cloud, 2),
                })

    # Coverage trend & rate of change (% per minute)
    rate_pct_per_min = 0.0
    trend = "stable"
    if recent_coverages and len(recent_coverages) >= 2:
        # recent_coverages = [(elapsed_minutes_from_oldest, coverage_pct), ...]
        xs = np.array([p[0] for p in recent_coverages] + [recent_coverages[-1][0] + 0.25], dtype=np.float64)
        ys = np.array([p[1] for p in recent_coverages] + [cloud_pct], dtype=np.float64)
        span = float(xs[-1] - xs[0])
        if span > 0.05:
            slope, _ = np.polyfit(xs, ys, 1)
            rate_pct_per_min = float(slope)
            if rate_pct_per_min >= 1.5:
                trend = "rapidly_increasing"
            elif rate_pct_per_min >= 0.35:
                trend = "increasing"
            elif rate_pct_per_min <= -1.5:
                trend = "rapidly_decreasing"
            elif rate_pct_per_min <= -0.35:
                trend = "decreasing"

    out: dict[str, Any] = {
        "valid_sky_pixels": valid_count,
        "cloud_coverage_pct": round(cloud_pct, 2),
        "raw_cloud_coverage_pct": round(raw_cloud_pct, 2),
        "tilt_correction_applied": use_weights,
        "clear_sky_pct": round(clear_pct, 2),
        "thin_cloud_pct": round(thin_pct, 2),
        "dense_cloud_pct": round(dense_pct, 2),
        "dark_cloud_pct": round(dark_pct, 2),
        "sun_glare_pct": round(sun_glare_pct, 2),
        "haze_fog_pct": round(haze_pct, 2),
        "uncertainty_pct": round(uncertain_pct, 2),
        "confidence": round(mean_confidence, 4),
        "trend": trend,
        "rate_of_change_pct_per_min": round(rate_pct_per_min, 3),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "grid": grid,
    }
    if geometry_meta:
        out.update(geometry_meta)
    return out
