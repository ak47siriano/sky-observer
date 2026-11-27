"""7.7 Color & Radiometric Sky Analysis module."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _estimate_cct_kelvin(mean_r: float, mean_g: float, mean_b: float) -> float:
    """Estimate approximate correlated color temperature (McCamy's formula via sRGB -> XYZ -> xy)."""
    r, g, b = [max(0.0, min(1.0, c / 255.0)) for c in (mean_r, mean_g, mean_b)]
    # sRGB to CIE XYZ (D65)
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    denom = X + Y + Z
    if denom < 1e-6:
        return 6500.0
    x = X / denom
    y = Y / denom
    if abs(0.1858 - y) < 1e-4:
        return 6500.0
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449.0 * (n ** 3) + 3525.0 * (n ** 2) + 6823.3 * n + 5520.33
    return float(max(2000.0, min(15000.0, cct)))


def normalize_ae_awb_bgr(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray | None = None,
    target_p85_luma: float = 175.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Bounded smartphone AE/AWB normalization for radiometric stability:
    - Uses neutral/brightest low-saturation cloud pixels (if present) or clamped shades-of-gray
      gains in [0.75, 1.35] so blue sky hue is preserved while AWB shifts are damped.
    - Applies bounded luminance scaling in [0.70, 1.40] so auto-exposure jumps do not corrupt
      radiometric features.
    """
    img_f = image_bgr.astype(np.float32)
    mask = (
        (valid_mask > 0)
        if (valid_mask is not None and valid_mask.shape[:2] == image_bgr.shape[:2])
        else np.ones(image_bgr.shape[:2], dtype=bool)
    )
    if int(mask.sum()) < 20:
        mask = np.ones(image_bgr.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32)
    # Candidate neutral/cloud pixels: bright enough and low-to-moderate saturation
    # (allows catching warm/cool AWB tints on clouds up to sat < 0.30 while excluding saturated blue sky).
    b_ch = img_f[:, :, 0]
    r_ch = img_f[:, :, 2]
    not_deep_blue = (b_ch - r_ch) < (0.22 * np.maximum(val, 1.0))
    neutral = mask & (sat < 0.30) & (val > 95.0) & not_deep_blue

    if int(neutral.sum()) >= 50:
        mb = float(np.mean(img_f[:, :, 0][neutral]))
        mg = float(np.mean(img_f[:, :, 1][neutral]))
        mr = float(np.mean(img_f[:, :, 2][neutral]))
        gray_ref = (mb + mg + mr) / 3.0
        b_gain = float(np.clip(gray_ref / max(mb, 1.0), 0.75, 1.35))
        g_gain = float(np.clip(gray_ref / max(mg, 1.0), 0.85, 1.20))
        r_gain = float(np.clip(gray_ref / max(mr, 1.0), 0.75, 1.35))
    else:
        b_gain, g_gain, r_gain = 1.0, 1.0, 1.0

    wb = np.empty_like(img_f)
    wb[:, :, 0] = img_f[:, :, 0] * b_gain
    wb[:, :, 1] = img_f[:, :, 1] * g_gain
    wb[:, :, 2] = img_f[:, :, 2] * r_gain

    luma = 0.114 * wb[:, :, 0] + 0.587 * wb[:, :, 1] + 0.299 * wb[:, :, 2]
    p85 = float(np.percentile(luma[mask], 85))
    luma_scale = float(np.clip(target_p85_luma / max(p85, 25.0), 0.70, 1.40)) if p85 > 25.0 else 1.0
    out = np.clip(wb * luma_scale, 0.0, 255.0).astype(np.uint8)

    nb = out[:, :, 0].astype(np.float32)
    nr = out[:, :, 2].astype(np.float32)
    norm_bsi = float(np.mean(((nb - nr) / (nb + nr + 1e-4))[mask]))
    return out, {
        "r_gain": round(r_gain, 3),
        "g_gain": round(g_gain, 3),
        "b_gain": round(b_gain, 3),
        "luma_scale": round(luma_scale, 3),
        "normalized_blue_sky_index": round(norm_bsi, 4),
    }


def analyze_color(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    prev_color_metrics: dict[str, Any] | None = None,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> dict[str, Any]:
    """Analyze RGB, HSV, Lab color space, blue sky index, grayness, CCT proxy, and color patterns."""
    small = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    m_small = cv2.resize(valid_mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    if int(m_small.sum()) < 20:
        m_small = np.ones((256, 256), dtype=bool)

    b_ch = small[:, :, 0].astype(np.float32)
    g_ch = small[:, :, 1].astype(np.float32)
    r_ch = small[:, :, 2].astype(np.float32)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    b_vals = b_ch[m_small]
    g_vals = g_ch[m_small]
    r_vals = r_ch[m_small]
    sat_vals = hsv[:, :, 1][m_small] / 255.0
    val_vals = hsv[:, :, 2][m_small]
    hue_vals = hsv[:, :, 0][m_small] * 2.0  # degrees 0..360

    l_vals = lab[:, :, 0][m_small] * (100.0 / 255.0)
    a_vals = lab[:, :, 1][m_small] - 128.0
    b_lab_vals = lab[:, :, 2][m_small] - 128.0

    # Normalized Blue-Red Ratio: (B - R) / (B + R + 1e-4)
    bsi_map = (b_ch - r_ch) / (b_ch + r_ch + 1e-4)
    blue_sky_index = float(np.mean(bsi_map[m_small]))

    # Grayness: low saturation pixels where R, G, B are close
    rgb_spread = np.max(small, axis=2).astype(np.float32) - np.min(small, axis=2).astype(np.float32)
    grayness = float(np.mean(1.0 - np.clip(rgb_spread[m_small] / 80.0, 0.0, 1.0)))

    mean_r = float(np.mean(r_vals))
    mean_g = float(np.mean(g_vals))
    mean_b = float(np.mean(b_vals))
    mean_lum = float(np.mean(l_vals))
    mean_sat = float(np.mean(sat_vals))
    cct_k = _estimate_cct_kelvin(mean_r, mean_g, mean_b)

    # Brightness distribution percentiles (0..100 Lab L*)
    brightness_dist = {
        "p10": round(float(np.percentile(l_vals, 10)), 2),
        "p25": round(float(np.percentile(l_vals, 25)), 2),
        "p50": round(float(np.percentile(l_vals, 50)), 2),
        "p75": round(float(np.percentile(l_vals, 75)), 2),
        "p90": round(float(np.percentile(l_vals, 90)), 2),
        "std": round(float(np.std(l_vals)), 2),
    }

    # Color pattern indicators (proportions of valid sky pixels)
    blue_sky_pixel_ratio = float(np.mean((bsi_map[m_small] > 0.08) & (sat_vals > 0.16) & (val_vals > 65)))
    white_cloud_pixel_ratio = float(np.mean((l_vals > 68) & (sat_vals < 0.24)))
    gray_cloud_pixel_ratio = float(np.mean((l_vals >= 35) & (l_vals <= 68) & (sat_vals < 0.22)))
    dark_cloud_pixel_ratio = float(np.mean((l_vals < 35) & (sat_vals < 0.30)))
    warm_twilight_ratio = float(np.mean((r_vals > b_vals + 15) & (sat_vals > 0.22)))
    haze_pixel_ratio = float(np.mean((l_vals > 52) & (sat_vals < 0.16) & (np.abs(bsi_map[m_small]) < 0.07)))

    # Temporal deltas
    lum_delta = 0.0
    bsi_delta = 0.0
    if prev_color_metrics:
        lum_delta = mean_lum - float(prev_color_metrics.get("avg_luminance", mean_lum))
        bsi_delta = blue_sky_index - float(prev_color_metrics.get("blue_sky_index", blue_sky_index))

    patterns: list[str] = []
    if blue_sky_pixel_ratio > 0.45:
        patterns.append("blue_sky_dominant")
    if white_cloud_pixel_ratio > 0.30:
        patterns.append("white_clouds_present")
    if gray_cloud_pixel_ratio > 0.35:
        patterns.append("gray_cloud_cover")
    if dark_cloud_pixel_ratio > 0.22:
        patterns.append("dark_cloud_formations")
    if haze_pixel_ratio > 0.40:
        patterns.append("haze_like_conditions")
    if warm_twilight_ratio > 0.25:
        patterns.append("sunrise_sunset_coloration")
    if lum_delta < -8.0:
        patterns.append("rapid_sky_darkening")

    # Regional grid color statistics
    regional_stats: list[dict[str, float]] = []
    rh = 256 // grid_rows
    cw = 256 // grid_cols
    for r_idx in range(grid_rows):
        for c_idx in range(grid_cols):
            y0, y1 = r_idx * rh, (r_idx + 1) * rh if r_idx < grid_rows - 1 else 256
            x0, x1 = c_idx * cw, (c_idx + 1) * cw if c_idx < grid_cols - 1 else 256
            cell_mask = m_small[y0:y1, x0:x1]
            if int(cell_mask.sum()) > 5:
                c_lum = float(np.mean(lab[y0:y1, x0:x1, 0][cell_mask] * (100.0 / 255.0)))
                c_bsi = float(np.mean(bsi_map[y0:y1, x0:x1][cell_mask]))
            else:
                c_lum = mean_lum
                c_bsi = blue_sky_index
            regional_stats.append({
                "row": r_idx,
                "col": c_idx,
                "luminance": round(c_lum, 2),
                "blue_sky_index": round(c_bsi, 3),
            })

    _, ae_awb_meta = normalize_ae_awb_bgr(small, m_small.astype(np.uint8) * 255)

    return {
        "rgb_mean": {"r": round(mean_r, 2), "g": round(mean_g, 2), "b": round(mean_b, 2)},
        "ae_awb_normalization": ae_awb_meta,
        "hsv_mean": {
            "h_deg": round(float(np.mean(hue_vals)), 2),
            "s": round(mean_sat, 3),
            "v": round(float(np.mean(val_vals)), 2),
        },
        "lab_mean": {
            "l": round(mean_lum, 2),
            "a": round(float(np.mean(a_vals)), 2),
            "b": round(float(np.mean(b_lab_vals)), 2),
        },
        "avg_luminance": round(mean_lum, 2),
        "blue_sky_index": round(blue_sky_index, 4),
        "saturation": round(mean_sat, 4),
        "grayness": round(grayness, 4),
        "color_temperature_k": round(cct_k, 0),
        "luminance_delta": round(lum_delta, 2),
        "blue_sky_index_delta": round(bsi_delta, 4),
        "brightness_distribution": brightness_dist,
        "pattern_ratios": {
            "blue_sky": round(blue_sky_pixel_ratio, 3),
            "white_cloud": round(white_cloud_pixel_ratio, 3),
            "gray_cloud": round(gray_cloud_pixel_ratio, 3),
            "dark_cloud": round(dark_cloud_pixel_ratio, 3),
            "haze": round(haze_pixel_ratio, 3),
            "twilight_warm": round(warm_twilight_ratio, 3),
        },
        "patterns": patterns,
        "regional_stats": regional_stats,
    }
