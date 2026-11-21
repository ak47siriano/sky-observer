"""7.1 Image Quality Analysis module."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class QualityResult:
    quality_score: float
    is_usable: bool
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    frame_hash: str = ""
    camera_shift_px: float = 0.0


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if b else "0" for b in diff.flatten())
    return f"{int(bits, 2):016x}"


def assess_image_quality(
    image_bgr: np.ndarray | None,
    prev_bgr: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    min_quality_score: float = 0.40,
    duplicate_threshold: float = 0.015,
    camera_move_shift_px: float = 18.0,
) -> QualityResult:
    """Detect blur, darkness, overexposure, glare, obstruction, corruption, duplicates, and camera shift."""
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return QualityResult(
            quality_score=0.0,
            is_usable=False,
            flags=["corrupted_frame"],
            metrics={"reason": "Invalid or empty BGR frame"},
        )

    h, w = image_bgr.shape[:2]
    if h < 32 or w < 32:
        return QualityResult(
            quality_score=0.0,
            is_usable=False,
            flags=["corrupted_frame"],
            metrics={"reason": "Resolution too small"},
        )

    # Check for mid-tone flat / corrupted single-color bands or NaN
    mean_raw = float(np.mean(image_bgr))
    if not np.isfinite(image_bgr).all() or (float(np.std(image_bgr)) < 0.2 and 15.0 <= mean_raw <= 240.0):
        return QualityResult(
            quality_score=0.05,
            is_usable=False,
            flags=["corrupted_frame"],
            metrics={"std_dev": float(np.std(image_bgr))},
            frame_hash=compute_dhash(image_bgr),
        )

    work = cv2.resize(image_bgr, (320, 240), interpolation=cv2.INTER_AREA)
    if valid_mask is not None:
        m_small = cv2.resize(valid_mask, (320, 240), interpolation=cv2.INTER_NEAREST) > 0
        if int(m_small.sum()) < 50:
            m_small = np.ones((240, 320), dtype=bool)
    else:
        m_small = np.ones((240, 320), dtype=bool)

    # Neutralize excluded zones (e.g., Iriun watermark or road traffic) before grayscale/edge/motion analysis
    if not np.all(m_small):
        mean_bgr_sky = np.mean(work[m_small], axis=0).astype(np.uint8)
        work = work.copy()
        work[~m_small] = mean_bgr_sky

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    sky_gray = gray[m_small]
    sky_sat = hsv[:, :, 1][m_small]
    sky_val = hsv[:, :, 2][m_small]

    mean_lum = float(np.mean(sky_gray))
    p5_lum = float(np.percentile(sky_gray, 5))
    p95_lum = float(np.percentile(sky_gray, 95))
    std_lum = float(np.std(sky_gray))

    # Laplacian variance for sharpness (note: clear blue sky is naturally smooth, so evaluate full frame edges too)
    lap_var_full = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_var_sky = float(cv2.Laplacian(gray, cv2.CV_64F)[m_small].var())

    # Glare: saturated near-white pixels
    glare_ratio = float(np.mean((sky_val >= 248) & (sky_sat <= 28)))

    # Obstruction check: dark low-entropy regions or excessive dark edge-free blob covering >55% of sky
    dark_blob_ratio = float(np.mean(sky_gray < 25))
    hist = cv2.calcHist([sky_gray], [0], None, [32], [0, 256]).flatten()
    hist_prob = hist / max(hist.sum(), 1.0)
    entropy = float(-np.sum(hist_prob[hist_prob > 0] * np.log2(hist_prob[hist_prob > 0])))

    flags: list[str] = []
    penalties = 0.0

    # Darkness
    if mean_lum < 22.0 or p95_lum < 45.0:
        flags.append("excessively_dark")
        penalties += 0.55
    elif mean_lum < 42.0:
        flags.append("low_light")
        penalties += 0.20

    # Overexposure
    if mean_lum > 238.0 and std_lum < 18.0:
        flags.append("overexposed")
        penalties += 0.50

    # Glare
    if glare_ratio > 0.28:
        flags.append("severe_glare")
        penalties += 0.30

    # Lens obstruction (e.g., hand/cap/mud blocking lens)
    if (dark_blob_ratio > 0.60 and entropy < 2.2) or (entropy < 1.35 and std_lum < 8.0 and mean_lum < 90.0):
        flags.append("lens_obstruction")
        penalties += 0.65

    # Blur (only penalize heavily when high-contrast regions also lack sharpness)
    if lap_var_full < 12.0 and std_lum > 14.0:
        flags.append("blurry_image")
        penalties += 0.35

    # Duplicate frame & camera movement check against previous frame
    frame_mad = 1.0
    camera_shift_px = 0.0
    orientation_change_score = 0.0

    if prev_bgr is not None and prev_bgr.shape[:2] == image_bgr.shape[:2]:
        prev_small = cv2.resize(prev_bgr, (320, 240), interpolation=cv2.INTER_AREA)
        if not np.all(m_small):
            prev_small = prev_small.copy()
            prev_small[~m_small] = np.mean(prev_small[m_small], axis=0).astype(np.uint8)
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        frame_mad = float(np.mean(np.abs(gray[m_small].astype(np.float32) - prev_gray[m_small].astype(np.float32))) / 255.0)
        if frame_mad < duplicate_threshold:
            flags.append("repeated_identical_frame")

        # Estimate global shift strictly inside eroded valid-sky mask using windowed 1st-order gradients
        eroded = cv2.erode(m_small.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        eroded_f = eroded.astype(np.float32)
        hann = cv2.createHanningWindow((320, 240), cv2.CV_32F)
        gx_c = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gx_p = cv2.Sobel(prev_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_p = cv2.Sobel(prev_gray, cv2.CV_32F, 0, 1, ksize=3)
        edges_curr = cv2.magnitude(gx_c, gy_c) * eroded_f * hann
        edges_prev = cv2.magnitude(gx_p, gy_p) * eroded_f * hann
        if float(np.std(edges_curr)) > 3.0 and float(np.std(edges_prev)) > 3.0:
            (dx, dy), response = cv2.phaseCorrelate(edges_prev, edges_curr)
            scale_x = w / 320.0
            scale_y = h / 240.0
            shift_small_px = float(np.hypot(dx, dy))
            camera_shift_px = float(np.hypot(dx * scale_x, dy * scale_y))
            orientation_change_score = float(response)
            effective_shift_thresh = max(float(camera_move_shift_px), 0.035 * float(min(w, h)))
            # Require meaningful shift on both native and analysis grids + clean phase peak to avoid sky drift false alerts
            if (
                camera_shift_px > effective_shift_thresh
                and shift_small_px >= 4.0
                and response > 0.42
                and frame_mad > 0.025
            ):
                flags.append("camera_movement")
                penalties += 0.45
                if camera_shift_px > effective_shift_thresh * 2.5:
                    flags.append("sudden_orientation_change")
                    penalties += 0.20

    # Hash strictly valid sky pixels so excluded logo changes do not alter hash
    frame_hash = hashlib.sha256(gray[m_small].tobytes()).hexdigest()[:16]
    dhash = compute_dhash(work)

    quality_score = float(max(0.0, min(1.0, 1.0 - penalties)))
    usable_blocking = {"corrupted_frame", "excessively_dark", "lens_obstruction", "overexposed"}
    is_usable = (quality_score >= min_quality_score) and not any(f in usable_blocking for f in flags)

    metrics = {
        "mean_luminance": round(mean_lum, 2),
        "p5_luminance": round(p5_lum, 2),
        "p95_luminance": round(p95_lum, 2),
        "std_luminance": round(std_lum, 2),
        "laplacian_var_full": round(lap_var_full, 2),
        "laplacian_var_sky": round(lap_var_sky, 2),
        "glare_ratio": round(glare_ratio, 4),
        "dark_blob_ratio": round(dark_blob_ratio, 4),
        "entropy_bits": round(entropy, 3),
        "frame_mad": round(frame_mad, 4),
        "camera_shift_px": round(camera_shift_px, 2),
        "phase_corr_response": round(orientation_change_score, 3),
        "dhash": dhash,
        "flags": flags,
    }
    return QualityResult(
        quality_score=round(quality_score, 3),
        is_usable=is_usable,
        flags=flags,
        metrics=metrics,
        frame_hash=frame_hash,
        camera_shift_px=round(camera_shift_px, 2),
    )
