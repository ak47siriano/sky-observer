"""Adaptive Night & Twilight Sky Analysis (Moon Position, Star Visibility & Urban/Dark-Site Segmentation).

Design references (methodology cited, zero third-party source code copied):
- `allclear` / `pynephoscope`: point-source star detection for clear-sky night verification
- Astronomical lunar ephemeris (low-precision Meeus/USNO equations, 100% offline)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from app.vision.segmentation import SEG_PALETTE_BGR, SegmentationOutput


def classify_sky_regime(solar_elevation_deg: float) -> str:
    """Classify diurnal sky illumination regime from solar elevation angle."""
    el = float(solar_elevation_deg)
    if el > 6.0:
        return "day"
    if el >= -6.0:
        return "twilight"
    return "night"


def compute_lunar_position_and_phase(
    lat: float,
    lon: float,
    dt_utc: datetime | None = None,
) -> dict[str, Any]:
    """
    Compute approximate Moon elevation, azimuth, and illuminated phase fraction (100% offline).
    Accuracy ~1-2 deg, sufficient for sky-camera moonlight glare & attenuation cues.
    """
    ts = dt_utc or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)

    # Days since J2000.0 (2000-01-01 12:00 UTC)
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
    d = (ts - j2000).total_seconds() / 86400.0

    # Mean lunar elements (degrees)
    L = math.radians((218.316 + 13.176396 * d) % 360.0)
    M = math.radians((134.963 + 13.064993 * d) % 360.0)
    F = math.radians((93.272 + 13.229350 * d) % 360.0)

    # Ecliptic longitude & latitude
    lam = L + math.radians(6.289 * math.sin(M))
    beta = math.radians(5.128 * math.sin(F))
    eps = math.radians(23.4397)

    # Right ascension & declination
    ra = math.atan2(math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps), math.cos(lam))
    sin_dec = math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam)
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))

    # Local sidereal time
    gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    lst = math.radians((gmst_deg + float(lon)) % 360.0)
    ha = lst - ra
    lat_r = math.radians(max(-89.9, min(89.9, float(lat))))

    sin_el = max(-1.0, min(1.0, math.sin(lat_r) * math.sin(dec) + math.cos(lat_r) * math.cos(dec) * math.cos(ha)))
    el_deg = math.degrees(math.asin(sin_el))
    cos_el = max(1e-6, math.cos(math.radians(el_deg)))
    sin_az = -math.cos(dec) * math.sin(ha) / cos_el
    cos_az = (math.sin(dec) - math.sin(lat_r) * sin_el) / (max(1e-6, math.cos(lat_r)) * cos_el)
    az_deg = math.degrees(math.atan2(sin_az, cos_az)) % 360.0

    # Synodic phase fraction (0=new moon, 1=full moon)
    synodic = 29.53058867
    known_new_moon = datetime(2000, 1, 6, 18, 14, 0, tzinfo=UTC)
    age_days = ((ts - known_new_moon).total_seconds() / 86400.0) % synodic
    phase_frac = 0.5 * (1.0 - math.cos(2.0 * math.pi * age_days / synodic))

    return {
        "elevation_deg": round(el_deg, 2),
        "azimuth_deg": round(az_deg, 2),
        "phase_fraction": round(phase_frac, 3),
        "above_horizon": bool(el_deg > 2.0),
    }


def detect_bright_stars(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    max_stars: int = 120,
) -> dict[str, Any]:
    """
    Detect crisp star-like point sources inside the valid night sky mask using a
    Difference-of-Gaussians (DoG) peak detector. Auto-enables star clarity cues only
    when enough crisp point sources survive background subtraction (inspired by allclear/pynephoscope).
    """
    h, w = image_bgr.shape[:2]
    work_w = min(480, w)
    work_h = max(120, int(round(work_w * float(h) / max(1.0, float(w)))))
    small = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    vm = cv2.resize(valid_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST) > 0
    # Erode slightly to avoid horizon streetlamp / roof edge false stars
    vm_eroded = cv2.erode(vm.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    if int(vm_eroded.sum()) < 100:
        return {"star_count": 0, "stars_usable": False, "sky_clarity_cue": "unavailable"}

    # Background subtraction via Difference of Gaussians
    small_blur = cv2.GaussianBlur(gray, (3, 3), 0.7)
    bg_blur = cv2.GaussianBlur(gray, (15, 15), 3.5)
    dog = small_blur - bg_blur

    sky_dog = dog[vm_eroded]
    med = float(np.median(sky_dog))
    mad = float(np.median(np.abs(sky_dog - med))) + 1e-3
    sigma = 1.4826 * mad
    thresh = max(8.0, med + 5.5 * sigma)

    # Local 5x5 maxima that are compact point sources (not large cloud/lamp blobs)
    dilated = cv2.dilate(dog, np.ones((5, 5), np.uint8))
    peaks = (dog == dilated) & (dog > thresh) & vm_eroded & (gray < 252.0)

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
    star_count = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 1 <= area <= 12:
            star_count += 1
            if star_count >= max_stars:
                break

    stars_usable = star_count >= 5
    clarity = (
        "clear_stars_visible"
        if star_count >= 12
        else ("partial_stars" if star_count >= 5 else "stars_obscured_or_bright_sky")
    )
    return {
        "star_count": star_count,
        "stars_usable": stars_usable,
        "sky_clarity_cue": clarity,
    }


def segment_night_sky_adaptive(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    inference_resolution: int = 384,
    night_baseline_luminance: float | None = None,
    urban_light_pollution: bool | None = None,
    lunar_info: dict[str, Any] | None = None,
    star_info: dict[str, Any] | None = None,
) -> tuple[SegmentationOutput, dict[str, Any]]:
    """
    Per-camera adaptive night cloud segmentation supporting BOTH:
      - Urban light-polluted sites: low/mid clouds reflect city glow -> brighter & warmer than clear night sky.
      - Dark rural/mountain sites: clear sky shows stars and faint airglow; clouds appear smoother/darker
        (or slightly illuminated by moonlight).
    """
    orig_h, orig_w = image_bgr.shape[:2]
    res = max(128, min(inference_resolution, 512))
    small = cv2.resize(image_bgr, (res, res), interpolation=cv2.INTER_AREA)
    vm_small = cv2.resize(valid_mask, (res, res), interpolation=cv2.INTER_NEAREST) > 0
    if not np.any(vm_small):
        vm_small = np.ones((res, res), dtype=bool)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b_ch = small[:, :, 0].astype(np.float32)
    r_ch = small[:, :, 2].astype(np.float32)
    warmth = (r_ch - b_ch) / (r_ch + b_ch + 10.0)

    sky_L = L[vm_small]
    p20_L = float(np.percentile(sky_L, 20))
    med_L = float(np.median(sky_L))
    p85_L = float(np.percentile(sky_L, 85))
    mean_warmth = float(np.mean(warmth[vm_small]))

    # Auto-infer urban light-pollution regime if not explicitly configured:
    # Urban skies show elevated ambient L* (>10) or warm sodium/LED glow (r > b)
    if urban_light_pollution is None:
        is_urban = (med_L >= 11.0) or (p85_L >= 20.0 and mean_warmth > 0.02)
    else:
        is_urban = bool(urban_light_pollution)

    # Adaptive clear-sky baseline luminance (tracks dark clear-sky floor)
    clear_ref_L = float(night_baseline_luminance) if night_baseline_luminance is not None else max(3.5, p20_L)

    stars_n = int((star_info or {}).get("star_count", 0))
    moon_up = bool((lunar_info or {}).get("above_horizon", False))
    moon_phase = float((lunar_info or {}).get("phase_fraction", 0.0))

    # Determine moon visibility cue from localized bright disk/halo when moon is above horizon
    moon_cue = "below_horizon"
    if moon_up:
        if p85_L > max(35.0, clear_ref_L + 22.0) and moon_phase >= 0.15:
            moon_cue = "visible"
        elif p85_L > clear_ref_L + 10.0:
            moon_cue = "attenuated"
        else:
            moon_cue = "obscured"

    class_small = np.zeros((res, res), dtype=np.uint8)
    conf_small = np.full((res, res), 0.55, dtype=np.float32)

    if is_urban:
        # Urban light-polluted night sky: clouds glow brighter & warmer than dark clear-sky background
        delta_L = L - clear_ref_L
        warm_boost = np.clip(warmth * 22.0, -4.0, 10.0)
        cloud_score = delta_L + warm_boost

        # Star visibility gently reduces false cloud classification in clear urban skies
        star_offset = 4.5 if stars_n >= 10 else (2.0 if stars_n >= 5 else 0.0)
        thin_m = (cloud_score >= (4.5 + star_offset)) & (cloud_score < (12.0 + star_offset))
        dense_m = (cloud_score >= (12.0 + star_offset)) & (L >= 18.0)
        dark_m = (cloud_score >= (12.0 + star_offset)) & (L < 18.0)

        class_small[thin_m] = 1
        class_small[dense_m] = 2
        class_small[dark_m] = 3
        conf_small = np.clip(0.45 + np.abs(cloud_score - 6.0) / 30.0, 0.35, 0.78).astype(np.float32)
    else:
        # Dark rural/astronomical site: clear sky has stars and low uniform glow; clouds obscure stars
        if stars_n >= 12:
            # Many crisp stars visible -> predominantly clear sky; flag only brighter moonlit cloud banks
            cloud_m = (L > clear_ref_L + 10.0) & (L > 14.0)
            class_small[cloud_m] = 1
            conf_small = np.full((res, res), 0.68, dtype=np.float32)
        else:
            # Few/no stars: compare against local texture/luminance elevation
            smooth_L = cv2.GaussianBlur(L, (11, 11), 3.0)
            # In dark sites without stars, moonlit or horizon-lit decks have elevated smooth L*,
            # while thick overcast is uniformly opaque (very low star count + muted local variance)
            local_std = np.sqrt(np.maximum(0.0, cv2.GaussianBlur(L * L, (9, 9), 2.5) - smooth_L * smooth_L))
            if med_L < 4.5 and stars_n == 0:
                # Pitch-black sky without stars: likely dark overcast OR camera noise floor -> low-confidence dark cloud
                class_small[vm_small] = 3
                conf_small = np.full((res, res), 0.38, dtype=np.float32)
            else:
                thin_m = (L >= clear_ref_L + 3.5) & (L < clear_ref_L + 9.0)
                dense_m = L >= clear_ref_L + 9.0
                dark_m = (local_std < 0.8) & (L >= 3.0) & (~thin_m) & (~dense_m) & (stars_n < 3)
                class_small[thin_m] = 1
                class_small[dense_m] = 2
                class_small[dark_m] = 3
                conf_small = np.full((res, res), 0.48, dtype=np.float32)

    class_map = cv2.resize(class_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    conf_map = cv2.resize(conf_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    valid_bool = valid_mask > 0
    mean_conf = float(np.mean(conf_map[valid_bool])) if np.any(valid_bool) else float(np.mean(conf_map))
    cloud_binary = np.where((class_map >= 1) & (class_map <= 3) & valid_bool, 255, 0).astype(np.uint8)

    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls_idx, bgr in SEG_PALETTE_BGR.items():
        color_mask[class_map == cls_idx] = bgr
    color_mask[~valid_bool] = (18, 22, 30)

    overlay = cv2.addWeighted(image_bgr, 0.58, color_mask, 0.42, 0.0)
    overlay[~valid_bool] = (image_bgr[~valid_bool] * 0.35).astype(np.uint8)

    seg_out = SegmentationOutput(
        class_map=class_map,
        cloud_binary_mask=cloud_binary,
        confidence_map=conf_map,
        mean_confidence=round(mean_conf, 4),
        overlay_bgr=overlay,
        color_mask_bgr=color_mask,
        model_used="adaptive-night-urban-v1" if is_urban else "adaptive-night-darksite-v1",
    )
    meta = {
        "night_site_mode": "urban_light_pollution" if is_urban else "dark_site",
        "clear_baseline_L": round(clear_ref_L, 2),
        "median_sky_L": round(med_L, 2),
        "moon_visibility_cue": moon_cue,
        "lunar": lunar_info or {},
        "stars": star_info or {},
    }
    return seg_out, meta
