"""Stage 1: 5-Class Cloud Segmentation (MobileNetV3/EfficientNet UNet + Calibrated Radiometric Fallback)."""

from __future__ import annotations

from dataclasses import dataclass
import gc

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.ai.device import to_torch_device
from app.vision.masking import neutralize_excluded_pixels

NOT_SKY_IGNORE_INDEX = 255

# 5 clean pixel-level classes (Haze/Fog and Uncertain removed to eliminate labeling noise)
SEG_CLASSES = [
    "clear_sky",    # 0
    "thin_cloud",   # 1
    "dense_cloud",  # 2
    "dark_cloud",   # 3
    "sun_glare",    # 4
]

# High-contrast Windy-style BGR color palette for visual overlays
SEG_PALETTE_BGR = {
    0: (190, 95, 22),    # Deep clear sky blue
    1: (248, 189, 56),   # Vivid sky-cyan / thin cloud (#38BDF8)
    2: (252, 250, 248),  # Crisp bright white / dense cloud (#F8FAFC)
    3: (235, 75, 195),   # Storm violet-magenta / dark cloud (#C34BEB)
    4: (36, 191, 251),   # Warm solar gold / sun glare (#FBBF24)
}


@dataclass
class SegmentationOutput:
    class_map: np.ndarray          # uint8 (H, W) with class indices 0..4
    cloud_binary_mask: np.ndarray  # uint8 (H, W) 255 where thin/dense/dark cloud (1..3)
    confidence_map: np.ndarray     # float32 (H, W) in [0..1]
    mean_confidence: float
    overlay_bgr: np.ndarray
    color_mask_bgr: np.ndarray
    model_used: str


def _detect_optical_bloom(image_bgr: np.ndarray, L: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """
    Detect bright clipped solar disk & optical lens bloom even without camera tilt geometry.
    Requires genuine highlight clipping and local/frame contrast so bright overcast cloud tops
    (e.g. L=89 over L=82 clouds) are not mislabeled as sun glare.
    """
    img_f = image_bgr.astype(np.float32)
    min_rgb = np.min(img_f, axis=-1)
    max_rgb = np.max(img_f, axis=-1)

    l_p75 = float(np.percentile(L, 75))
    # True solar disk/glare is sensor-clipped (252+) and markedly brighter than ambient cloud deck,
    # or near-total saturation (min_rgb >= 238, L >= 95).
    is_clipped_sun = (
        ((max_rgb >= 251.0) & (min_rgb >= 228.0) & (L >= 93.0) & (sat <= 0.25) & (L >= l_p75 + 9.0))
        | ((min_rgb >= 242.0) & (L >= 96.5) & (sat <= 0.20))
    )
    core = is_clipped_sun.astype(np.float32)
    core_frac = float(np.mean(core))
    if core_frac < 1e-4 or core_frac > 0.35:
        return core

    h, w = image_bgr.shape[:2]
    k = max(9, (min(h, w) // 10) | 1)
    halo = cv2.GaussianBlur(core, (k, k), sigmaX=k * 0.28, sigmaY=k * 0.28)
    halo_max = float(np.max(halo))
    if halo_max > 1e-4:
        halo = np.clip(halo / halo_max, 0.0, 1.0)
    bright_gate = np.clip((L - 82.0) / 14.0, 0.0, 1.0)
    return np.clip(np.maximum(core, halo * bright_gate), 0.0, 1.0).astype(np.float32)


def _radiometric_probabilities_with_sun(
    image_bgr: np.ndarray,
    sun_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute smooth 5-class probability distribution (H, W, 5) over:
      [clear_sky, thin_cloud, dense_cloud, dark_cloud, sun_glare]
    with adaptive sky white-balance compensation so cool-tinted shaded cloud bases
    classify as dark_cloud / dense_cloud rather than clear_sky, and thick bright
    clouds classify as dense_cloud rather than thin_cloud.
    """
    img_f = image_bgr.astype(np.float32)
    b = img_f[:, :, 0]
    g = img_f[:, :, 1]
    r = img_f[:, :, 2]

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    L = lab[:, :, 0] * (100.0 / 255.0)
    sat = hsv[:, :, 1] / 255.0
    bsi = (b - r) / (b + r + 1e-4)  # High (>0.15) for clear blue sky
    min_rgb = np.minimum(np.minimum(b, g), r)
    max_rgb = np.maximum(np.maximum(b, g), r)

    # Estimate cool-AWB / Rayleigh veil bias from bright low-saturation cloud pixels (if present).
    # Pure clear blue sky has high saturation (>0.28), so cloud_ref remains empty and bias = 0.0.
    cloud_ref = (L >= 70.0) & (L <= 95.0) & (sat <= 0.20)
    bsi_bias = 0.0
    sat_bias = 0.0
    if float(np.mean(cloud_ref)) >= 0.08:
        bsi_bias = float(np.clip(np.median(bsi[cloud_ref]), 0.0, 0.11))
        sat_bias = float(np.clip(np.median(sat[cloud_ref]) - 0.04, 0.0, 0.14))

    bsi_rel = bsi - bsi_bias
    sat_rel = np.maximum(0.0, sat - sat_bias)

    # Relative shadow depth inside cloudy skies: shaded cloud bases sit well below upper cloud L*
    l_p80 = float(np.percentile(L, 80))
    shade_depth = np.maximum(0.0, l_p80 - L) if float(np.mean(cloud_ref)) >= 0.10 else np.zeros_like(L)

    bloom_prior = _detect_optical_bloom(image_bgr, L, sat)
    if sun_prior is not None and sun_prior.shape == L.shape:
        eff_sun = np.clip(np.maximum(sun_prior.astype(np.float32), bloom_prior), 0.0, 1.0)
    else:
        eff_sun = bloom_prior

    # Clear blue sky: requires strong corrected blue excess + real saturation, NOT dark/shaded cloud bases
    clear_bsi_adj = bsi_rel + 0.04 * eff_sun
    dark_shade_pen = 0.14 * np.maximum(0.0, 66.0 - L) + 0.08 * np.maximum(0.0, shade_depth - 8.0)
    logit_clear = (
        9.0 * (clear_bsi_adj - 0.11)
        + 4.5 * (sat_rel - 0.14)
        - dark_shade_pen
        - 0.03 * np.maximum(0.0, L - 84.0)
        - 4.0 * np.maximum(0.0, eff_sun - 0.55)
    )

    # Thin cloud: translucent veil (moderate brightness + residual blue sky signal 0.035..0.11)
    logit_thin = (
        2.4 * np.exp(-((bsi_rel - 0.065) ** 2) / (2.0 * 0.045 * 0.045))
        + 0.03 * (L - 56.0)
        - 0.06 * np.maximum(0.0, L - 78.0)
        - 3.5 * np.maximum(0.0, 0.035 - bsi_rel) * 10.0
        - 2.2 * sat_rel
        - 2.0 * np.maximum(0.0, eff_sun - 0.50)
    )

    # Dense cloud: opaque white/light-gray clouds (low bsi_rel, low sat_rel, bright-to-mid L)
    logit_dense = (
        0.10 * (L - 60.0)
        + 2.6 * np.clip(1.0 - np.maximum(0.0, bsi_rel) / 0.08, 0.0, 1.0)
        - 5.0 * np.maximum(0.0, bsi_rel - 0.06)
        - 4.5 * np.maximum(0.0, sat_rel - 0.12)
        - 0.08 * np.maximum(0.0, min_rgb - 244.0)
        - 4.5 * eff_sun * np.clip((L - 86.0) / 10.0, 0.0, 1.0)
    )

    # Dark cloud: convective / thick shaded cloud bases (L < 68 or strong shade_depth below cloud deck)
    logit_dark = (
        0.14 * (66.0 - L)
        + 0.09 * np.maximum(0.0, shade_depth - 6.0)
        + 1.4 * np.clip(1.0 - np.maximum(0.0, bsi_rel) / 0.10, 0.0, 1.0)
        - 5.5 * np.maximum(0.0, bsi_rel - 0.10)
        - 3.0 * np.maximum(0.0, sat_rel - 0.16)
        - 4.0 * eff_sun
    )

    # Sun glare: clipped core + warm/bright circumsolar aureole
    clipped_boost = 0.28 * np.maximum(0.0, max_rgb - 246.0) + 0.25 * np.maximum(0.0, min_rgb - 232.0)
    aureole_boost = 4.6 * eff_sun * np.clip((L - 82.0) / 14.0, 0.0, 1.0)
    logit_glare = clipped_boost + aureole_boost + 0.22 * (L - 93.5) - 3.5 * sat_rel - 1.5 * np.abs(bsi_rel) - 1.5

    logits = np.stack([logit_clear, logit_thin, logit_dense, logit_dark, logit_glare], axis=-1)
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(np.clip(logits, -20.0, 20.0))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    return probs.astype(np.float32), eff_sun


def _radiometric_probabilities(
    image_bgr: np.ndarray,
    sun_prior: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper returning only the (H, W, 5) radiometric probability map."""
    return _radiometric_probabilities_with_sun(image_bgr, sun_prior=sun_prior)[0]


def _adapt_nn_probs_to_5(nn_probs: np.ndarray, rad_probs: np.ndarray) -> np.ndarray:
    """Adapt 5-class, 2-class (Phase 1A binary), or legacy 6-class logits to the 5 target classes."""
    ch = nn_probs.shape[2]
    if ch == 5:
        return nn_probs
    if ch == 2:
        # Phase 1A binary (0=clear, 1=cloud): distribute cloud probability across thin/dense/dark via radiometric prior
        out = np.zeros_like(rad_probs)
        out[:, :, 0] = nn_probs[:, :, 0]
        cloud_p = nn_probs[:, :, 1:2]
        rad_cloud = rad_probs[:, :, 1:4]
        rad_cloud_sum = np.maximum(np.sum(rad_cloud, axis=-1, keepdims=True), 1e-5)
        out[:, :, 1:4] = cloud_p * (rad_cloud / rad_cloud_sum)
        out[:, :, 4] = rad_probs[:, :, 4] * 0.25
        return out
    # Legacy 6-class checkpoint: take first 5 channels
    return nn_probs[:, :, :5]


def run_segmentation(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    inference_resolution: int = 512,
    uncertainty_entropy_threshold: float = 0.72,
    seg_model: torch.nn.Module | None = None,
    device: str = "cpu",
    model_name: str = "radiometric-physics-v1",
    sun_prior_map: np.ndarray | None = None,
    nn_weight: float = 0.75,
) -> SegmentationOutput:
    """
    Segment sky pixels into 5 clean classes (clear_sky, thin_cloud, dense_cloud, dark_cloud, sun_glare).
    Static Not_Sky areas (valid_mask == 0) are neutralized and masked out cleanly.
    """
    _ = uncertainty_entropy_threshold  # kept in signature for backward compatibility
    orig_h, orig_w = image_bgr.shape[:2]
    clean_bgr = neutralize_excluded_pixels(image_bgr, valid_mask)
    res = max(128, min(inference_resolution, 1024))
    # Keep DirectML (Windows WDDM shared display GPU) inference fast (<40ms) to avoid DX12 TDR display flashes
    if device == "directml":
        res = min(res, 384)
    small_bgr = cv2.resize(clean_bgr, (res, res), interpolation=cv2.INTER_AREA)
    sun_small = (
        cv2.resize(sun_prior_map.astype(np.float32), (res, res), interpolation=cv2.INTER_LINEAR)
        if sun_prior_map is not None and sun_prior_map.shape[:2] == (orig_h, orig_w)
        else None
    )

    rad_probs, eff_sun = _radiometric_probabilities_with_sun(small_bgr, sun_prior=sun_small)  # (res, res, 5)
    probs_5 = rad_probs.copy()

    base_nn_w = float(np.clip(nn_weight, 0.0, 0.90))

    def _blend_nn_with_sun_guard(nn_arr: np.ndarray) -> np.ndarray:
        adapted = _adapt_nn_probs_to_5(nn_arr, rad_probs)
        # Normal weighting: base_nn_w NN + (1-base_nn_w) physics; inside intense sun glare or
        # unmistakable dark cloud undersides (where physics says zero clear sky), trust physics
        glare_strength = np.clip(np.maximum(rad_probs[:, :, 4], eff_sun * rad_probs[:, :, 4] * 1.4), 0.0, 1.0)
        dark_base_guard = np.where(
            (rad_probs[:, :, 3] > 0.45) & (rad_probs[:, :, 0] < 0.12),
            rad_probs[:, :, 3],
            0.0,
        )
        override = np.maximum(glare_strength, 0.70 * dark_base_guard)
        nn_w = np.clip(base_nn_w - 0.60 * override, 0.12, base_nn_w)[..., None]
        return nn_w * adapted + (1.0 - nn_w) * rad_probs

    used_label = model_name
    torch_dev = to_torch_device(device)
    if seg_model is not None:
        try:
            rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(torch_dev)
            seg_model.eval()
            with torch.no_grad():
                logits = seg_model(tensor)
                nn_raw = F.softmax(logits, dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            del tensor, logits
            probs_5 = _blend_nn_with_sun_guard(nn_raw)
        except RuntimeError as exc:
            err_l = str(exc).lower()
            if device == "directml" or "memory" in err_l or "alloc" in err_l or "sospesa" in err_l or "removed" in err_l:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if device == "directml":
                    probs_5 = rad_probs
                    used_label = "radiometric-physics-v1 (directml-safe-fallback)"
                else:
                    try:
                        tiny_bgr = cv2.resize(clean_bgr, (256, 256), interpolation=cv2.INTER_AREA)
                        rgb_t = cv2.cvtColor(tiny_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        t_in = torch.from_numpy(rgb_t).permute(2, 0, 1).unsqueeze(0).to(torch_dev)
                        with torch.no_grad():
                            nn_small = F.softmax(seg_model(t_in), dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                        del t_in
                        nn_up = cv2.resize(nn_small, (res, res), interpolation=cv2.INTER_LINEAR)
                        probs_5 = _blend_nn_with_sun_guard(nn_up)
                    except Exception:
                        probs_5 = rad_probs
                        used_label = "radiometric-physics-v1 (low-ram-fallback)"
            else:
                raise
    else:
        used_label = "radiometric-physics-v1"

    probs_5_sum = np.maximum(np.sum(probs_5, axis=-1, keepdims=True), 1e-6)
    probs_5_norm = probs_5 / probs_5_sum

    max_prob = np.max(probs_5_norm, axis=-1)
    class_small = np.argmax(probs_5_norm, axis=-1).astype(np.uint8)

    # Resize back to original dimensions
    class_map = cv2.resize(class_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    conf_map = cv2.resize(max_prob.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    valid_bool = valid_mask > 0
    mean_confidence = float(np.mean(conf_map[valid_bool])) if np.any(valid_bool) else float(np.mean(conf_map))

    # Cloud binary mask: thin (1), dense (2), dark (3) — excludes clear (0) and sun_glare (4)
    cloud_binary = np.where((class_map >= 1) & (class_map <= 3) & valid_bool, 255, 0).astype(np.uint8)

    # Build color mask & semi-transparent overlay
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls_idx, bgr in SEG_PALETTE_BGR.items():
        color_mask[class_map == cls_idx] = bgr
    color_mask[~valid_bool] = (18, 22, 30)

    overlay = cv2.addWeighted(image_bgr, 0.58, color_mask, 0.42, 0.0)
    overlay[~valid_bool] = (image_bgr[~valid_bool] * 0.35).astype(np.uint8)

    return SegmentationOutput(
        class_map=class_map,
        cloud_binary_mask=cloud_binary,
        confidence_map=conf_map,
        mean_confidence=round(mean_confidence, 4),
        overlay_bgr=overlay,
        color_mask_bgr=color_mask,
        model_used=used_label,
    )
