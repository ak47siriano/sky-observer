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


def _radiometric_probabilities(image_bgr: np.ndarray) -> np.ndarray:
    """
    Compute smooth 5-class probability distribution (H, W, 5) over:
      [clear_sky, thin_cloud, dense_cloud, dark_cloud, sun_glare]
    preserving the physical Red/Blue Ratio (RBR) and CIE Lab luminance structure.
    """
    img_f = image_bgr.astype(np.float32)
    b = img_f[:, :, 0]
    g = img_f[:, :, 1]
    r = img_f[:, :, 2]

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    L = lab[:, :, 0] * (100.0 / 255.0)
    sat = hsv[:, :, 1] / 255.0
    bsi = (b - r) / (b + r + 1e-4)  # High (>0.08) for clear blue sky
    min_rgb = np.minimum(np.minimum(b, g), r)

    logit_clear = 5.5 * (bsi - 0.04) + 2.5 * (sat - 0.14) - 0.02 * np.maximum(0.0, L - 80.0)
    logit_thin = 2.6 * (1.0 - np.abs(bsi - 0.01) * 5.0) + 0.04 * (L - 52.0) - 2.2 * sat
    logit_dense = 0.09 * (L - 62.0) - 4.0 * bsi - 3.5 * (sat - 0.10) - 0.08 * np.maximum(0.0, min_rgb - 244.0)
    logit_dark = 0.11 * (45.0 - L) - 3.0 * bsi - 1.5 * sat
    # Sun glare: clipped/near-saturated bright solar disk & aureole
    logit_glare = 0.35 * (min_rgb - 242.0) + 0.18 * (L - 94.0) - 4.0 * sat - 2.0 * np.abs(bsi)

    logits = np.stack([logit_clear, logit_thin, logit_dense, logit_dark, logit_glare], axis=-1)
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(np.clip(logits, -20.0, 20.0))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    return probs.astype(np.float32)


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

    rad_probs = _radiometric_probabilities(small_bgr)  # (res, res, 5)
    probs_5 = rad_probs.copy()

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
            nn_probs = _adapt_nn_probs_to_5(nn_raw, rad_probs)
            # Ensemble 75% fine-tuned neural network + 25% radiometric physical prior
            probs_5 = 0.75 * nn_probs + 0.25 * rad_probs
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
                        probs_5 = 0.75 * _adapt_nn_probs_to_5(nn_up, rad_probs) + 0.25 * rad_probs
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
