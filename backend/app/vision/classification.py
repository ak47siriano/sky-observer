"""Stage 2: Cloud Structure & WMO Visual Genus Classification (18-dim Tabular Fusion + Optional WMO Grouping)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CLOUD_TYPE_CLASSES = [
    "clear_sky",
    "cirrus",
    "cirrostratus",
    "cumulus",
    "stratocumulus",
    "stratus",
    "altocumulus",
    "altostratus",
    "cumulonimbus_like",
    "mixed_clouds",
    "unknown",
]

# Optional macro-grouping of visually ambiguous single-camera WMO genera
# (enabled when val_macro_f1 is imbalanced across fine-grained WMO classes)
WMO_GROUPED_CLASSES = [
    "clear_sky",
    "high_cirriform",     # cirrus + cirrostratus
    "mid_alto",           # altocumulus + altostratus
    "low_cumuliform",     # cumulus
    "low_stratiform",     # stratus + stratocumulus + mixed_clouds
    "convective_storm",   # cumulonimbus_like
]

WMO_TO_GROUPED_MAP: dict[str, str] = {
    "clear_sky": "clear_sky",
    "cirrus": "high_cirriform",
    "cirrostratus": "high_cirriform",
    "altocumulus": "mid_alto",
    "altostratus": "mid_alto",
    "cumulus": "low_cumuliform",
    "stratocumulus": "low_stratiform",
    "stratus": "low_stratiform",
    "mixed_clouds": "low_stratiform",
    "cumulonimbus_like": "convective_storm",
    "unknown": "low_stratiform",
}


def group_wmo_label(label: str) -> str:
    """Map a fine-grained WMO cloud genus to its balanced single-camera macro group."""
    return WMO_TO_GROUPED_MAP.get(str(label or "").lower().strip(), "low_stratiform")


def _extract_feature_vector(
    coverage_metrics: dict[str, Any],
    color_metrics: dict[str, Any],
    texture_metrics: dict[str, Any],
    solar_elevation_deg: float | None = None,
) -> np.ndarray:
    """
    Build 18-dim tabular feature vector for Stage-2 Fusion Classifier:
      [0..4]  5 Stage-1 segmentation class fractions (clear, thin, dense, dark, sun_glare)
      [5]     total cloud coverage fraction
      [6]     normalized solar elevation (deg / 90.0)
      [7..12] radiometric descriptors (RBR, R-B normalized, luminance, BSI, grayness, saturation)
      [13..17] texture descriptors (edge_density, local_contrast, fragmentation, GLCM contrast, GLCM homogeneity)
    """
    bsi = float(color_metrics.get("blue_sky_index", 0.0))
    # Derive physical Red/Blue Ratio (RBR) & R-B from BSI when not explicitly provided
    # BSI = (B - R) / (B + R) => R/B = (1 - BSI) / (1 + BSI)
    default_rbr = float(np.clip((1.0 - bsi) / max(1.0 + bsi, 0.15), 0.1, 3.0))
    rbr = float(color_metrics.get("rbr", default_rbr))
    r_minus_b = float(color_metrics.get("r_minus_b_norm", -bsi))
    sol_norm = float(np.clip((float(solar_elevation_deg) if solar_elevation_deg is not None else 35.0) / 90.0, -0.5, 1.0))

    vec = np.array(
        [
            float(coverage_metrics.get("clear_sky_pct", 100.0)) / 100.0,
            float(coverage_metrics.get("thin_cloud_pct", 0.0)) / 100.0,
            float(coverage_metrics.get("dense_cloud_pct", 0.0)) / 100.0,
            float(coverage_metrics.get("dark_cloud_pct", 0.0)) / 100.0,
            float(coverage_metrics.get("sun_glare_pct", coverage_metrics.get("haze_fog_pct", 0.0))) / 100.0,
            float(coverage_metrics.get("cloud_coverage_pct", 0.0)) / 100.0,
            sol_norm,
            rbr,
            r_minus_b,
            float(color_metrics.get("avg_luminance", 50.0)) / 100.0,
            bsi,
            float(color_metrics.get("grayness", 0.5)),
            float(color_metrics.get("saturation", 0.25)),
            float(texture_metrics.get("edge_density", 0.0)) * 5.0,
            float(texture_metrics.get("local_contrast", 0.0)) / 40.0,
            float(texture_metrics.get("fragmentation_index", 0.0)),
            float(texture_metrics.get("glcm_contrast", 2.0)) / 20.0,
            float(texture_metrics.get("glcm_homogeneity", 0.6)),
        ],
        dtype=np.float32,
    )
    return vec


def _heuristic_cloud_probs(
    coverage_metrics: dict[str, Any],
    color_metrics: dict[str, Any],
    texture_metrics: dict[str, Any],
) -> dict[str, float]:
    """Compute calibrated physical prior probabilities over visual cloud classes."""
    cov = float(coverage_metrics.get("cloud_coverage_pct", 0.0))
    thin = float(coverage_metrics.get("thin_cloud_pct", 0.0))
    dense = float(coverage_metrics.get("dense_cloud_pct", 0.0))
    dark = float(coverage_metrics.get("dark_cloud_pct", 0.0))
    lum = float(color_metrics.get("avg_luminance", 55.0))
    frag = float(texture_metrics.get("fragmentation_index", 0.0))
    contrast = float(texture_metrics.get("local_contrast", 0.0))
    edge = float(texture_metrics.get("edge_density", 0.0))

    scores = {c: 0.05 for c in CLOUD_TYPE_CLASSES[:-1]}
    if cov < 12.0:
        scores["clear_sky"] += 4.2
    elif cov > 82.0 and dark > 25.0 and contrast > 10.0:
        scores["cumulonimbus_like"] += 3.2
        scores["stratus"] += 1.2
    elif cov > 80.0 and contrast < 10.0 and dark < 20.0:
        scores["stratus"] += 3.0
        scores["altostratus"] += 1.6
    elif thin > dense and cov < 55.0 and edge > 0.04:
        scores["cirrus"] += 2.6
        scores["cirrostratus"] += 1.4
    elif thin > dense and cov >= 55.0:
        scores["cirrostratus"] += 2.5
        scores["altostratus"] += 1.8
    elif frag > 0.35 and dense > 15.0 and cov < 65.0:
        scores["cumulus"] += 2.8
        scores["altocumulus"] += 1.3
    elif frag > 0.25 and cov >= 55.0 and dark < 25.0:
        scores["stratocumulus"] += 2.6
        scores["altocumulus"] += 1.4
    else:
        scores["mixed_clouds"] += 2.3
        scores["stratocumulus"] += 1.2

    if dark > 35.0 and lum < 45.0:
        scores["cumulonimbus_like"] += 2.1

    vals = np.array(list(scores.values()), dtype=np.float64)
    vals = np.exp(vals - np.max(vals))
    vals /= np.sum(vals)
    return {k: float(v) for k, v in zip(scores.keys(), vals)}


def _collapse_probs_to_grouped(probs: dict[str, float]) -> dict[str, float]:
    """Aggregate fine-grained WMO class probabilities into the 6 balanced single-camera macro groups."""
    grouped = {g: 0.0 for g in WMO_GROUPED_CLASSES}
    for k, v in probs.items():
        if k == "unknown":
            continue
        grouped[group_wmo_label(k)] += float(v)
    total = sum(grouped.values()) or 1.0
    return {k: round(v / total, 4) for k, v in grouped.items()}


def classify_cloud_type(
    image_bgr: np.ndarray,
    coverage_metrics: dict[str, Any],
    color_metrics: dict[str, Any],
    texture_metrics: dict[str, Any],
    min_confidence: float = 0.35,
    cls_model: torch.nn.Module | None = None,
    device: str = "cpu",
    model_name: str = "physical-heuristic-v1",
    solar_elevation_deg: float | None = None,
    group_ambiguous_wmo: bool = False,
) -> dict[str, Any]:
    """
    Estimate probability distribution across WMO cloud formation classes.
    Returns 'unknown' as dominant label when confidence falls below min_confidence.
    """
    prior_probs = _heuristic_cloud_probs(coverage_metrics, color_metrics, texture_metrics)
    probs = dict(prior_probs)
    used_model = model_name

    if cls_model is not None:
        from app.ai.device import to_torch_device

        try:
            torch_dev = to_torch_device(device)
            rgb = cv2.cvtColor(cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(torch_dev)
            feat_vec = torch.from_numpy(
                _extract_feature_vector(coverage_metrics, color_metrics, texture_metrics, solar_elevation_deg=solar_elevation_deg)
            ).unsqueeze(0).to(torch_dev)
            cls_model.eval()
            with torch.no_grad():
                logits = cls_model(img_tensor, feat_vec)
                nn_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            if len(nn_probs) == len(WMO_GROUPED_CLASSES):
                # Model was explicitly trained in 6-class grouped WMO mode
                grouped_prior = _collapse_probs_to_grouped(prior_probs)
                probs = {
                    g: 0.80 * float(nn_probs[idx]) + 0.20 * float(grouped_prior.get(g, 0.0))
                    for idx, g in enumerate(WMO_GROUPED_CLASSES)
                }
            else:
                for idx, cls_name in enumerate(CLOUD_TYPE_CLASSES):
                    nn_p = float(nn_probs[idx]) if idx < len(nn_probs) else 0.0
                    pr_p = float(prior_probs.get(cls_name, 0.0))
                    probs[cls_name] = 0.80 * nn_p + 0.20 * pr_p
        except Exception:
            probs["unknown"] = 0.03
            used_model = "physical-heuristic-v1 (fallback)"
    else:
        probs["unknown"] = 0.03
        used_model = "physical-heuristic-v1"

    if group_ambiguous_wmo and len(probs) != len(WMO_GROUPED_CLASSES):
        probs = _collapse_probs_to_grouped(probs)

    total = sum(probs.values()) or 1.0
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    ranked = sorted(((k, v) for k, v in probs.items() if k != "unknown"), key=lambda x: x[1], reverse=True)
    top_label, top_prob = ranked[0]
    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = float(min(0.99, max(0.0, top_prob * 0.85 + (top_prob - second_prob) * 0.45)))

    dominant = "unknown" if confidence < min_confidence else top_label

    return {
        "dominant_cloud_type": dominant,
        "confidence": round(confidence, 4),
        "probabilities": probs,
        "grouped_wmo": bool(group_ambiguous_wmo or len(probs) == len(WMO_GROUPED_CLASSES)),
        "solar_elevation_deg": round(float(solar_elevation_deg), 2) if solar_elevation_deg is not None else None,
        "model_used": used_model,
    }
