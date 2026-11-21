"""7.8 Texture, Edge Density, Fragmentation & Structural Analysis module."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _fast_glcm_features(gray_u8: np.ndarray, levels: int = 16) -> dict[str, float]:
    """Compute horizontal + vertical GLCM contrast, homogeneity, energy, and entropy quickly via NumPy."""
    q = (gray_u8.astype(np.int32) * levels) // 256
    q = np.clip(q, 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)
    # Horizontal offset (dx=2)
    a_h = q[:, :-2].ravel()
    b_h = q[:, 2:].ravel()
    np.add.at(glcm, (a_h, b_h), 1.0)
    # Vertical offset (dy=2)
    a_v = q[:-2, :].ravel()
    b_v = q[2:, :].ravel()
    np.add.at(glcm, (a_v, b_v), 1.0)

    glcm = glcm + glcm.T
    total = glcm.sum()
    if total <= 0:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 1.0, "glcm_energy": 1.0, "glcm_entropy": 0.0}
    P = glcm / total
    idx = np.arange(levels)
    diff = idx[:, None] - idx[None, :]
    contrast = float(np.sum((diff ** 2) * P))
    homogeneity = float(np.sum(P / (1.0 + np.abs(diff))))
    energy = float(np.sum(P ** 2))
    nz = P[P > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    return {
        "glcm_contrast": round(contrast, 3),
        "glcm_homogeneity": round(homogeneity, 4),
        "glcm_energy": round(energy, 4),
        "glcm_entropy": round(entropy, 3),
    }


def analyze_texture(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    cloud_binary_mask: np.ndarray,
    prev_cloud_binary_mask: np.ndarray | None = None,
    prev_texture_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze edge density, local contrast, GLCM texture, cloud fragmentation, and region growth/disappearance."""
    small_bgr = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    m_small = cv2.resize(valid_mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    c_small = (cv2.resize(cloud_binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 0) & m_small

    if int(m_small.sum()) < 20:
        m_small = np.ones((256, 256), dtype=bool)

    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130) > 0
    edge_density = float(np.mean(edges[m_small]))

    # Local contrast via sliding std-dev (9x9 window)
    gray_f = gray.astype(np.float32)
    local_mean = cv2.boxFilter(gray_f, -1, (9, 9))
    local_sq_mean = cv2.boxFilter(gray_f * gray_f, -1, (9, 9))
    local_var = np.maximum(0.0, local_sq_mean - local_mean * local_mean)
    local_contrast = float(np.mean(np.sqrt(local_var)[m_small]))

    glcm = _fast_glcm_features(gray)

    # Fragmentation via connected components on cloud mask
    c_u8 = c_small.astype(np.uint8)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(c_u8, connectivity=8)
    # Filter tiny 1-3px noise blobs
    valid_sky_area = max(int(m_small.sum()), 1)
    cloud_area_total = int(c_small.sum())
    component_areas = [
        int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if int(stats[i, cv2.CC_STAT_AREA]) >= 12
    ]
    fragment_count = len(component_areas)
    largest_fragment_ratio = float(max(component_areas) / max(cloud_area_total, 1)) if component_areas else 0.0
    # Fragmentation index: 0 = completely uniform/clear, 1 = many dispersed puffs
    fragmentation_index = float(min(1.0, (fragment_count / 24.0) * (1.0 - 0.5 * largest_fragment_ratio)))

    # Cloud-region growth vs disappearance against previous frame
    cloud_growth_pct = 0.0
    cloud_disappearance_pct = 0.0
    net_cloud_change_pct = 0.0
    if prev_cloud_binary_mask is not None:
        prev_c = (cv2.resize(prev_cloud_binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 0) & m_small
        newly_clouded = np.logical_and(c_small, np.logical_not(prev_c))
        newly_cleared = np.logical_and(prev_c, np.logical_not(c_small))
        cloud_growth_pct = float(np.sum(newly_clouded) / valid_sky_area * 100.0)
        cloud_disappearance_pct = float(np.sum(newly_cleared) / valid_sky_area * 100.0)
        net_cloud_change_pct = cloud_growth_pct - cloud_disappearance_pct

    texture_delta = 0.0
    if prev_texture_metrics:
        texture_delta = local_contrast - float(prev_texture_metrics.get("local_contrast", local_contrast))

    structure_type = "uniform"
    if cloud_area_total / valid_sky_area < 0.08:
        structure_type = "clear_smooth"
    elif fragmentation_index > 0.42:
        structure_type = "fragmented_developing"
    elif edge_density > 0.08 and local_contrast > 14.0:
        structure_type = "cellular_textured"
    elif largest_fragment_ratio > 0.78 and local_contrast < 9.0:
        structure_type = "uniform_stratiform"
    else:
        structure_type = "mixed_structure"

    return {
        "edge_density": round(edge_density, 4),
        "local_contrast": round(local_contrast, 3),
        "texture_delta": round(texture_delta, 3),
        "fragment_count": fragment_count,
        "largest_fragment_ratio": round(largest_fragment_ratio, 3),
        "fragmentation_index": round(fragmentation_index, 3),
        "cloud_growth_pct": round(cloud_growth_pct, 2),
        "cloud_disappearance_pct": round(cloud_disappearance_pct, 2),
        "net_cloud_change_pct": round(net_cloud_change_pct, 2),
        "structure_type": structure_type,
        **glcm,
    }
