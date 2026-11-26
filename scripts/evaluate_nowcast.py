"""Reproducible Nowcasting Benchmark (Persistence vs Optical-Flow Advection vs Hybrid).

Inspired by the LenghuSky-8 benchmark (Rui et al., CVPR Findings 2025):
compares Persistence, Optical-Flow Mask Advection, and Sky Observer's hybrid forecast
using Forecast Skill Score = 1 - MAE_model / MAE_persistence and pixel-level Cloud Mask IoU.

Usage:
  PYTHONPATH=backend python scripts/evaluate_nowcast.py
  PYTHONPATH=backend python scripts/evaluate_nowcast.py --from-db --json-out nowcast_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.ai.datasets import _render_synthetic_sky_frame  # noqa: E402
from app.forecast.advection import (  # noqa: E402
    advect_cloud_mask_optical_flow,
    compute_forecast_skill,
    persistence_baseline_cover,
)
from app.vision.coverage import compute_cloud_coverage  # noqa: E402
from app.vision.motion import estimate_cloud_motion  # noqa: E402
from app.vision.segmentation import run_segmentation  # noqa: E402


def _binary_iou(pred_prob: np.ndarray, true_mask: np.ndarray, thresh: float = 0.5) -> float:
    pred_bin = pred_prob >= thresh
    true_resized = cv2.resize(
        ((true_mask >= 1) & (true_mask <= 3)).astype(np.uint8),
        (pred_prob.shape[1], pred_prob.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    inter = int(np.logical_and(pred_bin, true_resized).sum())
    union = int(np.logical_or(pred_bin, true_resized).sum())
    return 1.0 if union == 0 else float(inter / union)


def _build_panoramic_sky_canvas(w: int, h: int) -> np.ndarray:
    """Create a 3x-wide panoramic sky: clear sky on left transitioning smoothly into a dense cloud front on right."""
    clear_img, _, _, _, _ = _render_synthetic_sky_frame("clear_sky", t_step=0, seed=11, width=w, height=h)
    cum_img, _, _, _, _ = _render_synthetic_sky_frame("cumulus", t_step=0, seed=22, width=w, height=h)
    ovc_img, _, _, _, _ = _render_synthetic_sky_frame("stratus", t_step=0, seed=33, width=w, height=h)

    pano = np.zeros((h, w * 3, 3), dtype=np.float32)
    pano[:, 0:w] = clear_img.astype(np.float32)
    pano[:, w : 2 * w] = cum_img.astype(np.float32)
    pano[:, 2 * w : 3 * w] = ovc_img.astype(np.float32)

    # Smooth horizontal feathering across tile seams so optical flow tracks real cloud edges without seam artifacts
    blend = 48
    for seam in (w, 2 * w):
        alpha = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
        left_patch = pano[:, seam - blend : seam].copy()
        right_patch = pano[:, seam : seam + blend].copy()
        pano[:, seam - blend // 2 : seam + blend // 2] = (
            left_patch * (1.0 - alpha) + right_patch * alpha
        )
    return np.clip(pano, 0, 255).astype(np.uint8)


def evaluate_bundled_sequence(horizons_steps: list[int] | None = None) -> dict[str, Any]:
    """
    Generate a deterministic 14-frame panoramic drifting cloud-front sequence (2.5-min steps -> +5/+10/+15/+20m)
    and measure Cover MAE, Skill vs Persistence, and pixel-level Advected Mask IoU.
    """
    horizons_steps = horizons_steps or [2, 4, 6, 8]  # 5m, 10m, 15m, 20m at 2.5-min steps
    step_minutes = 2.5
    n_frames = 14
    w, h = 256, 192
    valid = np.full((h, w), 255, dtype=np.uint8)
    pano = _build_panoramic_sky_canvas(w, h)

    # Pan camera window smoothly from right to left across panorama (clouds drift right-to-left across FOV)
    max_x0 = pano.shape[1] - w
    step_px = max(12, max_x0 // (n_frames + 1))

    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    covers: list[float] = []
    motions: list[tuple[float, float, float]] = []

    for t in range(n_frames):
        x0 = min(max_x0, t * step_px)
        img = pano[:, x0 : x0 + w].copy()
        seg = run_segmentation(img, valid, inference_resolution=256)
        cov = compute_cloud_coverage(seg.class_map, valid, seg.mean_confidence)["cloud_coverage_pct"]
        if t == 0:
            motions.append((0.0, 0.0, 0.0))
        else:
            c_bin = ((seg.class_map >= 1) & (seg.class_map <= 3)).astype(np.uint8) * 255
            mot, _ = estimate_cloud_motion(
                frames[-1],
                img,
                valid,
                c_bin,
                dt_seconds=step_minutes * 60.0,
                tilt_correction_enabled=False,
            )
            motions.append((
                float(mot.get("dx_px_s", 0.0)),
                float(mot.get("dy_px_s", 0.0)),
                float(mot.get("consistency", 0.5)),
            ))
        frames.append(img)
        masks.append(seg.class_map)
        covers.append(float(cov))

    results: dict[str, Any] = {
        "dataset": "bundled_panoramic_cloud_front_14f",
        "step_minutes": step_minutes,
        "horizons": {},
    }
    for k in horizons_steps:
        h_min = int(round(k * step_minutes))
        pers_errs: list[float] = []
        of_errs: list[float] = []
        hybrid_errs: list[float] = []
        pers_ious: list[float] = []
        of_ious: list[float] = []
        fov_exits: list[float] = []

        for t in range(1, n_frames - k):
            actual = covers[t + k]
            curr = covers[t]
            rate_per_min = (covers[t] - covers[t - 1]) / step_minutes
            dx, dy, cons = motions[t]

            pred_pers = persistence_baseline_cover(curr)
            adv = advect_cloud_mask_optical_flow(
                seg_mask=masks[t],
                dx_px_s=dx,
                dy_px_s=dy,
                horizon_minutes=float(h_min),
                orig_width=w,
                orig_height=h,
                valid_mask=valid,
                resolution=96,
                ensemble_members=5,
                motion_consistency=max(0.4, cons),
            )
            pred_of = float(adv["advected_cover_pct"])
            damp = {5: 4.5, 10: 8.0, 15: 10.5, 20: 13.0}.get(h_min, h_min * 0.5)
            pred_rule = float(np.clip(curr + rate_per_min * damp, 0.0, 100.0))
            pred_hybrid = float(np.clip(0.55 * pred_rule + 0.45 * pred_of, 0.0, 100.0))

            pers_errs.append(abs(pred_pers - actual))
            of_errs.append(abs(pred_of - actual))
            hybrid_errs.append(abs(pred_hybrid - actual))
            fov_exits.append(float(adv["fov_exit_fraction"]))

            prob_map = adv["cloud_probability_map"]
            pers_prob = cv2.resize(
                ((masks[t] >= 1) & (masks[t] <= 3)).astype(np.float32),
                (prob_map.shape[1], prob_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            pers_ious.append(_binary_iou(pers_prob, masks[t + k]))
            of_ious.append(_binary_iou(prob_map, masks[t + k]))

        p_mae = round(float(np.mean(pers_errs)), 2)
        o_mae = round(float(np.mean(of_errs)), 2)
        h_mae = round(float(np.mean(hybrid_errs)), 2)
        results["horizons"][str(h_min)] = {
            "samples": len(pers_errs),
            "persistence_mae_pct": p_mae,
            "optical_flow_mae_pct": o_mae,
            "optical_flow_skill": compute_forecast_skill(o_mae, p_mae),
            "hybrid_mae_pct": h_mae,
            "hybrid_skill": compute_forecast_skill(h_mae, p_mae),
            "persistence_mask_iou": round(float(np.mean(pers_ious)), 3),
            "optical_flow_mask_iou": round(float(np.mean(of_ious)), 3),
            "mean_fov_exit_fraction": round(float(np.mean(fov_exits)), 3),
        }
    return results


def evaluate_from_local_db() -> dict[str, Any]:
    """Summarize real chronologically verified forecasts stored in the local SQLite database."""
    from app.database import get_db_session, init_database
    from app.forecast.verification import compute_verification_summary

    init_database()
    with get_db_session() as db:
        summary = compute_verification_summary(db)
    return {"dataset": "local_sqlite_verified_history", "verification": summary}


def format_markdown_table(report: dict[str, Any]) -> str:
    header = (
        "| Horizon | n | Pers MAE (%) | OF MAE (%) | OF Skill | Hybrid MAE (%) | "
        "Hybrid Skill | Pers IoU | OF IoU | FOV Exit |"
    )
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    lines = [f"### Nowcast Benchmark — `{report.get('dataset')}`", "", header, sep]
    for h_key, row in report.get("horizons", {}).items():
        lines.append(
            f"| +{h_key} min | {row['samples']} | {row['persistence_mae_pct']:.2f} | "
            f"{row['optical_flow_mae_pct']:.2f} | {row['optical_flow_skill']} | "
            f"{row['hybrid_mae_pct']:.2f} | {row['hybrid_skill']} | "
            f"{row['persistence_mask_iou']:.3f} | {row['optical_flow_mask_iou']:.3f} | "
            f"{row['mean_fov_exit_fraction']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Sky Observer nowcasting baselines vs persistence.")
    parser.add_argument(
        "--from-db", action="store_true", help="Report measured skill from local SQLite verifications."
    )
    parser.add_argument("--json-out", type=str, default="", help="Optional path to save JSON benchmark output.")
    args = parser.parse_args()

    report = evaluate_from_local_db() if args.from_db else evaluate_bundled_sequence()
    if not args.from_db:
        print(format_markdown_table(report))
    else:
        print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
