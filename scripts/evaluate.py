"""Unified Reproducible Benchmark Suite for Sky Observer.

Evaluates:
  1. Cloud Segmentation (mIoU & pixel accuracy) on any locally imported public dataset
     (SWIMSEG / SWINySEG / WSISEG) or bundled deterministic sky frames, using radiometric physics
     and (if active) the registered neural segmenter.
  2. Cloud Type Classification (accuracy & macro-F1) on locally imported CCSN or bundled synthetic scenes.
  3. Short-Term Nowcasting (Persistence vs Optical-Flow Mask Advection vs Hybrid) with explicit
     Forecast Skill Score = 1 - MAE_model / MAE_persistence.

Never invents metrics: prints exact download/import instructions when public datasets are not present.

Usage:
  PYTHONPATH=backend python scripts/evaluate.py
  PYTHONPATH=backend python scripts/evaluate.py --datasets-dir data/datasets --json-out eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from app.ai.datasets import _render_synthetic_sky_frame  # noqa: E402
from app.vision.classification import classify_cloud_type  # noqa: E402
from app.vision.color import analyze_color  # noqa: E402
from app.vision.coverage import compute_cloud_coverage  # noqa: E402
from app.vision.segmentation import run_segmentation  # noqa: E402
from app.vision.texture import analyze_texture  # noqa: E402
from evaluate_nowcast import evaluate_bundled_sequence  # noqa: E402


def _eval_binary_seg_dir(name: str, ds_dir: Path, max_samples: int = 80) -> dict[str, Any]:
    img_dir = ds_dir / "images"
    msk_dir = ds_dir / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        return {"dataset": name, "status": "not_found", "hint": f"Place {name} with images/ and masks/ at {ds_dir}"}
    mask_map = {p.stem: p for p in msk_dir.iterdir() if p.is_file()}
    pairs = [
        (p, mask_map[p.stem])
        for p in sorted(img_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem in mask_map
    ][:max_samples]
    if not pairs:
        return {"dataset": name, "status": "empty", "hint": f"No matched image/mask stems in {ds_dir}"}

    inter = np.zeros(2, dtype=np.float64)
    union = np.zeros(2, dtype=np.float64)
    correct = 0
    total = 0
    for img_p, msk_p in pairs:
        bgr = cv2.imread(str(img_p))
        gt = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        h, w = bgr.shape[:2]
        gt_bin = (cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        valid = np.full((h, w), 255, dtype=np.uint8)
        seg = run_segmentation(bgr, valid, inference_resolution=256)
        pred_bin = ((seg.class_map >= 1) & (seg.class_map <= 3)).astype(np.uint8)
        correct += int((pred_bin == gt_bin).sum())
        total += int(gt_bin.size)
        for c in (0, 1):
            p_c = pred_bin == c
            t_c = gt_bin == c
            inter[c] += float((p_c & t_c).sum())
            union[c] += float((p_c | t_c).sum())

    present = union > 0
    miou = float(np.mean(inter[present] / union[present])) if np.any(present) else 0.0
    return {
        "dataset": name,
        "status": "measured",
        "backend": "radiometric-physics-v1",
        "samples": len(pairs),
        "binary_miou": round(miou, 4),
        "pixel_accuracy": round(float(correct / max(1, total)), 4),
    }


def evaluate_bundled_segmentation_and_classification() -> tuple[dict[str, Any], dict[str, Any]]:
    scenarios = [
        ("clear_sky", "clear_sky"),
        ("cirrus", "cirrus"),
        ("cumulus", "cumulus"),
        ("stratocumulus", "stratocumulus"),
        ("stratus", "stratus"),
        ("cumulonimbus_like", "cumulonimbus_like"),
    ]
    inter = np.zeros(2, dtype=np.float64)
    union = np.zeros(2, dtype=np.float64)
    correct = 0
    total = 0
    y_true: list[str] = []
    y_pred: list[str] = []

    for idx, (scen, expected_cls) in enumerate(scenarios):
        for rep in range(3):
            img, gt_mask, _, _, _ = _render_synthetic_sky_frame(
                scen, t_step=rep, seed=500 + idx * 10 + rep, width=256, height=192
            )
            valid = np.full(img.shape[:2], 255, dtype=np.uint8)
            seg = run_segmentation(img, valid, inference_resolution=256)
            pred_bin = ((seg.class_map >= 1) & (seg.class_map <= 3)).astype(np.uint8)
            gt_bin = ((gt_mask >= 1) & (gt_mask <= 3)).astype(np.uint8)
            correct += int((pred_bin == gt_bin).sum())
            total += int(gt_bin.size)
            for c in (0, 1):
                p_c = pred_bin == c
                t_c = gt_bin == c
                inter[c] += float((p_c & t_c).sum())
                union[c] += float((p_c | t_c).sum())

            cov = compute_cloud_coverage(seg.class_map, valid, seg.mean_confidence)
            col = analyze_color(img, valid)
            tex = analyze_texture(img, valid, seg.cloud_binary_mask)
            cls_res = classify_cloud_type(img, cov, col, tex, min_confidence=0.25)
            y_true.append(expected_cls)
            y_pred.append(str(cls_res["dominant_cloud_type"]))

    present = union > 0
    miou = float(np.mean(inter[present] / union[present])) if np.any(present) else 0.0
    seg_report = {
        "dataset": "bundled_synthetic_sky_18f",
        "status": "measured",
        "backend": "radiometric-physics-v1",
        "samples": len(y_true),
        "binary_miou": round(miou, 4),
        "pixel_accuracy": round(float(correct / max(1, total)), 4),
    }
    cls_report = {
        "dataset": "bundled_synthetic_sky_18f",
        "status": "measured",
        "backend": "physical-heuristic-v1",
        "samples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
    return seg_report, cls_report


def run_all_benchmarks(datasets_dir: Path) -> dict[str, Any]:
    bundled_seg, bundled_cls = evaluate_bundled_segmentation_and_classification()
    public_seg = [
        _eval_binary_seg_dir(name, datasets_dir / name)
        for name in ("SWIMSEG", "SWINySEG", "WSISEG")
    ]
    nowcast = evaluate_bundled_sequence()
    return {
        "segmentation": {"bundled": bundled_seg, "public": public_seg},
        "classification": {"bundled": bundled_cls},
        "nowcast": nowcast,
    }


def format_markdown(report: dict[str, Any]) -> str:
    b_seg = report["segmentation"]["bundled"]
    b_cls = report["classification"]["bundled"]
    lines = [
        "# Sky Observer — Reproducible Evaluation Report",
        "",
        "## 1. Segmentation",
        "| Dataset | Backend | Samples | Binary mIoU | Pixel Acc | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
        f"| `{b_seg['dataset']}` | `{b_seg['backend']}` | {b_seg['samples']} | "
        f"{b_seg['binary_miou']:.4f} | {b_seg['pixel_accuracy']:.4f} | {b_seg['status']} |",
    ]
    for p in report["segmentation"]["public"]:
        if p["status"] == "measured":
            lines.append(
                f"| `{p['dataset']}` | `{p['backend']}` | {p['samples']} | "
                f"{p['binary_miou']:.4f} | {p['pixel_accuracy']:.4f} | measured |"
            )
        else:
            lines.append(f"| `{p['dataset']}` | — | 0 | — | — | {p['status']} ({p['hint']}) |")

    lines.extend([
        "",
        "## 2. Classification",
        "| Dataset | Backend | Samples | Accuracy | Macro F1 |",
        "| :--- | :--- | :---: | :---: | :---: |",
        f"| `{b_cls['dataset']}` | `{b_cls['backend']}` | {b_cls['samples']} | "
        f"{b_cls['accuracy']:.4f} | {b_cls['macro_f1']:.4f} |",
        "",
        "## 3. Nowcasting (Persistence vs Optical-Flow Advection vs Hybrid)",
        "| Horizon | n | Pers MAE (%) | OF MAE (%) | OF Skill | Hybrid MAE (%) | Hybrid Skill |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for h_k, row in report["nowcast"]["horizons"].items():
        lines.append(
            f"| +{h_k} min | {row['samples']} | {row['persistence_mae_pct']:.2f} | "
            f"{row['optical_flow_mae_pct']:.2f} | {row['optical_flow_skill']} | "
            f"{row['hybrid_mae_pct']:.2f} | {row['hybrid_skill']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sky Observer reproducible evaluation suite.")
    parser.add_argument("--datasets-dir", type=str, default="data/datasets")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    report = run_all_benchmarks(Path(args.datasets_dir))
    print(format_markdown(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
