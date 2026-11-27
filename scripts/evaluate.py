"""Unified Reproducible Benchmark Suite for Sky Observer (Whole-Day Splits, Full Baselines & Stratified Metrics).

Evaluates:
  1. Cloud Segmentation:
     - Baselines: Fixed R/B threshold & Otsu R/B threshold
     - Model: `radiometric-physics-v1`
     - Metrics: per-class IoU (sky, cloud, binary mIoU), pixel accuracy, cloud cover MAE (%),
       stratified by condition (`clear`, `partial`, `overcast`), plus any locally imported public
       datasets (`SWIMSEG`, `SWINySEG`, `WSISEG`) or locked Golden Set (`golden_eval`).
  2. Cloud Type Classification (strictly whole-day chronological split — zero same-day leakage):
     - Baselines: Majority class & kNN on the 12 radiometric/texture features
     - Model: `physical-heuristic-v1`
     - Metrics: Accuracy & Macro-F1.
  3. Short-Term Nowcasting:
     - Baselines: Persistence, Linear Trend, Optical-Flow Mask Advection, Hybrid
     - Metrics: MAE per horizon & Forecast Skill Score = 1 - MAE_model / MAE_persistence.
  4. Event Detection:
     - Objective future-window ground truth (`rapid_increase`, `clearing`, `darkening`)
     - Metrics: Precision, Recall, and False Alarms per Day.

Usage:
  PYTHONPATH=backend python scripts/evaluate.py
  PYTHONPATH=backend python scripts/evaluate.py --json-out eval_report.json --md-out eval_report.md
  PYTHONPATH=backend python scripts/evaluate.py --save-to-registry --min-seg-miou 0.80 --min-nowcast-skill 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from app.ai.datasets import _render_synthetic_sky_frame  # noqa: E402
from app.config import AppConfig  # noqa: E402
from app.models_db import Observation  # noqa: E402
from app.services.events import SkyEventEngine, evaluate_objective_future_events  # noqa: E402
from app.vision.classification import classify_cloud_type  # noqa: E402
from app.vision.color import analyze_color  # noqa: E402
from app.vision.coverage import compute_cloud_coverage  # noqa: E402
from app.vision.segmentation import run_segmentation  # noqa: E402
from app.vision.texture import analyze_texture  # noqa: E402
from evaluate_nowcast import evaluate_bundled_sequence  # noqa: E402


def skill_score(y_true: np.ndarray, y_model: np.ndarray, y_persistence: np.ndarray) -> float:
    """Compute Forecast Skill Score = 1 - MAE_model / MAE_persistence (> 0 beats persistence)."""
    mae_model = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_model, dtype=float))))
    mae_pers = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_persistence, dtype=float))))
    if mae_pers <= 1e-6:
        return 0.0
    return round(1.0 - (mae_model / mae_pers), 4)


def split_items_by_day(
    items: list[dict[str, Any]],
    day_key: str = "day",
    test_fraction: float = 0.34,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split records chronologically by WHOLE calendar days (never frame-level random split),
    guaranteeing zero same-day temporal leakage between train and test.
    """
    unique_days = sorted({str(it[day_key]) for it in items})
    if len(unique_days) < 2:
        return items, items
    n_test_days = max(1, int(round(len(unique_days) * test_fraction)))
    n_test_days = min(len(unique_days) - 1, n_test_days)
    test_days = set(unique_days[-n_test_days:])
    train_items = [it for it in items if str(it[day_key]) not in test_days]
    test_items = [it for it in items if str(it[day_key]) in test_days]
    return train_items, test_items


def _rb_fixed_mask(image_bgr: np.ndarray, threshold: float = 0.78) -> np.ndarray:
    """Baseline 1: Fixed Red/Blue ratio threshold (R / (B + eps) > threshold -> cloud)."""
    b = image_bgr[:, :, 0].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    rb = r / (b + 1.0)
    return (rb >= threshold).astype(np.uint8)


def _rb_otsu_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Baseline 2: Adaptive Otsu threshold on normalized Red/Blue ratio."""
    b = image_bgr[:, :, 0].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    rb = np.clip((r / (b + 1.0)) * 128.0, 0.0, 255.0).astype(np.uint8)
    thr, mask = cv2.threshold(rb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.std(rb)) < 6.0:
        return (rb >= int(0.78 * 128)).astype(np.uint8)
    _ = thr
    return (mask > 127).astype(np.uint8)


def _compute_seg_metrics(pred_bins: list[np.ndarray], gt_bins: list[np.ndarray]) -> dict[str, Any]:
    inter = np.zeros(2, dtype=np.float64)
    union = np.zeros(2, dtype=np.float64)
    correct = 0
    total = 0
    cov_errs: list[float] = []
    for pred, gt in zip(pred_bins, gt_bins, strict=True):
        correct += int((pred == gt).sum())
        total += int(gt.size)
        for c in (0, 1):
            p_c = pred == c
            t_c = gt == c
            inter[c] += float((p_c & t_c).sum())
            union[c] += float((p_c | t_c).sum())
        cov_errs.append(abs(float(np.mean(pred) * 100.0) - float(np.mean(gt) * 100.0)))
    sky_iou = float(inter[0] / union[0]) if union[0] > 0 else 1.0
    cloud_iou = float(inter[1] / union[1]) if union[1] > 0 else 1.0
    present = union > 0
    miou = float(np.mean(inter[present] / union[present])) if np.any(present) else 0.0
    return {
        "samples": len(gt_bins),
        "sky_iou": round(sky_iou, 4),
        "cloud_iou": round(cloud_iou, 4),
        "binary_miou": round(miou, 4),
        "pixel_accuracy": round(float(correct / max(1, total)), 4),
        "cover_mae_pct": round(float(np.mean(cov_errs)) if cov_errs else 0.0, 2),
    }


def _extract_12_features(cov: dict[str, Any], col: dict[str, Any], tex: dict[str, Any]) -> list[float]:
    return [
        float(cov.get("cloud_coverage_pct", 0.0)),
        float(cov.get("clear_sky_pct", 0.0)),
        float(cov.get("thin_cloud_pct", 0.0)),
        float(cov.get("dense_cloud_pct", 0.0)),
        float(cov.get("dark_cloud_pct", 0.0)),
        float(col.get("avg_luminance", 50.0)),
        float(col.get("avg_blue_red_ratio", 1.0)),
        float(col.get("sky_saturation_mean", 0.2)),
        float(col.get("red_blue_diff_mean", 0.0)),
        float(tex.get("edge_density", 0.0)),
        float(tex.get("texture_contrast", 0.0)),
        float(tex.get("fragmentation_index", 0.0)),
    ]


def _condition_bucket(gt_cover_pct: float) -> str:
    if gt_cover_pct < 15.0:
        return "clear"
    if gt_cover_pct <= 75.0:
        return "partial"
    return "overcast"


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

    gt_bins, phys_bins, fixed_bins, otsu_bins = [], [], [], []
    for img_p, msk_p in pairs:
        bgr = cv2.imread(str(img_p))
        gt = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        h, w = bgr.shape[:2]
        gt_bin = (cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        valid = np.full((h, w), 255, dtype=np.uint8)
        seg = run_segmentation(bgr, valid, inference_resolution=256)
        gt_bins.append(gt_bin)
        phys_bins.append(((seg.class_map >= 1) & (seg.class_map <= 3)).astype(np.uint8))
        fixed_bins.append(_rb_fixed_mask(bgr))
        otsu_bins.append(_rb_otsu_mask(bgr))

    return {
        "dataset": name,
        "status": "measured",
        "backend": "radiometric-physics-v1",
        **_compute_seg_metrics(phys_bins, gt_bins),
        "baselines": {
            "rb_fixed": _compute_seg_metrics(fixed_bins, gt_bins),
            "rb_otsu": _compute_seg_metrics(otsu_bins, gt_bins),
        },
    }


def evaluate_bundled_segmentation_and_classification() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluate 6 canonical sky conditions across 3 distinct synthetic calendar days (18 frames),
    using strict WHOLE-DAY chronological splits for supervised baselines (kNN on 12 features & majority class).
    """
    scenarios = [
        ("clear_sky", "clear_sky"),
        ("cirrus", "cirrus"),
        ("cumulus", "cumulus"),
        ("stratocumulus", "stratocumulus"),
        ("stratus", "stratus"),
        ("cumulonimbus_like", "cumulonimbus_like"),
    ]
    days = ["2026-05-01", "2026-05-02", "2026-05-03"]

    gt_bins: list[np.ndarray] = []
    phys_bins: list[np.ndarray] = []
    fixed_bins: list[np.ndarray] = []
    otsu_bins: list[np.ndarray] = []
    strat_gt: dict[str, list[np.ndarray]] = defaultdict(list)
    strat_phys: dict[str, list[np.ndarray]] = defaultdict(list)
    strat_fixed: dict[str, list[np.ndarray]] = defaultdict(list)
    cls_records: list[dict[str, Any]] = []

    for day_idx, day_str in enumerate(days):
        for idx, (scen, expected_cls) in enumerate(scenarios):
            img, gt_mask, _, _, _ = _render_synthetic_sky_frame(
                scen, t_step=day_idx, seed=500 + idx * 10 + day_idx, width=256, height=192
            )
            valid = np.full(img.shape[:2], 255, dtype=np.uint8)
            seg = run_segmentation(img, valid, inference_resolution=256)
            pred_phys = ((seg.class_map >= 1) & (seg.class_map <= 3)).astype(np.uint8)
            pred_fixed = _rb_fixed_mask(img)
            pred_otsu = _rb_otsu_mask(img)
            gt_bin = ((gt_mask >= 1) & (gt_mask <= 3)).astype(np.uint8)

            gt_bins.append(gt_bin)
            phys_bins.append(pred_phys)
            fixed_bins.append(pred_fixed)
            otsu_bins.append(pred_otsu)

            cond = _condition_bucket(float(np.mean(gt_bin) * 100.0))
            strat_gt[cond].append(gt_bin)
            strat_phys[cond].append(pred_phys)
            strat_fixed[cond].append(pred_fixed)

            cov = compute_cloud_coverage(seg.class_map, valid, seg.mean_confidence)
            col = analyze_color(img, valid)
            tex = analyze_texture(img, valid, seg.cloud_binary_mask)
            cls_res = classify_cloud_type(img, cov, col, tex, min_confidence=0.25)
            cls_records.append({
                "day": day_str,
                "condition": cond,
                "y_true": expected_cls,
                "y_phys": str(cls_res["dominant_cloud_type"]),
                "features": _extract_12_features(cov, col, tex),
            })

    # Whole-day split for classification baselines (Day 1-2 train, Day 3 test -> zero same-day leakage)
    train_cls, test_cls = split_items_by_day(cls_records, day_key="day", test_fraction=0.34)
    y_train = [r["y_true"] for r in train_cls]
    y_test = [r["y_true"] for r in test_cls]
    y_phys_test = [r["y_phys"] for r in test_cls]
    x_train = np.array([r["features"] for r in train_cls], dtype=np.float32)
    x_test = np.array([r["features"] for r in test_cls], dtype=np.float32)

    majority_label = Counter(y_train).most_common(1)[0][0]
    y_maj = [majority_label] * len(y_test)

    knn = KNeighborsClassifier(n_neighbors=min(3, len(train_cls)))
    knn.fit(x_train, y_train)
    y_knn = list(knn.predict(x_test))

    y_all_true = [r["y_true"] for r in cls_records]
    y_all_phys = [r["y_phys"] for r in cls_records]

    seg_report = {
        "dataset": "bundled_synthetic_sky_18f",
        "status": "measured",
        "backend": "radiometric-physics-v1",
        **_compute_seg_metrics(phys_bins, gt_bins),
        "baselines": {
            "rb_fixed": _compute_seg_metrics(fixed_bins, gt_bins),
            "rb_otsu": _compute_seg_metrics(otsu_bins, gt_bins),
        },
        "by_condition": {
            cond: {
                "radiometric_physics": _compute_seg_metrics(strat_phys[cond], strat_gt[cond]),
                "rb_fixed": _compute_seg_metrics(strat_fixed[cond], strat_gt[cond]),
            }
            for cond in sorted(strat_gt.keys())
        },
    }
    cls_report = {
        "dataset": "bundled_synthetic_sky_18f",
        "status": "measured",
        "split_strategy": "whole_day_chronological",
        "train_days": sorted({r["day"] for r in train_cls}),
        "test_days": sorted({r["day"] for r in test_cls}),
        "backend": "physical-heuristic-v1",
        "samples": len(y_all_true),
        "accuracy": round(float(accuracy_score(y_all_true, y_all_phys)), 4),
        "macro_f1": round(float(f1_score(y_all_true, y_all_phys, average="macro", zero_division=0)), 4),
        "heldout_day_metrics": {
            "samples": len(y_test),
            "physical_heuristic": {
                "accuracy": round(float(accuracy_score(y_test, y_phys_test)), 4),
                "macro_f1": round(float(f1_score(y_test, y_phys_test, average="macro", zero_division=0)), 4),
            },
            "baseline_majority": {
                "accuracy": round(float(accuracy_score(y_test, y_maj)), 4),
                "macro_f1": round(float(f1_score(y_test, y_maj, average="macro", zero_division=0)), 4),
            },
            "baseline_knn_12f": {
                "accuracy": round(float(accuracy_score(y_test, y_knn)), 4),
                "macro_f1": round(float(f1_score(y_test, y_knn, average="macro", zero_division=0)), 4),
            },
        },
    }
    return seg_report, cls_report


def evaluate_event_rules_benchmark() -> dict[str, Any]:
    """
    Evaluate deterministic SkyEventEngine rules against objective future-window ground truth
    across a 2-day synthetic observation series (precision, recall, false alarms per day).
    """
    cfg = AppConfig()
    engine = SkyEventEngine()
    base = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    # 2 simulated daytime cycles (each 12 steps x 10 min = 2h active monitoring -> span = 2 days)
    profile = [
        (10.0, 80.0, 0.0),
        (12.0, 79.0, 0.0),
        (48.0, 62.0, 12.0),  # rapid increase + darkening
        (72.0, 50.0, 28.0),
        (65.0, 55.0, 20.0),
        (16.0, 77.0, 0.0),   # clearing start
        (12.0, 79.0, 0.0),
        (10.0, 80.0, 0.0),   # clearing confirmed (>=15m <20%)
        (15.0, 78.0, 0.0),
        (52.0, 58.0, 15.0),  # rapid increase + darkening
        (60.0, 54.0, 20.0),
        (14.0, 79.0, 0.0),
    ]
    obs_list: list[Observation] = []
    idx = 1
    for day_offset in (0, 1):
        t_day = base + timedelta(days=day_offset)
        for step, (cov, lum, dark) in enumerate(profile):
            obs_list.append(
                Observation(
                    id=idx,
                    captured_at=t_day + timedelta(minutes=step * 10),
                    source_type="demo_simulation",
                    is_usable=True,
                    cloud_coverage_pct=cov,
                    avg_luminance=lum,
                    dark_cloud_pct=dark,
                    dominant_cloud_type="cumulus" if cov > 30 else "clear_sky",
                )
            )
            idx += 1

    gt_rows = evaluate_objective_future_events(obs_list, cfg)
    days_span = 2.0
    event_map = {
        "rapid_increase": {"rapid_cloud_development", "cloud_cover_increasing"},
        "clearing": {"sky_clearing"},
        "darkening": {"dark_cloud_development", "storm_like_structure"},
    }
    stats: dict[str, dict[str, Any]] = {}
    for ev_key, pred_labels in event_map.items():
        tp = fp = fn = 0
        for i, obs in enumerate(obs_list):
            prev = obs_list[i - 1] if i > 0 else None
            pred = engine.classify(obs, prev, cfg)["event_class"] in pred_labels
            truth = bool(gt_rows[i][ev_key])
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
        prec = round(tp / max(1, tp + fp), 4)
        rec = round(tp / max(1, tp + fn), 4)
        stats[ev_key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "false_alarms_per_day": round(fp / days_span, 2),
        }
    return {"dataset": "bundled_2day_objective_events_24f", "days": days_span, "events": stats}


def save_report_to_registry(report: dict[str, Any]) -> int:
    """Persist structured benchmark summary into active ModelRegistry rows."""
    from app.database import get_db_session, init_database
    from app.models_db import ModelRegistry

    init_database()
    updated = 0
    with get_db_session() as db:
        for row in db.query(ModelRegistry).filter(ModelRegistry.is_active.is_(True)).all():
            existing = json.loads(row.metrics_json or "{}")
            existing["last_eval"] = {
                "seg_miou": report["segmentation"]["bundled"]["binary_miou"],
                "cls_acc": report["classification"]["bundled"]["accuracy"],
                "nowcast_5m_skill": report["nowcast"]["horizons"].get("5", {}).get("hybrid_skill"),
            }
            row.metrics_json = json.dumps(existing)
            updated += 1
        db.commit()
    return updated


def run_all_benchmarks(datasets_dir: Path) -> dict[str, Any]:
    bundled_seg, bundled_cls = evaluate_bundled_segmentation_and_classification()
    public_seg = [
        _eval_binary_seg_dir(name, datasets_dir / name)
        for name in ("golden_eval", "SWIMSEG", "SWINySEG", "WSISEG")
    ]
    nowcast = evaluate_bundled_sequence()
    events = evaluate_event_rules_benchmark()
    return {
        "segmentation": {"bundled": bundled_seg, "public": public_seg},
        "classification": {"bundled": bundled_cls},
        "nowcast": nowcast,
        "events": events,
    }


def format_markdown(report: dict[str, Any]) -> str:
    b_seg = report["segmentation"]["bundled"]
    b_rb = b_seg["baselines"]["rb_fixed"]
    b_ot = b_seg["baselines"]["rb_otsu"]
    b_cls = report["classification"]["bundled"]
    ho = b_cls["heldout_day_metrics"]
    lines = [
        "# Sky Observer — Reproducible Evaluation Report",
        "",
        "## 1. Segmentation (Model vs Fixed R/B & Otsu Baselines)",
        "| Dataset | Method | Samples | Sky IoU | Cloud IoU | Binary mIoU | Pixel Acc | Cover MAE (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| `{b_seg['dataset']}` | **`{b_seg['backend']}`** | {b_seg['samples']} | {b_seg['sky_iou']:.4f} | "
        f"{b_seg['cloud_iou']:.4f} | **{b_seg['binary_miou']:.4f}** | "
        f"{b_seg['pixel_accuracy']:.4f} | {b_seg['cover_mae_pct']:.2f} |",
        f"| `{b_seg['dataset']}` | `baseline:rb_fixed` | {b_rb['samples']} | {b_rb['sky_iou']:.4f} | "
        f"{b_rb['cloud_iou']:.4f} | {b_rb['binary_miou']:.4f} | "
        f"{b_rb['pixel_accuracy']:.4f} | {b_rb['cover_mae_pct']:.2f} |",
        f"| `{b_seg['dataset']}` | `baseline:rb_otsu` | {b_ot['samples']} | {b_ot['sky_iou']:.4f} | "
        f"{b_ot['cloud_iou']:.4f} | {b_ot['binary_miou']:.4f} | "
        f"{b_ot['pixel_accuracy']:.4f} | {b_ot['cover_mae_pct']:.2f} |",
        "",
        "### Segmentation Stratified by Sky Condition",
        "| Condition | n | Physics mIoU | Physics Cover MAE (%) | R/B Fixed mIoU |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for cond, vals in b_seg["by_condition"].items():
        rp = vals["radiometric_physics"]
        rf = vals["rb_fixed"]
        lines.append(
            f"| `{cond}` | {rp['samples']} | {rp['binary_miou']:.4f} | "
            f"{rp['cover_mae_pct']:.2f} | {rf['binary_miou']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Classification (Whole-Day Split: Model vs Majority & 12-Feature kNN)",
        f"- **Split strategy**: `{b_cls['split_strategy']}` "
        f"(train days: `{b_cls['train_days']}`, test days: `{b_cls['test_days']}`)",
        "",
        "| Method | Held-Out n | Accuracy | Macro F1 |",
        "| :--- | :---: | :---: | :---: |",
        f"| **`{b_cls['backend']}`** | {ho['samples']} | "
        f"**{ho['physical_heuristic']['accuracy']:.4f}** | **{ho['physical_heuristic']['macro_f1']:.4f}** |",
        f"| `baseline:knn_12_features` | {ho['samples']} | "
        f"{ho['baseline_knn_12f']['accuracy']:.4f} | {ho['baseline_knn_12f']['macro_f1']:.4f} |",
        f"| `baseline:majority_class` | {ho['samples']} | "
        f"{ho['baseline_majority']['accuracy']:.4f} | {ho['baseline_majority']['macro_f1']:.4f} |",
        "",
        "## 3. Nowcasting (Persistence vs Linear Trend vs Optical-Flow Advection vs Hybrid)",
        "| Horizon | n | Pers MAE | Lin MAE | Lin Skill | OF MAE | OF Skill | Hybrid MAE | Hybrid Skill |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for h_k, row in report["nowcast"]["horizons"].items():
        lines.append(
            f"| +{h_k} min | {row['samples']} | {row['persistence_mae_pct']:.2f} | "
            f"{row.get('linear_trend_mae_pct', 0.0):.2f} | {row.get('linear_trend_skill', 0.0)} | "
            f"{row['optical_flow_mae_pct']:.2f} | {row['optical_flow_skill']} | "
            f"{row['hybrid_mae_pct']:.2f} | {row['hybrid_skill']} |"
        )

    lines.extend([
        "",
        "## 4. Event Detection (Objective Future-Window Ground Truth)",
        "| Event | Precision | Recall | False Alarms / Day |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for ev_k, st in report["events"]["events"].items():
        lines.append(
            f"| `{ev_k}` | {st['precision']:.4f} | {st['recall']:.4f} | {st['false_alarms_per_day']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sky Observer reproducible evaluation suite.")
    parser.add_argument("--datasets-dir", type=str, default="data/datasets")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--md-out", type=str, default="")
    parser.add_argument("--save-to-registry", action="store_true")
    parser.add_argument(
        "--min-seg-miou", type=float, default=None, help="Fail (exit 2) if bundled seg mIoU drops below threshold."
    )
    parser.add_argument(
        "--min-nowcast-skill",
        type=float,
        default=None,
        help="Fail (exit 2) if +5m hybrid skill drops below threshold.",
    )
    args = parser.parse_args()

    report = run_all_benchmarks(Path(args.datasets_dir))
    md = format_markdown(report)
    print(md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if args.save_to_registry:
        n_upd = save_report_to_registry(report)
        print(f"\nSaved evaluation snapshot to {n_upd} active model_registry record(s).")

    seg_miou = float(report["segmentation"]["bundled"]["binary_miou"])
    skill_5m = float(report["nowcast"]["horizons"].get("5", {}).get("hybrid_skill") or 0.0)
    if args.min_seg_miou is not None and seg_miou < args.min_seg_miou:
        print(f"REGRESSION GATE FAILED: seg_miou={seg_miou:.4f} < {args.min_seg_miou:.4f}", file=sys.stderr)
        sys.exit(2)
    if args.min_nowcast_skill is not None and skill_5m < args.min_nowcast_skill:
        msg = f"REGRESSION GATE FAILED: nowcast_5m_skill={skill_5m:.4f} < {args.min_nowcast_skill:.4f}"
        print(msg, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
