"""Stratified Golden Evaluation Set Builder (150-300 local camera frames, permanently excluded from training).

Selects usable observations from SQLite (stratified across time-of-day: morning / midday / sunset-twilight,
cloud cover: clear / partial / overcast, sun-near-cloud glare, and haze/low-contrast), copies raw frames
to data/datasets/golden_eval/, writes a `.golden_locked` sentinel + `manifest.json` (never used in training),
and generates `label_studio_tasks.json` ready for SAM-assisted annotation in Label Studio or CVAT.

Usage:
  PYTHONPATH=backend python scripts/build_golden_set.py --target-size 200 --out-dir data/datasets/golden_eval
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_config_manager  # noqa: E402
from app.database import get_db_session, init_database  # noqa: E402
from app.models_db import Observation, safe_json_loads  # noqa: E402

GOLDEN_LOCK_FILENAME = ".golden_locked"
GOLDEN_MANIFEST_FILENAME = "manifest.json"


def _stratify_observation(obs: Observation) -> dict[str, str]:
    ts = obs.captured_at
    hour = ts.hour if ts is not None else 12
    if 5 <= hour < 10:
        time_bin = "morning"
    elif 10 <= hour < 16:
        time_bin = "midday"
    elif 16 <= hour < 21:
        time_bin = "sunset_twilight"
    else:
        time_bin = "night"

    cov = float(obs.cloud_coverage_pct or 0.0)
    if cov < 15.0:
        cover_bin = "clear"
    elif cov <= 75.0:
        cover_bin = "partial"
    else:
        cover_bin = "overcast"

    q_meta = safe_json_loads(obs.quality_metrics_json, {})
    cov_meta = safe_json_loads(obs.coverage_metrics_json, {})
    flags = set(q_meta.get("flags", []) or [])
    sun_glare = float(cov_meta.get("sun_glare_pct", getattr(obs, "sun_glare_pct", 0.0)) or 0.0)
    contrast_val = float(q_meta.get("contrast_score", getattr(obs, "contrast_score", 100.0)) or 100.0)
    special = "standard"
    if sun_glare >= 3.0 or "sun_glare" in flags:
        special = "sun_near_cloud"
    elif contrast_val < 18.0 or "haze" in flags:
        special = "haze_low_contrast"

    day_key = ts.strftime("%Y-%m-%d") if ts is not None else "unknown_day"
    return {
        "time_bin": time_bin,
        "cover_bin": cover_bin,
        "special": special,
        "day_key": day_key,
        "stratum": f"{time_bin}|{cover_bin}|{special}",
    }


def build_golden_set(
    out_dir: Path,
    target_size: int = 200,
    copy_images: bool = True,
) -> dict[str, Any]:
    cfg = get_config_manager().get()
    init_database(cfg.db_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    msk_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    with get_db_session() as db:
        rows = (
            db.query(Observation)
            .filter(Observation.is_usable.is_(True), Observation.raw_path != "")
            .order_by(Observation.captured_at.asc(), Observation.id.asc())
            .all()
        )
        candidates: list[tuple[Observation, dict[str, str]]] = []
        for obs in rows:
            rp = Path(obs.raw_path)
            if rp.exists() and rp.is_file():
                candidates.append((obs, _stratify_observation(obs)))

    # Round-robin stratified sampling across strata while spacing frames from the same day
    buckets: dict[str, list[tuple[Observation, dict[str, str]]]] = defaultdict(list)
    for item in candidates:
        buckets[item[1]["stratum"]].append(item)

    selected: list[tuple[Observation, dict[str, str]]] = []
    stratum_keys = sorted(buckets.keys())
    while len(selected) < target_size and any(buckets[k] for k in stratum_keys):
        for k in stratum_keys:
            if len(selected) >= target_size:
                break
            if buckets[k]:
                # Pick evenly spaced elements from the stratum bucket
                bucket = buckets[k]
                idx = len(bucket) // 2 if len(bucket) > 2 else 0
                selected.append(bucket.pop(idx))

    manifest_items: list[dict[str, Any]] = []
    label_studio_tasks: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = defaultdict(int)
    day_counts: dict[str, int] = defaultdict(int)

    for obs, meta in selected:
        src = Path(obs.raw_path)
        dst_name = f"golden_obs_{obs.id:06d}_{meta['day_key']}{src.suffix.lower() or '.jpg'}"
        dst_path = img_dir / dst_name
        if copy_images and not dst_path.exists():
            shutil.copy2(src, dst_path)

        stratum_counts[meta["stratum"]] += 1
        day_counts[meta["day_key"]] += 1
        entry = {
            "observation_id": int(obs.id),
            "captured_at": obs.captured_at.isoformat() if obs.captured_at else None,
            "image_file": f"images/{dst_name}",
            "mask_file": f"masks/{Path(dst_name).stem}.png",
            "source_raw_path": str(src),
            "cloud_coverage_pct_auto": round(float(obs.cloud_coverage_pct or 0.0), 2),
            "dominant_cloud_type_auto": str(obs.dominant_cloud_type or "unknown"),
            "is_golden": True,
            **meta,
        }
        manifest_items.append(entry)
        label_studio_tasks.append({
            "data": {
                "image": f"/data/local-files/?d=golden_eval/images/{dst_name}",
                "observation_id": int(obs.id),
                "captured_at": entry["captured_at"],
                "stratum": meta["stratum"],
            },
            "meta": {
                "golden_set": True,
                "target_classes": ["clear_sky", "cloud", "thick_cloud", "excluded"],
            },
        })

    # Write permanent golden lock sentinel so training pipelines never ingest this folder
    (out_dir / GOLDEN_LOCK_FILENAME).write_text(
        "DO NOT USE FOR TRAINING — Golden Evaluation Set locked by Sky Observer.\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1.0",
        "golden_locked": True,
        "created_at": datetime.now(UTC).isoformat(),
        "target_size": target_size,
        "selected_count": len(manifest_items),
        "recommended_range": "150-300 stratified frames",
        "segmentation_classes_3": {
            "0": "clear_sky",
            "1": "cloud (thin/scattered)",
            "2": "thick_cloud (dense/dark)",
            "255": "excluded (obstacles/glare ignore)",
        },
        "annotation_tools": ["Label Studio + SAM", "CVAT + SAM"],
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "day_counts": dict(sorted(day_counts.items())),
        "items": manifest_items,
    }
    (out_dir / GOLDEN_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "label_studio_tasks.json").write_text(json.dumps(label_studio_tasks, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified Golden Evaluation Set (excluded from training).")
    parser.add_argument("--out-dir", type=str, default="data/datasets/golden_eval")
    parser.add_argument("--target-size", type=int, default=200)
    args = parser.parse_args()
    res = build_golden_set(Path(args.out_dir), target_size=max(10, int(args.target_size)))
    print(
        f"Golden set prepared at {args.out_dir}: {res['selected_count']} frames across "
        f"{len(res['stratum_counts'])} strata and {len(res['day_counts'])} days."
    )


if __name__ == "__main__":
    main()
