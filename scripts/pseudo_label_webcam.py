"""High-confidence & temporally stable Stage-1 pseudo-labeling script (Phase 4 domain adaptation).

Uses `radiometric-physics-v1` (never unvalidated drifting checkpoints by default), filters for BOTH
high pixel confidence AND multi-frame temporal stability across consecutive frames, strictly excludes
any frame locked inside `data/datasets/golden_eval/manifest.json`, and supports 3-class export
(`clear_sky=0`, `cloud=1`, `thick_cloud=2`, `excluded=255`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.ai.datasets import import_local_dataset  # noqa: E402
from app.config import get_config_manager  # noqa: E402
from app.database import get_db_session, init_database  # noqa: E402
from app.vision.masking import build_valid_sky_mask_from_config  # noqa: E402
from app.vision.segmentation import (  # noqa: E402
    NOT_SKY_IGNORE_INDEX,
    reduce_segmentation_to_3_classes,
    run_segmentation,
)


def compute_temporal_mask_stability(prev_mask: np.ndarray, curr_mask: np.ndarray) -> float:
    """Compute binary cloud-mask IoU agreement between two consecutive frames."""
    p_bin = (prev_mask >= 1) & (prev_mask <= 3)
    c_bin = (curr_mask >= 1) & (curr_mask <= 3)
    if prev_mask.shape != curr_mask.shape:
        p_bin = cv2.resize(p_bin.astype(np.uint8), (curr_mask.shape[1], curr_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    inter = int(np.logical_and(p_bin, c_bin).sum())
    union = int(np.logical_or(p_bin, c_bin).sum())
    return 1.0 if union == 0 else float(inter / union)


def _load_golden_filenames(datasets_dir: Path) -> set[str]:
    manifest_p = datasets_dir / "golden_eval" / "manifest.json"
    if not manifest_p.exists():
        return set()
    try:
        raw = json.loads(manifest_p.read_text(encoding="utf-8"))
        return {
            Path(str(it["source_raw_path"])).name
            for it in raw.get("items", []) or []
            if it.get("source_raw_path")
        }
    except Exception:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-confidence & temporally stable Stage-1 pseudo-labels from real webcam frames."
    )
    parser.add_argument("--input-dir", default=None, help="Folder with raw webcam images (defaults to data/images/raw).")
    parser.add_argument("--output-dir", default=None, help="Output folder (defaults to data/datasets/webcam_pseudolabels).")
    parser.add_argument("--min-conf", type=float, default=0.78, help="Minimum mean pixel confidence [0..1] to keep.")
    parser.add_argument("--min-stability", type=float, default=0.80, help="Minimum consecutive-frame mask IoU [0..1].")
    parser.add_argument("--classes-3", action="store_true", help="Export reduced 3-class masks (clear/cloud/thick_cloud).")
    parser.add_argument("--max-frames", type=int, default=400, help="Maximum frames to scan.")
    parser.add_argument("--register", action="store_true", help="Also register exported pseudo-labels in SQLite DB.")
    args = parser.parse_args()

    cfg = get_config_manager().get()
    cfg.ensure_directories()
    init_database(cfg.db_path)

    in_dir = Path(args.input_dir) if args.input_dir else cfg.raw_images_dir
    out_dir = Path(args.output_dir) if args.output_dir else (cfg.datasets_dir / "webcam_pseudolabels")
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    golden_names = _load_golden_filenames(cfg.datasets_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(in_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts][: args.max_frames]

    kept = 0
    skipped_golden = 0
    skipped_unstable = 0
    prev_class_map: np.ndarray | None = None

    for idx, fpath in enumerate(files):
        if fpath.name in golden_names:
            skipped_golden += 1
            continue
        bgr = cv2.imread(str(fpath))
        if bgr is None or float(bgr.mean()) < 10.0:
            continue
        valid_mask = build_valid_sky_mask_from_config(bgr.shape[:2], cfg.camera)
        # Always use calibrated deterministic radiometric-physics-v1 as teacher (prevents self-training drift)
        seg = run_segmentation(
            image_bgr=bgr,
            valid_mask=valid_mask,
            inference_resolution=min(512, cfg.vision.inference_resolution),
            seg_model=None,
            device="cpu",
            model_name="radiometric-physics-v1",
        )
        curr_map = seg.class_map
        stability = compute_temporal_mask_stability(prev_class_map, curr_map) if prev_class_map is not None else 1.0
        prev_class_map = curr_map

        if seg.mean_confidence < args.min_conf:
            continue
        if idx > 0 and stability < args.min_stability:
            skipped_unstable += 1
            continue

        if args.classes_3:
            pseudo_mask, _ = reduce_segmentation_to_3_classes(curr_map, valid_mask)
        else:
            pseudo_mask = curr_map.copy()
            pseudo_mask[valid_mask == 0] = NOT_SKY_IGNORE_INDEX
        pseudo_mask[seg.confidence_map < args.min_conf] = NOT_SKY_IGNORE_INDEX

        stem = f"webcam_{idx:05d}_{fpath.stem}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(msk_out / f"{stem}.png"), pseudo_mask)
        kept += 1

    print(
        f"Pseudo-labeling complete: scanned={len(files)}, kept={kept}, "
        f"skipped_golden={skipped_golden}, skipped_unstable={skipped_unstable} -> {out_dir}"
    )
    if args.register and kept > 0:
        with get_db_session() as db:
            rec = import_local_dataset(
                db=db,
                config=cfg,
                name="webcam_pseudolabels_radiometric",
                source_dir=out_dir,
                description=(
                    f"High-confidence (>={args.min_conf}) & temporally stable (>={args.min_stability}) "
                    "radiometric-physics-v1 pseudo-labels (golden set excluded)."
                ),
            )
            print(f"Registered DB dataset '{rec.name}' (id={rec.id}, items={rec.item_count})")


if __name__ == "__main__":
    main()
