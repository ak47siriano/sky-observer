"""High-confidence Stage-1 pseudo-labeling script for real local webcam frames (Phase 1B domain adaptation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.datasets import import_local_dataset
from app.ai.registry import get_model_cache
from app.config import get_config_manager
from app.database import get_db_session, init_database
from app.vision.masking import build_valid_sky_mask_from_config
from app.vision.segmentation import NOT_SKY_IGNORE_INDEX, run_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-confidence Stage-1 pseudo-labels from real webcam frames.")
    parser.add_argument("--input-dir", default=None, help="Folder with raw webcam images (defaults to data/images/raw).")
    parser.add_argument("--output-dir", default=None, help="Output folder (defaults to data/datasets/webcam_pseudolabels).")
    parser.add_argument("--min-conf", type=float, default=0.78, help="Minimum mean pixel confidence [0..1] to keep a frame.")
    parser.add_argument("--max-frames", type=int, default=400, help="Maximum frames to pseudo-label.")
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

    with get_db_session() as db:
        get_model_cache().refresh_from_db(db)
    seg_model, seg_meta, seg_dev = get_model_cache().get("segmentation")
    model_name = seg_meta["model_id"] if seg_meta else "radiometric-physics-v1"

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(in_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts][: args.max_frames]
    kept = 0
    for idx, fpath in enumerate(files):
        bgr = cv2.imread(str(fpath))
        if bgr is None or float(bgr.mean()) < 10.0:
            continue
        valid_mask = build_valid_sky_mask_from_config(bgr.shape[:2], cfg.camera)
        seg = run_segmentation(
            image_bgr=bgr,
            valid_mask=valid_mask,
            inference_resolution=min(512, cfg.vision.inference_resolution),
            seg_model=seg_model,
            device=seg_dev,
            model_name=model_name,
        )
        if seg.mean_confidence < args.min_conf:
            continue

        # Mark static Not_Sky obstacles AND low-confidence pixels as 255 (ignore_index)
        pseudo_mask = seg.class_map.copy()
        pseudo_mask[valid_mask == 0] = NOT_SKY_IGNORE_INDEX
        pseudo_mask[seg.confidence_map < args.min_conf] = NOT_SKY_IGNORE_INDEX

        stem = f"webcam_{idx:05d}_{fpath.stem}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(msk_out / f"{stem}.png"), pseudo_mask)
        kept += 1

    print(f"Pseudo-labeling complete: scanned={len(files)}, kept={kept} (min_conf={args.min_conf}) -> {out_dir}")
    if args.register and kept > 0:
        with get_db_session() as db:
            rec = import_local_dataset(
                db=db,
                config=cfg,
                name="webcam_pseudolabels_phase1b",
                source_dir=out_dir,
                description=f"High-confidence (>={args.min_conf}) Stage-1 pseudo-labels from local webcam.",
            )
            print(f"Registered DB dataset '{rec.name}' (id={rec.id}, items={rec.item_count})")


if __name__ == "__main__":
    main()
