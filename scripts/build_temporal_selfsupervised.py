"""Self-supervised Stage-3 (+5m/+15m/+20m) sequence builder from raw webcam timelapses or frame folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.registry import get_model_cache
from app.config import get_config_manager
from app.database import get_db_session, init_database
from app.vision.color import analyze_color
from app.vision.coverage import compute_cloud_coverage
from app.vision.masking import build_valid_sky_mask_from_config
from app.vision.segmentation import run_segmentation
from app.vision.texture import analyze_texture


def _extract_frame_vec(bgr: np.ndarray, cfg, seg_model, seg_dev: str, model_name: str) -> np.ndarray:
    valid_mask = build_valid_sky_mask_from_config(bgr.shape[:2], cfg.camera)
    seg = run_segmentation(
        image_bgr=bgr,
        valid_mask=valid_mask,
        inference_resolution=min(384, cfg.vision.inference_resolution),
        seg_model=seg_model,
        device=seg_dev,
        model_name=model_name,
    )
    cov = compute_cloud_coverage(seg.class_map, valid_mask, seg.mean_confidence)
    col = analyze_color(bgr, valid_mask)
    tex = analyze_texture(bgr, valid_mask, seg.cloud_binary_mask)
    return np.array(
        [
            cov["cloud_coverage_pct"] / 100.0,
            cov["clear_sky_pct"] / 100.0,
            cov["thin_cloud_pct"] / 100.0,
            cov["dense_cloud_pct"] / 100.0,
            cov["dark_cloud_pct"] / 100.0,
            cov.get("sun_glare_pct", 0.0) / 100.0,
            col["avg_luminance"] / 100.0,
            col["blue_sky_index"],
            col["saturation"],
            col["grayness"],
            col["pattern_ratios"]["blue_sky"],
            col["pattern_ratios"]["white_cloud"],
            col["pattern_ratios"]["dark_cloud"],
            tex["edge_density"] * 5.0,
            tex["local_contrast"] / 40.0,
            tex["fragmentation_index"],
            tex["glcm_contrast"] / 20.0,
            tex["glcm_homogeneity"],
            tex["glcm_entropy"] / 8.0,
            cov["cloud_coverage_pct"] / 100.0,
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build self-supervised (+5m, +15m, +20m) temporal training sequences from raw timelapses/frames."
    )
    parser.add_argument("--source", required=True, help="Path to an .mp4 timelapse video OR a folder of chronological sky frames.")
    parser.add_argument("--frame-interval-sec", type=float, default=60.0, help="Real-world seconds between consecutive sampled frames.")
    parser.add_argument("--seq-len", type=int, default=4, help="Input history sequence length T (default: 4).")
    parser.add_argument("--output", default=None, help="Output .npz path (defaults to data/datasets/temporal_selfsup.npz).")
    args = parser.parse_args()

    cfg = get_config_manager().get()
    cfg.ensure_directories()
    init_database(cfg.db_path)

    with get_db_session() as db:
        get_model_cache().refresh_from_db(db)
    seg_model, seg_meta, seg_dev = get_model_cache().get("segmentation")
    model_name = seg_meta["model_id"] if seg_meta else "radiometric-physics-v1"

    src = Path(args.source)
    vecs: list[np.ndarray] = []
    if src.is_file():
        cap = cv2.VideoCapture(str(src))
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if float(frame.mean()) >= 10.0:
                vecs.append(_extract_frame_vec(frame, cfg, seg_model, seg_dev, model_name))
        cap.release()
    elif src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in sorted(src.glob("*")):
            if p.suffix.lower() not in exts:
                continue
            frame = cv2.imread(str(p))
            if frame is not None and float(frame.mean()) >= 10.0:
                vecs.append(_extract_frame_vec(frame, cfg, seg_model, seg_dev, model_name))
    else:
        raise FileNotFoundError(f"Source not found: {src}")

    dt = max(1.0, float(args.frame_interval_sec))
    h_steps = [max(1, round((m * 60.0) / dt)) for m in (5, 15, 20)]
    max_h = max(h_steps)
    T = max(2, int(args.seq_len))

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    for i in range(T - 1, len(vecs) - max_h):
        seq_in = np.stack(vecs[i - T + 1 : i + 1], axis=0)  # (T, 20)
        curr_c = float(vecs[i][0])
        curr_l = float(vecs[i][6])
        target = np.zeros((3, 6), dtype=np.float32)
        for h_idx, step_off in enumerate(h_steps):
            fut = vecs[i + step_off]
            fut_c = float(fut[0])
            fut_dark = float(fut[4])
            lum_drop = float(fut[6]) - curr_l
            target[h_idx, 0] = fut_c
            target[h_idx, 1] = 1.0 if (fut_c < curr_c - 0.05) else 0.0
            target[h_idx, 2] = 1.0 if (fut_c > curr_c + 0.05) else 0.0
            target[h_idx, 3] = 1.0 if fut_c >= 0.80 else 0.0
            target[h_idx, 4] = 1.0 if (fut_dark > 0.20 or lum_drop < -0.08) else 0.0
            target[h_idx, 5] = 1.0 if (fut_dark > 0.30 and fut_c > 0.70) else 0.0
        X_list.append(seq_in)
        Y_list.append(target)

    out_path = Path(args.output) if args.output else (cfg.datasets_dir / "temporal_selfsup.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X_arr = np.stack(X_list, axis=0) if X_list else np.zeros((0, T, 20), dtype=np.float32)
    Y_arr = np.stack(Y_list, axis=0) if Y_list else np.zeros((0, 3, 6), dtype=np.float32)
    np.savez_compressed(str(out_path), X=X_arr, Y=Y_arr, horizons=np.array([5, 15, 20], dtype=np.int32))
    print(f"Built {len(X_list)} self-supervised (+5m/+15m/+20m) sequences from {len(vecs)} frames -> {out_path}")


if __name__ == "__main__":
    main()
