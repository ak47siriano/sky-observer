"""Train a LenghuSky-8-inspired DINOv3 Frozen-Backbone + 1x1 Linear Probe on 50-100 Day/Night Images.

Methodology inspired by LenghuSky-8 (Rui et al., CVPR Findings 2025, Apache-2.0):
  1. Freeze a Vision Transformer backbone (optional user-provided DINOv3 / DINOv2 weights via `timm`,
     or deterministic local patch-encoder fallback when running 100% offline without downloaded weights).
  2. Train ONLY the lightweight `1x1 Conv` linear probe on 50-100 day/night labeled sky images
     (`user_camera_labels` from SQLite or any local `--data-dir` with `images/` + `masks/`).
  3. Optionally register the resulting checkpoint in `model_registry` (standby by default unless `--activate`),
     and evaluate downstream optical-flow cloud advection using `scripts/evaluate_nowcast.py`.

License notice:
  - LenghuSky-8 reference methodology/benchmark is Apache-2.0 (no source code copied).
  - Optional DINOv3 backbone weights are governed separately by the DINOv3 License; never commit
    or redistribute weight files in this repository.

Usage:
  PYTHONPATH=backend python scripts/train_dinov3_probe.py --data-dir data/datasets/my_day_night_100 --epochs 20
  PYTHONPATH=backend python scripts/train_dinov3_probe.py --from-user-labels --epochs 15 --register
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.ai.architectures import DINOv3LinearProbeSeg  # noqa: E402
from app.ai.device import resolve_device, to_torch_device  # noqa: E402
from app.ai.trainer import _FocalDiceSegLoss, _physics_safe_augment  # noqa: E402
from app.vision.segmentation import NOT_SKY_IGNORE_INDEX  # noqa: E402


class _PairSegDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        resolution: int = 256,
        num_classes: int = 5,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.res = resolution
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_p, msk_p = self.pairs[idx]
        bgr = cv2.imread(str(img_p))
        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if bgr is None or msk is None:
            return (
                torch.zeros(3, self.res, self.res, dtype=torch.float32),
                torch.full((self.res, self.res), NOT_SKY_IGNORE_INDEX, dtype=torch.long),
            )
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.res, self.res), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        msk = cv2.resize(msk, (self.res, self.res), interpolation=cv2.INTER_NEAREST)
        not_sky = msk == NOT_SKY_IGNORE_INDEX
        valid_vals = msk[~not_sky]
        max_v = int(valid_vals.max()) if valid_vals.size > 0 else 0
        if self.num_classes == 2:
            out_m = (
                np.where(msk > 127, 1, 0).astype(np.uint8)
                if max_v > 5
                else np.where((msk >= 1) & (msk <= 3), 1, 0).astype(np.uint8)
            )
        else:
            out_m = (
                np.where(msk > 127, 2, 0).astype(np.uint8)
                if max_v > 5
                else np.clip(msk, 0, self.num_classes - 1).astype(np.uint8)
            )
        out_m[not_sky] = NOT_SKY_IGNORE_INDEX
        if self.augment:
            rgb, out_m_aug = _physics_safe_augment(rgb, out_m)
            assert out_m_aug is not None
            out_m = out_m_aug
        return (
            torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1),
            torch.from_numpy(out_m.astype(np.int64)),
        )


def collect_pairs_from_dir(data_dir: Path) -> list[tuple[Path, Path]]:
    img_dir = data_dir / "images"
    msk_dir = data_dir / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Expected images/ and masks/ subdirectories inside {data_dir}")
    mask_by_stem = {p.stem: p for p in msk_dir.iterdir() if p.is_file()}
    pairs: list[tuple[Path, Path]] = []
    for img_p in sorted(img_dir.iterdir()):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        m = mask_by_stem.get(img_p.stem)
        if m is not None:
            pairs.append((img_p, m))
    return pairs


def collect_pairs_from_user_labels() -> list[tuple[Path, Path]]:
    from app.database import get_db_session, init_database
    from app.models_db import DatasetItem, DatasetRecord

    init_database()
    with get_db_session() as db:
        ds = db.query(DatasetRecord).filter(DatasetRecord.name == "user_camera_labels").first()
        if ds is None:
            return []
        items = db.query(DatasetItem).filter(DatasetItem.dataset_id == ds.id).all()
        pairs: list[tuple[Path, Path]] = []
        for it in items:
            ip = Path(it.image_path)
            mp = Path(it.mask_path or "")
            if ip.exists() and mp.exists():
                pairs.append((ip, mp))
        return pairs


def train_probe(
    pairs: list[tuple[Path, Path]],
    output_path: Path,
    *,
    num_classes: int = 5,
    resolution: int = 256,
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 3e-3,
    device_pref: str = "auto",
    backbone_id: str = "vit_small_patch14_dinov2.lvd142m",
    weights_path: str = "",
    use_timm_backbone: bool = False,
) -> dict[str, Any]:
    if len(pairs) < 2:
        raise ValueError(f"Need at least 2 labeled (image, mask) pairs to train linear probe; got {len(pairs)}.")

    val_n = max(1, int(round(len(pairs) * 0.2)))
    train_pairs = pairs[:-val_n]
    val_pairs = pairs[-val_n:]

    device = resolve_device(device_pref, 0.85)
    if device == "directml":
        device = "cpu"
    torch_dev = to_torch_device(device)

    model = DINOv3LinearProbeSeg(
        num_classes=num_classes,
        backbone_id=backbone_id,
        weights_path=weights_path,
        use_timm_backbone=use_timm_backbone,
    ).to(torch_dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = _FocalDiceSegLoss(ignore_index=NOT_SKY_IGNORE_INDEX, gamma=1.5)

    tr_loader = DataLoader(
        _PairSegDataset(train_pairs, resolution=resolution, num_classes=num_classes, augment=True),
        batch_size=max(1, batch_size),
        shuffle=True,
    )
    va_loader = DataLoader(
        _PairSegDataset(val_pairs, resolution=resolution, num_classes=num_classes, augment=False),
        batch_size=max(1, batch_size),
        shuffle=False,
    )

    best_miou = -1.0
    best_acc = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, msks in tr_loader:
            imgs, msks = imgs.to(torch_dev), msks.to(torch_dev)
            optimizer.zero_grad()
            loss = criterion(model(imgs), msks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        inter = np.zeros(num_classes, dtype=np.float64)
        union = np.zeros(num_classes, dtype=np.float64)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, msks in va_loader:
                imgs, msks = imgs.to(torch_dev), msks.to(torch_dev)
                preds = torch.argmax(model(imgs), dim=1)
                valid = msks != NOT_SKY_IGNORE_INDEX
                correct += int(((preds == msks) & valid).sum().item())
                total += int(valid.sum().item())
                for c in range(num_classes):
                    p_c = (preds == c) & valid
                    t_c = (msks == c) & valid
                    inter[c] += float((p_c & t_c).sum().item())
                    union[c] += float((p_c | t_c).sum().item())

        present = union > 0
        miou = float(np.mean(inter[present] / union[present])) if np.any(present) else 0.0
        px_acc = float(correct / max(1, total))
        if miou >= best_miou:
            best_miou = miou
            best_acc = px_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "DINOv3-LinearProbe",
                    "backbone_id": backbone_id,
                    "using_timm": model.using_timm,
                    "num_classes": num_classes,
                    "val_miou": round(miou, 4),
                    "val_pixel_acc": round(px_acc, 4),
                    "epoch": epoch,
                },
                output_path,
            )

    return {
        "weights_path": str(output_path),
        "architecture": "DINOv3-LinearProbe",
        "using_timm_backbone": model.using_timm,
        "backbone_id": backbone_id if model.using_timm else "local_patch_fallback",
        "trainable_params": sum(p.numel() for p in trainable),
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "val_miou": round(best_miou, 4),
        "val_pixel_acc": round(best_acc, 4),
        "device": device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv3 linear-probe sky segmenter on 50-100 day/night frames.")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing images/ and masks/.")
    parser.add_argument("--from-user-labels", action="store_true", help="Load pairs from SQLite user_camera_labels.")
    parser.add_argument("--output", type=str, default="data/models/dinov3_linear_probe_best.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=5, choices=[2, 5])
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--backbone-id", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--weights-path", type=str, default="", help="Optional local DINOv3/DINOv2 checkpoint file.")
    parser.add_argument("--use-timm-backbone", action="store_true", help="Load timm pretrained ViT if cached/online.")
    parser.add_argument(
        "--register", action="store_true", help="Register trained probe in local SQLite model_registry."
    )
    parser.add_argument(
        "--activate", action="store_true", help="Also activate in model_registry (requires --register)."
    )
    args = parser.parse_args()

    pairs = collect_pairs_from_dir(Path(args.data_dir)) if args.data_dir else collect_pairs_from_user_labels()
    if not pairs:
        print(
            "No labeled (image, mask) pairs found. Label 50-100 day/night frames in Dataset Labeling "
            "(user_camera_labels) or pass --data-dir <path_with_images_and_masks>."
        )
        sys.exit(1)

    report = train_probe(
        pairs,
        Path(args.output),
        num_classes=args.num_classes,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_pref=args.device,
        backbone_id=args.backbone_id,
        weights_path=args.weights_path,
        use_timm_backbone=args.use_timm_backbone,
    )

    if args.register:
        from app.ai.registry import register_and_maybe_activate
        from app.database import get_db_session, init_database

        init_database()
        with get_db_session() as db:
            register_and_maybe_activate(
                db=db,
                model_id=f"segmentation-dinov3-probe-{Path(args.output).stem[-6:]}",
                model_family="segmentation",
                architecture="DINOv3-LinearProbe",
                version="v1.0.0",
                weights_path=Path(args.output),
                metrics={"val_miou": report["val_miou"], "val_pixel_acc": report["val_pixel_acc"]},
                activate=bool(args.activate),
                notes=f"Linear probe trained on {report['train_samples']} train / {report['val_samples']} val frames",
            )
        report["registered"] = True
        report["activated"] = bool(args.activate)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
