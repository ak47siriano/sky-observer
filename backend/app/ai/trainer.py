"""Non-blocking Local Fine-Tuning Pipeline with Mixed Precision, Gradient Accumulation, Checkpoints & Resume."""

from __future__ import annotations

import gc
import json
import sys
import threading
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from app.ai.architectures import SkyEventClassifier, build_model_for_family
from app.ai.device import resolve_device, to_torch_device
from app.ai.registry import register_and_maybe_activate
from app.config import get_config_manager
from app.database import get_db_session
from app.models_db import DatasetItem, DatasetRecord, ModelRegistry, TrainingSession
from app.vision.classification import (
    CLOUD_TYPE_CLASSES,
    WMO_GROUPED_CLASSES,
    _extract_feature_vector,
    group_wmo_label,
)
from app.vision.color import analyze_color
from app.vision.coverage import compute_cloud_coverage
from app.vision.segmentation import NOT_SKY_IGNORE_INDEX
from app.vision.texture import analyze_texture


def _physics_safe_augment(
    img_rgb: np.ndarray,
    msk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Physics-preserving sky augmentation:
      - Horizontal flip (p=0.5)
      - Mild random crop & scale (0.85..1.0)
      - Uniform brightness/contrast jitter across RGB (NO Hue shift, preserving physical R/B ratio!)
    """
    h, w = img_rgb.shape[:2]
    if np.random.rand() > 0.5:
        img_rgb = np.ascontiguousarray(img_rgb[:, ::-1, :])
        if msk is not None:
            msk = np.ascontiguousarray(msk[:, ::-1])

    if np.random.rand() > 0.45:
        scale = float(np.random.uniform(0.84, 1.0))
        ch, cw = max(16, int(h * scale)), max(16, int(w * scale))
        y0 = int(np.random.randint(0, max(1, h - ch + 1)))
        x0 = int(np.random.randint(0, max(1, w - cw + 1)))
        img_rgb = cv2.resize(img_rgb[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        if msk is not None:
            msk = cv2.resize(msk[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)

    if np.random.rand() > 0.4:
        # Equal scalar gain/offset across R, G, B -> preserves R/B chromaticity ratio (no Hue distortion)
        alpha = float(np.random.uniform(0.90, 1.10))
        beta = float(np.random.uniform(-10.0, 10.0))
        img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

    return img_rgb, msk


class _SegDataset(Dataset):
    """100% lazy-loading disk dataset for Stage 1 segmentation (Phase 1A binary or Phase 1B 5-class + Not_Sky=255)."""

    def __init__(
        self,
        items: list[DatasetItem],
        resolution: int = 384,
        augment: bool = False,
        binary_phase: bool = False,
        static_not_sky_mask_path: str | None = None,
    ) -> None:
        self.samples: list[tuple[str, str]] = [
            (it.image_path, it.mask_path)
            for it in items
            if it.mask_path and Path(it.mask_path).exists() and Path(it.image_path).exists()
        ]
        self.res = resolution
        self.augment = augment
        self.binary_phase = binary_phase
        self.static_not_sky: np.ndarray | None = None
        if static_not_sky_mask_path and Path(static_not_sky_mask_path).exists():
            raw_excl = cv2.imread(static_not_sky_mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_excl is not None and raw_excl.size > 0:
                self.static_not_sky = cv2.resize(raw_excl, (self.res, self.res), interpolation=cv2.INTER_NEAREST) > 127

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        bgr = cv2.imread(img_path)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if bgr is None or msk is None:
            return torch.zeros(3, self.res, self.res, dtype=torch.float32), torch.zeros(self.res, self.res, dtype=torch.long)

        img = cv2.cvtColor(cv2.resize(bgr, (self.res, self.res), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        msk = cv2.resize(msk, (self.res, self.res), interpolation=cv2.INTER_NEAREST)

        # Preserve ignore_index=255 (Not_Sky obstacles), normalize binary or legacy 6-class masks
        not_sky = msk == NOT_SKY_IGNORE_INDEX
        if self.static_not_sky is not None:
            not_sky = not_sky | self.static_not_sky

        valid_vals = msk[~not_sky]
        max_valid = int(valid_vals.max()) if valid_vals.size > 0 else 0

        if self.binary_phase:
            if max_valid > 5:
                out_m = np.where(msk > 127, 1, 0).astype(np.uint8)
            else:
                out_m = np.where((msk >= 1) & (msk <= 3), 1, 0).astype(np.uint8)
        else:
            if max_valid > 5:
                out_m = np.where(msk > 127, 2, 0).astype(np.uint8)
            else:
                out_m = np.clip(msk, 0, 4).astype(np.uint8)

        out_m[not_sky] = NOT_SKY_IGNORE_INDEX

        if self.augment:
            img, out_m_aug = _physics_safe_augment(img, out_m)
            assert out_m_aug is not None
            out_m = out_m_aug

        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        msk_t = torch.from_numpy(out_m.astype(np.int64))
        return img_t, msk_t


class _ClsDataset(Dataset):
    """100% lazy-loading disk dataset for Stage 2 WMO classification (safe for 19,000+ TJNU GCD/MGCD images)."""

    def __init__(
        self,
        items: list[DatasetItem],
        augment: bool = False,
        group_ambiguous_wmo: bool = False,
    ) -> None:
        self.group_ambiguous_wmo = group_ambiguous_wmo
        self.class_names = WMO_GROUPED_CLASSES if group_ambiguous_wmo else CLOUD_TYPE_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.augment = augment
        self.samples: list[tuple[str, str, int, float | None]] = []
        for it in items:
            if not it.image_path or not Path(it.image_path).exists():
                continue
            raw_lbl = str(it.cloud_type_label or "").strip()
            mapped_lbl = group_wmo_label(raw_lbl) if group_ambiguous_wmo else raw_lbl
            if mapped_lbl in self.class_to_idx:
                self.samples.append((it.image_path, it.mask_path or "", self.class_to_idx[mapped_lbl], None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, mask_path, label_idx, solar_el = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            return (
                torch.zeros(3, 224, 224, dtype=torch.float32),
                torch.zeros(18, dtype=torch.float32),
                torch.tensor(label_idx, dtype=torch.long),
            )
        small = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
        valid_mask = np.full(small.shape[:2], 255, dtype=np.uint8)
        col = analyze_color(small, valid_mask)
        if mask_path and Path(mask_path).exists():
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST)
                if m.max() > 5 and m.max() != NOT_SKY_IGNORE_INDEX:
                    m = np.where(m > 127, 2, 0).astype(np.uint8)
            else:
                from app.vision.segmentation import _radiometric_probabilities

                m = np.argmax(_radiometric_probabilities(cv2.resize(small, (128, 128))), axis=-1).astype(np.uint8)
                m = cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST)
        else:
            from app.vision.segmentation import _radiometric_probabilities

            m = np.argmax(_radiometric_probabilities(cv2.resize(small, (128, 128))), axis=-1).astype(np.uint8)
            m = cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST)
        cov = compute_cloud_coverage(m, valid_mask, 0.8)
        cb = np.where((m >= 1) & (m <= 3), 255, 0).astype(np.uint8)
        tex = analyze_texture(small, valid_mask, cb)
        fvec = _extract_feature_vector(cov, col, tex, solar_elevation_deg=solar_el)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.augment:
            rgb, _ = _physics_safe_augment(rgb, None)
        return (
            torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1),
            torch.from_numpy(fvec.copy()),
            torch.tensor(label_idx, dtype=torch.long),
        )


def _frame_to_seq_vector(it: DatasetItem) -> np.ndarray:
    """Extract 20-dim temporal feature vector from a dataset frame."""
    bgr = cv2.imread(it.image_path)
    if bgr is None:
        return np.zeros(20, dtype=np.float32)
    h, w = bgr.shape[:2]
    valid_mask = np.full((h, w), 255, dtype=np.uint8)
    col = analyze_color(bgr, valid_mask)
    if it.mask_path and Path(it.mask_path).exists():
        m = cv2.resize(cv2.imread(it.mask_path, cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST)
        if m.max() > 5:
            m = np.where(m > 127, 2, 0).astype(np.uint8)
    else:
        from app.vision.segmentation import _radiometric_probabilities

        m = cv2.resize(
            np.argmax(_radiometric_probabilities(cv2.resize(bgr, (128, 128))), axis=-1).astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
    cov = compute_cloud_coverage(m, valid_mask, 0.8)
    cb = np.where((m >= 1) & (m <= 3), 255, 0).astype(np.uint8)
    tex = analyze_texture(bgr, valid_mask, cb)
    return np.array(
        [
            cov["cloud_coverage_pct"] / 100.0,
            cov["clear_sky_pct"] / 100.0,
            cov["thin_cloud_pct"] / 100.0,
            cov["dense_cloud_pct"] / 100.0,
            cov["dark_cloud_pct"] / 100.0,
            cov["haze_fog_pct"] / 100.0,
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
            (it.cloud_coverage_gt or cov["cloud_coverage_pct"]) / 100.0,
        ],
        dtype=np.float32,
    )


def _build_temporal_sequences(items: list[DatasetItem]) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Group items by group_key ordered by sequence_index to construct real temporal sequences."""
    grouped: dict[str, list[DatasetItem]] = {}
    for it in items:
        if Path(it.image_path).exists():
            grouped.setdefault(it.group_key, []).append(it)

    event_to_idx = {c: i for i, c in enumerate(SkyEventClassifier.EVENT_CLASSES)}
    samples: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _gk, seq_items in grouped.items():
        seq_items.sort(key=lambda x: x.sequence_index)
        if len(seq_items) < 4:
            continue
        vecs = [_frame_to_seq_vector(x) for x in seq_items]
        covs = [float(v[0]) for v in vecs]
        lums = [float(v[6]) for v in vecs]

        # Predict final trajectory states across 4 horizons using actual sequence progression
        in_len = max(3, len(vecs) - 2)
        seq_in = np.stack(vecs[:in_len], axis=0)
        # Pad or trim to T=4
        if seq_in.shape[0] > 4:
            seq_in = seq_in[-4:]
        elif seq_in.shape[0] < 4:
            pad = np.repeat(seq_in[:1], 4 - seq_in.shape[0], axis=0)
            seq_in = np.concatenate([pad, seq_in], axis=0)

        curr_c = covs[in_len - 1]
        fut_c = covs[-1]
        slope = fut_c - curr_c
        dark_last = vecs[-1][4]
        lum_drop = lums[-1] - lums[in_len - 1]

        # Predict 3 camera-only horizons (+5m, +15m, +20m)
        targets = np.zeros((3, 6), dtype=np.float32)
        for h_idx, scale in enumerate([0.35, 0.75, 1.0]):
            proj = float(np.clip(curr_c + slope * scale, 0.0, 1.0))
            targets[h_idx, 0] = proj
            targets[h_idx, 1] = 1.0 if (proj < curr_c - 0.05) else 0.0
            targets[h_idx, 2] = 1.0 if (proj > curr_c + 0.05) else 0.0
            targets[h_idx, 3] = 1.0 if proj >= 0.80 else 0.0
            targets[h_idx, 4] = 1.0 if (dark_last > 0.20 or lum_drop < -0.08) else 0.0
            targets[h_idx, 5] = 1.0 if (dark_last > 0.30 and proj > 0.70) else 0.0

        ev_lbl = seq_items[-1].event_label
        ev_idx = event_to_idx.get(ev_lbl, 0)
        samples.append((seq_in, targets, ev_idx))
    return samples


class _FocalDiceSegLoss(nn.Module):
    """Combined Focal Cross-Entropy + Multi-Class Soft Dice Loss respecting ignore_index=255."""

    def __init__(self, ignore_index: int = NOT_SKY_IGNORE_INDEX, gamma: float = 1.5) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        valid_f = valid.float()
        pt = torch.exp(-ce.clamp(max=20.0))
        focal = (((1.0 - pt) ** self.gamma) * ce * valid_f).sum() / valid_f.sum().clamp(min=1.0)

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1) * valid_f.unsqueeze(1)
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets)).clamp(0, num_classes - 1)
        one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float() * valid_f.unsqueeze(1)

        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dim=dims)
        denom = torch.sum(probs + one_hot, dim=dims)
        present = torch.sum(one_hot, dim=dims) > 0
        if torch.any(present):
            dice_scores = (2.0 * inter[present] + 1e-5) / (denom[present] + 1e-5)
            dice_loss = 1.0 - dice_scores.mean()
        else:
            dice_loss = logits.sum() * 0.0

        return 0.55 * focal + 0.45 * dice_loss


class TrainingManager:
    """Manages local background model training, resume, stop, and checkpoint evaluation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_flags: dict[str, bool] = {}
        self._active_threads: dict[str, threading.Thread] = {}

    def has_active_jobs(self) -> bool:
        with self._lock:
            return any(t.is_alive() for t in self._active_threads.values())

    def request_stop(self, job_id: str | None = None) -> list[str]:
        with self._lock:
            stopped: list[str] = []
            for jid in list(self._stop_flags.keys()):
                if job_id is None or jid == job_id:
                    self._stop_flags[jid] = True
                    stopped.append(jid)
            return stopped

    def start_training(
        self,
        model_family: str,
        dataset_id: int,
        epochs: int = 8,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        resume_from_job_id: str | None = None,
        auto_activate: bool = True,
        async_run: bool = True,
        warm_start_weights: str | None = None,
        device_override: str | None = None,
        architecture_override: str | None = None,
    ) -> dict[str, Any]:
        cfg = get_config_manager().get()
        pref_dev = device_override or cfg.ai.device
        device = resolve_device(pref_dev, cfg.ai.max_gpu_memory_fraction, cpu_threads=cfg.ai.cpu_threads)
        if batch_size is not None:
            bs = batch_size
        elif model_family == "segmentation":
            bs = int(getattr(cfg.ai, "seg_batch_size", cfg.ai.batch_size) or 8)
        elif model_family == "classification":
            bs = int(getattr(cfg.ai, "cls_batch_size", 32) or 32)
        else:
            bs = cfg.ai.batch_size
        lr = learning_rate or cfg.ai.learning_rate
        job_id = resume_from_job_id or f"{model_family}-{uuid.uuid4().hex[:8]}"

        with get_db_session() as db:
            ds = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
            if not ds:
                raise ValueError(f"Dataset {dataset_id} not found")
            existing = db.query(TrainingSession).filter(TrainingSession.job_id == job_id).first()
            if existing and resume_from_job_id:
                existing.status = "running"
                existing.total_epochs = epochs
                existing.error_message = ""
                session_id = existing.id
                arch = existing.architecture
            else:
                _, arch = build_model_for_family(
                    model_family,
                    architecture_override,
                    seg_backbone=str(getattr(cfg.ai, "seg_backbone", "convnext_aspp")),
                    cls_backbone=str(getattr(cfg.ai, "cls_backbone", "efficientnet_b0")),
                )
                sess = TrainingSession(
                    job_id=job_id,
                    model_family=model_family,
                    architecture=arch,
                    dataset_id=dataset_id,
                    status="queued",
                    current_epoch=0,
                    total_epochs=epochs,
                    batch_size=bs,
                    learning_rate=lr,
                    device_used=device,
                    mixed_precision=bool(cfg.ai.mixed_precision and device == "cuda"),
                    checkpoint_path=str(cfg.models_dir / f"{job_id}_best.pt"),
                )
                db.add(sess)
                db.flush()
                session_id = sess.id

        with self._lock:
            self._stop_flags[job_id] = False

        if async_run:
            t = threading.Thread(
                target=self._run_job,
                args=(
                    job_id,
                    session_id,
                    model_family,
                    dataset_id,
                    epochs,
                    bs,
                    lr,
                    resume_from_job_id is not None,
                    auto_activate,
                    warm_start_weights,
                    pref_dev,
                    architecture_override,
                ),
                daemon=True,
            )
            with self._lock:
                self._active_threads[job_id] = t
            t.start()
        else:
            self._run_job(
                job_id,
                session_id,
                model_family,
                dataset_id,
                epochs,
                bs,
                lr,
                resume_from_job_id is not None,
                auto_activate,
                warm_start_weights,
                pref_dev,
                architecture_override,
            )

        return {"job_id": job_id, "session_id": session_id, "model_family": model_family, "device": device}

    def _should_stop(self, job_id: str) -> bool:
        with self._lock:
            return self._stop_flags.get(job_id, False)

    def _run_job(
        self,
        job_id: str,
        session_id: int,
        model_family: str,
        dataset_id: int,
        epochs: int,
        batch_size: int,
        lr: float,
        resume: bool,
        auto_activate: bool,
        warm_start_weights: str | None = None,
        device_override: str | None = None,
        architecture_override: str | None = None,
    ) -> None:
        cfg = get_config_manager().get()
        device = resolve_device(
            device_override or cfg.ai.device,
            cfg.ai.max_gpu_memory_fraction,
            cpu_threads=cfg.ai.cpu_threads,
        )
        # On Windows WDDM, torch-directml training (backward passes + AdamW / weighted CE) can
        # exceed the 2s DX12 TDR timeout or miscompute weighted losses. Keep DirectML for fast live
        # inference, and run 'auto' training safely on Ryzen CPU unless explicitly forced to 'directml'.
        explicit_directml = (device_override or "").lower().strip() == "directml"
        if device == "directml" and (not explicit_directml or model_family in {"classification", "temporal", "event"}):
            device = "cpu"
        torch_dev = to_torch_device(device)
        use_amp = bool(cfg.ai.mixed_precision and device == "cuda")
        grad_accum = max(1, cfg.ai.gradient_accumulation_steps)

        best_ckpt_path = cfg.models_dir / f"{job_id}_best.pt"
        last_ckpt_path = cfg.models_dir / f"{job_id}_last.pt"
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with get_db_session() as db:
                sess = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
                assert sess is not None
                sess.status = "running"
                sess.device_used = device
                items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()

            train_items = [x for x in items if x.split == "train"]
            val_items = [x for x in items if x.split == "val"] or train_items[-max(1, len(train_items) // 5):]

            seg_binary_phase = False
            if model_family == "segmentation":
                # Auto-detect Phase 1A binary dataset (e.g., SWINySEG) if dataset name mentions swinyseg/binary
                with get_db_session() as db:
                    ds_rec = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
                    if ds_rec and any(k in (ds_rec.name or "").lower() for k in ("swinyseg", "phase1a", "binary")):
                        seg_binary_phase = True
                model, arch_name = build_model_for_family(
                    model_family,
                    architecture_override,
                    num_classes=2 if seg_binary_phase else 5,
                    pretrained=bool(getattr(cfg.ai, "pretrained_backbone", True)),
                    seg_backbone=str(getattr(cfg.ai, "seg_backbone", "convnext_aspp")),
                )
            elif model_family == "classification":
                grp_wmo = bool(getattr(cfg.ai, "group_ambiguous_wmo", False))
                model, arch_name = build_model_for_family(
                    model_family,
                    architecture_override,
                    num_classes=len(WMO_GROUPED_CLASSES) if grp_wmo else len(CLOUD_TYPE_CLASSES),
                    pretrained=bool(getattr(cfg.ai, "pretrained_backbone", True)),
                    cls_backbone=str(getattr(cfg.ai, "cls_backbone", "efficientnet_b0")),
                )
            else:
                model, arch_name = build_model_for_family(model_family, architecture_override, num_horizons=3)
            model = model.to(torch_dev)
            if not resume and warm_start_weights and Path(warm_start_weights).exists():
                try:
                    ws_ckpt = torch.load(str(warm_start_weights), map_location="cpu", weights_only=False)
                    ws_state = (
                        ws_ckpt["model_state_dict"]
                        if isinstance(ws_ckpt, dict) and "model_state_dict" in ws_ckpt
                        else ws_ckpt
                    )
                    # Filter out mismatched head shapes (supports Phase 1A binary -> Phase 1B 5-class transfer)
                    curr_state = model.state_dict()
                    compat_state = {
                        k: v for k, v in ws_state.items()
                        if k in curr_state and curr_state[k].shape == v.shape
                    }
                    model.load_state_dict(compat_state, strict=False)
                except Exception:
                    pass
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params or list(model.parameters()),
                lr=lr,
                weight_decay=1e-4,
                foreach=(device != "directml"),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            start_epoch = 0
            history: list[dict[str, Any]] = []
            best_metric = -1.0

            if resume and last_ckpt_path.exists():
                ckpt = torch.load(str(last_ckpt_path), map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0))
                history = ckpt.get("history", [])
                best_metric = float(ckpt.get("best_metric", -1.0))

            patience_left = cfg.ai.early_stopping_patience
            final_metrics: dict[str, Any] = {}
            conf_mat_payload: dict[str, Any] = {}

            if model_family == "segmentation":
                seg_res = max(256, min(512, cfg.vision.inference_resolution))
                tr_ds = _SegDataset(
                    train_items,
                    resolution=seg_res,
                    augment=True,
                    binary_phase=seg_binary_phase,
                    static_not_sky_mask_path=cfg.camera.exclusion_mask_path or None,
                )
                va_ds = _SegDataset(
                    val_items,
                    resolution=seg_res,
                    augment=False,
                    binary_phase=seg_binary_phase,
                    static_not_sky_mask_path=cfg.camera.exclusion_mask_path or None,
                )
                if len(tr_ds) == 0:
                    raise ValueError("No segmented mask items available in dataset for training")
                # On Windows, multiprocessing DataLoader spawn reloads torch\lib\shm.dll per worker
                # inside the live Uvicorn process, exhausting virtual commit RAM; keep num_workers=0 on win32.
                n_workers = 0 if sys.platform == "win32" else (int(getattr(cfg.ai, "dataloader_num_workers", 2) or 0) if len(tr_ds) >= 64 else 0)
                pin_mem = device == "cuda"
                tr_loader = DataLoader(tr_ds, batch_size=max(1, batch_size), shuffle=True, num_workers=n_workers, pin_memory=pin_mem)
                va_loader = DataLoader(va_ds, batch_size=max(1, batch_size), shuffle=False, num_workers=n_workers, pin_memory=pin_mem)
                criterion = _FocalDiceSegLoss(ignore_index=NOT_SKY_IGNORE_INDEX, gamma=1.5)
                num_seg_classes = 2 if seg_binary_phase else 5

                for epoch in range(start_epoch + 1, epochs + 1):
                    if self._should_stop(job_id):
                        break
                    model.train()
                    tr_loss = 0.0
                    optimizer.zero_grad()
                    for step, (imgs, msks) in enumerate(tr_loader):
                        if self._should_stop(job_id):
                            break
                        imgs, msks = imgs.to(torch_dev), msks.to(torch_dev)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits = model(imgs)
                            loss = criterion(logits, msks) / grad_accum
                        scaler.scale(loss).backward()
                        if (step + 1) % grad_accum == 0 or (step + 1) == len(tr_loader):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                        tr_loss += float(loss.item()) * grad_accum
                    if self._should_stop(job_id):
                        break
                    scheduler.step()

                    model.eval()
                    va_loss = 0.0
                    correct_px = 0
                    total_px = 0
                    inter_total = np.zeros(num_seg_classes, dtype=np.float64)
                    union_total = np.zeros(num_seg_classes, dtype=np.float64)
                    with torch.no_grad():
                        for imgs, msks in va_loader:
                            imgs, msks = imgs.to(torch_dev), msks.to(torch_dev)
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                logits = model(imgs)
                                va_loss += float(criterion(logits, msks).item())
                            preds = torch.argmax(logits, dim=1)
                            valid_px = msks != NOT_SKY_IGNORE_INDEX
                            correct_px += int(((preds == msks) & valid_px).sum().item())
                            total_px += int(valid_px.sum().item())
                            for c in range(num_seg_classes):
                                p_c = (preds == c) & valid_px
                                t_c = (msks == c) & valid_px
                                inter_total[c] += float((p_c & t_c).sum().item())
                                union_total[c] += float((p_c | t_c).sum().item())

                    valid_c = union_total > 0
                    miou = float(np.mean(inter_total[valid_c] / union_total[valid_c])) if np.any(valid_c) else 0.0
                    px_acc = float(correct_px / max(total_px, 1))
                    ep_row = {
                        "epoch": epoch,
                        "train_loss": round(tr_loss / max(len(tr_loader), 1), 4),
                        "val_loss": round(va_loss / max(len(va_loader), 1), 4),
                        "val_pixel_acc": round(px_acc, 4),
                        "val_miou": round(miou, 4),
                    }
                    history.append(ep_row)
                    score = miou
                    if score >= best_metric:
                        best_metric = score
                        patience_left = cfg.ai.early_stopping_patience
                        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_miou": miou}, best_ckpt_path)
                    else:
                        patience_left -= 1

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "history": history,
                            "best_metric": best_metric,
                        },
                        last_ckpt_path,
                    )
                    self._update_session_progress(session_id, epoch, history, best_metric)
                    if patience_left <= 0:
                        break

                final_metrics = {
                    "val_miou": round(best_metric, 4),
                    "val_pixel_acc": history[-1]["val_pixel_acc"] if history else 0.0,
                    "val_loss": history[-1]["val_loss"] if history else 0.0,
                }

            elif model_family == "classification":
                grp_wmo = bool(getattr(cfg.ai, "group_ambiguous_wmo", False))
                tr_ds = _ClsDataset(train_items, augment=True, group_ambiguous_wmo=grp_wmo)
                va_ds = _ClsDataset(val_items, augment=False, group_ambiguous_wmo=grp_wmo)
                if len(tr_ds) == 0:
                    raise ValueError("No cloud classification labeled items found in dataset")
                num_cls = len(tr_ds.class_names)
                counts = np.bincount([s[2] for s in tr_ds.samples], minlength=num_cls)
                weights = np.where(counts > 0, len(tr_ds) / (np.maximum(counts, 1) * np.count_nonzero(counts)), 1.0)
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=torch_dev))

                n_workers = 0 if sys.platform == "win32" else (int(getattr(cfg.ai, "dataloader_num_workers", 2) or 0) if len(tr_ds) >= 128 else 0)
                pin_mem = device == "cuda"
                tr_loader = DataLoader(tr_ds, batch_size=max(1, batch_size), shuffle=True, num_workers=n_workers, pin_memory=pin_mem)
                va_loader = DataLoader(va_ds, batch_size=max(1, batch_size), shuffle=False, num_workers=n_workers, pin_memory=pin_mem)

                for epoch in range(start_epoch + 1, epochs + 1):
                    if self._should_stop(job_id):
                        break
                    model.train()
                    tr_loss = 0.0
                    for imgs, feats, labels in tr_loader:
                        imgs, feats, labels = imgs.to(torch_dev), feats.to(torch_dev), labels.to(torch_dev)
                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits = model(imgs, feats)
                            loss = criterion(logits, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        tr_loss += float(loss.item())
                        if self._should_stop(job_id):
                            break
                    if self._should_stop(job_id):
                        break
                    scheduler.step()

                    model.eval()
                    va_loss = 0.0
                    all_preds: list[int] = []
                    all_true: list[int] = []
                    with torch.no_grad():
                        for imgs, feats, labels in va_loader:
                            imgs, feats, labels = imgs.to(torch_dev), feats.to(torch_dev), labels.to(torch_dev)
                            logits = model(imgs, feats)
                            va_loss += float(criterion(logits, labels).item())
                            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                            all_true.extend(labels.cpu().tolist())

                    acc = float(accuracy_score(all_true, all_preds)) if all_true else 0.0
                    f1 = float(f1_score(all_true, all_preds, average="macro", zero_division=0)) if all_true else 0.0
                    prec = float(precision_score(all_true, all_preds, average="macro", zero_division=0)) if all_true else 0.0
                    rec = float(recall_score(all_true, all_preds, average="macro", zero_division=0)) if all_true else 0.0

                    ep_row = {
                        "epoch": epoch,
                        "train_loss": round(tr_loss / max(len(tr_loader), 1), 4),
                        "val_loss": round(va_loss / max(len(va_loader), 1), 4),
                        "val_accuracy": round(acc, 4),
                        "val_macro_f1": round(f1, 4),
                    }
                    history.append(ep_row)
                    score = 0.5 * acc + 0.5 * f1
                    if score >= best_metric:
                        best_metric = score
                        patience_left = cfg.ai.early_stopping_patience
                        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_f1": f1}, best_ckpt_path)
                        present_labels = sorted(set(all_true) | set(all_preds))
                        cm = confusion_matrix(all_true, all_preds, labels=present_labels).tolist() if present_labels else []
                        conf_mat_payload = {
                            "labels": [tr_ds.class_names[i] for i in present_labels],
                            "matrix": cm,
                        }
                        final_metrics = {
                            "val_accuracy": round(acc, 4),
                            "val_macro_f1": round(f1, 4),
                            "val_precision": round(prec, 4),
                            "val_recall": round(rec, 4),
                        }
                    else:
                        patience_left -= 1

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "history": history,
                            "best_metric": best_metric,
                        },
                        last_ckpt_path,
                    )
                    self._update_session_progress(session_id, epoch, history, best_metric)
                    if patience_left <= 0:
                        break

            elif model_family in {"temporal", "event"}:
                tr_seq = _build_temporal_sequences(train_items)
                va_seq = _build_temporal_sequences(val_items) or tr_seq
                if not tr_seq:
                    raise ValueError("Not enough sequential frames (need >=4 consecutive frames per group) for sequence training")

                X_tr = torch.tensor(np.stack([s[0] for s in tr_seq]), dtype=torch.float32)
                Y_tr = torch.tensor(np.stack([s[1] for s in tr_seq]), dtype=torch.float32)
                E_tr = torch.tensor([s[2] for s in tr_seq], dtype=torch.long)

                X_va = torch.tensor(np.stack([s[0] for s in va_seq]), dtype=torch.float32)
                Y_va = torch.tensor(np.stack([s[1] for s in va_seq]), dtype=torch.float32)
                E_va = torch.tensor([s[2] for s in va_seq], dtype=torch.long)

                mse = nn.MSELoss()
                bce = nn.BCEWithLogitsLoss()
                ce = nn.CrossEntropyLoss()

                for epoch in range(start_epoch + 1, epochs + 1):
                    if self._should_stop(job_id):
                        break
                    model.train()
                    optimizer.zero_grad()
                    xb = X_tr.to(torch_dev)
                    if model_family == "temporal":
                        out = model(xb)
                        yb = Y_tr.to(torch_dev)
                        cov_pred = torch.sigmoid(out[:, :, 0])
                        loss = mse(cov_pred, yb[:, :, 0]) + 0.5 * bce(out[:, :, 1:], yb[:, :, 1:])
                    else:
                        out = model(xb)
                        loss = ce(out, E_tr.to(torch_dev))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    model.eval()
                    with torch.no_grad():
                        xv = X_va.to(torch_dev)
                        if model_family == "temporal":
                            vout = model(xv)
                            yv = Y_va.to(torch_dev)
                            vcov = torch.sigmoid(vout[:, :, 0])
                            vloss = float((mse(vcov, yv[:, :, 0]) + 0.5 * bce(vout[:, :, 1:], yv[:, :, 1:])).item())
                            mae_pct = float(torch.mean(torch.abs(vcov - yv[:, :, 0])).item() * 100.0)
                            score = 1.0 / (1.0 + mae_pct)
                            ep_row = {
                                "epoch": epoch,
                                "train_loss": round(float(loss.item()), 4),
                                "val_loss": round(vloss, 4),
                                "val_mae_pct": round(mae_pct, 2),
                            }
                            final_metrics = {"val_mae_pct": round(mae_pct, 2), "val_loss": round(vloss, 4), "experimental": True}
                        else:
                            vout = model(xv)
                            vloss = float(ce(vout, E_va.to(torch_dev)).item())
                            preds = torch.argmax(vout, dim=1).cpu().tolist()
                            trues = E_va.tolist()
                            acc = float(accuracy_score(trues, preds))
                            f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
                            score = 0.5 * acc + 0.5 * f1
                            ep_row = {
                                "epoch": epoch,
                                "train_loss": round(float(loss.item()), 4),
                                "val_loss": round(vloss, 4),
                                "val_accuracy": round(acc, 4),
                                "val_macro_f1": round(f1, 4),
                            }
                            final_metrics = {"val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4)}

                    history.append(ep_row)
                    if score >= best_metric:
                        best_metric = score
                        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_ckpt_path)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "history": history,
                            "best_metric": best_metric,
                        },
                        last_ckpt_path,
                    )
                    self._update_session_progress(session_id, epoch, history, best_metric)
            else:
                raise ValueError(f"Unsupported model_family: {model_family}")

            was_stopped = self._should_stop(job_id)
            with get_db_session() as db:
                sess = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
                assert sess is not None
                sess.status = "stopped" if was_stopped else "completed"
                sess.finished_at = datetime.now(UTC)
                sess.best_val_metric = round(best_metric, 4)

                if best_ckpt_path.exists():
                    existing_count = db.query(ModelRegistry).filter(ModelRegistry.model_family == model_family).count()
                    ver = f"v1.{existing_count}.0"
                    register_and_maybe_activate(
                        db=db,
                        model_id=f"{model_family}-{ver}-{job_id[-6:]}",
                        model_family=model_family,
                        architecture=arch_name,
                        version=ver,
                        weights_path=best_ckpt_path,
                        metrics=final_metrics,
                        confusion_matrix=conf_mat_payload,
                        training_session_id=session_id,
                        activate=auto_activate,
                        notes=f"Trained on dataset {dataset_id} ({len(train_items)} train / {len(val_items)} val items)",
                    )
        except Exception as exc:
            # Automatic GPU/DirectML fallback: if any non-CPU accelerator fails with OOM,
            # DXGI_ERROR_DEVICE_SUSPENDED, unsupported op, or locale decode error, retry cleanly on CPU.
            if device != "cpu" and not isinstance(exc, ValueError):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                is_oom = "out of memory" in str(exc).lower()
                self._run_job(
                    job_id=job_id,
                    session_id=session_id,
                    model_family=model_family,
                    dataset_id=dataset_id,
                    epochs=epochs,
                    batch_size=max(1, batch_size // 2) if is_oom else batch_size,
                    lr=lr,
                    resume=False,
                    auto_activate=auto_activate,
                    warm_start_weights=warm_start_weights,
                    device_override="cpu",
                )
                return
            with get_db_session() as db:
                sess = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
                if sess:
                    sess.status = "failed"
                    sess.error_message = str(exc)
                    sess.finished_at = datetime.now(UTC)
        finally:
            gc.collect()

    def _update_session_progress(
        self, session_id: int, epoch: int, history: list[dict[str, Any]], best_metric: float
    ) -> None:
        with get_db_session() as db:
            sess = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
            if sess:
                sess.current_epoch = epoch
                sess.best_val_metric = round(best_metric, 4)
                sess.metrics_history_json = json.dumps(history)


_training_manager = TrainingManager()


def get_training_manager() -> TrainingManager:
    return _training_manager
