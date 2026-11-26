"""Local Model Registry & Runtime Model Loader/Activation/Rollback manager."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import torch
from sqlalchemy.orm import Session

from app.ai.architectures import build_model_for_family
from app.ai.device import resolve_device, to_torch_device
from app.config import get_config_manager
from app.models_db import ModelRegistry, TrainingSession

BASELINE_DEFAULT_MARKER = "[baseline-default]"


def is_default_baseline_model(rec: ModelRegistry) -> bool:
    """Return True if this checkpoint is a protected default/baseline model (never wiped by 'purge trained models')."""
    notes_l = (rec.notes or "").lower()
    mid_l = (rec.model_id or "").lower()
    return (
        BASELINE_DEFAULT_MARKER in notes_l
        or mid_l.endswith("-default")
        or "-baseline-" in mid_l
    )


class ActiveModelCache:
    """Thread-safe in-memory cache of active PyTorch models per model_family."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._models: dict[str, torch.nn.Module] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._devices: dict[str, str] = {}
        self._device: str = "cpu"

    def refresh_from_db(self, db: Session) -> None:
        cfg = get_config_manager().get()
        device = resolve_device(cfg.ai.device, cfg.ai.max_gpu_memory_fraction)
        active_rows = db.query(ModelRegistry).filter(ModelRegistry.is_active.is_(True)).all()
        with self._lock:
            self._device = device
            self._models.clear()
            self._meta.clear()
            self._devices.clear()
            for row in active_rows:
                weights = Path(row.weights_path)
                if not weights.exists():
                    continue
                try:
                    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
                    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                    inferred_nc: int | None = None
                    inferred_h = 3
                    inferred_arch = row.architecture
                    if row.model_family == "segmentation" and isinstance(state, dict):
                        if "net.classifier.4.weight" in state:
                            inferred_nc = int(state["net.classifier.4.weight"].shape[0])
                            inferred_arch = "DeepLabV3-MobileNetV3"
                        elif "probe_norm.weight" in state and "head.weight" in state:
                            inferred_nc = int(state["head.weight"].shape[0])
                            inferred_arch = "DINOv3-LinearProbe"
                        elif "head.weight" in state:
                            inferred_nc = int(state["head.weight"].shape[0])
                            if any(k.startswith("aspp.") for k in state):
                                inferred_arch = "CloudSegUNet-ConvNeXt-ASPP"
                    elif row.model_family == "classification" and isinstance(state, dict):
                        if "head.3.weight" in state:
                            inferred_nc = int(state["head.3.weight"].shape[0])
                            if "vis_proj.0.weight" in state and int(state["vis_proj.0.weight"].shape[1]) == 768:
                                inferred_arch = "CloudTypeClassifier-ConvNeXt"
                            else:
                                inferred_arch = "CloudTypeClassifier-EfficientNetB0"
                        elif "classifier.3.weight" in state:
                            inferred_nc = int(state["classifier.3.weight"].shape[0])
                            inferred_arch = "CloudTypeClassifier-Fusion"
                    elif row.model_family == "temporal" and isinstance(state, dict) and "head.2.weight" in state:
                        inferred_h = max(1, int(state["head.2.weight"].shape[0]) // 6)
                    model, arch_name = build_model_for_family(
                        row.model_family,
                        inferred_arch,
                        num_classes=inferred_nc,
                        pretrained=False,
                        num_horizons=inferred_h,
                    )
                    model.load_state_dict(state)
                    # Route GRU-based temporal/event inference on CPU when using DirectML for DX12 stability
                    use_cpu_dml = device == "directml" and row.model_family in {"temporal", "event"}
                    model_dev = "cpu" if use_cpu_dml else device
                    try:
                        model.to(to_torch_device(model_dev))
                    except Exception:
                        model_dev = "cpu"
                        model.to(torch.device("cpu"))
                    model.eval()
                    self._models[row.model_family] = model
                    self._devices[row.model_family] = model_dev
                    self._meta[row.model_family] = {
                        "model_id": row.model_id,
                        "model_family": row.model_family,
                        "architecture": arch_name,
                        "version": row.version,
                        "weights_path": row.weights_path,
                        "metrics": json.loads(row.metrics_json or "{}"),
                        "device": model_dev,
                    }
                except Exception:
                    continue

    def get(self, family: str) -> tuple[torch.nn.Module | None, dict[str, Any] | None, str]:
        with self._lock:
            return self._models.get(family), self._meta.get(family), self._devices.get(family, self._device)

    def active_summary(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._meta)


_model_cache = ActiveModelCache()


def get_model_cache() -> ActiveModelCache:
    return _model_cache


def register_and_maybe_activate(
    db: Session,
    model_id: str,
    model_family: str,
    architecture: str,
    version: str,
    weights_path: Path,
    metrics: dict[str, Any],
    confusion_matrix: dict[str, Any] | None = None,
    training_session_id: int | None = None,
    activate: bool = True,
    notes: str = "",
) -> ModelRegistry:
    if activate:
        db.query(ModelRegistry).filter(ModelRegistry.model_family == model_family).update({"is_active": False})

    rec = db.query(ModelRegistry).filter(ModelRegistry.model_id == model_id).first()
    if rec:
        rec.weights_path = str(weights_path)
        rec.metrics_json = json.dumps(metrics)
        rec.confusion_matrix_json = json.dumps(confusion_matrix or {})
        rec.is_active = activate
        rec.notes = notes
    else:
        rec = ModelRegistry(
            model_id=model_id,
            model_family=model_family,
            architecture=architecture,
            version=version,
            weights_path=str(weights_path),
            training_session_id=training_session_id,
            metrics_json=json.dumps(metrics),
            confusion_matrix_json=json.dumps(confusion_matrix or {}),
            is_active=activate,
            notes=notes,
        )
        db.add(rec)
    db.commit()
    db.refresh(rec)
    if activate:
        _model_cache.refresh_from_db(db)
    return rec


def activate_model_by_id(db: Session, model_id: str) -> ModelRegistry:
    target = db.query(ModelRegistry).filter(ModelRegistry.model_id == model_id).first()
    if not target:
        raise ValueError(f"Model {model_id} not found")
    if not Path(target.weights_path).exists():
        raise FileNotFoundError(f"Weights file missing: {target.weights_path}")
    db.query(ModelRegistry).filter(ModelRegistry.model_family == target.model_family).update({"is_active": False})
    target.is_active = True
    db.commit()
    db.refresh(target)
    _model_cache.refresh_from_db(db)
    return target


def rollback_model_family(db: Session, model_family: str) -> ModelRegistry:
    rows = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_family == model_family)
        .order_by(ModelRegistry.created_at.desc(), ModelRegistry.id.desc())
        .all()
    )
    if len(rows) < 2:
        raise ValueError(f"No previous version available to rollback for {model_family}")
    active_idx = next((i for i, r in enumerate(rows) if r.is_active), 0)
    prev_idx = active_idx + 1 if active_idx + 1 < len(rows) else 1
    target = rows[prev_idx]
    return activate_model_by_id(db, target.model_id)


def delete_model_by_id(db: Session, model_id: str, force: bool = False) -> dict[str, Any]:
    """Delete a single registered model and its weights file on disk (protects active & default baseline models)."""
    target = db.query(ModelRegistry).filter(ModelRegistry.model_id == model_id).first()
    if not target:
        raise ValueError(f"Model {model_id} not found")
    if target.is_active and not force:
        raise PermissionError(f"Cannot delete active model '{model_id}'. Activate another model first.")
    if is_default_baseline_model(target) and not force:
        raise PermissionError(f"Cannot delete default baseline model '{model_id}'.")

    weights_path = target.weights_path
    sess_id = target.training_session_id
    db.delete(target)
    db.flush()

    deleted_file = False
    if weights_path:
        other_ref = db.query(ModelRegistry).filter(ModelRegistry.weights_path == weights_path).count()
        if other_ref == 0:
            wp = Path(weights_path)
            if wp.exists() and wp.is_file():
                try:
                    wp.unlink(missing_ok=True)
                    deleted_file = True
                except Exception:
                    pass

    if sess_id is not None:
        other_sess = db.query(ModelRegistry).filter(ModelRegistry.training_session_id == sess_id).count()
        if other_sess == 0:
            sess = db.query(TrainingSession).filter(TrainingSession.id == sess_id).first()
            if sess and sess.status not in {"running", "queued"}:
                db.delete(sess)

    db.commit()
    _model_cache.refresh_from_db(db)
    return {"deleted": True, "model_id": model_id, "deleted_file": deleted_file}


def prune_old_models(
    db: Session,
    keep_per_family: int = 1,
    model_family: str | None = None,
    also_prune_sessions: bool = True,
) -> dict[str, Any]:
    """
    Delete old standby models while keeping the active model (plus up to keep_per_family - 1 latest backups)
    per model family, and remove unreferenced weight files from disk.
    """
    keep_n = max(1, int(keep_per_family))
    families = [model_family] if model_family else ["segmentation", "classification", "temporal", "event"]
    deleted_models = 0
    deleted_files = 0
    deleted_ids: list[str] = []

    for fam in families:
        rows = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_family == fam)
            .order_by(ModelRegistry.is_active.desc(), ModelRegistry.created_at.desc(), ModelRegistry.id.desc())
            .all()
        )
        to_remove = [
            r
            for i, r in enumerate(rows)
            if i >= keep_n and not r.is_active and not is_default_baseline_model(r)
        ]
        for r in to_remove:
            wp_str = r.weights_path
            deleted_ids.append(r.model_id)
            db.delete(r)
            deleted_models += 1
            db.flush()
            if wp_str:
                still_used = db.query(ModelRegistry).filter(ModelRegistry.weights_path == wp_str).count()
                if still_used == 0:
                    wp = Path(wp_str)
                    if wp.exists() and wp.is_file():
                        try:
                            wp.unlink(missing_ok=True)
                            deleted_files += 1
                        except Exception:
                            pass

    deleted_sessions = 0
    if also_prune_sessions:
        kept_sess_ids = {
            int(r.training_session_id)
            for r in db.query(ModelRegistry).all()
            if r.training_session_id is not None
        }
        old_sessions = (
            db.query(TrainingSession)
            .filter(TrainingSession.status.notin_(["running", "queued"]))
            .all()
        )
        for s in old_sessions:
            if s.id not in kept_sess_ids:
                if s.checkpoint_path:
                    cp = Path(s.checkpoint_path)
                    if cp.exists() and cp.is_file():
                        still_ref = db.query(ModelRegistry).filter(ModelRegistry.weights_path == str(cp)).count()
                        if still_ref == 0:
                            try:
                                cp.unlink(missing_ok=True)
                                deleted_files += 1
                            except Exception:
                                pass
                db.delete(s)
                deleted_sessions += 1

    # Also remove any orphaned .pt files in models_dir not referenced by any kept ModelRegistry row
    cfg = get_config_manager().get()
    if cfg.models_dir.exists():
        kept_paths = {
            str(Path(r.weights_path).resolve())
            for r in db.query(ModelRegistry).all()
            if r.weights_path
        }
        for pt_file in list(cfg.models_dir.rglob("*.pt")):
            if pt_file.is_file():
                try:
                    if str(pt_file.resolve()) not in kept_paths:
                        pt_file.unlink(missing_ok=True)
                        deleted_files += 1
                except Exception:
                    pass

    db.commit()
    _model_cache.refresh_from_db(db)
    remaining = int(db.query(ModelRegistry).count())
    return {
        "deleted_models": deleted_models,
        "deleted_model_ids": deleted_ids,
        "deleted_files": deleted_files,
        "deleted_sessions": deleted_sessions,
        "remaining_models": remaining,
    }
