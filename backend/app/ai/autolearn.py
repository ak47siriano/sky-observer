"""Local Auto-Labeling & Warm-Start Auto-Training Engine for site-specific model reinforcement."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.ai.datasets import _assign_group_splits, bootstrap_local_sky_dataset
from app.ai.registry import activate_model_by_id
from app.ai.trainer import get_training_manager
from app.config import AppConfig, get_config_manager
from app.database import get_db_session
from app.models_db import (
    DatasetItem,
    DatasetRecord,
    EventAlert,
    ModelRegistry,
    Observation,
    TrainingSession,
    safe_json_loads,
)
from app.vision.classification import CLOUD_TYPE_CLASSES
from app.vision.masking import build_valid_sky_mask_from_config, neutralize_excluded_pixels
from app.vision.segmentation import NOT_SKY_IGNORE_INDEX, _radiometric_probabilities

AUTO_DATASET_NAME = "location_auto_reinforce"
FIXED_HOLDOUT_FILENAME = "fixed_holdout.json"
GOLDEN_DATASET_DIRNAME = "golden_eval"


def _load_golden_set_excluded_ids(config: AppConfig) -> tuple[set[int], set[str]]:
    """Load observation IDs and image names locked inside data/datasets/golden_eval/manifest.json."""
    obs_ids: set[int] = set()
    img_names: set[str] = set()
    manifest_p = config.datasets_dir / GOLDEN_DATASET_DIRNAME / "manifest.json"
    if not manifest_p.exists():
        return obs_ids, img_names
    try:
        raw = json.loads(manifest_p.read_text(encoding="utf-8"))
        for it in raw.get("items", []) or []:
            if it.get("observation_id") is not None:
                obs_ids.add(int(it["observation_id"]))
            if it.get("source_raw_path"):
                img_names.add(Path(str(it["source_raw_path"])).name)
    except Exception:
        pass
    return obs_ids, img_names


def enforce_pseudo_label_ratio(
    items_meta: list[dict[str, Any]],
    max_pseudo_fraction: float = 0.50,
) -> list[dict[str, Any]]:
    """
    Cap auto/pseudo-labeled training items to at most `max_pseudo_fraction` (<= 50%) of the
    training split whenever anchor items (human-labeled or baseline replay) exist.
    Validation/test items are never dropped.
    """
    cap = float(np.clip(max_pseudo_fraction, 0.10, 0.50))
    anchors_train: list[dict[str, Any]] = []
    pseudo_train: list[dict[str, Any]] = []
    non_train: list[dict[str, Any]] = []

    for it in items_meta:
        if it.get("split") != "train":
            non_train.append(it)
        elif it.get("user_labeled") or str(it.get("group_key", "")).startswith("replay_"):
            anchors_train.append(it)
        else:
            pseudo_train.append(it)

    if not anchors_train or not pseudo_train:
        return items_meta

    # Ensure len(pseudo) / (len(anchors) + len(pseudo)) <= cap  =>  len(pseudo) <= len(anchors) * cap / (1 - cap)
    max_pseudo = max(1, int(len(anchors_train) * (cap / max(1e-6, 1.0 - cap))))
    if len(pseudo_train) > max_pseudo:
        idxs = np.linspace(0, len(pseudo_train) - 1, num=max_pseudo, dtype=int)
        pseudo_train = [pseudo_train[int(i)] for i in sorted(set(idxs.tolist()))]
    return anchors_train + pseudo_train + non_train


def should_promote_checkpoint(
    candidate_metric: float | None,
    active_metric: float | None,
    tolerance: float = 0.01,
    candidate_by_category: dict[str, float] | None = None,
    active_by_category: dict[str, float] | None = None,
    max_category_regression: float = 0.03,
) -> tuple[bool, str]:
    """
    Anti-drift promotion gate:
      1. If no active model exists, promote initial checkpoint.
      2. Overall validation metric must be >= active_metric - tolerance.
      3. No individual sky condition category (clear/partial/overcast/twilight) may regress
         by more than `max_category_regression`.
    """
    if active_metric is None:
        return True, "promoted_initial"
    if candidate_metric is None:
        return False, "rejected_missing_metric"
    if candidate_metric < (active_metric - tolerance):
        return False, f"rejected_overall_regression ({candidate_metric:.4f} < {active_metric:.4f} - {tolerance:.3f})"
    if candidate_by_category and active_by_category:
        for cat, prev_val in active_by_category.items():
            if cat in candidate_by_category:
                cur_val = float(candidate_by_category[cat])
                if cur_val < float(prev_val) - max_category_regression:
                    return (
                        False,
                        f"rejected_category_regression:{cat} ({cur_val:.4f} < {float(prev_val):.4f} - {max_category_regression:.3f})",
                    )
    return True, f"promoted_metric_ok ({candidate_metric:.4f} >= {active_metric:.4f} - {tolerance:.3f})"


def _load_or_update_fixed_holdout(
    root: Path,
    items_meta: list[dict[str, Any]],
    holdout_fraction: float = 0.20,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Maintain a deterministic, append-only fixed holdout manifest (`fixed_holdout.json`)
    so validation samples NEVER leak into training across Auto-Train cycles (anti-drift guard).
    Prioritizes human-labeled samples (`user_labeled=True`) when seeding the holdout.
    """
    manifest_path = root / FIXED_HOLDOUT_FILENAME
    locked_keys: set[str] = set()
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("locked_keys"), list):
                locked_keys = {str(k) for k in raw["locked_keys"]}
        except Exception:
            locked_keys = set()

    def _item_key(it: dict[str, Any]) -> str:
        if it.get("observation_id") is not None:
            return f"obs:{it['observation_id']}"
        return f"path:{Path(str(it.get('image_path', ''))).name}"

    # Apply standard group splits first, then lock deterministic holdout keys to 'val'
    items_meta = _assign_group_splits(items_meta)
    local_items = [it for it in items_meta if not str(it.get("group_key", "")).startswith("replay_")]
    target_holdout = max(1, int(round(len(local_items) * float(np.clip(holdout_fraction, 0.05, 0.50)))))

    present_locked = [it for it in local_items if _item_key(it) in locked_keys]
    if len(present_locked) < target_holdout and len(local_items) >= 2:
        # Prefer user-labeled frames, then evenly spaced local frames across time
        candidates = [it for it in local_items if _item_key(it) not in locked_keys]
        user_cands = [it for it in candidates if it.get("user_labeled")]
        other_cands = [it for it in candidates if not it.get("user_labeled")]
        needed = target_holdout - len(present_locked)
        picked: list[dict[str, Any]] = []
        for it in user_cands[: max(0, min(needed, len(user_cands)))]:
            picked.append(it)
        rem = needed - len(picked)
        if rem > 0 and other_cands:
            idxs = np.linspace(0, len(other_cands) - 1, num=min(rem, len(other_cands)), dtype=int)
            for idx in sorted(set(idxs.tolist())):
                picked.append(other_cands[int(idx)])
        for it in picked:
            locked_keys.add(_item_key(it))

    # Enforce: locked holdout items are ALWAYS val; ensure at least 1 train item remains
    train_count = 0
    for it in items_meta:
        k = _item_key(it)
        if k in locked_keys:
            it["split"] = "val"
        elif it.get("split") == "train":
            train_count += 1
    if train_count == 0 and items_meta:
        items_meta[0]["split"] = "train"

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now(UTC).isoformat(),
                    "locked_keys": sorted(locked_keys),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    return items_meta, sorted(locked_keys)


def _extract_primary_metric(family: str, metrics: dict[str, Any]) -> float | None:
    if not isinstance(metrics, dict) or not metrics:
        return None
    if family == "segmentation":
        val = metrics.get("val_miou")
        return float(val) if val is not None else None
    if family in {"classification", "event"}:
        acc = metrics.get("val_accuracy")
        f1 = metrics.get("val_macro_f1")
        if acc is not None and f1 is not None:
            return 0.5 * float(acc) + 0.5 * float(f1)
        if acc is not None:
            return float(acc)
        return None
    if family == "temporal":
        mae = metrics.get("val_mae_pct")
        if mae is not None:
            return 1.0 / (1.0 + max(0.0, float(mae)))
    return None

_EVENT_TYPE_TO_LABEL = {
    "storm_like_development": "storm_like_structure",
    "rapid_cloud_increase": "rapid_cloud_development",
    "rapid_sky_clearing": "sky_clearing",
    "rapid_sky_darkening": "dark_cloud_development",
    "dense_cloud_region": "cloud_cover_increasing",
    "camera_obstruction": "camera_obstruction",
}


def _derive_cloud_genus_label(obs: Observation) -> str:
    if (
        obs.dominant_cloud_type in CLOUD_TYPE_CLASSES
        and obs.dominant_cloud_type != "unknown"
        and obs.classification_confidence >= 0.40
    ):
        return obs.dominant_cloud_type
    if obs.cloud_coverage_pct < 12.0:
        return "clear_sky"
    if obs.dark_cloud_pct >= 28.0:
        return "cumulonimbus_like"
    if obs.cloud_coverage_pct >= 85.0:
        return "stratus"
    if obs.thin_cloud_pct > obs.dense_cloud_pct * 1.4:
        return "cirrus"
    if obs.dense_cloud_pct >= 45.0:
        return "stratocumulus"
    return "cumulus"


def auto_label_local_observations(
    db: Session,
    config: AppConfig,
    max_samples: int | None = None,
    min_quality: float | None = None,
    include_baseline_replay: bool | None = None,
) -> dict[str, Any]:
    """
    Build/refresh the site-specific `location_auto_reinforce` training dataset using
    real captured local camera frames, active exclusion masks, verified user Confirm/Deny
    event feedback, manual user annotations, and a compact baseline replay anchor.
    """
    at_cfg = config.ai.auto_train
    limit_samples = max_samples if max_samples is not None else at_cfg.max_local_samples
    qual_thresh = min_quality if min_quality is not None else at_cfg.min_quality_score
    use_replay = include_baseline_replay if include_baseline_replay is not None else at_cfg.include_baseline_replay

    candidates = (
        db.query(Observation)
        .filter(Observation.is_usable.is_(True), Observation.quality_score >= qual_thresh)
        .order_by(Observation.captured_at.desc(), Observation.id.desc())
        .all()
    )

    _QC_EXCLUDE_FLAGS = {
        "window_reflection",
        "ae_awb_jump",
        "raindrop_smudge",
        "heavy_jpeg_blocking",
        "camera_movement",
    }
    golden_obs_ids, golden_img_names = _load_golden_set_excluded_ids(config)
    seen_hashes: set[str] = set()
    unique_valid: list[Observation] = []
    for obs in candidates:
        if obs.id in golden_obs_ids:
            continue
        if not obs.raw_path or not Path(obs.raw_path).exists():
            continue
        if Path(obs.raw_path).name in golden_img_names:
            continue
        q_meta = safe_json_loads(obs.quality_metrics_json, {})
        q_flags = set(q_meta.get("flags") or [])
        if q_flags & _QC_EXCLUDE_FLAGS:
            continue
        fhash = (obs.frame_hash or "").strip()
        if fhash:
            if fhash in seen_hashes:
                continue
            seen_hashes.add(fhash)
        unique_valid.append(obs)

    if not unique_valid:
        raise ValueError("No usable local camera observations available on disk yet.")

    if len(unique_valid) > limit_samples:
        step = np.linspace(0, len(unique_valid) - 1, num=limit_samples, dtype=int)
        selected = [unique_valid[int(i)] for i in sorted(set(step.tolist()))]
    else:
        selected = list(unique_valid)

    # Sort chronologically ascending so temporal sequences reflect real progression
    selected.sort(key=lambda o: (o.captured_at, o.id))

    obs_ids = [o.id for o in selected]
    ev_rows = (
        db.query(EventAlert)
        .filter(EventAlert.observation_id.in_(obs_ids))
        .order_by(EventAlert.id.desc())
        .all()
    )
    events_by_obs: dict[int, list[EventAlert]] = {}
    for ev in ev_rows:
        if ev.observation_id is not None:
            events_by_obs.setdefault(ev.observation_id, []).append(ev)

    # Map existing manual user annotations so we never discard explicit human corrections
    user_ds = db.query(DatasetRecord).filter(DatasetRecord.name == "user_camera_labels").first()
    user_by_obs: dict[int, DatasetItem] = {}
    extra_user_items: list[DatasetItem] = []
    selected_id_set = set(obs_ids)
    if user_ds:
        for ui in db.query(DatasetItem).filter(DatasetItem.dataset_id == user_ds.id).all():
            if not Path(ui.image_path).exists():
                continue
            if ui.observation_id and ui.observation_id in selected_id_set:
                user_by_obs[ui.observation_id] = ui
            else:
                extra_user_items.append(ui)

    root = config.datasets_dir / AUTO_DATASET_NAME
    img_dir = root / "images"
    msk_dir = root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    items_meta: list[dict[str, Any]] = []
    local_count = 0
    user_labeled_count = 0
    confirmed_events_used = 0
    max_obs_id = max(obs_ids)

    prev_obs: Observation | None = None
    for idx, obs in enumerate(selected):
        bgr = cv2.imread(obs.raw_path)
        if bgr is None:
            continue

        h, w = bgr.shape[:2]
        scale = min(1.0, 512.0 / max(h, w))
        save_w = max(64, int(round(w * scale)))
        save_h = max(64, int(round(h * scale)))

        valid_mask = build_valid_sky_mask_from_config((h, w), config.camera)
        clean_bgr = neutralize_excluded_pixels(bgr, valid_mask)
        train_bgr = cv2.resize(clean_bgr, (save_w, save_h), interpolation=cv2.INTER_AREA)

        valid_small = cv2.resize(valid_mask, (save_w, save_h), interpolation=cv2.INTER_NEAREST)
        physics_small = np.argmax(
            _radiometric_probabilities(cv2.resize(clean_bgr, (256, 256), interpolation=cv2.INTER_AREA)),
            axis=-1,
        ).astype(np.uint8)
        physics_resized = cv2.resize(physics_small, (save_w, save_h), interpolation=cv2.INTER_NEAREST)

        manual_item = user_by_obs.get(obs.id)
        if manual_item and manual_item.mask_path and Path(manual_item.mask_path).exists():
            m_disk = cv2.imread(manual_item.mask_path, cv2.IMREAD_GRAYSCALE)
            if m_disk is not None:
                m_res = cv2.resize(m_disk, (save_w, save_h), interpolation=cv2.INTER_NEAREST)
                pseudo_mask = np.where(
                    m_res == NOT_SKY_IGNORE_INDEX,
                    NOT_SKY_IGNORE_INDEX,
                    np.where(m_res > 5, np.where(m_res > 127, 2, 0), np.where(m_res == 5, physics_resized, m_res)),
                ).astype(np.uint8)
            else:
                pseudo_mask = physics_resized
        else:
            # Always use freshly calibrated radiometric physics for unlabeled camera frames
            # rather than recycling stale predicted masks (prevents self-reinforcing model collapse).
            pseudo_mask = physics_resized

        pseudo_mask[valid_small == 0] = NOT_SKY_IGNORE_INDEX

        dst_img = img_dir / f"loc_obs_{obs.id}.jpg"
        dst_msk = msk_dir / f"loc_obs_{obs.id}.png"
        cv2.imwrite(str(dst_img), train_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(dst_msk), pseudo_mask)

        # Resolve event label honoring user Confirm / Deny feedback first
        obs_events = events_by_obs.get(obs.id, [])
        confirmed_mapped = next(
            (
                _EVENT_TYPE_TO_LABEL[e.event_type]
                for e in obs_events
                if (e.user_status or ("confirmed" if e.acknowledged else "pending")) == "confirmed"
                and e.event_type in _EVENT_TYPE_TO_LABEL
            ),
            None,
        )
        all_rejected = bool(obs_events) and all(
            (e.user_status or "pending") == "rejected" for e in obs_events
        )

        if manual_item:
            cloud_lbl = manual_item.cloud_type_label or _derive_cloud_genus_label(obs)
            event_lbl = manual_item.event_label or "normal"
            quality_lbl = manual_item.quality_label or "usable"
            is_user_lbl = True
            user_labeled_count += 1
        else:
            cloud_lbl = _derive_cloud_genus_label(obs)
            quality_lbl = "usable"
            is_user_lbl = False
            if confirmed_mapped:
                event_lbl = confirmed_mapped
                confirmed_events_used += 1
            elif all_rejected:
                event_lbl = "normal"
            elif obs.dark_cloud_pct >= 35.0:
                event_lbl = "storm_like_structure"
            elif prev_obs is not None:
                cov_delta = float(obs.cloud_coverage_pct) - float(prev_obs.cloud_coverage_pct)
                lum_delta = float(obs.avg_luminance) - float(prev_obs.avg_luminance)
                if cov_delta >= 18.0:
                    event_lbl = "rapid_cloud_development"
                elif cov_delta <= -18.0:
                    event_lbl = "sky_clearing"
                elif lum_delta <= -15.0:
                    event_lbl = "dark_cloud_development"
                elif cov_delta >= 8.0:
                    event_lbl = "cloud_cover_increasing"
                else:
                    event_lbl = "normal"
            else:
                event_lbl = "normal"

        cov_gt = round(float(np.mean((pseudo_mask >= 1) & (pseudo_mask <= 3)) * 100.0), 2)
        items_meta.append({
            "observation_id": obs.id,
            "group_key": f"loc_seq_{idx // 6}",
            "sequence_index": idx % 6,
            "image_path": str(dst_img),
            "mask_path": str(dst_msk),
            "cloud_type_label": cloud_lbl,
            "event_label": event_lbl,
            "quality_label": quality_lbl,
            "cloud_coverage_gt": cov_gt,
            "user_labeled": is_user_lbl,
        })
        local_count += 1
        prev_obs = obs

    if local_count == 0:
        raise ValueError("Selected local observation images could not be read from disk.")

    for ui in extra_user_items:
        items_meta.append({
            "observation_id": ui.observation_id,
            "group_key": ui.group_key or "user_manual",
            "sequence_index": ui.sequence_index,
            "image_path": ui.image_path,
            "mask_path": ui.mask_path,
            "cloud_type_label": ui.cloud_type_label or "mixed_clouds",
            "event_label": ui.event_label or "normal",
            "quality_label": ui.quality_label or "usable",
            "cloud_coverage_gt": ui.cloud_coverage_gt,
            "user_labeled": True,
        })
        user_labeled_count += 1

    replay_count = 0
    if use_replay:
        base_ds = bootstrap_local_sky_dataset(db, config)
        base_items = (
            db.query(DatasetItem)
            .filter(DatasetItem.dataset_id == base_ds.id)
            .order_by(DatasetItem.group_key.asc(), DatasetItem.sequence_index.asc())
            .all()
        )
        # Keep up to 8 balanced scenario sequences (4 frames each) so temporal/event sequences stay intact
        by_scenario: dict[str, list[DatasetItem]] = {}
        for bi in base_items:
            if not Path(bi.image_path).exists():
                continue
            scen_key = bi.group_key.rsplit("_seq", 1)[0]
            if bi.group_key not in by_scenario and any(k.startswith(scen_key + "_seq") for k in by_scenario):
                continue
            grp = by_scenario.setdefault(bi.group_key, [])
            if len(grp) < 4:
                grp.append(bi)
            if len(by_scenario) >= 8 and all(len(v) >= 4 for v in by_scenario.values()):
                break

        for gk, seq in by_scenario.items():
            for ri in seq:
                items_meta.append({
                    "observation_id": None,
                    "group_key": f"replay_{gk}",
                    "sequence_index": ri.sequence_index,
                    "image_path": ri.image_path,
                    "mask_path": ri.mask_path,
                    "cloud_type_label": ri.cloud_type_label,
                    "event_label": ri.event_label,
                    "quality_label": ri.quality_label,
                    "cloud_coverage_gt": ri.cloud_coverage_gt,
                    "user_labeled": False,
                })
                replay_count += 1

    items_meta, locked_keys = _load_or_update_fixed_holdout(
        root=root,
        items_meta=items_meta,
        holdout_fraction=float(getattr(at_cfg, "fixed_holdout_fraction", 0.20)),
    )
    items_meta = enforce_pseudo_label_ratio(
        items_meta,
        max_pseudo_fraction=float(getattr(at_cfg, "max_pseudo_label_fraction", 0.50)),
    )

    existing = db.query(DatasetRecord).filter(DatasetRecord.name == AUTO_DATASET_NAME).first()
    if existing:
        db.query(DatasetItem).filter(DatasetItem.dataset_id == existing.id).delete()
        rec = existing
        rec.item_count = len(items_meta)
        rec.root_path = str(root)
        rec.source_type = "location_auto_labeled"
        rec.metadata_json = json.dumps({
            "updated_at": datetime.now(UTC).isoformat(),
            "local_items": local_count,
            "user_labeled_items": user_labeled_count,
            "confirmed_event_items": confirmed_events_used,
            "replay_items": replay_count,
            "fixed_holdout_count": len(locked_keys),
        })
    else:
        rec = DatasetRecord(
            name=AUTO_DATASET_NAME,
            version="1.0.0",
            source_type="location_auto_labeled",
            license_name="Local Camera Auto-Reinforcement",
            description="Auto-labeled site-specific sky observations merged with user corrections and replay anchors.",
            root_path=str(root),
            task_types_json=json.dumps(["segmentation", "classification", "temporal", "event"]),
            item_count=len(items_meta),
            integrity_verified=True,
            metadata_json=json.dumps({
                "updated_at": datetime.now(UTC).isoformat(),
                "local_items": local_count,
                "user_labeled_items": user_labeled_count,
                "confirmed_event_items": confirmed_events_used,
                "replay_items": replay_count,
                "fixed_holdout_count": len(locked_keys),
            }),
        )
        db.add(rec)
        db.flush()

    for it in items_meta:
        db.add(
            DatasetItem(
                dataset_id=rec.id,
                observation_id=it["observation_id"],
                split=it["split"],
                group_key=it["group_key"],
                sequence_index=it["sequence_index"],
                image_path=it["image_path"],
                mask_path=it["mask_path"],
                cloud_type_label=it["cloud_type_label"],
                event_label=it["event_label"],
                quality_label=it["quality_label"],
                cloud_coverage_gt=it["cloud_coverage_gt"],
                user_labeled=it["user_labeled"],
            )
        )
    db.commit()
    db.refresh(rec)

    return {
        "dataset_id": rec.id,
        "dataset_name": rec.name,
        "total_items": len(items_meta),
        "local_items": local_count,
        "user_labeled_items": user_labeled_count,
        "confirmed_event_items": confirmed_events_used,
        "replay_items": replay_count,
        "fixed_holdout_count": len(locked_keys),
        "latest_observation_id": max_obs_id,
    }


class AutoLearnManager:
    """Coordinates site-specific Auto-Label + Auto-Train cycles safely on background threads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_requested = False
        self._running = False
        self._stage: str = "idle"
        self._current_family: str | None = None
        self._last_run_at: str | None = None
        self._last_run_ts: float = 0.0
        self._last_obs_id: int = 0
        self._last_summary: dict[str, Any] | None = None
        self._last_jobs: list[dict[str, Any]] = []
        self._last_error: str = ""

    def get_status(self, db: Session, config: AppConfig) -> dict[str, Any]:
        at_cfg = config.ai.auto_train
        eligible_rows = (
            db.query(Observation)
            .filter(
                Observation.is_usable.is_(True),
                Observation.quality_score >= at_cfg.min_quality_score,
                Observation.raw_path != "",
            )
            .all()
        )
        eligible = sum(1 for r in eligible_rows if Path(r.raw_path).exists())
        new_since_last = sum(
            1 for r in eligible_rows if r.id > self._last_obs_id and Path(r.raw_path).exists()
        )
        with self._lock:
            return {
                "enabled": at_cfg.enabled,
                "config": at_cfg.model_dump(),
                "running": self._running,
                "stage": self._stage,
                "current_family": self._current_family,
                "last_run_at": self._last_run_at,
                "last_summary": self._last_summary,
                "last_jobs": list(self._last_jobs),
                "last_error": self._last_error,
                "eligible_local_observations": eligible,
                "new_since_last_run": new_since_last,
            }

    def request_stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_requested = True
        stopped_jobs = get_training_manager().request_stop()
        return {"stop_requested": True, "stopped_training_jobs": stopped_jobs}

    def trigger_run(
        self,
        families: list[str] | None = None,
        epochs: int | None = None,
        only_label: bool = False,
        async_run: bool = True,
    ) -> dict[str, Any]:
        cfg = get_config_manager().get()
        if not only_label and not cfg.ai.training_enabled:
            raise RuntimeError("Training is disabled in configuration.")

        with self._lock:
            if self._running:
                raise RuntimeError("Auto-train cycle is already running.")
            if get_training_manager().has_active_jobs():
                raise RuntimeError("Another training job is currently running.")

            with get_db_session() as db:
                has_usable = (
                    db.query(Observation)
                    .filter(
                        Observation.is_usable.is_(True),
                        Observation.quality_score >= cfg.ai.auto_train.min_quality_score,
                        Observation.raw_path != "",
                    )
                    .all()
                )
                if not any(Path(r.raw_path).exists() for r in has_usable):
                    raise ValueError("No usable local camera images captured yet for auto-training.")

            self._stop_requested = False
            self._running = True
            self._stage = "auto_labeling"
            self._current_family = None
            self._last_error = ""
            self._last_jobs = []

        target_families = list(families or cfg.ai.auto_train.families or ["segmentation", "classification"])
        run_epochs = epochs if epochs is not None else cfg.ai.auto_train.epochs

        if async_run:
            t = threading.Thread(
                target=self._execute_cycle,
                args=(target_families, run_epochs, only_label),
                name="sky-observer-autotrain",
                daemon=True,
            )
            t.start()
        else:
            self._execute_cycle(target_families, run_epochs, only_label)

        return {
            "started": True,
            "only_label": only_label,
            "families": target_families,
            "epochs": run_epochs,
            "async": async_run,
        }

    def _execute_cycle(self, target_families: list[str], run_epochs: int, only_label: bool) -> None:
        cfg = get_config_manager().get()
        try:
            with self._lock:
                self._stage = "auto_labeling"
                self._current_family = None

            with get_db_session() as db:
                summary = auto_label_local_observations(db, cfg)

            with self._lock:
                self._last_summary = summary
                self._last_obs_id = int(summary.get("latest_observation_id", self._last_obs_id))

            if only_label or self._stop_requested:
                with self._lock:
                    self._stage = "completed"
                    self._current_family = None
                    self._last_run_at = datetime.now(UTC).isoformat()
                    self._last_run_ts = time.time()
                    self._running = False
                return

            tm = get_training_manager()
            ds_id = int(summary["dataset_id"])
            jobs_out: list[dict[str, Any]] = []

            for fam in target_families:
                if self._stop_requested:
                    break
                with self._lock:
                    self._stage = "training"
                    self._current_family = fam

                warm_path: str | None = None
                prev_active_metric: float | None = None
                with get_db_session() as db:
                    active_rec = (
                        db.query(ModelRegistry)
                        .filter(ModelRegistry.model_family == fam, ModelRegistry.is_active.is_(True))
                        .first()
                    )
                    if active_rec:
                        prev_active_metric = _extract_primary_metric(
                            fam, safe_json_loads(active_rec.metrics_json, {})
                        )
                        if cfg.ai.auto_train.warm_start and Path(active_rec.weights_path).exists():
                            warm_path = active_rec.weights_path

                # Train candidate into standby first, then apply fixed-holdout anti-drift promotion gate
                res = tm.start_training(
                    model_family=fam,
                    dataset_id=ds_id,
                    epochs=run_epochs,
                    learning_rate=cfg.ai.auto_train.learning_rate,
                    auto_activate=False,
                    async_run=False,
                    warm_start_weights=warm_path,
                )
                promoted = False
                gate_reason = "no_checkpoint"
                with get_db_session() as db:
                    sess = db.query(TrainingSession).filter(TrainingSession.id == res["session_id"]).first()
                    cand = (
                        db.query(ModelRegistry)
                        .filter(ModelRegistry.training_session_id == res["session_id"])
                        .order_by(ModelRegistry.id.desc())
                        .first()
                    )
                    if cand is not None:
                        cand_metric = _extract_primary_metric(
                            fam, safe_json_loads(cand.metrics_json, {})
                        )
                        if cand_metric is None and sess is not None:
                            cand_metric = float(sess.best_val_metric or 0.0)
                        tol = float(getattr(cfg.ai.auto_train, "promotion_tolerance", 0.01))
                        if prev_active_metric is None:
                            activate_model_by_id(db, cand.model_id)
                            cand.notes = f"{cand.notes} | Promoted: initial active model"
                            db.commit()
                            promoted = True
                            gate_reason = "promoted_initial"
                        elif cand_metric is not None and cand_metric >= (prev_active_metric - tol):
                            activate_model_by_id(db, cand.model_id)
                            cand.notes = (
                                f"{cand.notes} | Promoted by anti-drift gate "
                                f"({cand_metric:.4f} >= {prev_active_metric:.4f} - {tol:.3f})"
                            )
                            db.commit()
                            promoted = True
                            gate_reason = "promoted_metric_ok"
                        else:
                            cand.notes = (
                                f"{cand.notes} | Kept standby by anti-drift gate "
                                f"(candidate={cand_metric} < active={prev_active_metric:.4f} - {tol:.3f})"
                            )
                            db.commit()
                            promoted = False
                            gate_reason = "rejected_anti_drift_regression"
                    jobs_out.append({
                        "job_id": res["job_id"],
                        "model_family": fam,
                        "device": res["device"],
                        "status": sess.status if sess else "unknown",
                        "best_val_metric": sess.best_val_metric if sess else 0.0,
                        "promoted": promoted,
                        "gate_reason": gate_reason,
                        "error_message": sess.error_message if sess else "",
                    })
                with self._lock:
                    self._last_jobs = list(jobs_out)

            with self._lock:
                self._stage = "completed"
                self._current_family = None
                self._last_run_at = datetime.now(UTC).isoformat()
                self._last_run_ts = time.time()
                self._running = False
        except Exception as exc:
            with self._lock:
                self._stage = "error"
                self._current_family = None
                self._last_error = str(exc)
                self._running = False

    def maybe_trigger_scheduled(self, config: AppConfig) -> bool:
        if not config.ai.auto_train.enabled or not config.ai.training_enabled:
            return False
        interval_sec = float(config.ai.auto_train.interval_minutes) * 60.0
        with self._lock:
            if self._running or get_training_manager().has_active_jobs():
                return False
            if (time.time() - self._last_run_ts) < interval_sec:
                return False

        # Persist cooldown across process restarts / prior OOM crashes by checking latest TrainingSession in DB
        with get_db_session() as db:
            last_sess = db.query(TrainingSession).order_by(TrainingSession.id.desc()).first()
            if last_sess is not None and last_sess.started_at is not None:
                started = last_sess.started_at
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                age_sec = (datetime.now(UTC) - started).total_seconds()
                if age_sec < interval_sec:
                    with self._lock:
                        self._last_run_ts = time.time() - max(0.0, age_sec)
                    return False

            rows = (
                db.query(Observation)
                .filter(
                    Observation.is_usable.is_(True),
                    Observation.quality_score >= config.ai.auto_train.min_quality_score,
                    Observation.id > self._last_obs_id,
                    Observation.raw_path != "",
                )
                .all()
            )
            ready = sum(1 for r in rows if Path(r.raw_path).exists())
            if ready < config.ai.auto_train.min_new_observations:
                return False

        try:
            self.trigger_run(async_run=True)
            return True
        except Exception:
            return False


_auto_learn_manager = AutoLearnManager()


def get_auto_learn_manager() -> AutoLearnManager:
    return _auto_learn_manager
