"""Storage Monitoring, Retention Policy Cleanup, Granular Data Purge & Full System Reset."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import AppConfig, get_config_manager
from app.models_db import (
    DatasetItem,
    DatasetRecord,
    EventAlert,
    Forecast,
    ForecastVerification,
    ModelRegistry,
    Observation,
    TrainingSession,
)


def _dir_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total_b = 0
    count = 0
    for f in path.rglob("*"):
        if f.is_file():
            total_b += f.stat().st_size
            count += 1
    return total_b, count


def _dir_size_bytes(path: Path) -> int:
    return _dir_stats(path)[0]


def _clear_directory_files(path: Path, protected_paths: set[str] | None = None) -> int:
    if not path.exists():
        return 0
    protected = protected_paths or set()
    deleted = 0
    for f in list(path.rglob("*")):
        if f.is_file():
            try:
                if str(f.resolve()) in protected:
                    continue
                f.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                continue
    return deleted


def get_storage_stats(config: AppConfig, db: Session | None = None) -> dict[str, Any]:
    config.ensure_directories()
    raw_b, raw_cnt = _dir_stats(config.raw_images_dir)
    proc_b, proc_cnt = _dir_stats(config.processed_images_dir)
    ev_b, ev_cnt = _dir_stats(config.event_images_dir)
    tl_b, tl_cnt = _dir_stats(config.timelapse_dir)
    ds_b, ds_cnt = _dir_stats(config.datasets_dir)
    mod_b, mod_cnt = _dir_stats(config.models_dir)
    tiles_b, tiles_cnt = _dir_stats(config.data_dir / "map_tiles")
    masks_b, masks_cnt = _dir_stats(config.data_dir / "masks")
    db_b = config.db_path.stat().st_size if config.db_path.exists() else 0

    total_used_b = raw_b + proc_b + ev_b + tl_b + ds_b + mod_b + tiles_b + masks_b + db_b
    disk = shutil.disk_usage(config.data_dir)

    db_counts: dict[str, int] = {}
    if db is not None:
        try:
            db_counts = {
                "observations": int(db.query(Observation).count()),
                "forecasts": int(db.query(Forecast).count()),
                "verifications": int(db.query(ForecastVerification).count()),
                "events": int(db.query(EventAlert).count()),
                "datasets": int(db.query(DatasetRecord).count()),
                "dataset_items": int(db.query(DatasetItem).count()),
                "models": int(db.query(ModelRegistry).count()),
                "training_sessions": int(db.query(TrainingSession).count()),
            }
        except Exception:
            db_counts = {}

    return {
        "data_dir": str(config.data_dir),
        "database_size_mb": round(db_b / (1024 ** 2), 2),
        "raw_images_mb": round(raw_b / (1024 ** 2), 2),
        "processed_images_mb": round(proc_b / (1024 ** 2), 2),
        "event_images_mb": round(ev_b / (1024 ** 2), 2),
        "timelapse_mb": round(tl_b / (1024 ** 2), 2),
        "datasets_mb": round(ds_b / (1024 ** 2), 2),
        "models_mb": round(mod_b / (1024 ** 2), 2),
        "map_tiles_mb": round(tiles_b / (1024 ** 2), 2),
        "exclusion_masks_mb": round(masks_b / (1024 ** 2), 2),
        "file_counts": {
            "raw_images": raw_cnt,
            "processed_images": proc_cnt,
            "event_images": ev_cnt,
            "timelapse": tl_cnt,
            "datasets": ds_cnt,
            "models": mod_cnt,
            "map_tiles": tiles_cnt,
            "exclusion_masks": masks_cnt,
        },
        "db_counts": db_counts,
        "total_app_storage_mb": round(total_used_b / (1024 ** 2), 2),
        "total_app_storage_gb": round(total_used_b / (1024 ** 3), 3),
        "max_storage_gb": config.storage.max_storage_gb,
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
    }


def apply_retention_policy(db: Session, config: AppConfig) -> dict[str, int]:
    """
    Delete expired raw/processed/timelapse files according to configured retention days.
    NEVER deletes images belonging to a training dataset or marked is_training_sample.
    """
    now = datetime.now(UTC)
    protected_paths: set[str] = {
        str(Path(it.image_path).resolve())
        for it in db.query(DatasetItem.image_path).all()
        if it.image_path
    }

    deleted_raw = 0
    deleted_processed = 0
    deleted_timelapse = 0

    raw_cutoff = now - timedelta(days=config.storage.raw_retention_days)
    proc_cutoff = now - timedelta(days=config.storage.processed_retention_days)
    ev_cutoff = now - timedelta(days=config.storage.event_retention_days)
    tl_cutoff = now - timedelta(days=config.storage.timelapse_retention_days)

    for obs in db.query(Observation).all():
        if obs.is_training_sample:
            continue
        ts = obs.captured_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        cutoff = ev_cutoff if obs.is_event_snapshot else raw_cutoff
        if obs.raw_path and ts < cutoff:
            rp = Path(obs.raw_path)
            if rp.exists() and str(rp.resolve()) not in protected_paths:
                rp.unlink(missing_ok=True)
                obs.raw_path = ""
                deleted_raw += 1

        if ts < proc_cutoff:
            for attr in ("seg_mask_path", "seg_overlay_path", "motion_overlay_path"):
                p_str = getattr(obs, attr)
                if p_str:
                    pp = Path(p_str)
                    if pp.exists() and str(pp.resolve()) not in protected_paths:
                        pp.unlink(missing_ok=True)
                        setattr(obs, attr, "")
                        deleted_processed += 1

    if config.timelapse_dir.exists():
        for vid in config.timelapse_dir.glob("*.mp4"):
            mtime = datetime.fromtimestamp(vid.stat().st_mtime, tz=UTC)
            if mtime < tl_cutoff:
                vid.unlink(missing_ok=True)
                deleted_timelapse += 1

    db.commit()
    return {
        "deleted_raw": deleted_raw,
        "deleted_processed": deleted_processed,
        "deleted_timelapse": deleted_timelapse,
    }


def delete_single_observation(db: Session, config: AppConfig, observation_id: int) -> dict[str, Any]:
    """Delete a single Observation and its raw/processed files on disk (unless protected by a dataset)."""
    obs = db.query(Observation).filter(Observation.id == observation_id).first()
    if obs is None:
        return {"deleted": False, "id": observation_id, "deleted_files": 0}

    protected_paths: set[str] = {
        str(Path(it.image_path).resolve())
        for it in db.query(DatasetItem).all()
        if it.image_path
    } | {
        str(Path(it.mask_path).resolve())
        for it in db.query(DatasetItem).all()
        if it.mask_path
    }

    deleted_files = 0
    for attr in ("raw_path", "seg_mask_path", "seg_overlay_path", "motion_overlay_path"):
        p_str = getattr(obs, attr, "")
        if p_str:
            p = Path(p_str)
            if p.exists() and str(p.resolve()) not in protected_paths:
                try:
                    p.unlink(missing_ok=True)
                    deleted_files += 1
                except Exception:
                    pass

    db.query(ForecastVerification).filter(ForecastVerification.observation_id == observation_id).delete()
    db.query(Forecast).filter(Forecast.observation_id == observation_id).delete()
    db.delete(obs)
    db.commit()
    return {"deleted": True, "id": observation_id, "deleted_files": deleted_files}


def purge_storage_targets(
    db: Session,
    config: AppConfig,
    targets: list[str],
    preserve_training_samples: bool = True,
    preserve_camera_connection: bool = True,
) -> dict[str, Any]:
    """
    Granular or full-system purge across disk directories, SQLite tables, and configuration.
    """
    target_set = set(targets)
    wipe_datasets = "datasets" in target_set

    protected_paths: set[str] = set()
    if preserve_training_samples and not wipe_datasets:
        for it in db.query(DatasetItem).all():
            if it.image_path:
                try:
                    protected_paths.add(str(Path(it.image_path).resolve()))
                except Exception:
                    pass
            if it.mask_path:
                try:
                    protected_paths.add(str(Path(it.mask_path).resolve()))
                except Exception:
                    pass
        for o in db.query(Observation).filter(Observation.is_training_sample.is_(True)).all():
            for attr in ("raw_path", "seg_mask_path"):
                p_str = getattr(o, attr, "")
                if p_str:
                    try:
                        protected_paths.add(str(Path(p_str).resolve()))
                    except Exception:
                        pass

    summary: dict[str, int | bool] = {
        "deleted_raw_files": 0,
        "deleted_processed_files": 0,
        "deleted_observations": 0,
        "deleted_forecasts": 0,
        "deleted_events": 0,
        "deleted_timelapse_files": 0,
        "deleted_map_tiles": 0,
        "deleted_exclusion_masks": 0,
        "deleted_datasets": 0,
        "deleted_models": 0,
        "config_reset": False,
    }

    if "datasets" in target_set:
        del_items = db.query(DatasetItem).delete()
        del_ds = db.query(DatasetRecord).delete()
        db.query(Observation).update({Observation.is_training_sample: False})
        ds_files = _clear_directory_files(config.datasets_dir)
        summary["deleted_datasets"] = int(del_ds + del_items + ds_files)

    if "models" in target_set:
        from app.ai.registry import get_model_cache, is_default_baseline_model

        all_models = db.query(ModelRegistry).order_by(ModelRegistry.created_at.asc(), ModelRegistry.id.asc()).all()
        kept_models: list[ModelRegistry] = []
        del_models: list[ModelRegistry] = []
        for rec in all_models:
            if is_default_baseline_model(rec):
                kept_models.append(rec)
            else:
                del_models.append(rec)

        kept_session_ids = {
            int(r.training_session_id) for r in kept_models if r.training_session_id is not None
        }
        protected_model_paths: set[str] = set()
        for r in kept_models:
            if r.weights_path:
                try:
                    protected_model_paths.add(str(Path(r.weights_path).resolve()))
                except Exception:
                    pass

        for r in del_models:
            db.delete(r)
        db.flush()

        # Delete only training sessions belonging to removed non-default models
        del_sess = 0
        for s in db.query(TrainingSession).all():
            if s.id not in kept_session_ids and s.status not in {"running", "queued"}:
                db.delete(s)
                del_sess += 1
        db.flush()

        # Re-activate preserved default baseline models for classification/temporal/event if active model was removed
        for fam in ("classification", "temporal", "event"):
            fam_kept = [r for r in kept_models if r.model_family == fam]
            if fam_kept and not any(r.is_active for r in fam_kept):
                fam_kept[0].is_active = True

        mod_files = _clear_directory_files(config.models_dir, protected_model_paths)
        db.commit()
        get_model_cache().refresh_from_db(db)
        summary["deleted_models"] = int(len(del_models) + del_sess + mod_files)
        summary["preserved_default_models"] = len(kept_models)

    if "raw_images" in target_set or "observations" in target_set:
        summary["deleted_raw_files"] = _clear_directory_files(config.raw_images_dir, protected_paths)
        for obs in db.query(Observation).all():
            if obs.raw_path and not Path(obs.raw_path).exists():
                obs.raw_path = ""

    if "processed_images" in target_set or "observations" in target_set:
        summary["deleted_processed_files"] = _clear_directory_files(config.processed_images_dir, protected_paths)
        for obs in db.query(Observation).all():
            for attr in ("seg_mask_path", "seg_overlay_path", "motion_overlay_path"):
                p_str = getattr(obs, attr, "")
                if p_str and not Path(p_str).exists():
                    setattr(obs, attr, "")

    if "forecasts" in target_set or "observations" in target_set:
        del_ver = db.query(ForecastVerification).delete()
        del_fc = db.query(Forecast).delete()
        summary["deleted_forecasts"] = int(del_fc + del_ver)

    if "events" in target_set:
        del_ev = db.query(EventAlert).delete()
        ev_files = _clear_directory_files(config.event_images_dir)
        db.query(Observation).update({Observation.is_event_snapshot: False})
        summary["deleted_events"] = int(del_ev + ev_files)

    if "timelapse" in target_set:
        tl_files = _clear_directory_files(config.timelapse_dir)
        db.query(Observation).filter(Observation.is_archived.is_(True)).update({Observation.is_archived: False})
        summary["deleted_timelapse_files"] = int(tl_files)

    if "observations" in target_set:
        if preserve_training_samples and not wipe_datasets:
            del_obs = db.query(Observation).filter(Observation.is_training_sample.is_(False)).delete()
        else:
            del_obs = db.query(Observation).delete()
        summary["deleted_observations"] = int(del_obs)

    if "map_tiles" in target_set:
        summary["deleted_map_tiles"] = _clear_directory_files(config.data_dir / "map_tiles")

    if "exclusion_masks" in target_set:
        summary["deleted_exclusion_masks"] = _clear_directory_files(config.data_dir / "masks")
        get_config_manager().update(
            {
                "camera": {
                    "exclusion_mask_path": "",
                    "exclusion_polygons": [],
                    "sky_roi_polygon": [],
                }
            }
        )

    if "reset_config" in target_set:
        default_cfg = AppConfig()
        patch = default_cfg.model_dump()
        if preserve_camera_connection:
            cur_cam = get_config_manager().get().camera
            patch["camera"]["name"] = cur_cam.name
            patch["camera"]["source_type"] = cur_cam.source_type
            patch["camera"]["source_url"] = cur_cam.source_url
            patch["camera"]["device_index"] = cur_cam.device_index
            patch["camera"]["latitude"] = cur_cam.latitude
            patch["camera"]["longitude"] = cur_cam.longitude
            patch["camera"]["location_description"] = cur_cam.location_description
            patch["camera"]["orientation_elevation_deg"] = cur_cam.orientation_elevation_deg
            patch["camera"]["orientation_azimuth_deg"] = cur_cam.orientation_azimuth_deg
            patch["camera"]["orientation_roll_deg"] = cur_cam.orientation_roll_deg
            patch["camera"]["fov_horizontal_deg"] = cur_cam.fov_horizontal_deg
            patch["camera"]["fov_vertical_deg"] = cur_cam.fov_vertical_deg
        get_config_manager().update(patch)
        summary["config_reset"] = True

    db.commit()
    return summary
