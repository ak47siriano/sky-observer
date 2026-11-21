"""Timelapse frame retrieval, Meteo HUD stamping, local MP4 export, and automatic Daily Sky Digest service."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.config import AppConfig
from app.models_db import DatasetItem, Observation
from app.vision.geometry import compute_solar_position


def _render_meteo_hud(
    img: np.ndarray,
    obs: Observation,
    location_label: str = "",
    temperature_c: float | None = None,
) -> np.ndarray:
    """Stamp a sleek translucent meteorological HUD strip along the bottom edge of a timelapse frame."""
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = max(34, min(56, int(h * 0.068)))
    y0 = max(0, h - bar_h)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (11, 14, 20), -1)
    cv2.addWeighted(overlay, 0.74, out, 0.26, 0.0, out)
    cv2.line(out, (0, y0), (w, y0), (248, 189, 56), 1, cv2.LINE_AA)

    ts = obs.captured_at if obs.captured_at.tzinfo else obs.captured_at.replace(tzinfo=UTC)
    ts_str = ts.strftime("%Y-%m-%d %H:%M")
    cloud_type = (obs.dominant_cloud_type or "unknown").replace("_", " ").title()
    left_txt = (
        f"SKY OBSERVER  |  {ts_str}  |  "
        f"Clouds {obs.cloud_coverage_pct:.1f}% ({cloud_type})  |  "
        f"Motion {obs.motion_speed_px_s:.1f} px/s {obs.motion_direction_cardinal}"
    )

    right_parts: list[str] = [f"L* {obs.avg_luminance:.0f}"]
    if temperature_c is not None:
        right_parts.append(f"{temperature_c:.1f} C")
    if location_label:
        right_parts.append(location_label[:26])
    right_txt = "  |  ".join(right_parts)

    scale = max(0.40, min(0.62, w / 1920.0 * 0.62))
    thickness = 1 if scale < 0.55 else 2
    baseline_y = y0 + int(bar_h * 0.65)

    cv2.putText(out, left_txt, (14, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (244, 245, 247), thickness, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if w - rw - 16 > int(w * 0.55):
        cv2.putText(out, right_txt, (w - rw - 14, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (248, 189, 56), thickness, cv2.LINE_AA)
    return out


def export_timelapse_video(
    db: Session,
    config: AppConfig,
    layer: Literal["raw", "segmentation", "motion"] = "raw",
    fps: int = 12,
    date_str: str | None = None,
    max_frames: int = 600,
    include_hud: bool = True,
    temperature_c: float | None = None,
    output_filename: str | None = None,
) -> Path:
    """Assemble captured frames (raw, segmentation overlay, or motion overlay) into an MP4 file locally."""
    q = (
        db.query(Observation)
        .filter(Observation.is_archived.is_(True))
        .order_by(Observation.captured_at.asc())
    )
    obs_list = q.all() if date_str else q.limit(max_frames).all()

    items: list[tuple[Path, Observation]] = []
    for obs in obs_list:
        if date_str and obs.captured_at.strftime("%Y-%m-%d") != date_str:
            continue
        if layer == "segmentation":
            cand = obs.seg_overlay_path or obs.raw_path
        elif layer == "motion":
            cand = obs.motion_overlay_path or obs.raw_path
        else:
            cand = obs.raw_path
        if cand and Path(cand).exists():
            items.append((Path(cand), obs))
            if len(items) >= max_frames:
                break

    if not items:
        raise ValueError("No stored frames match the requested timelapse filter.")

    first = cv2.imread(str(items[0][0]))
    if first is None:
        raise ValueError("Failed to read first frame for timelapse export.")
    h, w = first.shape[:2]

    config.timelapse_dir.mkdir(parents=True, exist_ok=True)
    if output_filename:
        safe_name = Path(output_filename).name
        out_path = config.timelapse_dir / safe_name
    else:
        ts_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_path = config.timelapse_dir / f"timelapse_{layer}_{ts_tag}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(max(1, min(60, fps))), (w, h))
    loc_label = config.camera.location_description or ""
    try:
        for fp, obs in items:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            if include_hud:
                img = _render_meteo_hud(img, obs, location_label=loc_label, temperature_c=temperature_c)
            writer.write(img)
    finally:
        writer.release()

    return out_path


def list_daily_digests(config: AppConfig) -> list[dict[str, Any]]:
    """Return metadata for all generated daily sky digest MP4 files."""
    if not config.timelapse_dir.exists():
        return []
    digests: list[dict[str, Any]] = []
    for mp4 in sorted(config.timelapse_dir.glob("*_daily.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = mp4.stat()
        date_part = mp4.name.replace("_daily.mp4", "")
        digests.append({
            "filename": mp4.name,
            "date": date_part,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            "download_url": f"/api/timelapse/digests/{mp4.name}",
        })
    return digests


def generate_daily_digest(
    db: Session,
    config: AppConfig,
    date_str: str | None = None,
    layer: Literal["raw", "segmentation", "motion"] = "raw",
    fps: int = 12,
    include_hud: bool = True,
    temperature_c: float | None = None,
) -> dict[str, Any]:
    """Generate a YYYY-MM-DD_daily.mp4 Daily Sky Digest with optional Meteo HUD."""
    target_date = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = export_timelapse_video(
        db=db,
        config=config,
        layer=layer,
        fps=fps,
        date_str=target_date,
        max_frames=1200,
        include_hud=include_hud,
        temperature_c=temperature_c,
        output_filename=f"{target_date}_daily.mp4",
    )
    stat = out_path.stat()
    return {
        "filename": out_path.name,
        "date": target_date,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "download_url": f"/api/timelapse/digests/{out_path.name}",
    }


def maybe_generate_sunset_daily_digest(db: Session, config: AppConfig) -> dict[str, Any] | None:
    """Automatically generate today's YYYY-MM-DD_daily.mp4 after sunset (solar elevation < -4 deg)."""
    now = datetime.now(UTC)
    solar = compute_solar_position(config.camera.latitude, config.camera.longitude, now)
    if float(solar.get("elevation_deg", 10.0)) >= -4.0:
        return None
    today_str = now.strftime("%Y-%m-%d")
    expected = config.timelapse_dir / f"{today_str}_daily.mp4"
    if expected.exists():
        return None
    rows = db.query(Observation).filter(Observation.is_archived.is_(True)).all()
    today_count = sum(1 for r in rows if r.captured_at.strftime("%Y-%m-%d") == today_str and r.raw_path and Path(r.raw_path).exists())
    if today_count < 5:
        return None
    try:
        return generate_daily_digest(db=db, config=config, date_str=today_str, layer="raw", fps=12, include_hud=True)
    except Exception:
        return None


def clear_timelapse_frames(
    db: Session,
    config: AppConfig,
    date_str: str | None = None,
    delete_files: bool = False,
) -> dict[str, Any]:
    """
    Clear archived timelapse frames (marking Observation.is_archived = False) so the
    timelapse restarts cleanly from subsequent captures.
    Optionally deletes non-protected image files from disk while ALWAYS protecting
    training dataset images, event snapshots, and the most recent live observation.
    """
    rows = db.query(Observation).filter(Observation.is_archived.is_(True)).all()
    if date_str:
        rows = [r for r in rows if r.captured_at.strftime("%Y-%m-%d") == date_str]

    latest_obs = db.query(Observation).order_by(Observation.captured_at.desc(), Observation.id.desc()).first()
    latest_id = latest_obs.id if latest_obs else None

    protected_paths: set[str] = set()
    if delete_files:
        protected_paths = {
            str(Path(it.image_path).resolve())
            for it in db.query(DatasetItem.image_path).all()
            if it.image_path
        }

    cleared_frames = 0
    deleted_files = 0
    for obs in rows:
        obs.is_archived = False
        cleared_frames += 1
        if delete_files and obs.id != latest_id and not obs.is_training_sample and not obs.is_event_snapshot:
            for attr in ("raw_path", "seg_mask_path", "seg_overlay_path", "motion_overlay_path"):
                p_str = getattr(obs, attr)
                if p_str:
                    pp = Path(p_str)
                    if pp.exists() and str(pp.resolve()) not in protected_paths:
                        pp.unlink(missing_ok=True)
                        setattr(obs, attr, "")
                        deleted_files += 1

    deleted_exports = 0
    if config.timelapse_dir.exists():
        date_tag = date_str.replace("-", "") if date_str else None
        for vid in config.timelapse_dir.glob("*.mp4"):
            if date_tag is None or date_tag in vid.name or (date_str and date_str in vid.name):
                vid.unlink(missing_ok=True)
                deleted_exports += 1

    db.commit()
    return {
        "cleared": True,
        "cleared_frames": cleared_frames,
        "deleted_files": deleted_files,
        "deleted_exports": deleted_exports,
        "date": date_str,
    }
