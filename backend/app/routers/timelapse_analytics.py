"""Timelapse & Analytics Export API router."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.config import get_config_manager
from app.database import get_db
from app.forecast.verification import compute_verification_summary
from app.models_db import Forecast, Observation
from app.routers.forecast_events import forecast_to_dict
from app.schemas import DailyDigestRequest, TimelapseClearRequest, TimelapseExportRequest
from app.services.timelapse import (
    clear_timelapse_frames,
    export_timelapse_video,
    generate_daily_digest,
    list_daily_digests,
)
from app.services.weather_online import get_online_weather_service
from app.services.worker import observation_to_dict

router = APIRouter()


@router.get("/api/timelapse/frames")
def api_timelapse_frames(
    date: str | None = None,
    limit: int = Query(300, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    q = (
        db.query(Observation)
        .filter(Observation.is_archived.is_(True))
        .order_by(Observation.captured_at.desc(), Observation.id.desc())
    )
    if date:
        rows = [r for r in q.all() if r.captured_at.strftime("%Y-%m-%d") == date][:limit]
    else:
        rows = q.limit(limit).all()
    rows.reverse()
    frames = []
    for r in rows:
        has_raw = bool(r.raw_path and Path(r.raw_path).exists())
        has_segmentation = bool(r.seg_overlay_path and Path(r.seg_overlay_path).exists())
        has_motion = bool(r.motion_overlay_path and Path(r.motion_overlay_path).exists())
        if not (has_raw or has_segmentation or has_motion):
            continue
        frames.append({
            "observation_id": r.id,
            "captured_at": r.captured_at.isoformat(),
            "cloud_coverage_pct": r.cloud_coverage_pct,
            "visual_condition": r.visual_condition,
            "has_raw": has_raw,
            "has_segmentation": has_segmentation,
            "has_motion": has_motion,
        })
    return {"count": len(frames), "frames": frames}


@router.post("/api/timelapse/export")
def api_timelapse_export(req: TimelapseExportRequest, db: Session = Depends(get_db)):
    cfg = get_config_manager().get()
    temp_c = None
    if req.include_hud and not cfg.server.offline_mode:
        try:
            wx = get_online_weather_service().get_weather(cfg.camera.latitude, cfg.camera.longitude, cfg)
            temp_c = (wx.get("current") or {}).get("temperature_c")
        except Exception:
            temp_c = None
    try:
        out_path = export_timelapse_video(
            db=db,
            config=cfg,
            layer=req.layer,
            fps=req.fps,
            date_str=req.date,
            max_frames=req.max_frames,
            include_hud=req.include_hud,
            temperature_c=float(temp_c) if temp_c is not None else None,
        )
        return FileResponse(str(out_path), media_type="video/mp4", filename=out_path.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/timelapse/digests")
def api_timelapse_digests() -> dict[str, Any]:
    cfg = get_config_manager().get()
    return {"digests": list_daily_digests(cfg)}


@router.post("/api/timelapse/daily-digest")
def api_timelapse_generate_digest(req: DailyDigestRequest | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    payload = req or DailyDigestRequest()
    temp_c = None
    if payload.include_hud and not cfg.server.offline_mode:
        try:
            wx = get_online_weather_service().get_weather(cfg.camera.latitude, cfg.camera.longitude, cfg)
            temp_c = (wx.get("current") or {}).get("temperature_c")
        except Exception:
            temp_c = None
    try:
        digest = generate_daily_digest(
            db=db,
            config=cfg,
            date_str=payload.date,
            layer=payload.layer,
            fps=payload.fps,
            include_hud=payload.include_hud,
            temperature_c=float(temp_c) if temp_c is not None else None,
        )
        return {"generated": True, "digest": digest, "digests": list_daily_digests(cfg)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/timelapse/digests/{filename}")
def api_timelapse_download_digest(filename: str):
    cfg = get_config_manager().get()
    safe_name = Path(filename).name
    if not safe_name.endswith("_daily.mp4"):
        raise HTTPException(status_code=400, detail="Invalid digest filename")
    target = (cfg.timelapse_dir / safe_name).resolve()
    if not target.exists() or target.parent != cfg.timelapse_dir.resolve():
        raise HTTPException(status_code=404, detail="Daily digest not found")
    return FileResponse(str(target), media_type="video/mp4", filename=safe_name)


@router.get("/api/analytics/export")
def api_analytics_export(
    format: str = Query(default="csv", pattern="^(csv|json)$"),
    hours: int | None = Query(default=None, ge=1, le=24 * 365),
    limit: int = Query(default=2000, ge=1, le=20000),
    db: Session = Depends(get_db),
):
    """Export historical observations, forecasts, and verification metrics as CSV (Excel) or JSON."""
    q = db.query(Observation).order_by(Observation.captured_at.desc(), Observation.id.desc())
    if hours is not None:
        cutoff = datetime.now(UTC).timestamp() - hours * 3600.0
        rows = [
            r
            for r in q.limit(limit * 2).all()
            if (r.captured_at if r.captured_at.tzinfo else r.captured_at.replace(tzinfo=UTC)).timestamp() >= cutoff
        ][:limit]
    else:
        rows = q.limit(limit).all()
    rows.reverse()

    ts_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    if format == "json":
        ver = compute_verification_summary(db)
        fcs = db.query(Forecast).order_by(Forecast.generated_at.desc()).limit(200).all()
        payload = {
            "exported_at": datetime.now(UTC).isoformat(),
            "observations_count": len(rows),
            "observations": [observation_to_dict(r) for r in rows],
            "recent_forecasts": [forecast_to_dict(f) for f in reversed(fcs)],
            "verification_summary": ver,
        }
        return Response(
            content=json.dumps(payload, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="sky_observer_export_{ts_tag}.json"'},
        )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id",
        "captured_at",
        "cloud_coverage_pct",
        "clear_sky_pct",
        "thin_cloud_pct",
        "dense_cloud_pct",
        "dark_cloud_pct",
        "haze_fog_pct",
        "dominant_cloud_type",
        "classification_confidence",
        "visual_condition",
        "motion_speed_px_s",
        "motion_direction_deg",
        "motion_direction_cardinal",
        "motion_consistency",
        "avg_luminance",
        "blue_sky_index",
        "quality_score",
        "is_usable",
    ])
    for r in rows:
        writer.writerow([
            r.id,
            r.captured_at.isoformat() if r.captured_at else "",
            round(float(r.cloud_coverage_pct), 2),
            round(float(r.clear_sky_pct), 2),
            round(float(r.thin_cloud_pct), 2),
            round(float(r.dense_cloud_pct), 2),
            round(float(r.dark_cloud_pct), 2),
            round(float(r.haze_fog_pct), 2),
            r.dominant_cloud_type,
            round(float(r.classification_confidence), 4),
            r.visual_condition,
            round(float(r.motion_speed_px_s), 3),
            round(float(r.motion_direction_deg), 1),
            r.motion_direction_cardinal,
            round(float(r.motion_consistency), 3),
            round(float(r.avg_luminance), 2),
            round(float(r.blue_sky_index), 4),
            round(float(r.quality_score), 3),
            bool(r.is_usable),
        ])
    csv_bytes = ("\ufeff" + buf.getvalue()).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="sky_observer_export_{ts_tag}.csv"'},
    )


@router.post("/api/timelapse/clear")
def api_timelapse_clear(
    req: TimelapseClearRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    cfg = get_config_manager().get()
    payload = req or TimelapseClearRequest()
    return clear_timelapse_frames(
        db=db,
        config=cfg,
        date_str=payload.date,
        delete_files=payload.delete_files,
    )
