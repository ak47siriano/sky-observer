"""Sky Observer FastAPI Application — 100% Local AI Sky Monitoring & Weather Nowcasting."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.ai.autolearn import get_auto_learn_manager
from app.ai.datasets import (
    PUBLIC_DATASET_CATALOG,
    bootstrap_local_sky_dataset,
    get_dataset_summary,
    import_local_dataset,
)
from app.ai.device import HARDWARE_PROFILES, get_hardware_status
from app.ai.registry import (
    BUILTIN_RADIOMETRIC_MODEL_ID,
    activate_model_by_id,
    deactivate_model_family,
    delete_model_by_id,
    get_model_cache,
    prune_old_models,
    rollback_model_family,
)
from app.ai.trainer import get_training_manager
from app.camera.manager import get_camera_manager, mask_url_credentials
from app.config import get_config_manager
from app.database import get_db, get_db_session, init_database
from app.forecast.engine import generate_forecasts_for_observation
from app.models_db import (
    DatasetItem,
    DatasetRecord,
    EventAlert,
    Forecast,
    ModelRegistry,
    Observation,
    TrainingSession,
    safe_json_loads,
)
from app.schemas import (
    ActivateModelRequest,
    AutoTrainConfigRequest,
    AutoTrainRunRequest,
    CameraAutoTiltRequest,
    CameraGeometryRequest,
    CameraTestRequest,
    DataPurgeRequest,
    DatasetExportRequest,
    DatasetImportRequest,
    DatasetLabelRequest,
    HardwareProfileRequest,
    MapPrefetchRequest,
    MasksUpdateRequest,
    PaintedMaskRequest,
    PruneModelsRequest,
    RadarLocationRequest,
    RollbackModelRequest,
    SecondaryCamerasUpdateRequest,
    SystemModeRequest,
    TrainModelRequest,
)
from app.services.map_tiles import get_tile_bytes, get_tile_cache_stats, prefetch_tiles_around
from app.services.storage import (
    apply_retention_policy,
    delete_single_observation,
    get_storage_stats,
    purge_storage_targets,
)
from app.services.weather_online import get_online_weather_service
from app.services.worker import get_broadcaster, get_worker, observation_to_dict, process_and_store_frame
from app.vision.coverage import compute_cloud_coverage
from app.vision.geometry import (
    CameraGeometry,
    compute_pixel_geometry_maps,
    compute_radar_footprint,
    compute_solar_position,
    estimate_camera_tilt_from_frame,
    geometry_from_camera_config,
    project_mask_to_radar_bgra,
    project_sky_angles_to_image,
    with_cloud_altitude,
)
from app.vision.masking import build_valid_sky_mask_from_config, render_mask_overlay
from app.vision.motion import estimate_cloud_motion


def _forecast_to_dict(fc: Forecast) -> dict[str, Any]:
    return {
        "id": fc.id,
        "generated_at": fc.generated_at.isoformat() if fc.generated_at else None,
        "observation_id": fc.observation_id,
        "horizon_minutes": fc.horizon_minutes,
        "target_time": fc.target_time.isoformat() if fc.target_time else None,
        "status": fc.status,
        "forecast": fc.forecast_condition,
        "cloud_cover_prediction": fc.cloud_cover_prediction,
        "clearing_prob": fc.clearing_prob,
        "increasing_prob": fc.increasing_prob,
        "overcast_prob": fc.overcast_prob,
        "darkening_prob": fc.darkening_prob,
        "precip_signal_prob": fc.precip_signal_prob,
        "storm_signal_prob": fc.storm_signal_prob,
        "confidence": fc.confidence,
        "signals": safe_json_loads(fc.signals_json, []),
        "reasons": safe_json_loads(fc.reasons_json, []),
        "limitations": safe_json_loads(fc.limitations_json, []),
        "model_version": fc.model_version,
        "verified": fc.verified,
    }


def _event_to_dict(ev: EventAlert) -> dict[str, Any]:
    return {
        "id": ev.id,
        "created_at": ev.created_at.isoformat() if ev.created_at else None,
        "event_type": ev.event_type,
        "severity": ev.severity,
        "title": ev.title,
        "description": ev.description,
        "observation_id": ev.observation_id,
        "metrics": safe_json_loads(ev.metrics_json, {}),
        "acknowledged": ev.acknowledged,
        "user_status": ev.user_status or ("confirmed" if ev.acknowledged else "pending"),
        "user_note": ev.user_note or "",
    }


def _apply_event_feedback(
    db: Session,
    ev: EventAlert,
    status: str,
    note: str | None = None,
) -> dict[str, Any]:
    ev.user_status = status
    ev.acknowledged = status != "pending"
    if note is not None:
        ev.user_note = note.strip()
    if ev.observation_id is not None:
        obs = db.query(Observation).filter(Observation.id == ev.observation_id).first()
        if obs:
            if status == "rejected":
                remaining = (
                    db.query(EventAlert)
                    .filter(
                        EventAlert.observation_id == ev.observation_id,
                        EventAlert.id != ev.id,
                        EventAlert.user_status != "rejected",
                    )
                    .count()
                )
                if remaining == 0:
                    obs.is_event_snapshot = False
            else:
                obs.is_event_snapshot = True
    db.commit()
    return _event_to_dict(ev)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = get_config_manager().get()
    cfg.ensure_directories()
    init_database(cfg.db_path)
    with get_db_session() as db:
        get_model_cache().refresh_from_db(db)
    try:
        get_broadcaster().set_loop(asyncio.get_running_loop())
    except RuntimeError:
        pass
    if os.environ.get("SKYOBSERVER_DISABLE_WORKER", "").lower() not in {"1", "true"}:
        get_worker().start()
    yield
    get_worker().stop()


from app.routers.forecast_events import router as forecast_events_router
from app.routers.timelapse_analytics import router as timelapse_analytics_router
from app.security import (
    ApiTokenAuthMiddleware,
    extract_token_from_request_parts,
    get_configured_api_token,
    is_valid_api_token,
)

app = FastAPI(
    title="Sky Observer API",
    description="100% Local AI Sky Monitoring, Cloud Vision & Weather Nowcasting System",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(ApiTokenAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_config_manager().get().server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(forecast_events_router)
app.include_router(timelapse_analytics_router)


@app.get("/api/auth/status")
def api_auth_status(request: Request) -> dict[str, Any]:
    expected = get_configured_api_token()
    presented = extract_token_from_request_parts(request.headers, request.query_params)
    return {
        "auth_required": bool(expected),
        "authenticated": is_valid_api_token(presented, expected),
    }


def _encode_jpeg_response(img_bgr: np.ndarray, quality: int = 88) -> Response:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# =====================================================================
# CAMERA ENDPOINTS
# =====================================================================


def _sanitize_camera_dump(cam_dump: dict[str, Any]) -> dict[str, Any]:
    out = dict(cam_dump)
    if isinstance(out.get("source_url"), str):
        out["source_url"] = mask_url_credentials(out["source_url"])
    if isinstance(out.get("secondary_cameras"), list):
        sec_out = []
        for sc in out["secondary_cameras"]:
            sc_copy = dict(sc) if isinstance(sc, dict) else sc
            if isinstance(sc_copy, dict) and isinstance(sc_copy.get("source_url"), str):
                sc_copy["source_url"] = mask_url_credentials(sc_copy["source_url"])
            sec_out.append(sc_copy)
        out["secondary_cameras"] = sec_out
    return out


def _restore_masked_url(submitted_url: Any, existing_url: str) -> Any:
    if isinstance(submitted_url, str) and existing_url and submitted_url == mask_url_credentials(existing_url):
        return existing_url
    return submitted_url


@app.get("/api/camera/status")
def api_camera_status() -> dict[str, Any]:
    return get_camera_manager().get_status()


@app.get("/api/camera/config")
def api_camera_config_get() -> dict[str, Any]:
    return _sanitize_camera_dump(get_config_manager().get().camera.model_dump())


@app.put("/api/camera/config")
def api_camera_config_put(payload: dict[str, Any]) -> dict[str, Any]:
    current_cam = get_config_manager().get().camera
    safe_payload = dict(payload)
    if "source_url" in safe_payload:
        safe_payload["source_url"] = _restore_masked_url(safe_payload["source_url"], current_cam.source_url)
    try:
        updated = get_config_manager().replace_section("camera", safe_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid camera configuration") from exc
    get_camera_manager().close()
    return _sanitize_camera_dump(updated.camera.model_dump())


@app.put("/api/camera/masks")
def api_camera_masks_put(req: MasksUpdateRequest) -> dict[str, Any]:
    try:
        updated = get_config_manager().update(
            {
                "camera": {
                    "sky_roi_polygon": req.sky_roi_polygon,
                    "exclusion_polygons": req.exclusion_polygons,
                }
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return updated.camera.model_dump()


@app.put("/api/camera/exclusion-mask")
def api_camera_exclusion_mask_put(req: PaintedMaskRequest) -> dict[str, Any]:
    """Save a user-painted permanent exclusion mask PNG (e.g. Iriun logo, road, cars, trees)."""
    cfg = get_config_manager().get()
    raw_b64 = req.mask_png_base64.split(",", 1)[-1]
    try:
        raw_bytes = base64.b64decode(raw_b64)
        arr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PNG mask payload: {exc}") from exc
    if arr is None or arr.size == 0:
        raise HTTPException(status_code=400, detail="Empty or undecodable exclusion mask PNG")

    if arr.ndim == 3 and arr.shape[2] == 4:
        binary = np.where(arr[:, :, 3] > 20, 255, 0).astype(np.uint8)
    elif arr.ndim == 3:
        binary = np.where(np.max(arr, axis=2) > 20, 255, 0).astype(np.uint8)
    else:
        binary = np.where(arr > 20, 255, 0).astype(np.uint8)

    masks_dir = cfg.data_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / "camera_exclusion_mask.png"
    cv2.imwrite(str(out_path), binary)

    get_config_manager().update({"camera": {"exclusion_mask_path": str(out_path)}})
    excluded_pct = round(float(np.mean(binary > 0) * 100.0), 2)
    return {"saved": True, "path": str(out_path), "excluded_pct": excluded_pct}


@app.get("/api/camera/exclusion-mask.png")
def api_camera_exclusion_mask_get():
    """Return the saved painted exclusion mask as translucent crimson RGBA PNG."""
    cfg = get_config_manager().get()
    w, h = 640, 360
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if cfg.camera.exclusion_mask_path:
        mp = Path(cfg.camera.exclusion_mask_path)
        if mp.exists():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.size > 0:
                mr = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
                rgba[mr] = [68, 68, 239, 170]  # Crimson BGRA for canvas overlay
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode exclusion mask")
    return Response(content=buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.delete("/api/camera/exclusion-mask")
def api_camera_exclusion_mask_delete() -> dict[str, Any]:
    cfg = get_config_manager().get()
    if cfg.camera.exclusion_mask_path:
        Path(cfg.camera.exclusion_mask_path).unlink(missing_ok=True)
    get_config_manager().update({"camera": {"exclusion_mask_path": ""}})
    return {"cleared": True}


@app.post("/api/camera/exclusion-mask/propose")
def api_camera_exclusion_mask_propose(
    observation_id: int | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Auto-propose a static sky-vs-obstacle exclusion mask from the latest (or selected) camera frame.
    Returns a translucent crimson PNG data URL for user review/refinement; does NOT auto-save.
    """
    from app.vision.masking import propose_sky_obstacle_mask

    q = db.query(Observation)
    obs = (
        q.filter(Observation.id == observation_id).first()
        if observation_id is not None
        else q.filter(Observation.raw_path != "").order_by(Observation.id.desc()).first()
    )
    if obs is None or not obs.raw_path or not Path(obs.raw_path).exists():
        raise HTTPException(status_code=404, detail="No raw camera frame available to propose obstacle mask")
    img = cv2.imread(obs.raw_path)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="Could not read camera frame from disk")

    cfg = get_config_manager().get()
    excl_u8, stats = propose_sky_obstacle_mask(
        img,
        elevation_deg=cfg.camera.orientation_elevation_deg if cfg.camera.tilt_correction_enabled else None,
        fov_vertical_deg=cfg.camera.fov_vertical_deg,
    )
    w, h = 640, 360
    mr = cv2.resize(excl_u8, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mr] = [68, 68, 239, 170]
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode proposed exclusion mask")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "observation_id": obs.id,
        "mask_png_base64": f"data:image/png;base64,{b64}",
        **stats,
    }


@app.post("/api/camera/test")
def api_camera_test(req: CameraTestRequest | None = None) -> dict[str, Any]:
    return get_camera_manager().test_connection(
        source_type=req.source_type if req else None,
        source_url=req.source_url if req else None,
        device_index=req.device_index if req else None,
    )


@app.get("/api/camera/devices")
def api_camera_devices() -> dict[str, Any]:
    return {"devices": get_camera_manager().list_local_video_devices()}


@app.get("/api/camera/latest")
def api_camera_latest(
    layer: str = Query("raw", pattern="^(raw|segmentation|motion|mask)$"),
    db: Session = Depends(get_db),
):
    obs = db.query(Observation).order_by(Observation.captured_at.desc(), Observation.id.desc()).first()
    if not obs:
        raise HTTPException(status_code=404, detail="No observations captured yet")

    if layer == "segmentation" and obs.seg_overlay_path and Path(obs.seg_overlay_path).exists():
        return FileResponse(obs.seg_overlay_path, media_type="image/jpeg")
    if layer == "motion" and obs.motion_overlay_path and Path(obs.motion_overlay_path).exists():
        return FileResponse(obs.motion_overlay_path, media_type="image/jpeg")
    if obs.raw_path and Path(obs.raw_path).exists():
        if layer == "mask":
            img = cv2.imread(obs.raw_path)
            if img is not None:
                cfg = get_config_manager().get().camera
                vm = build_valid_sky_mask_from_config(img.shape[:2], cfg)
                return _encode_jpeg_response(render_mask_overlay(img, vm, cfg))
        return FileResponse(obs.raw_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Image file missing on disk")


@app.get("/api/camera/snapshot")
def api_camera_snapshot(analyze: bool = True, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Trigger an immediate local camera capture (and optional CV analysis)."""
    cfg = get_config_manager().get()
    cap = get_camera_manager().capture_once(cfg.camera, bypass_backoff=True)
    if not cap.success or cap.image_bgr is None:
        raise HTTPException(status_code=503, detail=cap.error or "Camera snapshot failed")
    if not analyze:
        return {"captured_at": cap.captured_at.isoformat(), "latency_ms": round(cap.latency_ms, 2), "analyzed": False}
    obs = process_and_store_frame(
        db=db,
        image_bgr=cap.image_bgr,
        config=cfg,
        source_type=cap.source_type,
        captured_at=cap.captured_at,
        capture_latency_ms=cap.latency_ms,
        archive_raw=True,
        run_forecast=True,
    )
    return {"analyzed": True, "observation": observation_to_dict(obs)}


@app.post("/api/camera/upload")
async def api_camera_upload(
    file: UploadFile = File(...),
    max_video_frames: int = Query(12, ge=1, le=60),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Method C: Import local image (.jpg/.png) or video (.mp4/.avi) directly into the analysis pipeline."""
    cfg = get_config_manager().get()
    data = await file.read()
    filename = (file.filename or "upload.jpg").lower()

    if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        tmp_path = cfg.raw_images_dir / f"upload_tmp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.mp4"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(data)
        cap = cv2.VideoCapture(str(tmp_path))
        frames: list[np.ndarray] = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // max_video_frames) if total > max_video_frames else 1
        idx = 0
        while cap.isOpened() and len(frames) < max_video_frames:
            ok, frm = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frm)
            idx += 1
        cap.release()
        tmp_path.unlink(missing_ok=True)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode frames from uploaded video")
        stored = []
        base_dt = datetime.now(UTC)
        for i, frm in enumerate(frames):
            from datetime import timedelta

            obs = process_and_store_frame(
                db=db,
                image_bgr=frm,
                config=cfg,
                source_type="local_import",
                captured_at=base_dt + timedelta(seconds=i * cfg.camera.capture_interval_seconds),
                archive_raw=True,
                run_forecast=(i == len(frames) - 1),
            )
            stored.append(observation_to_dict(obs))
        return {"imported_frames": len(stored), "latest_observation": stored[-1]}

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image file")
    obs = process_and_store_frame(
        db=db,
        image_bgr=img,
        config=cfg,
        source_type="local_import",
        captured_at=datetime.now(UTC),
        archive_raw=True,
        run_forecast=True,
    )
    return {"imported_frames": 1, "latest_observation": observation_to_dict(obs)}


# =====================================================================
# ANALYSIS ENDPOINTS
# =====================================================================


@app.get("/api/analysis/latest")
def api_analysis_latest(db: Session = Depends(get_db)) -> dict[str, Any]:
    obs = db.query(Observation).order_by(Observation.captured_at.desc(), Observation.id.desc()).first()
    if not obs:
        return {"available": False, "observation": None}
    return {"available": True, "observation": observation_to_dict(obs)}


@app.get("/api/sky-state")
def api_sky_state(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Ultra-fast lightweight sky-state endpoint (<5ms) for Home Assistant, ESP32, and local automations."""
    obs = db.query(Observation).order_by(Observation.id.desc()).first()
    if not obs:
        return {
            "available": False,
            "state": "unavailable",
            "sky_regime": "unknown",
            "cloud_coverage_pct": None,
            "dominant_cloud_type": "unknown",
            "confidence": 0.0,
            "is_usable": False,
        }
    cov_data = safe_json_loads(obs.coverage_metrics_json, {})
    q_data = safe_json_loads(obs.quality_metrics_json, {})
    return {
        "available": True,
        "observation_id": obs.id,
        "captured_at": obs.captured_at.isoformat() if obs.captured_at else None,
        "state": obs.visual_condition,
        "sky_regime": cov_data.get("sky_regime", "day"),
        "cloud_coverage_pct": round(float(obs.cloud_coverage_pct), 1),
        "clear_sky_pct": round(float(obs.clear_sky_pct), 1),
        "dark_cloud_pct": round(float(obs.dark_cloud_pct), 1),
        "dominant_cloud_type": obs.dominant_cloud_type,
        "confidence": round(float(obs.classification_confidence), 3),
        "quality_score": round(float(obs.quality_score), 3),
        "qc_flags": q_data.get("flags", []),
        "is_usable": bool(obs.is_usable),
    }


@app.get("/api/solar/pv")
def api_solar_pv(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Optional clear-sky GHI, cloud-transmittance PV power output (kW), and short-term solar ramp alerts."""
    from app.services.solar_pv import estimate_solar_pv_and_ramps

    cfg = get_config_manager().get()
    pv_cfg = cfg.solar_pv
    obs = db.query(Observation).filter(Observation.is_usable.is_(True)).order_by(Observation.id.desc()).first()
    if obs is None:
        return {"available": False, "enabled": pv_cfg.enabled, "reason": "No usable sky observation yet"}

    latest_fc = db.query(Forecast).order_by(Forecast.generated_at.desc(), Forecast.id.desc()).first()
    fc_list: list[dict[str, Any]] = []
    if latest_fc is not None:
        rows = (
            db.query(Forecast)
            .filter(Forecast.generated_at == latest_fc.generated_at)
            .order_by(Forecast.horizon_minutes.asc())
            .all()
        )
        fc_list = [
            {"horizon_minutes": r.horizon_minutes, "cloud_cover_prediction": r.cloud_cover_prediction}
            for r in rows
        ]

    res = estimate_solar_pv_and_ramps(
        lat=cfg.camera.latitude,
        lon=cfg.camera.longitude,
        thin_cloud_pct=float(obs.thin_cloud_pct),
        dense_cloud_pct=float(obs.dense_cloud_pct),
        dark_cloud_pct=float(obs.dark_cloud_pct),
        cloud_coverage_pct=float(obs.cloud_coverage_pct),
        pv_kwp=pv_cfg.pv_kwp,
        panel_tilt_deg=pv_cfg.tilt_deg,
        panel_azimuth_deg=pv_cfg.azimuth_deg,
        system_loss_fraction=pv_cfg.system_loss_fraction,
        ramp_alert_pct=pv_cfg.ramp_alert_pct,
        forecast_horizons=fc_list,
        dt_utc=obs.captured_at,
    )
    return {
        "available": True,
        "enabled": pv_cfg.enabled,
        "observation_id": obs.id,
        "captured_at": obs.captured_at.isoformat() if obs.captured_at else None,
        **res,
    }


@app.get("/api/analysis/history")
def api_analysis_history(
    limit: int = Query(120, ge=1, le=2000),
    hours: float | None = Query(None, gt=0, le=720),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    q = db.query(Observation)
    if hours is not None:
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        q = q.filter(Observation.captured_at >= cutoff)
    rows = q.order_by(Observation.captured_at.desc(), Observation.id.desc()).limit(limit).all()
    rows.reverse()
    return {"count": len(rows), "items": [observation_to_dict(r) for r in rows]}


@app.get("/api/analysis/cloud-mask")
def api_analysis_cloud_mask(
    observation_id: int | None = None,
    overlay: bool = True,
    db: Session = Depends(get_db),
):
    q = db.query(Observation)
    obs = q.filter(Observation.id == observation_id).first() if observation_id else q.order_by(Observation.id.desc()).first()
    if not obs:
        raise HTTPException(status_code=404, detail="No observation found")
    target = obs.seg_overlay_path if overlay else obs.seg_mask_path
    if target and Path(target).exists():
        return FileResponse(target)
    raise HTTPException(status_code=404, detail="Cloud mask image not available for this observation")


@app.get("/api/analysis/motion")
def api_analysis_motion(
    observation_id: int | None = None,
    image: bool = False,
    db: Session = Depends(get_db),
):
    q = db.query(Observation)
    obs = q.filter(Observation.id == observation_id).first() if observation_id else q.order_by(Observation.id.desc()).first()
    if not obs:
        return {"available": False, "motion": None}
    if image:
        if obs.motion_overlay_path and Path(obs.motion_overlay_path).exists():
            return FileResponse(obs.motion_overlay_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="Motion overlay image missing")
    return {
        "available": True,
        "observation_id": obs.id,
        "captured_at": obs.captured_at.isoformat(),
        "motion": safe_json_loads(obs.motion_metrics_json, {}),
    }


@app.get("/api/analysis/quality")
def api_analysis_quality(observation_id: int | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    q = db.query(Observation)
    obs = q.filter(Observation.id == observation_id).first() if observation_id else q.order_by(Observation.id.desc()).first()
    if not obs:
        return {"available": False, "quality": None}
    return {
        "available": True,
        "observation_id": obs.id,
        "quality_score": obs.quality_score,
        "is_usable": obs.is_usable,
        "quality": safe_json_loads(obs.quality_metrics_json, {}),
    }


@app.get("/api/analysis/image/{observation_id}")
def api_observation_image(
    observation_id: int,
    layer: str = Query("raw", pattern="^(raw|segmentation|motion)$"),
    db: Session = Depends(get_db),
):
    obs = db.query(Observation).filter(Observation.id == observation_id).first()
    if not obs:
        raise HTTPException(status_code=404, detail="Observation not found")
    path = obs.raw_path
    if layer == "segmentation":
        path = obs.seg_overlay_path or obs.raw_path
    elif layer == "motion":
        path = obs.motion_overlay_path or obs.raw_path
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Image not found on disk")
    return FileResponse(path, media_type="image/jpeg")


# =====================================================================
# MODELS & TRAINING ENDPOINTS
# =====================================================================


@app.get("/api/models")
def api_models_list(db: Session = Depends(get_db)) -> dict[str, Any]:
    from app.ai.registry import is_default_baseline_model

    rows = db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc(), ModelRegistry.id.desc()).all()
    return {
        "count": len(rows),
        "items": [
            {
                "model_id": r.model_id,
                "model_family": r.model_family,
                "architecture": r.architecture,
                "version": r.version,
                "weights_path": r.weights_path,
                "weights_exists": Path(r.weights_path).exists(),
                "metrics": safe_json_loads(r.metrics_json, {}),
                "confusion_matrix": safe_json_loads(r.confusion_matrix_json, {}),
                "is_active": r.is_active,
                "is_default": is_default_baseline_model(r),
                "notes": r.notes,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
    }


@app.get("/api/models/active")
def api_models_active(db: Session = Depends(get_db)) -> dict[str, Any]:
    get_model_cache().refresh_from_db(db)
    return {"active": get_model_cache().active_summary()}


@app.post("/api/models/train")
def api_models_train(req: TrainModelRequest) -> dict[str, Any]:
    cfg = get_config_manager().get()
    if not cfg.ai.training_enabled:
        raise HTTPException(status_code=403, detail="Training is disabled in configuration")
    try:
        return get_training_manager().start_training(
            model_family=req.model_family,
            dataset_id=req.dataset_id,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            resume_from_job_id=req.resume_from_job_id,
            auto_activate=req.auto_activate,
            async_run=True,
            device_override=req.device,
            architecture_override=req.architecture,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/training/status")
def api_models_training_status(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(TrainingSession).order_by(TrainingSession.started_at.desc(), TrainingSession.id.desc()).limit(limit).all()
    return {
        "sessions": [
            {
                "job_id": r.job_id,
                "model_family": r.model_family,
                "architecture": r.architecture,
                "dataset_id": r.dataset_id,
                "status": r.status,
                "current_epoch": r.current_epoch,
                "total_epochs": r.total_epochs,
                "batch_size": r.batch_size,
                "learning_rate": r.learning_rate,
                "device_used": r.device_used,
                "mixed_precision": r.mixed_precision,
                "best_val_metric": r.best_val_metric,
                "metrics_history": safe_json_loads(r.metrics_history_json, []),
                "error_message": r.error_message,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in rows
        ]
    }


@app.post("/api/models/training/stop")
def api_models_training_stop(job_id: str | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    stopped = get_training_manager().request_stop(job_id)
    # Also stop any active Auto-Train cycle so it does not launch the next model family
    try:
        get_auto_learn_manager().request_stop()
    except Exception:
        pass
    # Immediately mark any stuck/orphaned or active DB training sessions as stopped
    q = db.query(TrainingSession).filter(TrainingSession.status.in_(["running", "queued"]))
    if job_id:
        q = q.filter(TrainingSession.job_id == job_id)
    rows = q.all()
    db_stopped: list[str] = []
    now_dt = datetime.now(UTC)
    for r in rows:
        r.status = "stopped"
        r.finished_at = now_dt
        db_stopped.append(r.job_id)
    if rows:
        db.commit()
    all_stopped = sorted(set(stopped) | set(db_stopped))
    return {"stop_requested_for": all_stopped, "stopped_count": len(all_stopped)}


@app.post("/api/models/activate")
def api_models_activate(req: ActivateModelRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if req.model_id == BUILTIN_RADIOMETRIC_MODEL_ID:
        res = deactivate_model_family(db, "segmentation")
        return {
            "activated": BUILTIN_RADIOMETRIC_MODEL_ID,
            "model_family": "segmentation",
            "version": "built-in",
            "deactivated_count": res["deactivated_count"],
        }
    try:
        row = activate_model_by_id(db, req.model_id)
        return {"activated": row.model_id, "model_family": row.model_family, "version": row.version}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/deactivate")
def api_models_deactivate(req: RollbackModelRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Deactivate neural checkpoints for a model family and return to built-in engine (e.g. radiometric-physics-v1)."""
    try:
        return deactivate_model_family(db, req.model_family)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/rollback")
def api_models_rollback(req: RollbackModelRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        row = rollback_model_family(db, req.model_family)
        return {"rolled_back_to": row.model_id, "model_family": row.model_family, "version": row.version}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/prune")
def api_models_prune(req: PruneModelsRequest | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Delete old standby models while keeping the active model (plus optional backup count) per family."""
    payload = req or PruneModelsRequest()
    return prune_old_models(
        db=db,
        keep_per_family=payload.keep_per_family,
        model_family=payload.model_family,
        also_prune_sessions=payload.also_prune_sessions,
    )


@app.delete("/api/models/training/sessions")
def api_models_clear_sessions(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Clear finished/failed/stopped training session history logs."""
    deleted = (
        db.query(TrainingSession)
        .filter(TrainingSession.status.notin_(["running", "queued"]))
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted_sessions": int(deleted)}


@app.delete("/api/models/{model_id}")
def api_model_delete(model_id: str, force: bool = Query(default=False), db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return delete_model_by_id(db, model_id=model_id, force=force)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/models/auto-train/status")
def api_models_auto_train_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    return get_auto_learn_manager().get_status(db, cfg)


@app.put("/api/models/auto-train/config")
def api_models_auto_train_config(
    req: AutoTrainConfigRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        updated = get_config_manager().replace_section("ai", {"auto_train": patch})
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_auto_learn_manager().get_status(db, updated)


@app.post("/api/models/auto-train/run")
def api_models_auto_train_run(req: AutoTrainRunRequest | None = None) -> dict[str, Any]:
    payload = req or AutoTrainRunRequest()
    try:
        return get_auto_learn_manager().trigger_run(
            families=payload.families,
            epochs=payload.epochs,
            only_label=payload.only_label,
            async_run=True,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/auto-train/stop")
def api_models_auto_train_stop() -> dict[str, Any]:
    return get_auto_learn_manager().request_stop()


# =====================================================================
# DATASET & LABELING ENDPOINTS
# =====================================================================


@app.get("/api/dataset")
def api_dataset_list(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(DatasetRecord).order_by(DatasetRecord.created_at.desc()).all()
    summaries = [get_dataset_summary(db, r.id) for r in rows]
    return {"datasets": summaries, "public_catalog": PUBLIC_DATASET_CATALOG}


@app.get("/api/dataset/{dataset_id}/items")
def api_dataset_items(dataset_id: int, limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).limit(limit).all()
    return {
        "count": len(items),
        "items": [
            {
                "id": it.id,
                "split": it.split,
                "group_key": it.group_key,
                "sequence_index": it.sequence_index,
                "image_path": it.image_path,
                "mask_path": it.mask_path,
                "cloud_type_label": it.cloud_type_label,
                "event_label": it.event_label,
                "quality_label": it.quality_label,
                "cloud_coverage_gt": it.cloud_coverage_gt,
                "user_labeled": it.user_labeled,
            }
            for it in items
        ],
    }


@app.post("/api/dataset/bootstrap")
def api_dataset_bootstrap(db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    rec = bootstrap_local_sky_dataset(db, cfg)
    return get_dataset_summary(db, rec.id)


@app.post("/api/dataset/import")
def api_dataset_import(req: DatasetImportRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    allowed_root = os.path.normpath(os.path.realpath(str(cfg.data_dir)))
    datasets_root = os.path.normpath(os.path.realpath(str(cfg.datasets_dir)))
    raw_dir = req.directory_path.strip()
    if not raw_dir:
        raise HTTPException(status_code=400, detail="Dataset directory path cannot be empty")
    try:
        if os.path.isabs(raw_dir):
            rel_sub = os.path.relpath(os.path.normpath(raw_dir), allowed_root)
            normalized_dir = os.path.normpath(os.path.join(allowed_root, rel_sub))
        else:
            normalized_dir = os.path.normpath(os.path.join(datasets_root, raw_dir))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid dataset directory path") from exc
    if not normalized_dir.startswith(allowed_root + os.sep):
        raise HTTPException(status_code=400, detail="Dataset directory must reside inside local data directory")
    resolved_dir = os.path.realpath(normalized_dir)
    if not resolved_dir.startswith(allowed_root + os.sep):
        raise HTTPException(status_code=400, detail="Dataset directory must reside inside local data directory")

    try:
        rec = import_local_dataset(
            db=db,
            config=cfg,
            name=req.name,
            source_dir=Path(resolved_dir),
            license_name=req.license_name,
            description=req.description,
        )
        return get_dataset_summary(db, rec.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="Dataset directory does not exist inside local data directory") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid dataset directory path or no readable images found") from exc


@app.post("/api/dataset/label")
def api_dataset_label(req: DatasetLabelRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Save user-edited segmentation mask, cloud class, event label, and quality label for a captured frame."""
    cfg = get_config_manager().get()
    if req.observation_id is None and req.dataset_item_id is None:
        raise HTTPException(status_code=400, detail="Provide observation_id or dataset_item_id")

    ds = db.query(DatasetRecord).filter(DatasetRecord.name == req.dataset_name).first()
    if not ds:
        ds_root = cfg.datasets_dir / req.dataset_name
        ds_root.mkdir(parents=True, exist_ok=True)
        ds = DatasetRecord(
            name=req.dataset_name,
            version="1.0.0",
            source_type="user_labeled_camera",
            license_name="Local User Camera Observations",
            description="User-labeled sky camera observations for local fine-tuning",
            root_path=str(ds_root),
            task_types_json=json.dumps(["segmentation", "classification", "event"]),
            item_count=0,
        )
        db.add(ds)
        db.flush()

    obs = db.query(Observation).filter(Observation.id == req.observation_id).first() if req.observation_id else None
    if req.observation_id and not obs:
        raise HTTPException(status_code=404, detail="Observation not found")

    ds_root = Path(ds.root_path)
    (ds_root / "images").mkdir(parents=True, exist_ok=True)
    (ds_root / "masks").mkdir(parents=True, exist_ok=True)

    mask_path = ""
    cov_gt = None

    if obs:
        obs.is_training_sample = True
        src_img = Path(obs.raw_path)
        if not src_img.exists():
            raise HTTPException(status_code=404, detail="Raw observation image missing on disk")
        dst_img = ds_root / "images" / f"obs_{obs.id}.jpg"
        if not dst_img.exists():
            dst_img.write_bytes(src_img.read_bytes())

        if req.mask_png_base64:
            raw_b64 = req.mask_png_base64.split(",", 1)[-1]
            mask_bytes = base64.b64decode(raw_b64)
            m_arr = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if m_arr is not None:
                if m_arr.ndim == 3:
                    # Convert RGBA/BGR canvas paint into class indices (blue=0 clear, white=2 dense, dark=3 dark, pale=1 thin)
                    gray = cv2.cvtColor(m_arr[:, :, :3], cv2.COLOR_BGR2GRAY)
                    class_m = np.where(gray > 180, 2, np.where(gray > 100, 1, 0)).astype(np.uint8)
                else:
                    class_m = m_arr
                dst_msk = ds_root / "masks" / f"obs_{obs.id}.png"
                cv2.imwrite(str(dst_msk), class_m)
                mask_path = str(dst_msk)
                cov_gt = round(float(np.mean((class_m >= 1) & (class_m <= 3)) * 100.0), 2)
        elif obs.seg_mask_path and Path(obs.seg_mask_path).exists():
            dst_msk = ds_root / "masks" / f"obs_{obs.id}.png"
            dst_msk.write_bytes(Path(obs.seg_mask_path).read_bytes())
            mask_path = str(dst_msk)
            cov_gt = obs.cloud_coverage_pct

        item = db.query(DatasetItem).filter(DatasetItem.dataset_id == ds.id, DatasetItem.observation_id == obs.id).first()
        if not item:
            item = DatasetItem(
                dataset_id=ds.id,
                observation_id=obs.id,
                split=req.split,
                group_key=obs.captured_at.strftime("%Y-%m-%d"),
                sequence_index=obs.id,
                image_path=str(dst_img),
                mask_path=mask_path,
                cloud_type_label=req.cloud_type_label or obs.dominant_cloud_type,
                event_label=req.event_label,
                quality_label=req.quality_label,
                cloud_coverage_gt=cov_gt,
                user_labeled=True,
            )
            db.add(item)
        else:
            item.split = req.split
            if mask_path:
                item.mask_path = mask_path
            item.cloud_type_label = req.cloud_type_label or item.cloud_type_label
            item.event_label = req.event_label
            item.quality_label = req.quality_label
            item.cloud_coverage_gt = cov_gt if cov_gt is not None else item.cloud_coverage_gt
            item.user_labeled = True

        ds.item_count = db.query(DatasetItem).filter(DatasetItem.dataset_id == ds.id).count()
        db.commit()
        return {"saved": True, "dataset_id": ds.id, "dataset_name": ds.name, "observation_id": obs.id}

    item = db.query(DatasetItem).filter(DatasetItem.id == req.dataset_item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")
    item.cloud_type_label = req.cloud_type_label or item.cloud_type_label
    item.event_label = req.event_label
    item.quality_label = req.quality_label
    item.split = req.split
    item.user_labeled = True
    db.commit()
    return {"saved": True, "dataset_item_id": item.id}


@app.post("/api/dataset/export")
def api_dataset_export(req: DatasetExportRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    ds = db.query(DatasetRecord).filter(DatasetRecord.id == req.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == ds.id).all()
    export_dir = cfg.datasets_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    name = req.export_name or f"{ds.name}_manifest.json"
    out_file = export_dir / name
    manifest = {
        "dataset": get_dataset_summary(db, ds.id),
        "items": [
            {
                "split": it.split,
                "group_key": it.group_key,
                "sequence_index": it.sequence_index,
                "image_path": it.image_path,
                "mask_path": it.mask_path,
                "cloud_type_label": it.cloud_type_label,
                "event_label": it.event_label,
                "quality_label": it.quality_label,
                "cloud_coverage_gt": it.cloud_coverage_gt,
            }
            for it in items
        ],
    }
    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"exported_path": str(out_file), "item_count": len(items)}


# =====================================================================
# SYSTEM & CONFIG ENDPOINTS
# =====================================================================


@app.get("/api/health")
@app.get("/api/system/health")
def api_system_health(db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    obs_count = db.query(Observation).count()
    return {
        "status": "healthy",
        "version": __version__,
        "offline_mode": cfg.server.offline_mode,
        "database_ok": True,
        "observations_count": obs_count,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/system/gpu")
def api_system_gpu() -> dict[str, Any]:
    cfg = get_config_manager().get()
    return get_hardware_status(
        cfg.ai.device,
        cfg.ai.max_gpu_memory_fraction,
        cfg.vision.inference_resolution,
        hardware_profile=cfg.ai.hardware_profile,
        cpu_threads=cfg.ai.cpu_threads,
    )


@app.put("/api/system/hardware-profile")
def api_system_hardware_profile(req: HardwareProfileRequest) -> dict[str, Any]:
    """Apply a universal hardware performance preset (low_end, balanced, high_end, ultra, auto) or manual CPU/GPU device selection."""
    cfg = get_config_manager().get()
    hw_now = get_hardware_status(
        cfg.ai.device,
        cfg.ai.max_gpu_memory_fraction,
        cfg.vision.inference_resolution,
        hardware_profile=cfg.ai.hardware_profile,
        cpu_threads=cfg.ai.cpu_threads,
    )

    ai_patch: dict[str, Any] = {}
    vision_patch: dict[str, Any] = {}

    if req.profile and req.profile != "custom":
        target_tier = hw_now["detected_tier"] if req.profile == "auto" else req.profile
        preset = HARDWARE_PROFILES.get(target_tier, HARDWARE_PROFILES["balanced"])
        ai_patch["hardware_profile"] = req.profile
        ai_patch["device"] = preset["device"]
        ai_patch["batch_size"] = preset["batch_size"]
        ai_patch["gradient_accumulation_steps"] = preset["gradient_accumulation_steps"]
        ai_patch["max_gpu_memory_fraction"] = preset["max_gpu_memory_fraction"]
        ai_patch["mixed_precision"] = preset["mixed_precision"]
        ai_patch["cpu_threads"] = preset["cpu_threads"]
        vision_patch["inference_resolution"] = preset["inference_resolution"]
    elif req.profile == "custom":
        ai_patch["hardware_profile"] = "custom"

    # Apply explicit manual overrides if provided
    if req.device is not None:
        ai_patch["device"] = req.device
    if req.batch_size is not None:
        ai_patch["batch_size"] = req.batch_size
    if req.gradient_accumulation_steps is not None:
        ai_patch["gradient_accumulation_steps"] = req.gradient_accumulation_steps
    if req.cpu_threads is not None:
        ai_patch["cpu_threads"] = req.cpu_threads
    if req.max_gpu_memory_fraction is not None:
        ai_patch["max_gpu_memory_fraction"] = req.max_gpu_memory_fraction
    if req.mixed_precision is not None:
        ai_patch["mixed_precision"] = req.mixed_precision
    if req.inference_resolution is not None:
        vision_patch["inference_resolution"] = req.inference_resolution

    patch_all: dict[str, Any] = {}
    if ai_patch:
        patch_all["ai"] = ai_patch
    if vision_patch:
        patch_all["vision"] = vision_patch

    updated = get_config_manager().update(patch_all) if patch_all else cfg
    hw_after = get_hardware_status(
        updated.ai.device,
        updated.ai.max_gpu_memory_fraction,
        updated.vision.inference_resolution,
        hardware_profile=updated.ai.hardware_profile,
        cpu_threads=updated.ai.cpu_threads,
    )
    return {
        "applied_profile": updated.ai.hardware_profile,
        "device": updated.ai.device,
        "active_device": hw_after["gpu"]["active_device"],
        "inference_resolution": updated.vision.inference_resolution,
        "batch_size": updated.ai.batch_size,
        "cpu_threads": updated.ai.cpu_threads,
        "hardware": hw_after,
    }


@app.get("/api/system/storage")
def api_system_storage(db: Session = Depends(get_db)) -> dict[str, Any]:
    return get_storage_stats(get_config_manager().get(), db=db)


@app.post("/api/system/storage/cleanup")
def api_system_storage_cleanup(db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    res = apply_retention_policy(db, cfg)
    return {"cleanup": res, "storage": get_storage_stats(cfg, db=db)}


@app.post("/api/system/storage/vacuum")
def api_system_storage_vacuum(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Run SQLite VACUUM to reclaim freed database pages cleanly."""
    cfg = get_config_manager().get()
    db.commit()
    bind = db.get_bind()
    with bind.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("VACUUM"))
    return {"vacuumed": True, "storage": get_storage_stats(cfg, db=db)}


@app.post("/api/system/storage/purge")
def api_system_storage_purge(req: DataPurgeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Granular or full-system reset of images, observations, forecasts, events, datasets, models & config."""
    if not req.targets:
        raise HTTPException(status_code=400, detail="No purge targets specified")
    cfg = get_config_manager().get()
    if "models" in req.targets or "datasets" in req.targets:
        try:
            get_auto_learn_manager().stop()
            get_training_manager().stop_job()
        except Exception:
            pass
    purged = purge_storage_targets(
        db=db,
        config=cfg,
        targets=list(req.targets),
        preserve_training_samples=req.preserve_training_samples,
        preserve_camera_connection=req.preserve_camera_connection,
    )
    if "models" in req.targets:
        get_model_cache().refresh_from_db(db)
    updated_cfg = get_config_manager().get()
    stats = get_storage_stats(updated_cfg, db=db)
    get_broadcaster().publish("storage_purged", {"purged": purged, "storage": stats})
    return {"purged": purged, "storage": stats}


@app.delete("/api/analysis/observation/{observation_id}")
def api_delete_observation(observation_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    res = delete_single_observation(db, cfg, observation_id)
    if not res["deleted"]:
        raise HTTPException(status_code=404, detail="Observation not found")
    return res


@app.get("/api/system/status")
def api_system_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    hw = get_hardware_status(
        cfg.ai.device,
        cfg.ai.max_gpu_memory_fraction,
        cfg.vision.inference_resolution,
        hardware_profile=cfg.ai.hardware_profile,
        cpu_threads=cfg.ai.cpu_threads,
    )
    st = get_storage_stats(cfg, db=db)
    cam = get_camera_manager().get_status()
    latest_obs = db.query(Observation).order_by(Observation.id.desc()).first()
    running_jobs = db.query(TrainingSession).filter(TrainingSession.status.in_(["queued", "running"])).count()
    online_wx = None
    if not cfg.server.offline_mode:
        online_wx = get_online_weather_service().get_weather(cfg.camera.latitude, cfg.camera.longitude, cfg)
    geom_sys, alt_src = _resolve_effective_geometry(cfg, cfg.camera.image_width, cfg.camera.image_height, online_wx)
    solar_sys = compute_solar_position(cfg.camera.latitude, cfg.camera.longitude)
    solar_proj = project_sky_angles_to_image(
        float(solar_sys["azimuth_deg"]),
        float(solar_sys["elevation_deg"]),
        100,
        100,
        geom_sys,
    )
    cam["geometry"] = {
        "orientation_elevation_deg": cfg.camera.orientation_elevation_deg,
        "orientation_azimuth_deg": cfg.camera.orientation_azimuth_deg,
        "orientation_roll_deg": cfg.camera.orientation_roll_deg,
        "fov_horizontal_deg": geom_sys.fov_horizontal_deg,
        "fov_vertical_deg": geom_sys.fov_vertical_deg,
        "cloud_altitude_m": geom_sys.cloud_altitude_m,
        "cloud_altitude_source": alt_src,
        "tilt_correction_enabled": cfg.camera.tilt_correction_enabled,
        "latitude": cfg.camera.latitude,
        "longitude": cfg.camera.longitude,
        "solar": {**solar_sys, **solar_proj},
    }
    return {
        "version": __version__,
        "offline_mode": cfg.server.offline_mode,
        "setup_completed": bool(cfg.server.setup_completed),
        "observations_count": db.query(Observation).count(),
        "privacy": (
            "100% local execution; zero external APIs or telemetry"
            if cfg.server.offline_mode
            else "Hybrid mode: local camera CV + Open-Meteo & RainViewer public APIs"
        ),
        "hardware": hw,
        "storage": st,
        "camera": cam,
        "latest_inference_ms": latest_obs.inference_duration_ms if latest_obs else None,
        "training_active_jobs": running_jobs,
        "active_models": get_model_cache().active_summary(),
        "online_weather": online_wx,
    }


@app.put("/api/system/mode")
def api_system_mode_put(req: SystemModeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    updated = get_config_manager().update({"server": {"offline_mode": req.offline_mode}})
    get_online_weather_service().clear_cache()
    latest_obs = db.query(Observation).filter(Observation.is_usable.is_(True)).order_by(Observation.id.desc()).first()
    if latest_obs is not None:
        generate_forecasts_for_observation(db, latest_obs, updated)
    return {
        "offline_mode": updated.server.offline_mode,
        "mode": "offline_only" if updated.server.offline_mode else "online_hybrid",
    }


# =====================================================================
# GEOLOCATION RADAR CLOUD MAP & ONLINE WEATHER ENDPOINTS
# =====================================================================


_CLOUD_TYPE_ALTITUDE_M: dict[str, float] = {
    "cirrus": 8500.0,
    "cirrostratus": 8000.0,
    "altocumulus": 4200.0,
    "altostratus": 3800.0,
    "cumulus": 1500.0,
    "stratocumulus": 1350.0,
    "cumulonimbus": 1100.0,
    "nimbostratus": 1000.0,
    "stratus": 650.0,
    "fog": 450.0,
    "haze": 550.0,
}


def _estimate_cloud_altitude_from_observation(obs: Observation | None) -> float | None:
    """Dynamically estimate cloud base altitude (m) directly from the latest camera image AI classification & cloud classes."""
    if obs is None:
        return None
    dom = str(getattr(obs, "dominant_cloud_type", "") or "").lower().strip()
    if dom in _CLOUD_TYPE_ALTITUDE_M and float(getattr(obs, "classification_confidence", 0.0) or 0.0) >= 0.30:
        return _CLOUD_TYPE_ALTITUDE_M[dom]

    thin = max(0.0, float(getattr(obs, "thin_cloud_pct", 0.0) or 0.0))
    dense = max(0.0, float(getattr(obs, "dense_cloud_pct", 0.0) or 0.0))
    dark = max(0.0, float(getattr(obs, "dark_cloud_pct", 0.0) or 0.0))
    haze = max(0.0, float(getattr(obs, "haze_fog_pct", 0.0) or 0.0))
    total = thin + dense + dark + haze
    if total < 1.0:
        return None
    # Weighted altitude from visual cloud morphology: thin high clouds (~4500m), dense cumulus (~1600m),
    # dark storm bases (~1100m), low haze/fog (~500m)
    est = (thin * 4500.0 + dense * 1600.0 + dark * 1100.0 + haze * 500.0) / total
    return round(max(400.0, min(9000.0, est)), 0)


def _resolve_effective_geometry(
    cfg: Any,
    width: int,
    height: int,
    online_wx: dict[str, Any] | None = None,
    obs: Observation | None = None,
) -> tuple[CameraGeometry, str]:
    geom = geometry_from_camera_config(cfg.camera, width=width, height=height)
    if cfg.camera.assumed_cloud_altitude_m is not None:
        return geom, "manual"

    wx = online_wx
    if wx is None and not cfg.server.offline_mode:
        wx = get_online_weather_service().get_weather(cfg.camera.latitude, cfg.camera.longitude, cfg)
    cur = (wx or {}).get("current") or {}
    lcl = cur.get("estimated_cloud_base_m")
    lcl_m = float(lcl) if lcl is not None and float(lcl) > 0 else None

    # 1) Online Dynamic Multi-Layer Altitude: weight Open-Meteo Low (LCL), Mid (~4000m) and High (~8500m) cloud decks
    c_low = float(cur.get("cloud_cover_low_pct") or 0.0)
    c_mid = float(cur.get("cloud_cover_mid_pct") or 0.0)
    c_high = float(cur.get("cloud_cover_high_pct") or 0.0)
    layer_sum = c_low + c_mid + c_high
    if layer_sum >= 5.0:
        h_low = lcl_m if lcl_m is not None else 1200.0
        h_mid = max(3600.0, h_low + 2000.0)
        h_high = 8500.0
        weighted_alt = round((c_low * h_low + c_mid * h_mid + c_high * h_high) / layer_sum, 0)
        return with_cloud_altitude(geom, weighted_alt), "open_meteo_layers"

    if lcl_m is not None:
        return with_cloud_altitude(geom, lcl_m), "open_meteo_lcl"

    # 2) Offline / Image-Based Dynamic Altitude from local AI cloud classification & segmentation classes
    vis_alt = _estimate_cloud_altitude_from_observation(obs)
    if vis_alt is not None:
        return with_cloud_altitude(geom, vis_alt), "vision_ai"

    return geom, "default_2000m"


def _extract_camera_cloud_grid(obs: Observation | None, grid_size: int = 12) -> list[list[dict[str, float]]]:
    if obs is None or not obs.seg_mask_path or not Path(obs.seg_mask_path).exists():
        return []
    mask = cv2.imread(obs.seg_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        return []
    h, w = mask.shape[:2]
    rh = max(1, h // grid_size)
    cw = max(1, w // grid_size)
    grid: list[list[dict[str, float]]] = []
    for r in range(grid_size):
        row_cells: list[dict[str, float]] = []
        y0, y1 = r * rh, (r + 1) * rh if r < grid_size - 1 else h
        for c in range(grid_size):
            x0, x1 = c * cw, (c + 1) * cw if c < grid_size - 1 else w
            patch = mask[y0:y1, x0:x1]
            total = max(1, patch.size)
            thin = float(np.count_nonzero(patch == 1) / total)
            dense = float(np.count_nonzero(patch == 2) / total)
            dark = float(np.count_nonzero(patch == 3) / total)
            haze = float(np.count_nonzero(patch == 4) / total)
            row_cells.append({
                "cloud": round(thin + dense + dark, 3),
                "thin": round(thin, 3),
                "dense": round(dense, 3),
                "dark": round(dark, 3),
                "haze": round(haze, 3),
            })
        grid.append(row_cells)
    return grid


def _apply_camera_geometry_and_recalculate(
    db: Session,
    patch: dict[str, Any],
    recalculate_latest: bool = True,
) -> dict[str, Any]:
    updated = get_config_manager().update({"camera": patch})
    recalculated_obs_id: int | None = None

    if recalculate_latest:
        obs = db.query(Observation).filter(Observation.is_usable.is_(True)).order_by(Observation.id.desc()).first()
        if obs is not None and obs.seg_mask_path and Path(obs.seg_mask_path).exists():
            seg_mask = cv2.imread(obs.seg_mask_path, cv2.IMREAD_GRAYSCALE)
            if seg_mask is not None and seg_mask.size > 0:
                mh, mw = seg_mask.shape[:2]
                geom = geometry_from_camera_config(updated.camera, width=mw, height=mh)
                geo_maps = compute_pixel_geometry_maps(mw, mh, geom)
                valid_mask = build_valid_sky_mask_from_config((mh, mw), updated.camera)
                if geom.tilt_correction_enabled:
                    hm = np.where(geo_maps["above_horizon"], valid_mask, 0).astype(np.uint8)
                    if int(np.count_nonzero(hm)) >= max(100, int(0.05 * mh * mw)):
                        valid_mask = hm

                prev_cov = safe_json_loads(obs.coverage_metrics_json, {})
                cov = compute_cloud_coverage(
                    class_map=seg_mask,
                    valid_mask=valid_mask,
                    mean_confidence=float(prev_cov.get("confidence", 0.8)),
                    grid_rows=updated.vision.grid_rows,
                    grid_cols=updated.vision.grid_cols,
                    pixel_weights=geo_maps["sky_weight"] if geom.tilt_correction_enabled else None,
                    geometry_meta={
                        "camera_elevation_deg": round(geom.elevation_deg, 2),
                        "tilt_from_vertical_deg": geom.tilt_from_vertical_deg,
                        "camera_azimuth_deg": round(geom.azimuth_deg, 2),
                        "below_horizon_pct": geo_maps["below_horizon_pct"],
                    }
                    if geom.tilt_correction_enabled
                    else None,
                )
                obs.cloud_coverage_pct = float(cov.get("cloud_coverage_pct", obs.cloud_coverage_pct))
                obs.clear_sky_pct = float(cov.get("clear_sky_pct", obs.clear_sky_pct))
                obs.thin_cloud_pct = float(cov.get("thin_cloud_pct", obs.thin_cloud_pct))
                obs.dense_cloud_pct = float(cov.get("dense_cloud_pct", obs.dense_cloud_pct))
                obs.dark_cloud_pct = float(cov.get("dark_cloud_pct", obs.dark_cloud_pct))
                obs.haze_fog_pct = float(cov.get("haze_fog_pct", obs.haze_fog_pct))
                obs.uncertainty_pct = float(cov.get("uncertainty_pct", obs.uncertainty_pct))
                obs.coverage_metrics_json = json.dumps(cov)

                # Recompute optical flow with new tilt/azimuth if consecutive raw frames exist
                prev_obs = (
                    db.query(Observation)
                    .filter(Observation.is_usable.is_(True), Observation.id < obs.id)
                    .order_by(Observation.id.desc())
                    .first()
                )
                if (
                    updated.vision.optical_flow_enabled
                    and obs.raw_path
                    and Path(obs.raw_path).exists()
                    and prev_obs is not None
                    and prev_obs.raw_path
                    and Path(prev_obs.raw_path).exists()
                ):
                    curr_bgr = cv2.imread(obs.raw_path)
                    prev_bgr = cv2.imread(prev_obs.raw_path)
                    if curr_bgr is not None and prev_bgr is not None and curr_bgr.shape[:2] == prev_bgr.shape[:2]:
                        ch, cw = curr_bgr.shape[:2]
                        m_fit = seg_mask if (mh, mw) == (ch, cw) else cv2.resize(seg_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
                        c_bin = np.where((m_fit >= 1) & (m_fit <= 3), 255, 0).astype(np.uint8)
                        v_fit = build_valid_sky_mask_from_config((ch, cw), updated.camera)
                        t_curr = obs.captured_at if obs.captured_at.tzinfo else obs.captured_at.replace(tzinfo=UTC)
                        t_prev = prev_obs.captured_at if prev_obs.captured_at.tzinfo else prev_obs.captured_at.replace(tzinfo=UTC)
                        dt_s = max(0.5, (t_curr - t_prev).total_seconds())
                        geom_m = geometry_from_camera_config(updated.camera, width=cw, height=ch)
                        mot, mot_overlay = estimate_cloud_motion(
                            prev_bgr=prev_bgr,
                            curr_bgr=curr_bgr,
                            valid_mask=v_fit,
                            cloud_binary_mask=c_bin,
                            dt_seconds=dt_s,
                            fov_horizontal_deg=geom_m.fov_horizontal_deg,
                            assumed_cloud_altitude_m=updated.camera.assumed_cloud_altitude_m,
                            grid_rows=updated.vision.grid_rows,
                            grid_cols=updated.vision.grid_cols,
                            fov_vertical_deg=geom_m.fov_vertical_deg,
                            orientation_elevation_deg=geom_m.elevation_deg,
                            orientation_azimuth_deg=geom_m.azimuth_deg,
                            orientation_roll_deg=geom_m.roll_deg,
                            tilt_correction_enabled=geom_m.tilt_correction_enabled,
                        )
                        obs.motion_speed_px_s = float(mot.get("magnitude_px_s", 0.0))
                        obs.motion_speed_deg_s = mot.get("angular_speed_deg_s")
                        obs.motion_direction_deg = float(mot.get("direction_deg", 0.0))
                        obs.motion_direction_cardinal = str(mot.get("direction_cardinal", "N"))
                        obs.motion_consistency = float(mot.get("consistency", 0.0))
                        obs.motion_confidence = float(mot.get("confidence", 0.0))
                        obs.motion_metrics_json = json.dumps(mot)
                        if obs.motion_overlay_path:
                            cv2.imwrite(obs.motion_overlay_path, mot_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

                db.commit()
                db.refresh(obs)
                recalculated_obs_id = obs.id
                get_broadcaster().publish("observation", observation_to_dict(obs))

    geom_out, alt_src = _resolve_effective_geometry(
        updated,
        width=updated.camera.image_width,
        height=updated.camera.image_height,
    )
    proj = compute_radar_footprint(
        updated.camera.latitude,
        updated.camera.longitude,
        geom_out,
        updated.camera.radar_radius_km,
        cloud_altitude_source=alt_src,
    )
    return {
        "orientation_elevation_deg": updated.camera.orientation_elevation_deg,
        "orientation_azimuth_deg": updated.camera.orientation_azimuth_deg,
        "orientation_roll_deg": updated.camera.orientation_roll_deg,
        "fov_horizontal_deg": updated.camera.fov_horizontal_deg,
        "fov_vertical_deg": updated.camera.fov_vertical_deg,
        "assumed_cloud_altitude_m": updated.camera.assumed_cloud_altitude_m,
        "tilt_correction_enabled": updated.camera.tilt_correction_enabled,
        "recalculated_observation_id": recalculated_obs_id,
        "camera_projection": proj,
    }


@app.put("/api/camera/geometry")
def api_camera_geometry_put(req: CameraGeometryRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "orientation_elevation_deg": req.orientation_elevation_deg,
        "orientation_azimuth_deg": req.orientation_azimuth_deg,
        "orientation_roll_deg": req.orientation_roll_deg,
        "tilt_correction_enabled": req.tilt_correction_enabled,
    }
    if req.fov_horizontal_deg is not None:
        patch["fov_horizontal_deg"] = req.fov_horizontal_deg
    if req.fov_vertical_deg is not None:
        patch["fov_vertical_deg"] = req.fov_vertical_deg
    if "assumed_cloud_altitude_m" in req.model_fields_set:
        patch["assumed_cloud_altitude_m"] = req.assumed_cloud_altitude_m

    return _apply_camera_geometry_and_recalculate(db, patch, recalculate_latest=req.recalculate_latest)


@app.post("/api/camera/geometry/auto-detect")
def api_camera_geometry_auto_detect(
    req: CameraAutoTiltRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    cfg = get_config_manager().get()
    req_obj = req or CameraAutoTiltRequest()
    frame_bgr: np.ndarray | None = None

    latest_obs = db.query(Observation).filter(Observation.is_usable.is_(True)).order_by(Observation.id.desc()).first()
    if latest_obs is not None and latest_obs.raw_path and Path(latest_obs.raw_path).exists():
        frame_bgr = cv2.imread(latest_obs.raw_path)

    if frame_bgr is None or frame_bgr.size == 0:
        cap = get_camera_manager().capture_once(cfg.camera, bypass_backoff=True)
        if cap.ok and cap.frame is not None and cap.frame.size > 0:
            frame_bgr = cap.frame

    if frame_bgr is None or frame_bgr.size == 0:
        raise HTTPException(status_code=404, detail="No camera frame available for automatic horizon detection")

    fov_h = float(req_obj.fov_horizontal_deg or cfg.camera.fov_horizontal_deg or 72.0)
    fov_v = req_obj.fov_vertical_deg if req_obj.fov_vertical_deg is not None else cfg.camera.fov_vertical_deg
    est = estimate_camera_tilt_from_frame(frame_bgr, fov_horizontal_deg=fov_h, fov_vertical_deg=fov_v)

    result: dict[str, Any] = {
        **est,
        "applied": bool(req_obj.apply),
    }
    if req_obj.apply:
        applied_res = _apply_camera_geometry_and_recalculate(
            db,
            {
                "orientation_elevation_deg": est["orientation_elevation_deg"],
                "orientation_roll_deg": est["orientation_roll_deg"],
                "tilt_correction_enabled": True,
            },
            recalculate_latest=req_obj.recalculate_latest,
        )
        result["geometry"] = applied_res
    return result


@app.post("/api/camera/geometry/from-sensors")
def api_camera_geometry_from_sensors(
    apply: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Read Android IP Webcam /sensors.json (accelerometer + compass) to propose camera tilt & heading."""
    cfg = get_config_manager().get()
    try:
        res = get_camera_manager().fetch_ipwebcam_orientation(cfg.camera.source_url, timeout=cfg.camera.timeout_seconds)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    out: dict[str, Any] = {**res, "applied": bool(apply)}
    if apply:
        patch: dict[str, Any] = {
            "orientation_elevation_deg": res["orientation_elevation_deg"],
            "orientation_roll_deg": res["orientation_roll_deg"],
            "tilt_correction_enabled": True,
        }
        if res.get("orientation_azimuth_deg") is not None:
            patch["orientation_azimuth_deg"] = res["orientation_azimuth_deg"]
        out["geometry"] = _apply_camera_geometry_and_recalculate(db, patch, recalculate_latest=True)
    return out


@app.post("/api/camera/geometry/refine-sun")
def api_camera_geometry_refine_sun(
    apply: bool = False,
    observation_id: int | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Refine compass azimuth using observed solar disk centroid inside FOV vs astronomical sun azimuth."""
    from app.vision.geometry import refine_azimuth_from_sun

    cfg = get_config_manager().get()
    q = db.query(Observation)
    obs = (
        q.filter(Observation.id == observation_id).first()
        if observation_id is not None
        else q.filter(Observation.is_usable.is_(True), Observation.raw_path != "").order_by(Observation.id.desc()).first()
    )
    if obs is None or not obs.raw_path or not Path(obs.raw_path).exists():
        raise HTTPException(status_code=404, detail="No usable camera frame available for solar calibration")
    img = cv2.imread(obs.raw_path)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="Could not read frame from disk")

    h, w = img.shape[:2]
    geom = geometry_from_camera_config(cfg.camera, width=w, height=h)
    valid_mask = build_valid_sky_mask_from_config((h, w), cfg.camera)
    res = refine_azimuth_from_sun(
        image_bgr=img,
        valid_mask=valid_mask,
        geom=geom,
        lat=cfg.camera.latitude,
        lon=cfg.camera.longitude,
        dt_utc=obs.captured_at,
    )
    out: dict[str, Any] = {**res, "observation_id": obs.id, "applied": False}
    if apply and res.get("ok"):
        out["applied"] = True
        out["geometry"] = _apply_camera_geometry_and_recalculate(
            db,
            {"orientation_azimuth_deg": res["proposed_azimuth_deg"], "tilt_correction_enabled": True},
            recalculate_latest=True,
        )
    return out


@app.put("/api/radar/location")
def api_radar_location_put(req: RadarLocationRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    cur_cfg = get_config_manager().get()
    patch: dict[str, Any] = {"latitude": req.latitude, "longitude": req.longitude}
    if req.location_description is not None:
        patch["location_description"] = req.location_description.strip()
    if not patch.get("location_description") and not cur_cfg.server.offline_mode:
        place = get_online_weather_service().reverse_geocode(req.latitude, req.longitude, cur_cfg)
        if place and place.get("label"):
            patch["location_description"] = place["label"]
    if req.radar_radius_km is not None:
        patch["radar_radius_km"] = req.radar_radius_km
    updated = get_config_manager().update({"camera": patch})
    get_online_weather_service().clear_cache()
    if not updated.server.offline_mode:
        latest_obs = db.query(Observation).filter(Observation.is_usable.is_(True)).order_by(Observation.id.desc()).first()
        if latest_obs is not None:
            generate_forecasts_for_observation(db, latest_obs, updated)

        def _bg_prefetch() -> None:
            try:
                prefetch_tiles_around(updated, updated.camera.latitude, updated.camera.longitude)
            except Exception:
                pass

        threading.Thread(target=_bg_prefetch, daemon=True).start()
    return {
        "latitude": updated.camera.latitude,
        "longitude": updated.camera.longitude,
        "location_description": updated.camera.location_description,
        "radar_radius_km": updated.camera.radar_radius_km,
    }


def _serialize_radar_obs(obs: Observation) -> dict[str, Any]:
    ts = obs.captured_at if obs.captured_at.tzinfo else obs.captured_at.replace(tzinfo=UTC)
    return {
        "id": obs.id,
        "time": int(ts.timestamp()),
        "captured_at": ts.isoformat(),
        "cloud_coverage_pct": obs.cloud_coverage_pct,
        "clear_sky_pct": obs.clear_sky_pct,
        "thin_cloud_pct": obs.thin_cloud_pct,
        "dense_cloud_pct": obs.dense_cloud_pct,
        "dark_cloud_pct": obs.dark_cloud_pct,
        "haze_fog_pct": obs.haze_fog_pct,
        "dominant_cloud_type": obs.dominant_cloud_type,
        "visual_condition": obs.visual_condition,
        "avg_luminance": obs.avg_luminance,
        "motion_speed_px_s": obs.motion_speed_px_s,
        "motion_speed_deg_s": obs.motion_speed_deg_s,
        "motion_direction_deg": obs.motion_direction_deg,
        "motion_direction_cardinal": obs.motion_direction_cardinal,
        "motion_consistency": obs.motion_consistency,
    }


def _build_camera_radar_history(db: Session, max_frames: int = 80, scan_limit: int = 600) -> list[dict[str, Any]]:
    rows = (
        db.query(Observation)
        .filter(Observation.is_usable.is_(True), Observation.seg_mask_path != "")
        .order_by(Observation.id.desc())
        .limit(scan_limit)
        .all()
    )
    valid_rows = [r for r in reversed(rows) if r.seg_mask_path and Path(r.seg_mask_path).exists()]
    n = len(valid_rows)
    if n > max_frames and max_frames >= 2:
        idxs = list(dict.fromkeys(int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)))
        valid_rows = [valid_rows[idx] for idx in idxs]
    return [_serialize_radar_obs(r) for r in valid_rows]


@app.get("/api/radar/state")
def api_radar_state(db: Session = Depends(get_db)) -> dict[str, Any]:
    cfg = get_config_manager().get()
    obs = db.query(Observation).filter(Observation.is_usable.is_(True)).order_by(Observation.id.desc()).first()
    wx_svc = get_online_weather_service()
    online_wx = None
    regional_grid: list[dict[str, Any]] = []
    rainviewer = None
    if not cfg.server.offline_mode:
        online_wx = wx_svc.get_weather(cfg.camera.latitude, cfg.camera.longitude, cfg)
        if not (cfg.camera.location_description or "").strip() and online_wx and online_wx.get("location_name"):
            cfg = get_config_manager().update(
                {"camera": {"location_description": str(online_wx["location_name"])}}
            )
        regional_grid = wx_svc.get_regional_cloud_grid(cfg.camera.latitude, cfg.camera.longitude, cfg, span_deg=2.2)
        rainviewer = wx_svc.get_radar_layers(cfg)

    span_half = 1.1
    regional_bounds = [
        [max(-85.0, cfg.camera.latitude - span_half), cfg.camera.longitude - span_half],
        [min(85.0, cfg.camera.latitude + span_half), cfg.camera.longitude + span_half],
    ]
    cam_w = int(obs.width) if (obs and obs.width and obs.width > 0) else cfg.camera.image_width
    cam_h = int(obs.height) if (obs and obs.height and obs.height > 0) else cfg.camera.image_height
    geom, alt_src = _resolve_effective_geometry(cfg, width=cam_w, height=cam_h, online_wx=online_wx, obs=obs)
    camera_projection = compute_radar_footprint(
        cfg.camera.latitude,
        cfg.camera.longitude,
        geom,
        cfg.camera.radar_radius_km,
        cloud_altitude_source=alt_src,
    )

    return {
        "offline_mode": cfg.server.offline_mode,
        "location": {
            "latitude": cfg.camera.latitude,
            "longitude": cfg.camera.longitude,
            "location_description": cfg.camera.location_description,
            "radar_radius_km": cfg.camera.radar_radius_km,
            "orientation_azimuth_deg": cfg.camera.orientation_azimuth_deg,
            "orientation_elevation_deg": cfg.camera.orientation_elevation_deg,
            "orientation_roll_deg": cfg.camera.orientation_roll_deg,
            "fov_horizontal_deg": cfg.camera.fov_horizontal_deg,
            "fov_vertical_deg": cfg.camera.fov_vertical_deg,
            "assumed_cloud_altitude_m": cfg.camera.assumed_cloud_altitude_m,
            "tilt_correction_enabled": cfg.camera.tilt_correction_enabled,
        },
        "camera_projection": camera_projection,
        "secondary_camera_projections": _build_secondary_projections(
            cfg,
            alt_m=float(camera_projection.get("cloud_altitude_m", 2000.0)),
            alt_src=str(camera_projection.get("cloud_altitude_source", "default")),
        ),
        "observation": _serialize_radar_obs(obs) if obs else None,
        "camera_history": _build_camera_radar_history(db, max_frames=80),
        "camera_cloud_grid": _extract_camera_cloud_grid(obs, grid_size=12),
        "online_weather": online_wx,
        "regional_cloud_grid": regional_grid,
        "regional_cloud_bounds": regional_bounds,
        "rainviewer": rainviewer,
        "tile_cache": get_tile_cache_stats(cfg),
    }


_secondary_camera_masks: dict[str, dict[str, Any]] = {}


def _build_secondary_projections(cfg, alt_m: float, alt_src: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sc in getattr(cfg.camera, "secondary_cameras", []):
        if not sc.enabled:
            continue
        geom_s = CameraGeometry(
            fov_horizontal_deg=float(sc.fov_horizontal_deg),
            fov_vertical_deg=float(sc.fov_vertical_deg),
            elevation_deg=float(sc.orientation_elevation_deg),
            azimuth_deg=float(sc.orientation_azimuth_deg),
            roll_deg=float(sc.orientation_roll_deg),
            cloud_altitude_m=float(alt_m),
            tilt_correction_enabled=True,
        )
        fp = compute_radar_footprint(
            cfg.camera.latitude,
            cfg.camera.longitude,
            geom_s,
            cfg.camera.radar_radius_km,
            cloud_altitude_source=alt_src,
        )
        cached = _secondary_camera_masks.get(sc.id)
        out.append({
            "id": sc.id,
            "name": sc.name,
            "enabled": sc.enabled,
            "azimuth_deg": sc.orientation_azimuth_deg,
            "elevation_deg": sc.orientation_elevation_deg,
            "fov_polygon": fp.get("fov_polygon", []),
            "view_center_latlon": fp.get("view_center_latlon"),
            "has_mask": cached is not None,
            "cloud_coverage_pct": cached["cloud_coverage_pct"] if cached else None,
            "captured_at": cached["captured_at"] if cached else None,
        })
    return out


@app.get("/api/cameras/secondary")
def api_secondary_cameras_get() -> dict[str, Any]:
    cfg = get_config_manager().get()
    cams = []
    for c in cfg.camera.secondary_cameras:
        d = c.model_dump()
        d["source_url"] = mask_url_credentials(d.get("source_url", ""))
        cams.append(d)
    return {
        "cameras": cams,
        "cached_ids": list(_secondary_camera_masks.keys()),
    }


@app.put("/api/cameras/secondary")
def api_secondary_cameras_put(req: SecondaryCamerasUpdateRequest) -> dict[str, Any]:
    existing_by_id = {c.id: c.source_url for c in get_config_manager().get().camera.secondary_cameras}
    sec_payloads = []
    for c in req.cameras:
        d = c.model_dump()
        d["source_url"] = _restore_masked_url(d.get("source_url", ""), existing_by_id.get(c.id, ""))
        sec_payloads.append(d)
    updated = get_config_manager().update({"camera": {"secondary_cameras": sec_payloads}})
    valid_ids = {c.id for c in updated.camera.secondary_cameras if c.enabled}
    for cid in list(_secondary_camera_masks.keys()):
        if cid not in valid_ids:
            _secondary_camera_masks.pop(cid, None)
    out_cams = []
    for c in updated.camera.secondary_cameras:
        d = c.model_dump()
        d["source_url"] = mask_url_credentials(d.get("source_url", ""))
        out_cams.append(d)
    return {"cameras": out_cams}


@app.post("/api/cameras/secondary/{cam_id}/snapshot")
def api_secondary_camera_snapshot(cam_id: str) -> dict[str, Any]:
    """Capture & segment one frame from a secondary camera and cache its mask for 360° Radar Fusion."""
    from app.config import CameraConfig
    from app.vision.segmentation import run_segmentation

    cfg = get_config_manager().get()
    target = next((c for c in cfg.camera.secondary_cameras if c.id == cam_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Secondary camera {cam_id} not found")

    temp_cam_cfg = CameraConfig(
        name=target.name,
        source_type=target.source_type,
        source_url=target.source_url,
        device_index=target.device_index,
        timeout_seconds=cfg.camera.timeout_seconds,
    )
    cap = get_camera_manager().capture_once(temp_cam_cfg)
    if not cap.success or cap.image_bgr is None:
        raise HTTPException(status_code=502, detail=f"Failed to capture from {target.name}: {cap.error}")

    h, w = cap.image_bgr.shape[:2]
    valid_mask = np.full((h, w), 255, dtype=np.uint8)
    cache = get_model_cache()
    seg_model, seg_meta, seg_dev = cache.get("segmentation")
    seg = run_segmentation(
        image_bgr=cap.image_bgr,
        valid_mask=valid_mask,
        inference_resolution=cfg.vision.inference_resolution,
        seg_model=seg_model,
        device=seg_dev,
        model_name=seg_meta.get("model_id", "radiometric-physics-v1") if seg_meta else "radiometric-physics-v1",
    )
    cov_pct = round(float(np.mean((seg.class_map >= 1) & (seg.class_map <= 3)) * 100.0), 1)
    _secondary_camera_masks[cam_id] = {
        "mask": seg.class_map,
        "cloud_coverage_pct": cov_pct,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    return {
        "captured": True,
        "id": cam_id,
        "name": target.name,
        "cloud_coverage_pct": cov_pct,
        "captured_at": _secondary_camera_masks[cam_id]["captured_at"],
    }


@app.get("/api/map/pmtiles")
def api_map_pmtiles_list() -> dict[str, Any]:
    """List locally installed PMTiles/Protomaps offline basemap archives."""
    from app.services.map_tiles import list_local_pmtiles

    cfg = get_config_manager().get()
    archives = list_local_pmtiles(cfg)
    return {"count": len(archives), "archives": archives}


@app.get("/api/map/pmtiles/{filename}")
def api_map_pmtiles_file(filename: str, request: Request) -> Response:
    """Serve a local .pmtiles archive with HTTP Range support (206 Partial Content) for offline Protomaps."""
    from app.services.map_tiles import resolve_pmtiles_path

    cfg = get_config_manager().get()
    try:
        path = resolve_pmtiles_path(cfg, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    file_size = path.stat().st_size
    range_hdr = (request.headers.get("range") or "").strip()
    if range_hdr.startswith("bytes="):
        spec = range_hdr[6:].split(",", 1)[0].strip()
        start_s, _, end_s = spec.partition("-")
        try:
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="Invalid Range header") from exc
        if start < 0 or end < start or start >= file_size:
            raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")
        end = min(end, file_size - 1)
        length = end - start + 1
        with path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(length)
        return Response(
            content=chunk,
            status_code=206,
            media_type="application/vnd.pmtiles",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Cache-Control": "public, max-age=86400",
            },
        )

    return FileResponse(
        str(path),
        media_type="application/vnd.pmtiles",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/radar/tiles/{style}/{z}/{x}/{y}.png")
def api_radar_tile(style: str, z: int, x: int, y: int):
    """Serve cached/local map tiles (zero outbound requests when offline_mode=True)."""
    cfg = get_config_manager().get()
    try:
        data, mime = get_tile_bytes(cfg, style, z, x, y)
        return Response(
            content=data,
            media_type=mime,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/radar/tiles/prefetch")
def api_radar_tiles_prefetch(req: MapPrefetchRequest | None = None) -> dict[str, Any]:
    """Download and cache map tiles locally around camera pin while in Online Hybrid mode."""
    cfg = get_config_manager().get()
    payload = req or MapPrefetchRequest()
    lat = payload.latitude if payload.latitude is not None else cfg.camera.latitude
    lon = payload.longitude if payload.longitude is not None else cfg.camera.longitude
    try:
        return prefetch_tiles_around(
            config=cfg,
            lat=lat,
            lon=lon,
            styles=payload.styles,
            zooms=payload.zooms,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/api/radar/regional-clouds.png")
def api_radar_regional_clouds():
    """Render a smooth interpolated BGRA cloud-cover raster from Open-Meteo's 5x5 regional mesh."""
    cfg = get_config_manager().get()
    size = 384
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    if not cfg.server.offline_mode:
        pts = get_online_weather_service().get_regional_cloud_grid(
            cfg.camera.latitude, cfg.camera.longitude, cfg, span_deg=2.2, grid_n=5
        )
        if len(pts) >= 25:
            cov_5x5 = np.array([float(p.get("cloud_cover_pct", 0.0)) for p in pts[:25]], dtype=np.float32).reshape(5, 5)
            # Flip vertically so north (higher lat) is at top row 0
            cov_5x5 = cov_5x5[::-1, :]
            precip_5x5 = np.array([float(p.get("precipitation_mm", 0.0)) for p in pts[:25]], dtype=np.float32).reshape(5, 5)[::-1, :]

            cov_up = cv2.resize(cov_5x5, (size, size), interpolation=cv2.INTER_CUBIC)
            cov_up = cv2.GaussianBlur(cov_up, (31, 31), 0)
            precip_up = cv2.resize(precip_5x5, (size, size), interpolation=cv2.INTER_CUBIC)

            cov_norm = np.clip(cov_up / 100.0, 0.0, 1.0)
            # Crisp cloud white-to-blue tint when precipitating
            blue_tint = np.clip(precip_up * 80.0, 0.0, 60.0)
            rgba[:, :, 0] = np.clip(245 + blue_tint * 0.15, 0, 255).astype(np.uint8)
            rgba[:, :, 1] = np.clip(245 - blue_tint * 0.4, 0, 255).astype(np.uint8)
            rgba[:, :, 2] = np.clip(248 - blue_tint * 0.9, 0, 255).astype(np.uint8)

            # Soft radial edge feathering so regional overlay blends cleanly into the basemap
            yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
            edge_dist = np.minimum(
                np.minimum(xx, size - 1 - xx),
                np.minimum(yy, size - 1 - yy),
            ) / (size * 0.18)
            feather = np.clip(edge_dist, 0.0, 1.0)
            alpha = np.where(cov_norm > 0.08, cov_norm * 190.0 * feather, 0.0)
            rgba[:, :, 3] = np.clip(alpha, 0.0, 200.0).astype(np.uint8)

    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode regional cloud raster")
    return Response(content=buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/radar/cloud-overlay.png")
def api_radar_cloud_overlay(
    obs_id: int | None = Query(default=None, ge=1),
    size: int = Query(default=384, ge=128, le=768),
    db: Session = Depends(get_db),
):
    """Render a smooth volumetric RGBA PNG sky cloud overlay projected onto the geographic radar grid."""
    cfg = get_config_manager().get()
    if obs_id is not None:
        obs = db.query(Observation).filter(Observation.id == obs_id, Observation.is_usable.is_(True)).first()
    else:
        obs = db.query(Observation).filter(Observation.is_usable.is_(True)).order_by(Observation.id.desc()).first()
    rgba = np.zeros((size, size, 4), dtype=np.uint8)

    if obs and obs.seg_mask_path and Path(obs.seg_mask_path).exists():
        mask = cv2.imread(obs.seg_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.size > 0:
            mh, mw = mask.shape[:2]
            geom, alt_src = _resolve_effective_geometry(cfg, width=mw, height=mh, obs=obs)
            footprint = compute_radar_footprint(
                cfg.camera.latitude,
                cfg.camera.longitude,
                geom,
                cfg.camera.radar_radius_km,
                cloud_altitude_source=alt_src,
            )
            raw_bgr = cv2.imread(obs.raw_path) if (obs.raw_path and Path(obs.raw_path).exists()) else None
            valid_mask = build_valid_sky_mask_from_config((mh, mw), cfg.camera)
            rgba = project_mask_to_radar_bgra(
                mask,
                geom,
                overlay_radius_km=float(footprint["overlay_radius_km"]),
                size=size,
                raw_bgr=raw_bgr,
                valid_mask=valid_mask,
            )
            # Multi-Camera 360° Fusion: composite any live cached secondary camera cloud masks on the same radar canvas
            if obs_id is None and _secondary_camera_masks:
                for sc in getattr(cfg.camera, "secondary_cameras", []):
                    if not sc.enabled or sc.id not in _secondary_camera_masks:
                        continue
                    sec_mask = _secondary_camera_masks[sc.id]["mask"]
                    sec_geom = CameraGeometry(
                        fov_horizontal_deg=float(sc.fov_horizontal_deg),
                        fov_vertical_deg=float(sc.fov_vertical_deg),
                        elevation_deg=float(sc.orientation_elevation_deg),
                        azimuth_deg=float(sc.orientation_azimuth_deg),
                        roll_deg=float(sc.orientation_roll_deg),
                        cloud_altitude_m=float(geom.cloud_altitude_m),
                        tilt_correction_enabled=True,
                    )
                    sec_rgba = project_mask_to_radar_bgra(
                        sec_mask,
                        sec_geom,
                        overlay_radius_km=float(footprint["overlay_radius_km"]),
                        size=size,
                    )
                    # Alpha-max composite so complementary sky sectors fuse seamlessly
                    sec_a = sec_rgba[:, :, 3] > rgba[:, :, 3]
                    rgba[sec_a] = sec_rgba[sec_a]

    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode radar cloud overlay")
    cache_hdr = "public, max-age=300" if obs_id is not None else "no-store"
    return Response(content=buf.tobytes(), media_type="image/png", headers={"Cache-Control": cache_hdr})


@app.get("/api/weather/online")
def api_weather_online() -> dict[str, Any]:
    cfg = get_config_manager().get()
    return get_online_weather_service().get_weather(cfg.camera.latitude, cfg.camera.longitude, cfg)


@app.get("/api/weather/geocode")
def api_weather_geocode(
    q: str = Query(..., min_length=2, max_length=120),
    lang: str = Query("en", pattern="^(en|it)$"),
) -> dict[str, Any]:
    cfg = get_config_manager().get()
    if cfg.server.offline_mode:
        return {"offline_mode": True, "results": []}
    return {"offline_mode": False, "results": get_online_weather_service().geocode_search(q, lang, cfg)}


@app.get("/api/system/config")
def api_system_config_get() -> dict[str, Any]:
    dumped = get_config_manager().get().model_dump()
    if isinstance(dumped.get("camera"), dict):
        dumped["camera"] = _sanitize_camera_dump(dumped["camera"])
    return dumped


@app.put("/api/system/config")
def api_system_config_put(payload: dict[str, Any]) -> dict[str, Any]:
    current_cam = get_config_manager().get().camera
    safe_payload = dict(payload)
    if isinstance(safe_payload.get("camera"), dict) and "source_url" in safe_payload["camera"]:
        cam_p = dict(safe_payload["camera"])
        cam_p["source_url"] = _restore_masked_url(cam_p["source_url"], current_cam.source_url)
        safe_payload["camera"] = cam_p
    try:
        updated = get_config_manager().update(safe_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid system configuration") from exc
    dumped = updated.model_dump()
    if isinstance(dumped.get("camera"), dict):
        dumped["camera"] = _sanitize_camera_dump(dumped["camera"])
    return dumped


# =====================================================================
# REAL-TIME WEBSOCKET
# =====================================================================


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket) -> None:
    expected = get_configured_api_token()
    if expected:
        presented = extract_token_from_request_parts(ws.headers, ws.query_params)
        if not is_valid_api_token(presented, expected):
            await ws.close(code=1008, reason="Unauthorized")
            return
    await ws.accept()
    hub = get_broadcaster()
    hub.register(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.unregister(ws)
    except Exception:
        hub.unregister(ws)


# =====================================================================
# STATIC FRONTEND SERVING (if built dist exists)
# =====================================================================

_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend_dist.exists():
    assets_dir = _frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        candidate = _frontend_dist / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_frontend_dist / "index.html"))
