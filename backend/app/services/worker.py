"""Background Capture, Analysis, Forecasting, Verification, and Retention Worker."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.ai.autolearn import get_auto_learn_manager
from app.ai.registry import get_model_cache
from app.camera.manager import get_camera_manager
from app.config import AppConfig, get_config_manager
from app.database import get_db_session
from app.forecast.engine import generate_forecasts_for_observation
from app.forecast.verification import verify_due_forecasts
from app.models_db import Observation, safe_json_loads
from app.services.events import evaluate_events_for_observation, record_operational_alert
from app.services.storage import apply_retention_policy
from app.vision.pipeline import run_vision_pipeline


class LiveBroadcaster:
    """Async WebSocket broadcast hub fed from background or API threads."""

    def __init__(self) -> None:
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self, ws: Any) -> None:
        self._clients.add(ws)

    def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self._clients or self._loop is None:
            return
        msg = json.dumps({"type": event_type, "payload": payload, "ts": datetime.now(UTC).isoformat()})

        async def _send_all() -> None:
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
        except Exception:
            pass


_broadcaster = LiveBroadcaster()


def get_broadcaster() -> LiveBroadcaster:
    return _broadcaster


def observation_to_dict(obs: Observation) -> dict[str, Any]:
    cov_data = safe_json_loads(obs.coverage_metrics_json, {})
    return {
        "id": obs.id,
        "captured_at": obs.captured_at.isoformat() if obs.captured_at else None,
        "source_type": obs.source_type,
        "width": obs.width,
        "height": obs.height,
        "quality_score": obs.quality_score,
        "is_usable": obs.is_usable,
        "quality": safe_json_loads(obs.quality_metrics_json, {}),
        "cloud_coverage_pct": obs.cloud_coverage_pct,
        "clear_sky_pct": obs.clear_sky_pct,
        "thin_cloud_pct": obs.thin_cloud_pct,
        "dense_cloud_pct": obs.dense_cloud_pct,
        "dark_cloud_pct": obs.dark_cloud_pct,
        "sun_glare_pct": float(cov_data.get("sun_glare_pct", 0.0)),
        "haze_fog_pct": obs.haze_fog_pct,
        "uncertainty_pct": obs.uncertainty_pct,
        "coverage": cov_data,
        "dominant_cloud_type": obs.dominant_cloud_type,
        "classification_confidence": obs.classification_confidence,
        "classification_probabilities": safe_json_loads(obs.classification_probs_json, {}),
        "motion": safe_json_loads(obs.motion_metrics_json, {}),
        "motion_speed_px_s": obs.motion_speed_px_s,
        "motion_speed_deg_s": obs.motion_speed_deg_s,
        "motion_direction_deg": obs.motion_direction_deg,
        "motion_direction_cardinal": obs.motion_direction_cardinal,
        "motion_consistency": obs.motion_consistency,
        "motion_confidence": obs.motion_confidence,
        "avg_luminance": obs.avg_luminance,
        "blue_sky_index": obs.blue_sky_index,
        "color": safe_json_loads(obs.color_metrics_json, {}),
        "texture": safe_json_loads(obs.texture_metrics_json, {}),
        "visual_condition": obs.visual_condition,
        "inference_duration_ms": obs.inference_duration_ms,
        "capture_latency_ms": obs.capture_latency_ms,
        "has_raw": bool(obs.raw_path and Path(obs.raw_path).exists()),
        "has_seg_overlay": bool(obs.seg_overlay_path and Path(obs.seg_overlay_path).exists()),
        "has_motion_overlay": bool(obs.motion_overlay_path and Path(obs.motion_overlay_path).exists()),
        "is_training_sample": obs.is_training_sample,
    }


def process_and_store_frame(
    db: Session,
    image_bgr: np.ndarray,
    config: AppConfig,
    source_type: str,
    captured_at: datetime | None = None,
    capture_latency_ms: float = 0.0,
    archive_raw: bool = True,
    run_forecast: bool = True,
) -> Observation:
    """Run vision pipeline, save artifacts to disk, persist Observation, detect events, forecast & verify."""
    ts = captured_at or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)

    prev_obs = db.query(Observation).order_by(Observation.captured_at.desc(), Observation.id.desc()).first()
    prev_bgr = None
    prev_cloud_mask = None
    prev_color = None
    prev_texture = None
    dt_seconds = config.camera.capture_interval_seconds

    if prev_obs is not None:
        if prev_obs.raw_path and Path(prev_obs.raw_path).exists():
            prev_bgr = cv2.imread(prev_obs.raw_path)
        if prev_obs.seg_mask_path and Path(prev_obs.seg_mask_path).exists():
            pm = cv2.imread(prev_obs.seg_mask_path, cv2.IMREAD_GRAYSCALE)
            if pm is not None:
                prev_cloud_mask = np.where((pm >= 1) & (pm <= 3), 255, 0).astype(np.uint8)
        prev_color = safe_json_loads(prev_obs.color_metrics_json, None)
        prev_texture = safe_json_loads(prev_obs.texture_metrics_json, None)
        prev_ts = prev_obs.captured_at
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.replace(tzinfo=UTC)
        dt_seconds = max(0.5, (ts - prev_ts).total_seconds())

    # Recent coverages for rolling rate-of-change calculation
    recent_rows = (
        db.query(Observation)
        .filter(Observation.is_usable.is_(True))
        .order_by(Observation.captured_at.desc(), Observation.id.desc())
        .limit(6)
        .all()
    )
    recent_coverages: list[tuple[float, float]] = []
    if recent_rows:
        oldest_ts = recent_rows[-1].captured_at
        if oldest_ts.tzinfo is None:
            oldest_ts = oldest_ts.replace(tzinfo=UTC)
        for r in reversed(recent_rows):
            rts = r.captured_at if r.captured_at.tzinfo else r.captured_at.replace(tzinfo=UTC)
            recent_coverages.append(((rts - oldest_ts).total_seconds() / 60.0, float(r.cloud_coverage_pct)))

    cache = get_model_cache()
    seg_model, seg_meta, seg_dev = cache.get("segmentation")
    if getattr(config.ai, "seg_backend", "auto") == "radiometric_physics":
        seg_model, seg_meta = None, None
    cls_model, cls_meta, cls_dev = cache.get("classification")
    seg_miou = float((seg_meta or {}).get("metrics", {}).get("val_miou", 0.5) or 0.0)
    # Gate weak/under-trained checkpoints so they do not override calibrated sky physics
    seg_nn_weight = 0.75 if seg_miou >= 0.35 else 0.25

    res = run_vision_pipeline(
        image_bgr=image_bgr,
        config=config,
        prev_bgr=prev_bgr,
        prev_cloud_mask=prev_cloud_mask,
        prev_color_metrics=prev_color,
        prev_texture_metrics=prev_texture,
        recent_coverages=recent_coverages,
        dt_seconds=dt_seconds,
        seg_model=seg_model,
        seg_model_name=seg_meta.get("model_id", "radiometric-physics-v1") if seg_meta else "radiometric-physics-v1",
        seg_nn_weight=seg_nn_weight,
        cls_model=cls_model,
        cls_model_name=cls_meta.get("model_id", "physical-heuristic-v1") if cls_meta else "physical-heuristic-v1",
        device=seg_dev,
    )

    config.ensure_directories()
    tag = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"
    raw_path = ""
    seg_mask_path = ""
    seg_overlay_path = ""
    motion_overlay_path = ""

    if config.storage.image_storage_enabled and archive_raw:
        raw_file = config.raw_images_dir / f"sky_{tag}.jpg"
        cv2.imwrite(str(raw_file), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.camera.jpeg_quality])
        raw_path = str(raw_file)

    if config.storage.image_storage_enabled and res.class_map is not None:
        sm_file = config.processed_images_dir / f"seg_mask_{tag}.png"
        cv2.imwrite(str(sm_file), res.class_map)
        seg_mask_path = str(sm_file)

    if config.storage.image_storage_enabled and res.seg_overlay_bgr is not None:
        so_file = config.processed_images_dir / f"seg_overlay_{tag}.jpg"
        cv2.imwrite(str(so_file), res.seg_overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        seg_overlay_path = str(so_file)

    if config.storage.image_storage_enabled and res.motion_overlay_bgr is not None:
        mo_file = config.processed_images_dir / f"motion_overlay_{tag}.jpg"
        cv2.imwrite(str(mo_file), res.motion_overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        motion_overlay_path = str(mo_file)

    cov = res.coverage
    mot = res.motion
    cls = res.classification
    col = res.color

    obs = Observation(
        captured_at=ts,
        source_type=source_type,
        frame_hash=res.frame_hash,
        width=image_bgr.shape[1],
        height=image_bgr.shape[0],
        raw_path=raw_path,
        seg_mask_path=seg_mask_path,
        seg_overlay_path=seg_overlay_path,
        motion_overlay_path=motion_overlay_path,
        quality_score=res.quality_score,
        is_usable=res.is_usable,
        quality_metrics_json=json.dumps(res.quality),
        cloud_coverage_pct=float(cov.get("cloud_coverage_pct", 0.0)),
        clear_sky_pct=float(cov.get("clear_sky_pct", 0.0)),
        thin_cloud_pct=float(cov.get("thin_cloud_pct", 0.0)),
        dense_cloud_pct=float(cov.get("dense_cloud_pct", 0.0)),
        dark_cloud_pct=float(cov.get("dark_cloud_pct", 0.0)),
        haze_fog_pct=float(cov.get("haze_fog_pct", 0.0)),
        uncertainty_pct=float(cov.get("uncertainty_pct", 0.0)),
        coverage_metrics_json=json.dumps(cov),
        dominant_cloud_type=str(cls.get("dominant_cloud_type", "unknown")),
        classification_confidence=float(cls.get("confidence", 0.0)),
        classification_probs_json=json.dumps(cls.get("probabilities", {})),
        motion_speed_px_s=float(mot.get("magnitude_px_s", 0.0)),
        motion_speed_deg_s=mot.get("angular_speed_deg_s"),
        motion_direction_deg=float(mot.get("direction_deg", 0.0)),
        motion_direction_cardinal=str(mot.get("direction_cardinal", "N")),
        motion_consistency=float(mot.get("consistency", 0.0)),
        motion_confidence=float(mot.get("confidence", 0.0)),
        motion_metrics_json=json.dumps(mot),
        avg_luminance=float(col.get("avg_luminance", 0.0)),
        blue_sky_index=float(col.get("blue_sky_index", 0.0)),
        color_metrics_json=json.dumps(col),
        texture_metrics_json=json.dumps(res.texture),
        visual_condition=res.visual_condition,
        inference_duration_ms=res.inference_duration_ms,
        capture_latency_ms=round(capture_latency_ms, 2),
        is_archived=archive_raw,
    )
    db.add(obs)
    db.commit()
    db.refresh(obs)

    verify_due_forecasts(db, obs)
    evaluate_events_for_observation(db, obs, prev_obs, config)
    if run_forecast:
        generate_forecasts_for_observation(db, obs, config)

    _broadcaster.publish("observation", observation_to_dict(obs))
    return obs


class SkyObserverWorker:
    """Non-blocking background scheduler handling capture, analysis, forecast updates, and storage retention."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_capture_ts = 0.0
        self._last_forecast_ts = 0.0
        self._last_retention_ts = 0.0
        self._last_autotrain_check_ts = 0.0
        self._offline_alert_sent = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="sky-observer-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        get_camera_manager().close()

    def _loop(self) -> None:
        cam = get_camera_manager()
        while not self._stop_event.is_set():
            try:
                cfg = get_config_manager().get()
                now = time.time()

                # Do not poll external camera URL if source_type is local_import or source_url is unconfigured (unless demo_simulation)
                can_poll = (
                    (cfg.camera.source_type in {"demo_simulation", "webcam_device"})
                    or (cfg.camera.source_type in {"http_snapshot", "mjpeg_stream", "rtsp", "web_stream"} and bool(cfg.camera.source_url.strip()))
                )

                if can_poll and (now - self._last_capture_ts) >= cfg.camera.capture_interval_seconds:
                    self._last_capture_ts = now
                    cap = cam.capture_once(cfg.camera)
                    if cap.success and cap.image_bgr is not None:
                        self._offline_alert_sent = False
                        run_fc = (now - self._last_forecast_ts) >= cfg.forecast.update_interval_seconds
                        with get_db_session() as db:
                            process_and_store_frame(
                                db=db,
                                image_bgr=cap.image_bgr,
                                config=cfg,
                                source_type=cap.source_type,
                                captured_at=cap.captured_at,
                                capture_latency_ms=cap.latency_ms,
                                archive_raw=True,
                                run_forecast=run_fc,
                            )
                        if run_fc:
                            self._last_forecast_ts = now
                    else:
                        status = cam.get_status()
                        if status["consecutive_failures"] == 3 and not self._offline_alert_sent:
                            self._offline_alert_sent = True
                            with get_db_session() as db:
                                record_operational_alert(
                                    db,
                                    "camera_offline",
                                    "warning",
                                    "Camera Offline / Unreachable",
                                    f"Failed to capture from {cfg.camera.source_type}: {cap.error}",
                                    {"source_url": cfg.camera.source_url},
                                )

                # Periodic storage retention cleanup & sunset daily sky digest every 15 minutes
                if (now - self._last_retention_ts) >= 900.0:
                    self._last_retention_ts = now
                    with get_db_session() as db:
                        apply_retention_policy(db, cfg)
                        try:
                            from app.services.timelapse import maybe_generate_sunset_daily_digest

                            maybe_generate_sunset_daily_digest(db, cfg)
                        except Exception:
                            pass

                # Periodic location auto-train check every 60 seconds
                if (now - self._last_autotrain_check_ts) >= 60.0:
                    self._last_autotrain_check_ts = now
                    get_auto_learn_manager().maybe_trigger_scheduled(cfg)

            except Exception as exc:
                import logging

                logging.getLogger("sky_observer.worker").warning("Worker cycle error: %s", exc)

            self._stop_event.wait(1.0)


_worker = SkyObserverWorker()


def get_worker() -> SkyObserverWorker:
    return _worker
