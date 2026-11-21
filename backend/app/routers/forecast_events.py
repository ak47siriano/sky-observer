"""Forecast & Events API router."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import get_config_manager
from app.database import get_db
from app.forecast.verification import compute_verification_summary
from app.models_db import EventAlert, Forecast, Observation, safe_json_loads
from app.schemas import EventFeedbackRequest, EventsClearRequest

router = APIRouter()


def forecast_to_dict(fc: Forecast) -> dict[str, Any]:
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


def event_to_dict(ev: EventAlert) -> dict[str, Any]:
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


def apply_event_feedback(
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
    return event_to_dict(ev)


@router.get("/api/forecast/latest")
def api_forecast_latest(db: Session = Depends(get_db)) -> dict[str, Any]:
    latest_obs = db.query(Forecast).order_by(Forecast.generated_at.desc(), Forecast.id.desc()).first()
    if not latest_obs:
        return {
            "status": "unavailable",
            "message": "No forecasts generated yet. Connect a camera or import sky images.",
            "horizons": [],
        }
    rows = (
        db.query(Forecast)
        .filter(Forecast.generated_at == latest_obs.generated_at)
        .order_by(Forecast.horizon_minutes.asc())
        .all()
    )
    return {
        "status": latest_obs.status,
        "generated_at": latest_obs.generated_at.isoformat(),
        "model_version": latest_obs.model_version,
        "horizons": [forecast_to_dict(r) for r in rows],
    }


@router.get("/api/forecast/history")
def api_forecast_history(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(Forecast).order_by(Forecast.generated_at.desc(), Forecast.horizon_minutes.asc()).limit(limit).all()
    return {"count": len(rows), "items": [forecast_to_dict(r) for r in rows]}


@router.get("/api/forecast/verification")
def api_forecast_verification(model_version: str | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    return compute_verification_summary(db, model_version=model_version)


@router.get("/api/events")
def api_events(
    limit: int = Query(100, ge=1, le=1000),
    status: str | None = Query(None, pattern="^(pending|confirmed|rejected)$"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    summary = {
        "pending": db.query(EventAlert).filter(EventAlert.user_status == "pending").count(),
        "confirmed": db.query(EventAlert).filter(EventAlert.user_status == "confirmed").count(),
        "rejected": db.query(EventAlert).filter(EventAlert.user_status == "rejected").count(),
    }
    q = db.query(EventAlert)
    if status is not None:
        q = q.filter(EventAlert.user_status == status)
    rows = q.order_by(EventAlert.created_at.desc(), EventAlert.id.desc()).limit(limit).all()
    return {"count": len(rows), "summary": summary, "items": [event_to_dict(r) for r in rows]}


@router.get("/api/events/latest")
def api_events_latest(limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(EventAlert).order_by(EventAlert.created_at.desc(), EventAlert.id.desc()).limit(limit).all()
    return {"count": len(rows), "items": [event_to_dict(r) for r in rows]}


@router.put("/api/events/config")
def api_events_config(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        updated = get_config_manager().replace_section("events", payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return updated.events.model_dump()


@router.post("/api/events/{event_id}/ack")
def api_event_ack(event_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    ev = db.query(EventAlert).filter(EventAlert.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return apply_event_feedback(db, ev, "confirmed")


@router.post("/api/events/{event_id}/reject")
def api_event_reject(event_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    ev = db.query(EventAlert).filter(EventAlert.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return apply_event_feedback(db, ev, "rejected")


@router.post("/api/events/{event_id}/feedback")
def api_event_feedback(event_id: int, req: EventFeedbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    ev = db.query(EventAlert).filter(EventAlert.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return apply_event_feedback(db, ev, req.status, req.note)


@router.delete("/api/events/{event_id}")
def api_event_delete(event_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    ev = db.query(EventAlert).filter(EventAlert.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    obs_id = ev.observation_id
    db.delete(ev)
    db.flush()
    if obs_id is not None:
        obs = db.query(Observation).filter(Observation.id == obs_id).first()
        if obs:
            remaining = (
                db.query(EventAlert)
                .filter(EventAlert.observation_id == obs_id, EventAlert.user_status != "rejected")
                .count()
            )
            if remaining == 0:
                obs.is_event_snapshot = False
    db.commit()
    return {"deleted": True, "id": event_id}


@router.post("/api/events/clear")
def api_events_clear(
    req: EventsClearRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    payload = req or EventsClearRequest()
    scope = payload.scope
    rows = db.query(EventAlert).order_by(EventAlert.created_at.desc(), EventAlert.id.desc()).all()

    def _status_of(ev: EventAlert) -> str:
        return ev.user_status or ("confirmed" if ev.acknowledged else "pending")

    def _ts(dt: datetime | None) -> float:
        if dt is None:
            return 0.0
        return (dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)).timestamp()

    to_delete: list[EventAlert] = []
    if scope == "all":
        to_delete = rows
    elif scope == "pending":
        to_delete = [ev for ev in rows if _status_of(ev) == "pending"]
    elif scope == "rejected":
        to_delete = [ev for ev in rows if _status_of(ev) == "rejected"]
    else:
        seen_pending_types: set[str] = set()
        last_kept_ts_by_type: dict[str, float] = {}
        for ev in rows:
            st = _status_of(ev)
            ev_t = _ts(ev.created_at)
            if st == "confirmed":
                last_kept_ts_by_type.setdefault(ev.event_type, ev_t)
                continue
            if st == "pending":
                if ev.event_type in seen_pending_types:
                    to_delete.append(ev)
                    continue
                prev_kept = last_kept_ts_by_type.get(ev.event_type)
                if prev_kept is not None and (prev_kept - ev_t) < 900.0:
                    to_delete.append(ev)
                    continue
                seen_pending_types.add(ev.event_type)
                last_kept_ts_by_type[ev.event_type] = ev_t
            else:
                prev_kept = last_kept_ts_by_type.get(ev.event_type)
                if prev_kept is not None and (prev_kept - ev_t) < 900.0:
                    to_delete.append(ev)
                    continue
                last_kept_ts_by_type[ev.event_type] = ev_t

    affected_obs_ids = {ev.observation_id for ev in to_delete if ev.observation_id is not None}
    for ev in to_delete:
        db.delete(ev)
    db.flush()

    for obs_id in affected_obs_ids:
        obs = db.query(Observation).filter(Observation.id == obs_id).first()
        if obs:
            remaining = (
                db.query(EventAlert)
                .filter(EventAlert.observation_id == obs_id, EventAlert.user_status != "rejected")
                .count()
            )
            if remaining == 0:
                obs.is_event_snapshot = False

    db.commit()
    return {"cleared": True, "scope": scope, "deleted": len(to_delete)}
