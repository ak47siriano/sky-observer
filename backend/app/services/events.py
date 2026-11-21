"""Local Event Detection & Configurable Alert Generator with Cooldown, Hysteresis & Rejection Backoff."""

from __future__ import annotations

import json
from datetime import datetime, UTC
from typing import Any

from sqlalchemy.orm import Session

from app.config import AppConfig, get_config_manager
from app.models_db import EventAlert, Observation, safe_json_loads


class SkyEventEngine:
    """
    Stage 4: Hybrid/Deterministic Temporal Rule Engine replacing (or complementing) SkyEventClassifier-GRU.
    Classifies all 8 canonical sky & operational events directly from:
      - Stage 1 frame-to-frame deltas (cloud_coverage, dense, dark, luminance, quality flags)
      - Stage 2 WMO cloud class (e.g. cumulonimbus_like / convective_storm)
      - Stage 3 temporal trend (rate_of_change_pct_per_min)
    """

    EVENT_CLASSES = (
        "normal",
        "rapid_cloud_development",
        "sky_clearing",
        "cloud_cover_increasing",
        "dark_cloud_development",
        "storm_like_structure",
        "camera_obstruction",
        "unusual_change",
    )

    def classify(
        self,
        obs: Observation,
        prev_obs: Observation | None,
        config: AppConfig,
        forecast_delta_15m: float | None = None,
    ) -> dict[str, Any]:
        ev_cfg = config.events
        q_metrics = safe_json_loads(obs.quality_metrics_json, {})
        flags = set(q_metrics.get("flags", []) or [])
        cov_meta = safe_json_loads(obs.coverage_metrics_json, {})
        rate_per_min = float(cov_meta.get("rate_of_change_pct_per_min", 0.0) or 0.0)
        dom = str(obs.dominant_cloud_type or "").lower()

        cov_delta = (
            float(obs.cloud_coverage_pct) - float(prev_obs.cloud_coverage_pct)
            if (prev_obs is not None and prev_obs.is_usable and obs.is_usable)
            else 0.0
        )
        lum_delta = (
            float(obs.avg_luminance) - float(prev_obs.avg_luminance)
            if (prev_obs is not None and prev_obs.is_usable and obs.is_usable)
            else 0.0
        )
        dark_delta = (
            float(obs.dark_cloud_pct) - float(prev_obs.dark_cloud_pct)
            if (prev_obs is not None and prev_obs.is_usable and obs.is_usable)
            else 0.0
        )

        # Priority-ordered deterministic rules over the 8 canonical events
        if "lens_obstruction" in flags:
            label, conf, reason = "camera_obstruction", 0.95, "Lens obstruction flag raised by quality analyzer."
        elif obs.is_usable and (
            float(obs.dark_cloud_pct) >= ev_cfg.storm_like_dark_cloud_pct
            or (dom in {"cumulonimbus_like", "convective_storm"} and float(obs.dark_cloud_pct) >= 25.0)
        ):
            label, conf, reason = (
                "storm_like_structure",
                0.88,
                f"Dark cloud base {obs.dark_cloud_pct:.1f}% with WMO genus '{dom or 'unknown'}'.",
            )
        elif obs.is_usable and (
            lum_delta <= ev_cfg.rapid_darkening_delta or dark_delta >= 12.0
        ):
            label, conf, reason = (
                "dark_cloud_development",
                0.84,
                f"Sky luminance delta {lum_delta:+.1f} L*, dark cloud delta {dark_delta:+.1f}%.",
            )
        elif obs.is_usable and (
            cov_delta >= ev_cfg.rapid_cloud_increase_pct or rate_per_min >= 2.0
        ):
            label, conf, reason = (
                "rapid_cloud_development",
                0.85,
                f"Rapid cloud expansion ({cov_delta:+.1f}%, {rate_per_min:+.2f}%/min).",
            )
        elif obs.is_usable and (
            cov_delta <= -ev_cfg.rapid_cloud_decrease_pct
            or (forecast_delta_15m is not None and forecast_delta_15m <= -15.0)
        ):
            label, conf, reason = (
                "sky_clearing",
                0.82,
                f"Sky clearing detected ({cov_delta:+.1f}%).",
            )
        elif obs.is_usable and (
            cov_delta >= 8.0 or rate_per_min >= 0.6 or (forecast_delta_15m is not None and forecast_delta_15m >= 12.0)
        ):
            label, conf, reason = (
                "cloud_cover_increasing",
                0.76,
                f"Sustained cloud increase ({cov_delta:+.1f}%, trend {rate_per_min:+.2f}%/min).",
            )
        elif "camera_movement" in flags or "sudden_orientation_change" in flags or abs(cov_delta) >= 35.0:
            label, conf, reason = "unusual_change", 0.78, "Abrupt sky/camera transition detected."
        else:
            label, conf, reason = "normal", 0.90, "Sky state within normal variation thresholds."

        return {
            "event_class": label,
            "confidence": round(conf, 3),
            "reason": reason,
            "engine": "SkyEventEngine-Deterministic-v2",
            "deltas": {
                "cov_delta": round(cov_delta, 2),
                "lum_delta": round(lum_delta, 2),
                "dark_delta": round(dark_delta, 2),
                "rate_per_min": round(rate_per_min, 3),
            },
        }


_sky_event_engine = SkyEventEngine()


def get_sky_event_engine() -> SkyEventEngine:
    return _sky_event_engine


def _normalize_utc(ts: datetime | None) -> datetime:
    if ts is None:
        return datetime.now(UTC)
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _create_event(
    db: Session,
    event_type: str,
    severity: str,
    title: str,
    description: str,
    observation_id: int | None = None,
    metrics: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> EventAlert:
    ev = EventAlert(
        event_type=event_type,
        severity=severity,
        title=title,
        description=description,
        observation_id=observation_id,
        metrics_json=json.dumps(metrics or {}),
    )
    if created_at is not None:
        ev.created_at = _normalize_utc(created_at)
    db.add(ev)
    return ev


def _recent_blocking_event(
    db: Session,
    event_type: str,
    now: datetime,
    cooldown_minutes: int,
    rejected_backoff_minutes: int,
) -> EventAlert | None:
    last = (
        db.query(EventAlert)
        .filter(EventAlert.event_type == event_type)
        .order_by(EventAlert.created_at.desc(), EventAlert.id.desc())
        .first()
    )
    if last is None:
        return None
    last_ts = _normalize_utc(last.created_at)
    now_utc = _normalize_utc(now)
    wait_minutes = max(cooldown_minutes, rejected_backoff_minutes) if last.user_status == "rejected" else cooldown_minutes
    elapsed_sec = (now_utc - last_ts).total_seconds()
    if elapsed_sec < float(wait_minutes) * 60.0:
        return last
    return None


def _maybe_create_event(
    db: Session,
    event_type: str,
    severity: str,
    title: str,
    description: str,
    now: datetime,
    cooldown_minutes: int,
    rejected_backoff_minutes: int,
    observation_id: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> EventAlert | None:
    if _recent_blocking_event(db, event_type, now, cooldown_minutes, rejected_backoff_minutes) is not None:
        return None
    return _create_event(
        db=db,
        event_type=event_type,
        severity=severity,
        title=title,
        description=description,
        observation_id=observation_id,
        metrics=metrics,
        created_at=now,
    )


def evaluate_events_for_observation(
    db: Session,
    obs: Observation,
    prev_obs: Observation | None,
    config: AppConfig,
) -> list[EventAlert]:
    """Check quality flags, rapid cloud changes, darkening, and storm-like developments with anti-spam cooldown."""
    ev_cfg = config.events
    cd = int(ev_cfg.cooldown_minutes)
    rej_cd = int(ev_cfg.rejected_backoff_minutes)
    now = _normalize_utc(obs.captured_at)

    created: list[EventAlert] = []
    q_metrics = safe_json_loads(obs.quality_metrics_json, {})
    flags = q_metrics.get("flags", [])

    def _add(ev: EventAlert | None) -> None:
        if ev is not None:
            created.append(ev)

    if ev_cfg.camera_movement_enabled and ("camera_movement" in flags or "sudden_orientation_change" in flags):
        _add(
            _maybe_create_event(
                db,
                "camera_movement",
                "warning",
                "Camera Movement Detected",
                f"Detected frame shift of {q_metrics.get('camera_shift_px', 0):.1f} px; verify phone mount stability.",
                now,
                cd,
                rej_cd,
                obs.id,
                {"shift_px": q_metrics.get("camera_shift_px", 0), "flags": flags},
            )
        )

    if "lens_obstruction" in flags:
        _add(
            _maybe_create_event(
                db,
                "camera_obstruction",
                "critical",
                "Possible Camera Obstruction",
                "Low entropy or dark blockage detected across sky field.",
                now,
                cd,
                rej_cd,
                obs.id,
                {"flags": flags},
            )
        )
    elif not obs.is_usable:
        _add(
            _maybe_create_event(
                db,
                "image_quality_problem",
                "warning",
                "Image Quality Degraded",
                f"Frame flagged unusable ({', '.join(flags) or 'low quality score'}).",
                now,
                cd,
                rej_cd,
                obs.id,
                {"quality_score": obs.quality_score, "flags": flags},
            )
        )

    if (
        obs.is_usable
        and obs.quality_score >= 0.55
        and prev_obs is not None
        and prev_obs.is_usable
        and prev_obs.quality_score >= 0.55
    ):
        cov_delta = float(obs.cloud_coverage_pct) - float(prev_obs.cloud_coverage_pct)
        lum_delta = float(obs.avg_luminance) - float(prev_obs.avg_luminance)

        if cov_delta >= ev_cfg.rapid_cloud_increase_pct:
            _add(
                _maybe_create_event(
                    db,
                    "rapid_cloud_increase",
                    "info",
                    "Rapid Cloud Coverage Increase",
                    f"Cloud coverage jumped by +{cov_delta:.1f}% (now {obs.cloud_coverage_pct:.1f}%).",
                    now,
                    cd,
                    rej_cd,
                    obs.id,
                    {"delta_pct": round(cov_delta, 2), "cloud_coverage_pct": obs.cloud_coverage_pct},
                )
            )
        elif cov_delta <= -ev_cfg.rapid_cloud_decrease_pct:
            _add(
                _maybe_create_event(
                    db,
                    "rapid_sky_clearing",
                    "info",
                    "Rapid Sky Clearing",
                    f"Cloud coverage dropped by {cov_delta:.1f}% (now {obs.cloud_coverage_pct:.1f}%).",
                    now,
                    cd,
                    rej_cd,
                    obs.id,
                    {"delta_pct": round(cov_delta, 2), "cloud_coverage_pct": obs.cloud_coverage_pct},
                )
            )

        if lum_delta <= ev_cfg.rapid_darkening_delta:
            _add(
                _maybe_create_event(
                    db,
                    "rapid_sky_darkening",
                    "warning",
                    "Significant Sky Darkening",
                    f"Average sky luminance dropped by {lum_delta:.1f} L* units.",
                    now,
                    cd,
                    rej_cd,
                    obs.id,
                    {"lum_delta": round(lum_delta, 2), "avg_luminance": obs.avg_luminance},
                )
            )

    sustained_dense = prev_obs is None or (prev_obs.is_usable and prev_obs.dense_cloud_pct >= ev_cfg.dense_cloud_alert_pct * 0.85)
    if obs.is_usable and sustained_dense and obs.dense_cloud_pct >= ev_cfg.dense_cloud_alert_pct:
        _add(
            _maybe_create_event(
                db,
                "dense_cloud_region",
                "info",
                "Dense Cloud Cover Detected",
                f"Dense clouds cover {obs.dense_cloud_pct:.1f}% of valid sky.",
                now,
                cd,
                rej_cd,
                obs.id,
                {"dense_cloud_pct": obs.dense_cloud_pct},
            )
        )

    sustained_dark = prev_obs is None or (prev_obs.is_usable and prev_obs.dark_cloud_pct >= ev_cfg.storm_like_dark_cloud_pct * 0.85)
    if obs.is_usable and sustained_dark and obs.dark_cloud_pct >= ev_cfg.storm_like_dark_cloud_pct:
        _add(
            _maybe_create_event(
                db,
                "storm_like_development",
                "warning",
                "Possible Storm-Like Cloud Development",
                f"Dark cloud bases cover {obs.dark_cloud_pct:.1f}% of valid sky (visual indicator only).",
                now,
                cd,
                rej_cd,
                obs.id,
                {"dark_cloud_pct": obs.dark_cloud_pct, "dominant_cloud_type": obs.dominant_cloud_type},
            )
        )

    if created:
        obs.is_event_snapshot = True
        db.commit()
    return created


def record_operational_alert(
    db: Session,
    event_type: str,
    severity: str,
    title: str,
    description: str,
    metrics: dict[str, Any] | None = None,
) -> EventAlert:
    ev_cfg = get_config_manager().get().events
    now = datetime.now(UTC)
    existing = _recent_blocking_event(
        db,
        event_type=event_type,
        now=now,
        cooldown_minutes=int(ev_cfg.cooldown_minutes),
        rejected_backoff_minutes=int(ev_cfg.rejected_backoff_minutes),
    )
    if existing is not None:
        return existing
    ev = _create_event(db, event_type, severity, title, description, None, metrics, created_at=now)
    db.commit()
    db.refresh(ev)
    return ev
