"""Chronological Forecast Verification & Calibration Metrics (MAE, RMSE, Precision, Recall, F1, Brier, Calibration)."""

from __future__ import annotations

from datetime import UTC
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sqlalchemy.orm import Session

from app.models_db import Forecast, ForecastVerification, Observation


def _coarse_sky_category(cover_pct: float) -> str:
    if cover_pct < 20.0:
        return "clear"
    if cover_pct < 60.0:
        return "partly_cloudy"
    if cover_pct < 82.0:
        return "mostly_cloudy"
    return "overcast"


def _map_forecast_to_coarse(forecast_cond: str, predicted_cover: float) -> str:
    if forecast_cond in {"clear", "mostly_clear"}:
        return "clear"
    if forecast_cond == "partly_cloudy":
        return "partly_cloudy"
    if forecast_cond in {"overcast", "dark_overcast", "storm_like_visual_development"}:
        return "overcast"
    return _coarse_sky_category(predicted_cover)


def verify_due_forecasts(db: Session, observation: Observation, tolerance_minutes: float = 10.0) -> int:
    """
    Chronologically verify unverified forecasts whose target_time has arrived at observation.captured_at.
    Returns count of newly verified forecasts.
    """
    if not observation.is_usable:
        return 0
    obs_time = observation.captured_at
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=UTC)

    pending = (
        db.query(Forecast)
        .filter(
            Forecast.verified.is_(False),
            Forecast.status == "available",
            Forecast.target_time <= obs_time,
        )
        .order_by(Forecast.target_time.asc())
        .all()
    )
    verified_count = 0
    actual_cover = float(observation.cloud_coverage_pct)
    actual_cat = _coarse_sky_category(actual_cover)
    actual_overcast = 1.0 if actual_cover >= 80.0 else 0.0

    for fc in pending:
        t_target = fc.target_time
        if t_target.tzinfo is None:
            t_target = t_target.replace(tzinfo=UTC)
        # Skip forecasts whose target time was very old (> 2x horizon + tolerance) without marking against wrong future
        age_diff_min = (obs_time - t_target).total_seconds() / 60.0
        if age_diff_min > max(tolerance_minutes, fc.horizon_minutes * 1.5):
            # Mark expired so it doesn't distort real-time metrics
            fc.verified = True
            continue

        abs_err = abs(float(fc.cloud_cover_prediction) - actual_cover)
        sq_err = abs_err ** 2
        pred_cat = _map_forecast_to_coarse(fc.forecast_condition, float(fc.cloud_cover_prediction))
        cond_ok = pred_cat == actual_cat
        brier = (float(fc.overcast_prob) - actual_overcast) ** 2

        ver = ForecastVerification(
            forecast_id=fc.id,
            observation_id=observation.id,
            verified_at=obs_time,
            horizon_minutes=fc.horizon_minutes,
            predicted_cover=float(fc.cloud_cover_prediction),
            actual_cover=actual_cover,
            abs_error=round(abs_err, 3),
            sq_error=round(sq_err, 3),
            predicted_condition=pred_cat,
            actual_condition=actual_cat,
            condition_correct=cond_ok,
            brier_score=round(brier, 4),
            confidence=float(fc.confidence),
            model_version=fc.model_version,
        )
        db.add(ver)
        fc.verified = True
        verified_count += 1

    if pending:
        db.commit()
    return verified_count


def compute_verification_summary(db: Session, model_version: str | None = None) -> dict[str, Any]:
    """Compute MAE, RMSE, accuracy, precision, recall, F1, confusion matrix, Brier score, and calibration curve."""
    q = db.query(ForecastVerification)
    if model_version:
        q = q.filter(ForecastVerification.model_version == model_version)
    rows = q.order_by(ForecastVerification.verified_at.asc()).all()

    if not rows:
        return {
            "total_verified": 0,
            "status": "insufficient_verified_forecasts",
            "message": "No forecasts have reached their verification horizon yet.",
            "overall": None,
            "by_horizon": {},
            "by_model_version": {},
            "confusion_matrix": {"labels": [], "matrix": []},
            "calibration_bins": [],
            "mae_trend": [],
            "recent_samples": [],
        }

    abs_errs = np.array([r.abs_error for r in rows], dtype=np.float64)
    sq_errs = np.array([r.sq_error for r in rows], dtype=np.float64)
    briers = np.array([r.brier_score for r in rows], dtype=np.float64)
    y_true = [r.actual_condition for r in rows]
    y_pred = [r.predicted_condition for r in rows]
    labels = sorted(set(y_true) | set(y_pred))

    overall = {
        "mae_pct": round(float(np.mean(abs_errs)), 2),
        "rmse_pct": round(float(np.sqrt(np.mean(sq_errs))), 2),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "brier_score": round(float(np.mean(briers)), 4),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if labels else []

    by_horizon: dict[str, Any] = {}
    for h in sorted({r.horizon_minutes for r in rows}):
        sub = [r for r in rows if r.horizon_minutes == h]
        h_true = [r.actual_condition for r in sub]
        h_pred = [r.predicted_condition for r in sub]
        by_horizon[str(h)] = {
            "count": len(sub),
            "mae_pct": round(float(np.mean([r.abs_error for r in sub])), 2),
            "rmse_pct": round(float(np.sqrt(np.mean([r.sq_error for r in sub]))), 2),
            "accuracy": round(float(accuracy_score(h_true, h_pred)), 4),
            "brier_score": round(float(np.mean([r.brier_score for r in sub])), 4),
        }

    by_model: dict[str, Any] = {}
    for mv in sorted({r.model_version for r in rows}):
        sub = [r for r in rows if r.model_version == mv]
        by_model[mv] = {
            "count": len(sub),
            "mae_pct": round(float(np.mean([r.abs_error for r in sub])), 2),
            "rmse_pct": round(float(np.sqrt(np.mean([r.sq_error for r in sub]))), 2),
            "accuracy": round(float(np.mean([1.0 if r.condition_correct else 0.0 for r in sub])), 4),
        }

    # 5-bin calibration plot data (confidence vs empirical condition accuracy)
    calibration_bins: list[dict[str, Any]] = []
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    for b_idx in range(len(edges) - 1):
        lo, hi = edges[b_idx], edges[b_idx + 1]
        in_bin = [r for r in rows if lo <= r.confidence < hi]
        if in_bin:
            calibration_bins.append({
                "bin": f"{lo:.1f}-{min(1.0, hi):.1f}",
                "mean_confidence": round(float(np.mean([r.confidence for r in in_bin])), 3),
                "empirical_accuracy": round(float(np.mean([1.0 if r.condition_correct else 0.0 for r in in_bin])), 3),
                "count": len(in_bin),
            })

    recent_samples = [
        {
            "forecast_id": r.forecast_id,
            "horizon_minutes": r.horizon_minutes,
            "verified_at": r.verified_at.isoformat(),
            "predicted_cover": r.predicted_cover,
            "actual_cover": r.actual_cover,
            "abs_error": r.abs_error,
            "predicted_condition": r.predicted_condition,
            "actual_condition": r.actual_condition,
            "model_version": r.model_version,
        }
        for r in rows[-25:]
    ]

    trend_rows = rows[-120:]
    mae_trend: list[dict[str, Any]] = []
    window_errs: list[float] = []
    for r in trend_rows:
        window_errs.append(float(r.abs_error))
        if len(window_errs) > 10:
            window_errs.pop(0)
        mae_trend.append({
            "verified_at": r.verified_at.isoformat(),
            "horizon_minutes": r.horizon_minutes,
            "abs_error": round(float(r.abs_error), 2),
            "rolling_mae_pct": round(float(np.mean(window_errs)), 2),
            "condition_correct": bool(r.condition_correct),
            "model_version": r.model_version,
        })

    return {
        "total_verified": len(rows),
        "status": "measured",
        "overall": overall,
        "by_horizon": by_horizon,
        "by_model_version": by_model,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "calibration_bins": calibration_bins,
        "mae_trend": mae_trend,
        "recent": recent_samples,
    }
