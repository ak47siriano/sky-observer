"""SQLAlchemy ORM models for Sky Observer SQLite database."""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    source_type: Mapped[str] = mapped_column(String(32), default="http_snapshot")
    frame_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)

    raw_path: Mapped[str] = mapped_column(Text, default="")
    seg_mask_path: Mapped[str] = mapped_column(Text, default="")
    seg_overlay_path: Mapped[str] = mapped_column(Text, default="")
    motion_overlay_path: Mapped[str] = mapped_column(Text, default="")

    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_usable: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    quality_metrics_json: Mapped[str] = mapped_column(Text, default="{}")

    cloud_coverage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    clear_sky_pct: Mapped[float] = mapped_column(Float, default=100.0)
    thin_cloud_pct: Mapped[float] = mapped_column(Float, default=0.0)
    dense_cloud_pct: Mapped[float] = mapped_column(Float, default=0.0)
    dark_cloud_pct: Mapped[float] = mapped_column(Float, default=0.0)
    haze_fog_pct: Mapped[float] = mapped_column(Float, default=0.0)
    uncertainty_pct: Mapped[float] = mapped_column(Float, default=0.0)
    coverage_metrics_json: Mapped[str] = mapped_column(Text, default="{}")

    dominant_cloud_type: Mapped[str] = mapped_column(String(64), default="unknown")
    classification_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    classification_probs_json: Mapped[str] = mapped_column(Text, default="{}")

    motion_speed_px_s: Mapped[float] = mapped_column(Float, default=0.0)
    motion_speed_deg_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    motion_direction_deg: Mapped[float] = mapped_column(Float, default=0.0)
    motion_direction_cardinal: Mapped[str] = mapped_column(String(8), default="N")
    motion_consistency: Mapped[float] = mapped_column(Float, default=0.0)
    motion_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    motion_metrics_json: Mapped[str] = mapped_column(Text, default="{}")

    avg_luminance: Mapped[float] = mapped_column(Float, default=0.0)
    blue_sky_index: Mapped[float] = mapped_column(Float, default=0.0)
    color_metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    texture_metrics_json: Mapped[str] = mapped_column(Text, default="{}")

    visual_condition: Mapped[str] = mapped_column(String(64), default="unknown")
    inference_duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    capture_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    is_training_sample: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_event_snapshot: Mapped[bool] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=True)


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    observation_id: Mapped[int | None] = mapped_column(
        ForeignKey("observations.id", ondelete="SET NULL"), nullable=True
    )
    horizon_minutes: Mapped[int] = mapped_column(Integer, index=True)
    target_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    status: Mapped[str] = mapped_column(String(32), default="available")
    forecast_condition: Mapped[str] = mapped_column(String(64), default="unknown")
    cloud_cover_prediction: Mapped[float] = mapped_column(Float, default=0.0)
    clearing_prob: Mapped[float] = mapped_column(Float, default=0.0)
    increasing_prob: Mapped[float] = mapped_column(Float, default=0.0)
    overcast_prob: Mapped[float] = mapped_column(Float, default=0.0)
    darkening_prob: Mapped[float] = mapped_column(Float, default=0.0)
    precip_signal_prob: Mapped[float] = mapped_column(Float, default=0.0)
    storm_signal_prob: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    signals_json: Mapped[str] = mapped_column(Text, default="[]")
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    limitations_json: Mapped[str] = mapped_column(Text, default="[]")
    baseline_metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    model_version: Mapped[str] = mapped_column(String(64), default="rules-v1.0.0")
    verified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class ForecastVerification(Base):
    __tablename__ = "forecast_verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_id: Mapped[int] = mapped_column(ForeignKey("forecasts.id", ondelete="CASCADE"), unique=True)
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id", ondelete="CASCADE"))
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, index=True)
    predicted_cover: Mapped[float] = mapped_column(Float)
    actual_cover: Mapped[float] = mapped_column(Float)
    abs_error: Mapped[float] = mapped_column(Float)
    sq_error: Mapped[float] = mapped_column(Float)
    persistence_abs_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    optical_flow_abs_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_condition: Mapped[str] = mapped_column(String(64))
    actual_condition: Mapped[str] = mapped_column(String(64))
    condition_correct: Mapped[bool] = mapped_column(Boolean)
    brier_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    model_version: Mapped[str] = mapped_column(String(64), default="rules-v1.0.0")


class EventAlert(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")
    title: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    observation_id: Mapped[int | None] = mapped_column(
        ForeignKey("observations.id", ondelete="SET NULL"), nullable=True
    )
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    user_status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    user_note: Mapped[str] = mapped_column(Text, default="")


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    source_type: Mapped[str] = mapped_column(String(32), default="local")
    license_name: Mapped[str] = mapped_column(String(128), default="Local User / Research")
    description: Mapped[str] = mapped_column(Text, default="")
    root_path: Mapped[str] = mapped_column(Text)
    task_types_json: Mapped[str] = mapped_column(Text, default='["segmentation","classification"]')
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    integrity_verified: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DatasetItem(Base):
    __tablename__ = "dataset_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    observation_id: Mapped[int | None] = mapped_column(
        ForeignKey("observations.id", ondelete="SET NULL"), nullable=True
    )
    split: Mapped[str] = mapped_column(String(16), default="train", index=True)
    group_key: Mapped[str] = mapped_column(String(64), default="default", index=True)
    sequence_index: Mapped[int] = mapped_column(Integer, default=0)

    image_path: Mapped[str] = mapped_column(Text)
    mask_path: Mapped[str] = mapped_column(Text, default="")
    cloud_type_label: Mapped[str] = mapped_column(String(64), default="")
    event_label: Mapped[str] = mapped_column(String(64), default="normal")
    quality_label: Mapped[str] = mapped_column(String(32), default="usable")
    cloud_coverage_gt: Mapped[float | None] = mapped_column(Float, nullable=True)
    user_labeled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_family: Mapped[str] = mapped_column(String(32), index=True)
    architecture: Mapped[str] = mapped_column(String(64))
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=10)
    batch_size: Mapped[int] = mapped_column(Integer, default=4)
    learning_rate: Mapped[float] = mapped_column(Float, default=0.001)
    device_used: Mapped[str] = mapped_column(String(32), default="cpu")
    mixed_precision: Mapped[bool] = mapped_column(Boolean, default=True)
    best_val_metric: Mapped[float] = mapped_column(Float, default=0.0)
    metrics_history_json: Mapped[str] = mapped_column(Text, default="[]")
    checkpoint_path: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_family: Mapped[str] = mapped_column(String(32), index=True)
    architecture: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    weights_path: Mapped[str] = mapped_column(Text)
    training_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="SET NULL"), nullable=True
    )
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    confusion_matrix_json: Mapped[str] = mapped_column(Text, default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def safe_json_loads(raw: str | None, fallback: Any) -> Any:
    import json

    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback
