"""Initial Sky Observer schema.

Revision ID: 001_initial
Revises: None
Create Date: 2026-11-21
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("frame_hash", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("raw_path", sa.Text(), nullable=False),
        sa.Column("seg_mask_path", sa.Text(), nullable=False),
        sa.Column("seg_overlay_path", sa.Text(), nullable=False),
        sa.Column("motion_overlay_path", sa.Text(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("is_usable", sa.Boolean(), nullable=False),
        sa.Column("quality_metrics_json", sa.Text(), nullable=False),
        sa.Column("cloud_coverage_pct", sa.Float(), nullable=False),
        sa.Column("clear_sky_pct", sa.Float(), nullable=False),
        sa.Column("thin_cloud_pct", sa.Float(), nullable=False),
        sa.Column("dense_cloud_pct", sa.Float(), nullable=False),
        sa.Column("dark_cloud_pct", sa.Float(), nullable=False),
        sa.Column("haze_fog_pct", sa.Float(), nullable=False),
        sa.Column("uncertainty_pct", sa.Float(), nullable=False),
        sa.Column("coverage_metrics_json", sa.Text(), nullable=False),
        sa.Column("dominant_cloud_type", sa.String(length=64), nullable=False),
        sa.Column("classification_confidence", sa.Float(), nullable=False),
        sa.Column("classification_probs_json", sa.Text(), nullable=False),
        sa.Column("motion_speed_px_s", sa.Float(), nullable=False),
        sa.Column("motion_speed_deg_s", sa.Float(), nullable=True),
        sa.Column("motion_direction_deg", sa.Float(), nullable=False),
        sa.Column("motion_direction_cardinal", sa.String(length=8), nullable=False),
        sa.Column("motion_consistency", sa.Float(), nullable=False),
        sa.Column("motion_confidence", sa.Float(), nullable=False),
        sa.Column("motion_metrics_json", sa.Text(), nullable=False),
        sa.Column("avg_luminance", sa.Float(), nullable=False),
        sa.Column("blue_sky_index", sa.Float(), nullable=False),
        sa.Column("color_metrics_json", sa.Text(), nullable=False),
        sa.Column("texture_metrics_json", sa.Text(), nullable=False),
        sa.Column("visual_condition", sa.String(length=64), nullable=False),
        sa.Column("inference_duration_ms", sa.Float(), nullable=False),
        sa.Column("capture_latency_ms", sa.Float(), nullable=False),
        sa.Column("is_training_sample", sa.Boolean(), nullable=False),
        sa.Column("is_event_snapshot", sa.Boolean(), nullable=False),
        sa.Column("is_archived", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_observations_captured_at", "observations", ["captured_at"])
    op.create_index("ix_observations_frame_hash", "observations", ["frame_hash"])

    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observation_id", sa.Integer(), nullable=True),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("target_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("forecast_condition", sa.String(length=64), nullable=False),
        sa.Column("cloud_cover_prediction", sa.Float(), nullable=False),
        sa.Column("clearing_prob", sa.Float(), nullable=False),
        sa.Column("increasing_prob", sa.Float(), nullable=False),
        sa.Column("overcast_prob", sa.Float(), nullable=False),
        sa.Column("darkening_prob", sa.Float(), nullable=False),
        sa.Column("precip_signal_prob", sa.Float(), nullable=False),
        sa.Column("storm_signal_prob", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("signals_json", sa.Text(), nullable=False),
        sa.Column("reasons_json", sa.Text(), nullable=False),
        sa.Column("limitations_json", sa.Text(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["observation_id"], ["observations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "forecast_verifications",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("forecast_id", sa.Integer(), nullable=False),
        sa.Column("observation_id", sa.Integer(), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("predicted_cover", sa.Float(), nullable=False),
        sa.Column("actual_cover", sa.Float(), nullable=False),
        sa.Column("abs_error", sa.Float(), nullable=False),
        sa.Column("sq_error", sa.Float(), nullable=False),
        sa.Column("predicted_condition", sa.String(length=64), nullable=False),
        sa.Column("actual_condition", sa.String(length=64), nullable=False),
        sa.Column("condition_correct", sa.Boolean(), nullable=False),
        sa.Column("brier_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["forecast_id"], ["forecasts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["observation_id"], ["observations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("forecast_id"),
    )
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("observation_id", sa.Integer(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("acknowledged", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["observation_id"], ["observations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("license_name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("root_path", sa.Text(), nullable=False),
        sa.Column("task_types_json", sa.Text(), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("integrity_verified", sa.Boolean(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "dataset_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("observation_id", sa.Integer(), nullable=True),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("group_key", sa.String(length=64), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("mask_path", sa.Text(), nullable=False),
        sa.Column("cloud_type_label", sa.String(length=64), nullable=False),
        sa.Column("event_label", sa.String(length=64), nullable=False),
        sa.Column("quality_label", sa.String(length=32), nullable=False),
        sa.Column("cloud_coverage_gt", sa.Float(), nullable=True),
        sa.Column("user_labeled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["observation_id"], ["observations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "training_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("model_family", sa.String(length=32), nullable=False),
        sa.Column("architecture", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("current_epoch", sa.Integer(), nullable=False),
        sa.Column("total_epochs", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("learning_rate", sa.Float(), nullable=False),
        sa.Column("device_used", sa.String(length=32), nullable=False),
        sa.Column("mixed_precision", sa.Boolean(), nullable=False),
        sa.Column("best_val_metric", sa.Float(), nullable=False),
        sa.Column("metrics_history_json", sa.Text(), nullable=False),
        sa.Column("checkpoint_path", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("model_family", sa.String(length=32), nullable=False),
        sa.Column("architecture", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("weights_path", sa.Text(), nullable=False),
        sa.Column("training_session_id", sa.Integer(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("confusion_matrix_json", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["training_session_id"], ["training_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id"),
    )


def downgrade() -> None:
    op.drop_table("model_registry")
    op.drop_table("training_sessions")
    op.drop_table("dataset_items")
    op.drop_table("datasets")
    op.drop_table("events")
    op.drop_table("forecast_verifications")
    op.drop_table("forecasts")
    op.drop_table("observations")
