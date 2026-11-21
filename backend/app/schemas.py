"""Pydantic request and response schemas for the Sky Observer FastAPI backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CameraTestRequest(BaseModel):
    source_type: str | None = None
    source_url: str | None = None
    device_index: int | None = None


class MasksUpdateRequest(BaseModel):
    sky_roi_polygon: list[list[float]] = Field(default_factory=list)
    exclusion_polygons: list[dict[str, Any]] = Field(default_factory=list)


class TrainModelRequest(BaseModel):
    model_family: Literal["segmentation", "classification", "temporal", "event"]
    dataset_id: int
    epochs: int = Field(default=8, ge=1, le=200)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    learning_rate: float | None = Field(default=None, gt=0.0, le=0.5)
    device: Literal["auto", "cuda", "rocm", "directml", "mps", "cpu"] | None = None
    resume_from_job_id: str | None = None
    auto_activate: bool = True


class HardwareProfileRequest(BaseModel):
    profile: Literal["auto", "low_end", "balanced", "high_end", "ultra", "custom"] | None = None
    device: Literal["auto", "cuda", "rocm", "directml", "mps", "cpu"] | None = None
    inference_resolution: int | None = Field(default=None, ge=128, le=1280)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    gradient_accumulation_steps: int | None = Field(default=None, ge=1, le=16)
    cpu_threads: int | None = Field(default=None, ge=0, le=64)
    max_gpu_memory_fraction: float | None = Field(default=None, ge=0.1, le=0.98)
    mixed_precision: bool | None = None


class ActivateModelRequest(BaseModel):
    model_id: str


class RollbackModelRequest(BaseModel):
    model_family: Literal["segmentation", "classification", "temporal", "event"]


class PruneModelsRequest(BaseModel):
    keep_per_family: int = Field(default=1, ge=1, le=20)
    model_family: Literal["segmentation", "classification", "temporal", "event"] | None = None
    also_prune_sessions: bool = True


class DatasetImportRequest(BaseModel):
    name: str
    directory_path: str
    license_name: str = "Local User / Research"
    description: str = ""


class DatasetLabelRequest(BaseModel):
    observation_id: int | None = None
    dataset_item_id: int | None = None
    dataset_name: str = "user_camera_labels"
    cloud_type_label: str = ""
    event_label: str = "normal"
    quality_label: str = "usable"
    mask_png_base64: str | None = None
    split: Literal["train", "val", "test"] = "train"


class DatasetExportRequest(BaseModel):
    dataset_id: int
    export_name: str | None = None


class TimelapseExportRequest(BaseModel):
    layer: Literal["raw", "segmentation", "motion"] = "raw"
    fps: int = Field(default=12, ge=1, le=60)
    date: str | None = None
    max_frames: int = Field(default=600, ge=2, le=5000)
    include_hud: bool = True


class DailyDigestRequest(BaseModel):
    date: str | None = None
    layer: Literal["raw", "segmentation", "motion"] = "raw"
    fps: int = Field(default=12, ge=1, le=60)
    include_hud: bool = True


class SecondaryCameraItem(BaseModel):
    id: str = "cam_2"
    name: str = "Secondary Sky Cam"
    enabled: bool = True
    source_type: Literal["http_snapshot", "mjpeg_stream", "rtsp", "web_stream", "webcam_device", "demo_simulation"] = "http_snapshot"
    source_url: str = ""
    device_index: int = Field(default=1, ge=0, le=16)
    orientation_azimuth_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    orientation_elevation_deg: float = Field(default=55.0, ge=-20.0, le=90.0)
    orientation_roll_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    fov_horizontal_deg: float = Field(default=74.0, ge=10.0, le=180.0)
    fov_vertical_deg: float = Field(default=54.0, ge=10.0, le=180.0)


class SecondaryCamerasUpdateRequest(BaseModel):
    cameras: list[SecondaryCameraItem] = Field(default_factory=list)


class TimelapseClearRequest(BaseModel):
    date: str | None = None
    delete_files: bool = False


class SystemModeRequest(BaseModel):
    offline_mode: bool


class RadarLocationRequest(BaseModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    location_description: str | None = None
    radar_radius_km: float | None = Field(default=None, ge=1.0, le=100.0)


class CameraGeometryRequest(BaseModel):
    orientation_elevation_deg: float = Field(ge=-20.0, le=90.0)
    orientation_azimuth_deg: float = Field(ge=0.0, le=360.0)
    orientation_roll_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    fov_horizontal_deg: float | None = Field(default=None, ge=10.0, le=170.0)
    fov_vertical_deg: float | None = Field(default=None, ge=10.0, le=170.0)
    assumed_cloud_altitude_m: float | None = Field(default=None, ge=100.0, le=15000.0)
    tilt_correction_enabled: bool = True
    recalculate_latest: bool = True


class PaintedMaskRequest(BaseModel):
    mask_png_base64: str


class EventFeedbackRequest(BaseModel):
    status: Literal["pending", "confirmed", "rejected"]
    note: str = Field(default="", max_length=500)


class AutoTrainConfigRequest(BaseModel):
    enabled: bool | None = None
    interval_minutes: int | None = Field(default=None, ge=5, le=1440)
    min_new_observations: int | None = Field(default=None, ge=2, le=500)
    min_quality_score: float | None = Field(default=None, ge=0.1, le=0.95)
    max_local_samples: int | None = Field(default=None, ge=4, le=1000)
    include_baseline_replay: bool | None = None
    epochs: int | None = Field(default=None, ge=1, le=30)
    learning_rate: float | None = Field(default=None, gt=0.0, le=0.1)
    families: list[Literal["segmentation", "classification", "temporal", "event"]] | None = None
    warm_start: bool | None = None


class AutoTrainRunRequest(BaseModel):
    families: list[Literal["segmentation", "classification", "temporal", "event"]] | None = None
    epochs: int | None = Field(default=None, ge=1, le=30)
    only_label: bool = False


class EventsClearRequest(BaseModel):
    scope: Literal["pending", "rejected", "all", "duplicates"] = "duplicates"


class MapPrefetchRequest(BaseModel):
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    styles: list[Literal["dark_base", "dark_ref", "osm", "sat_base"]] | None = None
    zooms: list[int] | None = None


PurgeTarget = Literal[
    "raw_images",
    "processed_images",
    "observations",
    "forecasts",
    "events",
    "timelapse",
    "map_tiles",
    "exclusion_masks",
    "datasets",
    "models",
    "reset_config",
]


class DataPurgeRequest(BaseModel):
    targets: list[PurgeTarget] = Field(default_factory=list)
    preserve_training_samples: bool = True
    preserve_camera_connection: bool = True
