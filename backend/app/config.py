"""Validated YAML-backed configuration management for Sky Observer."""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ExclusionPolygon(BaseModel):
    """Permanent exclusion mask polygon in normalized [0..1] image coordinates."""

    id: str
    label: str
    category: Literal["trees", "buildings", "poles", "roofs", "road", "logo", "other"] = "other"
    points: list[list[float]] = Field(default_factory=list)

    @field_validator("points")
    @classmethod
    def validate_points(cls, pts: list[list[float]]) -> list[list[float]]:
        cleaned: list[list[float]] = []
        for p in pts:
            if len(p) != 2:
                raise ValueError("Each polygon vertex must be [x, y]")
            cleaned.append([max(0.0, min(1.0, float(p[0]))), max(0.0, min(1.0, float(p[1])))])
        return cleaned


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    offline_mode: bool = True
    setup_completed: bool = False
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    @field_validator("cors_origins")
    @classmethod
    def forbid_wildcard_origin(cls, origins: list[str]) -> list[str]:
        cleaned = [o.strip() for o in origins if o and o.strip() and o.strip() != "*"]
        return cleaned or ["http://localhost:8080", "http://127.0.0.1:8080"]


class SecondaryCameraConfig(BaseModel):
    """Additional camera pointing in a complementary direction (e.g., North + South) for 360-degree Radar Fusion."""

    id: str = "cam_2"
    name: str = "Secondary Sky Cam"
    enabled: bool = True
    source_type: Literal[
        "http_snapshot", "mjpeg_stream", "rtsp", "web_stream", "webcam_device", "demo_simulation"
    ] = "http_snapshot"
    source_url: str = ""
    device_index: int = Field(default=1, ge=0, le=16)
    orientation_azimuth_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    orientation_elevation_deg: float = Field(default=55.0, ge=-20.0, le=90.0)
    orientation_roll_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    fov_horizontal_deg: float = Field(default=74.0, ge=10.0, le=180.0)
    fov_vertical_deg: float = Field(default=54.0, ge=10.0, le=180.0)


class CameraConfig(BaseModel):
    name: str = "Android Sky Cam"
    location_description: str = ""
    source_type: Literal[
        "http_snapshot",
        "mjpeg_stream",
        "rtsp",
        "web_stream",
        "webcam_device",
        "local_import",
        "demo_simulation",
    ] = "http_snapshot"
    source_url: str = ""
    device_index: int = Field(default=0, ge=0, le=16)
    capture_interval_seconds: float = Field(default=5.0, ge=1.0, le=3600.0)
    analysis_interval_seconds: float = Field(default=15.0, ge=1.0, le=3600.0)
    archive_interval_seconds: float = Field(default=30.0, ge=5.0, le=86400.0)
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=120.0)
    reconnect_enabled: bool = True
    reconnect_backoff_seconds: float = Field(default=5.0, ge=1.0, le=300.0)
    image_width: int = Field(default=1920, ge=160, le=4096)
    image_height: int = Field(default=1080, ge=120, le=4096)
    jpeg_quality: int = Field(default=88, ge=30, le=100)
    frame_sampling: int = Field(default=1, ge=1, le=60)
    duplicate_threshold: float = Field(default=0.015, ge=0.0, le=0.5)
    orientation_azimuth_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    orientation_elevation_deg: float = Field(default=65.0, ge=-20.0, le=90.0)
    orientation_roll_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    tilt_correction_enabled: bool = True
    fov_horizontal_deg: float | None = Field(default=72.0, ge=10.0, le=180.0)
    fov_vertical_deg: float | None = Field(default=45.0, ge=10.0, le=180.0)
    assumed_cloud_altitude_m: float | None = Field(default=None, ge=100.0, le=15000.0)
    latitude: float = Field(default=41.9028, ge=-90.0, le=90.0)
    longitude: float = Field(default=12.4964, ge=-180.0, le=180.0)
    radar_radius_km: float = Field(default=15.0, ge=1.0, le=100.0)
    sky_roi_polygon: list[list[float]] = Field(default_factory=list)
    exclusion_polygons: list[ExclusionPolygon] = Field(default_factory=list)
    exclusion_mask_path: str = ""
    secondary_cameras: list[SecondaryCameraConfig] = Field(default_factory=list)


class VisionConfig(BaseModel):
    inference_resolution: int = Field(default=512, ge=128, le=1280)
    cloud_segmentation_enabled: bool = True
    cloud_classification_enabled: bool = True
    optical_flow_enabled: bool = True
    image_quality_check_enabled: bool = True
    grid_rows: int = Field(default=3, ge=1, le=12)
    grid_cols: int = Field(default=3, ge=1, le=12)
    min_quality_score: float = Field(default=0.40, ge=0.0, le=1.0)
    uncertainty_entropy_threshold: float = Field(default=0.72, ge=0.1, le=1.0)
    classification_min_confidence: float = Field(default=0.35, ge=0.05, le=0.99)


class ForecastRulesConfig(BaseModel):
    rapid_increase_rate_pct_per_min: float = 1.5
    rapid_decrease_rate_pct_per_min: float = -1.5
    darkening_luminance_delta: float = -12.0
    overcast_threshold_pct: float = 82.0
    clear_threshold_pct: float = 18.0
    storm_dark_cloud_pct: float = 35.0
    high_motion_consistency: float = 0.65


class ForecastConfig(BaseModel):
    # Camera-only FOV horizons limited to +5/+15/+20 min; +30/+60 min added only with external NWP/wind input
    horizons_minutes: list[int] = Field(default_factory=lambda: [5, 15, 20])
    external_horizons_minutes: list[int] = Field(default_factory=lambda: [5, 15, 20, 30, 60])
    update_interval_seconds: float = Field(default=60.0, ge=5.0, le=3600.0)
    rain_prediction_enabled: bool = True
    experimental_predictions: bool = True
    ensemble_members: int = Field(default=5, ge=1, le=15)
    advection_resolution: int = Field(default=96, ge=48, le=256)
    rules: ForecastRulesConfig = Field(default_factory=ForecastRulesConfig)


class AutoTrainConfig(BaseModel):
    enabled: bool = False
    interval_minutes: int = Field(default=60, ge=5, le=1440)
    min_new_observations: int = Field(default=12, ge=2, le=500)
    min_quality_score: float = Field(default=0.50, ge=0.1, le=0.95)
    max_local_samples: int = Field(default=80, ge=4, le=1000)
    include_baseline_replay: bool = True
    epochs: int = Field(default=3, ge=1, le=30)
    learning_rate: float = Field(default=0.0005, gt=0.0, le=0.1)
    families: list[Literal["segmentation", "classification", "temporal", "event"]] = Field(
        default_factory=lambda: ["segmentation", "classification", "temporal"]
    )
    warm_start: bool = True
    fixed_holdout_fraction: float = Field(default=0.20, ge=0.05, le=0.50)
    promotion_tolerance: float = Field(default=0.01, ge=0.0, le=0.20)


class AIConfig(BaseModel):
    device: Literal["auto", "cuda", "rocm", "directml", "mps", "cpu"] = "auto"
    hardware_profile: Literal["auto", "low_end", "balanced", "high_end", "ultra", "custom"] = "auto"
    cpu_threads: int = Field(default=0, ge=0, le=64)
    mixed_precision: bool = True
    max_gpu_memory_fraction: float = Field(default=0.85, ge=0.1, le=0.98)
    batch_size: int = Field(default=8, ge=1, le=64)
    seg_batch_size: int = Field(default=8, ge=1, le=32)
    cls_batch_size: int = Field(default=32, ge=1, le=128)
    dataloader_num_workers: int = Field(default=2, ge=0, le=8)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=16)
    seg_backend: Literal[
        "auto", "radiometric_physics", "deeplabv3_daytime", "dinov3_linear_probe"
    ] = "auto"
    seg_backbone: Literal[
        "convnext_aspp",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "lightweight",
        "deeplabv3_mobilenet_v3",
        "dinov3_linear_probe",
    ] = "convnext_aspp"
    dinov3_backbone_id: str = "vit_small_patch14_dinov2.lvd142m"
    dinov3_weights_path: str = ""
    cls_backbone: Literal["efficientnet_b0", "convnext_tiny", "compact"] = "efficientnet_b0"
    pretrained_backbone: bool = True
    group_ambiguous_wmo: bool = False
    event_engine_mode: Literal["hybrid", "deterministic", "neural"] = "hybrid"
    training_enabled: bool = True
    default_epochs: int = Field(default=12, ge=1, le=200)
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.5)
    early_stopping_patience: int = Field(default=4, ge=1, le=50)
    auto_train: AutoTrainConfig = Field(default_factory=AutoTrainConfig)


class StorageConfig(BaseModel):
    base_dir: str = "./data"
    database_path: str = "./data/db/skyobserver.db"
    raw_retention_days: int = Field(default=14, ge=1, le=3650)
    processed_retention_days: int = Field(default=30, ge=1, le=3650)
    timelapse_retention_days: int = Field(default=60, ge=1, le=3650)
    event_retention_days: int = Field(default=90, ge=1, le=3650)
    training_retention_days: int = Field(default=3650, ge=30, le=36500)
    max_storage_gb: float = Field(default=50.0, ge=1.0, le=10000.0)
    image_storage_enabled: bool = True


class EventsConfig(BaseModel):
    rapid_cloud_increase_pct: float = Field(default=22.0, ge=1.0, le=100.0)
    rapid_cloud_decrease_pct: float = Field(default=22.0, ge=1.0, le=100.0)
    rapid_darkening_delta: float = Field(default=-18.0, ge=-100.0, le=-1.0)
    dense_cloud_alert_pct: float = Field(default=70.0, ge=5.0, le=100.0)
    storm_like_dark_cloud_pct: float = Field(default=45.0, ge=5.0, le=100.0)
    camera_move_shift_px: float = Field(default=40.0, ge=2.0, le=500.0)
    min_storage_free_gb: float = Field(default=2.0, ge=0.2, le=500.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=1440)
    rejected_backoff_minutes: int = Field(default=60, ge=5, le=1440)
    camera_movement_enabled: bool = True
    browser_notifications_enabled: bool = False


class OnlineWeatherConfig(BaseModel):
    provider: Literal["open_meteo"] = "open_meteo"
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    fusion_weight_online: float = Field(default=0.40, ge=0.0, le=0.80)
    rainviewer_enabled: bool = True


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    online_weather: OnlineWeatherConfig = Field(default_factory=OnlineWeatherConfig)

    def resolve_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return (_default_project_root() / p).resolve()

    @property
    def data_dir(self) -> Path:
        env_dir = os.environ.get("SKYOBSERVER_DATA_DIR")
        if env_dir:
            return Path(env_dir).resolve()
        return self.resolve_path(self.storage.base_dir)

    @property
    def db_path(self) -> Path:
        env_db = os.environ.get("SKYOBSERVER_DB_PATH")
        if env_db:
            return Path(env_db).resolve()
        if os.environ.get("SKYOBSERVER_DATA_DIR"):
            return self.data_dir / "db" / "skyobserver.db"
        return self.resolve_path(self.storage.database_path)

    @property
    def raw_images_dir(self) -> Path:
        return self.data_dir / "images" / "raw"

    @property
    def processed_images_dir(self) -> Path:
        return self.data_dir / "images" / "processed"

    @property
    def event_images_dir(self) -> Path:
        return self.data_dir / "images" / "events"

    @property
    def timelapse_dir(self) -> Path:
        return self.data_dir / "timelapse"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    def ensure_directories(self) -> None:
        for d in (
            self.data_dir,
            self.db_path.parent,
            self.raw_images_dir,
            self.processed_images_dir,
            self.event_images_dir,
            self.timelapse_dir,
            self.models_dir,
            self.datasets_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Thread-safe configuration loader and persister."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._lock = RLock()
        env_path = os.environ.get("SKYOBSERVER_CONFIG_PATH")
        if config_path is not None:
            self.config_path = config_path
        elif env_path:
            self.config_path = Path(env_path)
        else:
            self.config_path = _default_project_root() / "config" / "config.yaml"
        self._config = self._load()

    def _load(self) -> AppConfig:
        raw: dict[str, Any] = {}
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    raw = loaded
        cfg = AppConfig.model_validate(raw)
        host_env = os.environ.get("SKYOBS_HOST") or os.environ.get("SKYOBSERVER_HOST")
        if host_env and host_env.strip():
            cfg.server.host = host_env.strip()
        port_env = os.environ.get("SKYOBS_PORT") or os.environ.get("SKYOBSERVER_PORT")
        if port_env and port_env.strip().isdigit():
            p_int = int(port_env.strip())
            if 1 <= p_int <= 65535:
                cfg.server.port = p_int
        cors_env = os.environ.get("SKYOBS_CORS_ORIGINS") or os.environ.get("SKYOBSERVER_CORS_ORIGINS")
        if cors_env:
            parsed_origins = [x.strip() for x in cors_env.split(",") if x.strip() and x.strip() != "*"]
            if parsed_origins:
                cfg.server.cors_origins = parsed_origins
        if os.environ.get("SKYOBSERVER_OFFLINE_MODE"):
            cfg.server.offline_mode = os.environ["SKYOBSERVER_OFFLINE_MODE"].lower() in {"1", "true", "yes"}
        if os.environ.get("SKYOBSERVER_DEVICE"):
            dev = os.environ["SKYOBSERVER_DEVICE"].lower()
            if dev in {"auto", "cuda", "rocm", "directml", "mps", "cpu"}:
                cfg.ai.device = dev  # type: ignore[assignment]
        if os.environ.get("SKYOBSERVER_MAX_GPU_MEMORY_FRACTION"):
            cfg.ai.max_gpu_memory_fraction = float(os.environ["SKYOBSERVER_MAX_GPU_MEMORY_FRACTION"])
        cfg.ensure_directories()
        return cfg

    def get(self) -> AppConfig:
        with self._lock:
            return self._config.model_copy(deep=True)

    def update(self, patch: dict[str, Any]) -> AppConfig:
        with self._lock:
            current = self._config.model_dump()
            merged = _deep_merge(current, patch)
            updated = AppConfig.model_validate(merged)
            updated.ensure_directories()
            self._config = updated
            self.save()
            return updated.model_copy(deep=True)

    def replace_section(self, section: str, values: dict[str, Any]) -> AppConfig:
        return self.update({section: values})

    def save(self) -> None:
        with self._lock:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(self._config.model_dump(mode="json"), fh, sort_keys=False)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


_manager_singleton: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    global _manager_singleton
    if _manager_singleton is None:
        _manager_singleton = ConfigManager()
    return _manager_singleton


def reset_config_manager(config_path: Path | None = None) -> ConfigManager:
    global _manager_singleton
    _manager_singleton = ConfigManager(config_path=config_path)
    return _manager_singleton
