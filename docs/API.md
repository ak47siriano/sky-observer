# Sky Observer — REST & WebSocket API Reference

Interactive OpenAPI / Swagger UI is served locally at `http://localhost:8080/docs`.

## Camera & Calibration
- `GET /api/camera/status` — Connection state, masked source URL, source type, latency, consecutive failures.
- `GET /api/camera/config` & `PUT /api/camera/config` — Read/update camera settings (credentials in `source_url` are masked on read and preserved on save; SSRF guard validates URLs).
- `PUT /api/camera/masks` — Save polygonal sky ROI and permanent exclusion masks.
- `PUT /api/camera/exclusion-mask` & `GET /api/camera/exclusion-mask.png` & `DELETE /api/camera/exclusion-mask` — Save/read/clear painted exclusion masks.
- `POST /api/camera/exclusion-mask/propose` — Auto-propose a sky-vs-obstacle exclusion mask from the latest (or specified) frame for user review before saving.
- `PUT /api/camera/geometry` — Save camera tilt/azimuth/roll/FOV/zenith cutoff and recalculate latest observation.
- `POST /api/camera/geometry/auto-detect` — Estimate horizon elevation and roll from the latest live frame.
- `POST /api/camera/geometry/from-sensors?apply=false` — Read Android IP Webcam `/sensors.json` (accelerometer + compass) to propose tilt, roll, and heading.
- `POST /api/camera/geometry/refine-sun?apply=false` — Refine compass azimuth by matching an observed solar disk centroid inside the FOV against NOAA solar azimuth.
- `POST /api/camera/test` — Test camera connectivity without mutating saved config.
- `GET /api/camera/latest?layer=raw|segmentation|motion|mask` — Latest frame or CV overlay JPEG.
- `GET /api/camera/snapshot` — Trigger immediate capture & full CV analysis.
- `POST /api/camera/upload` — Upload local image or video.

## Analysis, Sky State & Solar PV
- `GET /api/sky-state` — Ultra-fast minimal JSON endpoint (`state`, `sky_regime`, `cloud_coverage_pct`, `dominant_cloud_type`, `confidence`, `qc_flags`, `is_usable`) designed for Home Assistant, ESP32, and local scripts.
- `GET /api/analysis/latest` — Latest full observation metrics (including `sky_regime`, `night_telemetry`, and hardware QC flags).
- `GET /api/analysis/history?limit=120&hours=24` — Time-series observation history.
- `GET /api/analysis/cloud-mask?overlay=true` — Latest segmentation mask or overlay image.
- `GET /api/analysis/motion?image=false` — Latest optical flow JSON or vector overlay image.
- `GET /api/analysis/quality` — Latest image quality assessment metrics and low-cost hardware QC flags.
- `GET /api/solar/pv` — Clear-sky GHI, cloud-transmittance rooftop PV power output (kW), and short-term solar ramp alerts (`ramp_down` / `ramp_up`).

## Forecast & Verification
- `GET /api/forecast/latest` — Multi-horizon predictions with `baselines` (`persistence_cover_pct`, `optical_flow_cover_pct`, `fov_exit_fraction`), uncertainty bounds (`cover_p10`, `cover_p90`), and `low_reliability` indicator.
- `GET /api/forecast/advected-mask.png?horizon=15` — Probabilistic optical-flow advected cloud mask PNG for `+1..60` min.
- `GET /api/forecast/history` — Stored historical forecasts.
- `GET /api/forecast/verification` — Chronological MAE, RMSE, Persistence MAE, Optical-Flow MAE, `skill_vs_persistence` (`1 - MAE_model / MAE_persistence`), Brier Score, and Calibration bins (overall & per horizon).

## Events & Alerts
- `GET /api/events` & `GET /api/events/latest` — Local weather & camera alerts.
- `PUT /api/events/config` — Update alert thresholds.
- `POST /api/events/{event_id}/ack` & `POST /api/events/{event_id}/reject` & `POST /api/events/{event_id}/feedback` — Confirm or deny alerts to feed Auto-Train.

## Models, Auto-Train & Datasets
- `GET /api/models` & `GET /api/models/active` — List registered checkpoints and active models per family.
- `POST /api/models/train` — Launch background fine-tuning job (supports `CloudSegUNet`, `DeepLabV3-MobileNetV3`, `DINOv3-LinearProbe`, `CloudTypeClassifier`, `TemporalSkyConvGRU`, `SkyEventClassifier`).
- `GET /api/models/training/status` & `POST /api/models/training/stop` — Monitor or stop active training jobs.
- `POST /api/models/activate` & `POST /api/models/rollback` & `POST /api/models/prune` — Manage checkpoint versions.
- `GET /api/models/auto-train/status` & `PUT /api/models/auto-train/config` & `POST /api/models/auto-train/run` — Control fixed-holdout anti-drift Location Auto-Train.
- `GET /api/dataset`, `POST /api/dataset/bootstrap`, `POST /api/dataset/import`, `POST /api/dataset/label`, `POST /api/dataset/export` — Local dataset lifecycle.

## System & Real-time
- `GET /api/health` & `GET /api/system/health` — Docker & service health check.
- `GET /api/system/status`, `GET /api/system/gpu`, `PUT /api/system/hardware-profile`, `GET /api/system/storage` — Hardware presets and disk telemetry.
- `WS /ws/live` — Real-time JSON WebSocket stream for new observations.
