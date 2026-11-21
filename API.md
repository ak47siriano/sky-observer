# Sky Observer — REST & WebSocket API Reference

Interactive OpenAPI / Swagger UI is served locally at `http://localhost:8080/docs`.

## Camera
- `GET /api/camera/status` — Connection state, source type, latency, consecutive failures, reconnect counter.
- `GET /api/camera/config` — Current camera settings and calibration.
- `PUT /api/camera/config` — Validate and persist camera settings.
- `PUT /api/camera/masks` — Save polygonal sky ROI and permanent exclusion masks.
- `POST /api/camera/test` — Test snapshot/RTSP connectivity without mutating saved config.
- `GET /api/camera/latest?layer=raw|segmentation|motion|mask` — Latest frame or CV overlay JPEG.
- `GET /api/camera/snapshot` — Trigger immediate capture & full CV analysis.
- `POST /api/camera/upload` — Upload local image or video (Method C).

## Analysis
- `GET /api/analysis/latest` — Latest observation JSON metrics.
- `GET /api/analysis/history?limit=120&hours=24` — Time-series observation history.
- `GET /api/analysis/cloud-mask?overlay=true` — Latest segmentation mask or overlay image.
- `GET /api/analysis/motion?image=false` — Latest optical flow JSON or vector overlay image.
- `GET /api/analysis/quality` — Latest image quality assessment metrics.

## Forecast & Verification
- `GET /api/forecast/latest` — Multi-horizon (+5m, +15m, +30m, +60m) predictions with signals, reasons, and limitations.
- `GET /api/forecast/history` — Stored historical forecasts.
- `GET /api/forecast/verification` — Measured MAE, RMSE, Accuracy, Precision, Recall, F1, Confusion Matrix, Brier Score, and Calibration bins.

## Events & Alerts
- `GET /api/events` & `GET /api/events/latest` — Local weather & camera alerts.
- `PUT /api/events/config` — Update alert thresholds.
- `POST /api/events/{event_id}/ack` — Acknowledge alert.

## Models & Training
- `GET /api/models` & `GET /api/models/active` — List registered models and active models per family.
- `POST /api/models/train` — Launch background fine-tuning job.
- `GET /api/models/training/status` — Poll epoch metrics, loss, device, and status.
- `POST /api/models/training/stop` — Gracefully stop running training job.
- `POST /api/models/activate` — Activate a specific model version.
- `POST /api/models/rollback` — Roll back a model family to its prior version.

## Datasets & Timelapse
- `GET /api/dataset` — List local datasets and public scientific dataset catalog.
- `POST /api/dataset/bootstrap` — Generate offline baseline training dataset.
- `POST /api/dataset/import` — Import local directory of images/masks/classes.
- `POST /api/dataset/label` — Save user mask/class/event annotation for an observation.
- `POST /api/dataset/export` — Export dataset JSON manifest locally.
- `GET /api/timelapse/frames` & `POST /api/timelapse/export` — Browse frames or export local MP4 timelapse.

## System & Real-time
- `GET /api/system/health` — Docker & service health check.
- `GET /api/system/status` — CPU, RAM, GPU, VRAM, storage, camera, and inference telemetry.
- `GET /api/system/gpu` & `GET /api/system/storage` — Dedicated hardware and disk usage endpoints.
- `POST /api/system/storage/cleanup` — Enforce retention policy (training data protected).
- `WS /ws/live` — Real-time JSON stream for new observations.
