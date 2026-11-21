# Sky Observer — Architecture

## 1. Design Philosophy
Sky Observer is a 100% local, privacy-first meteorological computer vision observatory. An Android smartphone mounted outdoors sends frames over the local Wi-Fi/LAN to a desktop PC (targeting 8 GB NVIDIA VRAM / 16 GB RAM, with automatic CPU fallback). Zero external weather APIs, cloud AI APIs, or telemetry services are called.

## 2. System Layers

```text
Android IP Camera (LAN) ──▶ CameraManager (HTTP / MJPEG / RTSP / Local Upload)
                                      │
                                      ▼
                            VisionPipeline (OpenCV + PyTorch)
                  ┌───────────┬──────────────┬───────────────┐
                  ▼           ▼              ▼               ▼
            Quality & Mask  SegUNet      CloudMotion      Classifier
            (7.1 & 7.2)    (7.3 & 7.4)   (7.5 Flow)      (7.6-7.8)
                  └───────────┴──────────────┴───────────────┘
                                      │
                                      ▼
                      SQLite (WAL) + Disk Image Store
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
      Forecasting & Verification  Event Alerts       Model Lab & Trainer
      (Rules + TemporalConvGRU)   (Local Thresholds) (GPU/AMP + Registry)
                │                     │                     │
                └─────────────────────┴─────────────────────┘
                                      │
                                      ▼
                    FastAPI REST + WebSocket (/ws/live)
                                      │
                                      ▼
                 React + TypeScript + Vite Dark Dashboard
```

## 3. Storage Architecture
- **Database (`skyobserver.db`)**: SQLite in WAL mode (`PRAGMA journal_mode=WAL`) storing observations, metrics JSON, forecasts, forecast verifications, events, datasets, dataset items, training sessions, and model registry versions.
- **Disk Images**: Raw snapshots, 6-class PNG segmentation masks, visual segmentation overlays, motion vector overlays, exported MP4 timelapses, and dataset images are stored on disk under `/data` (never as SQLite BLOBs).
- **Retention**: Configurable per category (`raw_retention_days`, `processed_retention_days`, `timelapse_retention_days`, `event_retention_days`). Training images (`is_training_sample=True` or referenced in `dataset_items`) are strictly protected from automatic deletion.
