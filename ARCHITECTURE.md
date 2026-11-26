# Sky Observer — Architecture

## 1. Design Philosophy
Sky Observer is a **local-first, 100% offline-capable** meteorological computer vision observatory. An Android smartphone, USB webcam, ESP32-CAM/Raspberry Pi, or LAN IP/RTSP camera sends sky frames to a local host (scaling from a **Low-End 4 GB RAM CPU-only mini-PC** up to **NVIDIA CUDA / AMD ROCm / DirectML / Apple MPS** workstations). All external network features (Open-Meteo NWP, RainViewer precipitation radar, offline tile prefetch) are strictly optional and disabled in Offline Only mode.

## 2. System Layers

```text
Camera Source (LAN HTTP / MJPEG / RTSP / USB / Local Import / Optional Web Stream)
                                       │
                                       ▼
                        CameraManager (SSRF Guard + Masking)
                                       │
                                       ▼
                            VisionPipeline (OpenCV + PyTorch)
             ┌──────────────┬──────────────┬──────────────┬──────────────┐
             ▼              ▼              ▼              ▼              ▼
        Quality & QC   Auto-Mask &     Stage 1 Seg    Optical Flow   Stage 2 Cls
        (7.1)          ROI (7.2)     (Physics / UNet  (Farneback)    (11 Sky Classes
                                    / DeepLabV3 /                    + 18-dim Fusion)
                                    DINOv3 Probe)
             └──────────────┴──────────────┴──────────────┴──────────────┘
                                       │
                                       ▼
                       SQLite (WAL) + Disk Image Store
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
      Forecasting & Verification  Event Alerts       Model Lab & Anti-Drift Auto-Train
      (Persistence + OF Mask      (Hybrid Rules      (Fixed Holdout + Promotion Gate
       Advection Ensemble +        + Confirm/Deny     + DINOv3 Probe CLI + Registry)
       TemporalSkyConvGRU)         Feedback)
                 │                     │                     │
                 └─────────────────────┴─────────────────────┘
                                       │
                                       ▼
                     FastAPI REST + WebSocket (/ws/live)
                                       │
                                       ▼
                  React + TypeScript + Vite Dark Dashboard
```

## 3. Pluggable Segmentation & Nowcasting Flow
- **Zero-GPU Default**: `radiometric-physics-v1` (CIE-Lab + `(B-R)/(B+R)` + AWB compensation + sun-glare guard) runs deterministically on CPU in tens of milliseconds.
- **Daytime Neural Option**: `DeepLabV3-MobileNetV3` (`torchvision`, BSD-3-Clause) provides an immediate daytime neural segmentation backbone ready for SWIMSEG/SWINySEG or local camera fine-tuning.
- **Day/Night Custom Adaptation**: `DINOv3-LinearProbe` (inspired by **LenghuSky-8**, Rui et al., CVPR Findings 2025, Apache-2.0 methodology; separate DINOv3 weights license) freezes a ViT patch encoder and trains only a `1×1 Conv` linear probe on 50–100 user-labeled day/night frames (`scripts/train_dinov3_probe.py`).
- **Optical-Flow Mask Advection**: `backend/app/forecast/advection.py` warps the Stage-1 cloud mask forward via semi-Lagrangian `cv2.remap` with a deterministic perturbed-flow ensemble (`cover_p10`, `cover_p50`, `cover_p90`, `fov_exit_fraction`, pixel probability map) and benchmarks every verified forecast against Persistence (`skill = 1 - MAE / MAE_persistence`).

## 4. Storage Architecture
- **Database (`skyobserver.db`)**: SQLite in WAL mode (`PRAGMA journal_mode=WAL`) with idempotent schema migrations (`_ensure_schema_migrations`) storing observations, metrics JSON, forecasts + baseline metrics, chronological verifications (including persistence and optical-flow errors), events, datasets, training sessions, and model registry versions.
- **Disk Images**: Raw snapshots, 5-class PNG segmentation masks, overlays, exclusion masks, exported MP4 timelapses, and dataset images live on disk under `./data` (never as SQLite BLOBs).
- **Retention**: Configurable per category; training samples (`is_training_sample=True` or referenced in `dataset_items`) and active models are strictly protected from cleanup.
