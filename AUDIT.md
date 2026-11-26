# Sky Observer — Codebase & Reality Audit

This audit compares the repository documentation (`README.md`, `ARCHITECTURE.md`, `MODEL_CARD.md`, `TRAINING.md`, `DATASETS.md`) against the **actual verified code** as of `2026-11-26` before executing the multi-phase engineering upgrade roadmap.

---

## 1. Implemented & Working Features

| Area | Status | Evidence in Code |
| :--- | :---: | :--- |
| **Camera Ingestion** | Working | `backend/app/camera/manager.py` supports HTTP snapshot, MJPEG, RTSP, local USB/virtual webcam (`DirectShow`/`V4L2`), file upload, `demo_simulation`, and `web_stream` (`.m3u8` HLS + SkylineWebcams page resolver). |
| **Radiometric Cloud Segmentation** | Working | `backend/app/vision/segmentation.py` implements `radiometric-physics-v1` (5 classes: `clear_sky`, `thin_cloud`, `dense_cloud`, `dark_cloud`, `sun_glare` + `Not_Sky=255` ignore mask) with AWB bias compensation, dark-cloud-base protection, and sun-aureole prior, plus optional PyTorch `CloudSegUNet` inference. |
| **Cloud Classification** | Working | `backend/app/vision/classification.py` + `backend/app/ai/architectures.py` (`CloudTypeClassifier` CNN + 18 tabular features, 11 sky classes with low-confidence fallback). |
| **Dense Optical Flow & Tilt Geometry** | Working | `backend/app/vision/motion.py` (Farneback + Lucas-Kanade + Phase Correlation, MAD outlier filtering, CLAHE stabilization, dual-deck separation) and `backend/app/vision/geometry.py` (ENU 3D camera basis, `-20°..90°` elevation, roll, azimuth, `estimate_camera_tilt_from_frame` auto-horizon/roll, NOAA solar position, bilinear ground-to-cloud projection). |
| **Local Persistence & Real-Time UI** | Working | `backend/app/database.py` (SQLite WAL, 8 SQLAlchemy tables), `backend/app/main.py` + routers + `/ws/live`, React 18 + TypeScript + Leaflet PWA (`frontend/src/`), `SkyHorizonOverlay` exact Sutherland-Hodgman horizon clipping, AI Lab built-in engine visibility. |
| **Basic Chronological Verification** | Working | `backend/app/forecast/verification.py` verifies matured forecasts chronologically (`MAE`, `RMSE`, `Accuracy`, `Precision`, `Recall`, `F1`, `Brier`, calibration bins, rolling MAE trend). |
| **Online Hybrid Weather Fusion** | Working | `backend/app/services/weather_online.py` (Open-Meteo NWP + LCL cloud-base + RainViewer radar tiles) with strict `offline_mode` lock. |
| **Core Automated Tests** | Working | 31 passing tests across `test_vision.py`, `test_radar_and_online_weather.py`, `test_api.py`, and `test_forecast_and_verification.py`. |

---

## 2. Partially Implemented Features (Gaps Identified)

| Area | What Exists | Concrete Gap to Fix |
| :--- | :--- | :--- |
| **Nowcasting Evaluation & Baselines** | Rule-based + `TemporalSkyConvGRU` predictions (`+5/+15/+20/+30/+60m`) and MAE verification. | Missing **Persistence baseline**, **Optical-Flow mask advection baseline**, **Forecast Skill Score** (`1 - MAE_model / MAE_persistence`), UI "low-reliability" warnings when `skill <= 0`, and a standalone reproducible benchmark script (`scripts/evaluate_nowcast.py`). |
| **Image Quality Control (`quality.py`)** | Checks blur, darkness, overexposure, glare, lens obstruction, duplicates, and camera shift. | Missing low-cost hardware artifacts: **window-glass indoor reflections**, **sudden smartphone AE/AWB jumps**, **persistent raindrops/smudges**, **heavy JPEG compression (ESP32-CAM)**, and explicit QC flags visible on the timeline. |
| **Network & Camera Security** | `ServerConfig.host` defaults to `127.0.0.1`; `docker-compose.yml` binds `127.0.0.1:8080`; dataset path-traversal guard is in place. | `CameraManager._is_private_url` does not resolve DNS (vulnerable to DNS rebinding and link-local `169.254.169.254` SSRF on `web_stream`); `source_url` credentials (`user:pass@`) are not masked in logs/API responses; `SECURITY.md` and `/api/health` alias are missing; `Dockerfile` defaults to `0.0.0.0`. |
| **Auto-Train (`autolearn.py`)** | Collects high-confidence observations, merges user labels + replay, and trains local models. | No **fixed holdout validation set** to block model promotion on regression/drift; `test_ai_and_training.py` hits a Windows PyTorch CPU OpenMP access violation during `test_auto_label_and_auto_train_local_reinforcement`. |
| **Projection Uncertainty & Zenith Cap** | Horizon taper (`2°..7.5°`) and radial fade exist in `geometry.py`. | No configurable **maximum zenith angle cutoff (default `70°`)** with explicit opacity decay and height-source legend in the UI. |
| **Terminology & Scientific Honesty** | Functional UI and docs exist. | Docs/UI still refer to **"11 WMO Cloud Genera"** (WMO defines **10** genera; `clear_sky`, `mixed_clouds`, `cumulonimbus_like`, `unknown` are sky states) and call camera sky projection **"3D Radar" / "Volumetric"** (radar applies only to RainViewer). `README.md` lacks an explicit **Limitations** section. |

---

## 3. Missing Features (Planned in Roadmap)

1. **Pixel-level & probabilistic nowcasting**: optical-flow advected future cloud masks on the map + lightweight perturbed-flow ensemble for per-pixel coverage probability and confidence bands.
2. **Optional DeepLabV3 (daytime, `torchvision` BSD-3) & DINOv3 linear-probe segmentation (day/night, inspired by LenghuSky-8)**: pluggable segmentation backends gated by hardware presets so `Low-End PC (4 GB RAM)` keeps zero-GPU `radiometric-physics-v1` as default, plus 50–100 frame day/night linear-probe fine-tuning workflow and sky/obstacle auto-mask proposals.
3. **Adaptive Night Mode**: automatic day/twilight/night switching from solar elevation, per-camera light-pollution vs dark-sky calibration (with optional Open-Meteo check), lunar visibility reference, and optional bright-star catalog detection.
4. **Hardware-free calibration extensions**: Android IP Webcam `/sensors.json` orientation/compass reading and solar-in-FOV azimuth refinement with residual degree error readout.
5. **Realistic classification states**: `contrails`, `rain`, `snow`, `fog`, explicit `uncertain` low-confidence output, and optional shared-encoder multitask evaluation.
6. **Optional Solar Irradiance & PV Estimation**: clear-sky GHI model + cloud attenuation + user PV array (`kWp`, tilt, azimuth) + solar ramp-event alerts.
7. **Low-cost sources & integrations**: ESP32-CAM / Raspberry Pi Camera presets, ultra-fast `GET /api/sky-state`, optional Home Assistant MQTT Auto-Discovery, and Android reuse guide.
8. **Packaging, Docs Layout & CI**: docs moved under `docs/`, `CONTRIBUTING.md`, `CHANGELOG.md`, Bash + PowerShell parity, unified `pyproject.toml` dependencies, GitHub Actions CI (`ruff`, `pytest`, `tsc`, `vite build`, Docker CPU build), `.github/dependabot.yml`, and `.pre-commit-config.yaml`.

---

## 4. Declared Metrics Not Yet Independently Reproduced

- **`MODEL_CARD.md` baseline v1.0.0 metrics** (`CloudSegUNet` `val_miou=0.6304`, `CloudTypeClassifier` `val_accuracy=0.8750`, `TemporalSkyConvGRU` `val_mae_pct=21.08%`, `SkyEventClassifier` `val_accuracy=0.7500`) were measured exclusively on the **synthetic** `skyobserver_local_baseline` dataset, not on standardized splits of SWIMSEG/SWINySEG/CCSN or manually labeled real webcam frames.
- **No persistence-normalized forecast skill score** is currently measured or published for `+5/+15/+20/+30/+60m`.
- All future metric claims will be generated strictly by running `scripts/evaluate.py` / `scripts/evaluate_nowcast.py`.
