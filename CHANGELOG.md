# Changelog

All notable changes to Sky Observer are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Honest Nowcasting & Probabilistic Mask Advection**:
  - Semi-Lagrangian optical-flow cloud mask advection (`backend/app/forecast/advection.py`) with deterministic perturbed-flow ensemble (`cover_p10`, `cover_p50`, `cover_p90`, `fov_exit_fraction`) and `GET /api/forecast/advected-mask.png`.
  - Persistence and Optical-Flow MAE baselines recorded in chronological verification alongside `skill_vs_persistence` (`1 - MAE_model / MAE_persistence`) and automatic `low_reliability` horizon flagging.
  - Reproducible evaluation CLIs: `scripts/evaluate_nowcast.py` and unified `scripts/evaluate.py`.
- **Pluggable Segmentation Backends & Anti-Drift Auto-Train**:
  - `DeepLabV3-MobileNetV3` (`torchvision`, BSD-3-Clause) daytime segmentation backend.
  - `DINOv3-LinearProbe` frozen ViT patch encoder + `1x1 Conv` linear probe (inspired by LenghuSky-8) with dedicated day/night fine-tuning CLI (`scripts/train_dinov3_probe.py`).
  - Automatic sky-vs-obstacle exclusion mask proposal (`POST /api/camera/exclusion-mask/propose`) and 1-click button in `ExclusionMaskPainter`.
  - Fixed-holdout validation manifest (`fixed_holdout.json`) and anti-drift promotion gate in `AutoLearnManager`.
- **Low-Cost Hardware QC, Calibration & Adaptive Night Mode**:
  - Hardware QC detectors for indoor window reflections, smartphone AE/AWB jumps, localized raindrops/smudges, and heavy 8x8 JPEG blockiness (excluded from Auto-Train).
  - Android IP Webcam `/sensors.json` orientation reader (`POST /api/camera/geometry/from-sensors`), solar-in-FOV azimuth refinement (`POST /api/camera/geometry/refine-sun`), and configurable `max_projection_zenith_deg` (default `70°`) with distance/zenith opacity decay.
  - Adaptive night/twilight sky regime (`backend/app/vision/night.py`) with lunar position/phase, DoG bright-star visibility cue, and urban light-pollution vs dark-site night segmentation.
- **Solar PV Estimation, Smart-Home & IoT Integrations**:
  - Optional clear-sky GHI, cloud-transmittance PV power output, and solar ramp-event detection (`GET /api/solar/pv`).
  - Ultra-fast `GET /api/sky-state` telemetry endpoint, optional Home Assistant MQTT Auto-Discovery publisher (`backend/app/services/mqtt_ha.py`), local `allsky_folder` watcher, and ESP32-CAM / Raspberry Pi / AllSky presets.
  - Sky-state visual tags (`contrails`, `rain`, `snow`, `fog`) alongside explicit `is_uncertain` reporting.
- **Security, Legal & DevOps**:
  - DNS-resolving SSRF guard on camera/web stream URLs, credential masking across API responses & logs, CORS wildcard rejection, `/api/health` alias, and `SECURITY.md`.
  - Repository audit (`AUDIT.md`), expanded dataset license catalog (`docs/DATASETS.md`), `web_stream` ToS notice, `CONTRIBUTING.md`, GitHub Actions CI, Dependabot, pre-commit config, and slim CPU-wheel Dockerfiles.

### Changed
- Aligned scientific terminology: "Projected Cloud Cover Map" (reserving "Radar" strictly for RainViewer precipitation radar) and "11 Sky Classes (10 WMO Genera + Sky States)".
- Reorganized technical documentation into `docs/` (`API.md`, `ARCHITECTURE.md`, `DATASETS.md`, `INSTALLATION.md`, `MODEL_CARD.md`, `TRAINING.md`, `TROUBLESHOOTING.md`).
