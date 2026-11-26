# Changelog

All notable changes to Sky Observer are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-11-26

### Added
- **Dual UI Modes (`Simple` ↔ `Expert`) & Adaptive Brand Identity**:
  - Always-accessible 1-click **`[ Simple | Expert ]`** switch in the top header (`localStorage`-persisted) with simplified visual Sky Now, Clean Cloud/Rain Map, and Basic Settings views alongside the full meteorological Expert workstation.
  - Dynamic **Doppler Scope** vector SVG brand logos (`logo-simple.svg` / `logo-expert.svg`) and matching live browser-tab favicon that switch automatically with UI mode.
- **Zero-GPU Default Segmentation & Reversible Neural Backends**:
  - Calibrated `radiometric-physics-v1` engine runs out of the box on CPU without GPU or downloaded weights.
  - Pluggable neural backends: daytime `DeepLabV3-MobileNetV3` (`torchvision`, BSD-3-Clause), day/night `DINOv3-LinearProbe` (LenghuSky-8-inspired frozen ViT + `1x1 Conv` probe via `scripts/train_dinov3_probe.py`), and `CloudSegUNet`.
  - 1-click deactivation of active neural segmentation (`POST /api/models/deactivate` or activating `radiometric-physics-v1`) to return immediately to built-in calibrated physics.
  - Protected `[baseline-default]` PyTorch checkpoints shielded from storage purges, manual deletion, and pruning.
- **Honest Nowcasting & Probabilistic Mask Advection**:
  - Semi-Lagrangian optical-flow cloud mask advection (`backend/app/forecast/advection.py`) with deterministic perturbed-flow ensemble (`cover_p10`, `cover_p50`, `cover_p90`, `fov_exit_fraction`) and `GET /api/forecast/advected-mask.png`.
  - Persistence and Optical-Flow MAE baselines recorded in chronological verification alongside `skill_vs_persistence` (`1 - MAE_model / MAE_persistence`) and automatic `low_reliability` horizon flagging.
  - Reproducible evaluation CLIs: `scripts/evaluate_nowcast.py` and unified `scripts/evaluate.py`.
- **Anti-Drift Local Auto-Train & Calibration**:
  - Fixed-holdout validation manifest (`fixed_holdout.json`) and anti-drift promotion gate in `AutoLearnManager`.
  - Automatic sky-vs-obstacle exclusion mask proposal (`POST /api/camera/exclusion-mask/propose`) and 1-click button in `ExclusionMaskPainter`.
  - Android IP Webcam `/sensors.json` orientation reader (`POST /api/camera/geometry/from-sensors`), solar-in-FOV azimuth refinement (`POST /api/camera/geometry/refine-sun`), and configurable `max_projection_zenith_deg` (default `70°`) with distance/zenith opacity decay.
- **Low-Cost Hardware QC & Adaptive Night Mode**:
  - Hardware QC detectors for indoor window reflections, smartphone AE/AWB jumps, localized raindrops/smudges, heavy 8x8 JPEG blockiness, and true frozen-frame detection (avoiding false positives on calm clear skies).
  - Adaptive night/twilight sky regime (`backend/app/vision/night.py`) with lunar position/phase, DoG bright-star visibility cue, and urban light-pollution vs dark-site night segmentation.
- **Solar PV Estimation, Smart-Home & IoT Integrations**:
  - Optional clear-sky GHI, cloud-transmittance PV power output, and solar ramp-event detection (`GET /api/solar/pv`).
  - Ultra-fast `GET /api/sky-state` telemetry endpoint, optional Home Assistant MQTT Auto-Discovery publisher (`backend/app/services/mqtt_ha.py`), local `allsky_folder` watcher, and ESP32-CAM / Raspberry Pi / AllSky presets.
  - Sky-state visual tags (`contrails`, `rain`, `snow`, `fog`) alongside explicit `is_uncertain` reporting.
- **Security, Legal & DevOps**:
  - DNS-resolving SSRF guard on camera/web stream URLs, credential masking across API responses & logs, CORS wildcard rejection, `/api/health` alias, `python-multipart==0.0.34`, and `SECURITY.md`.
  - Repository audit (`AUDIT.md`), expanded dataset license catalog (`docs/DATASETS.md`), `web_stream` ToS notice, `CONTRIBUTING.md`, GitHub Actions CI + automatic tag-driven Releases, grouped Dependabot config, pre-commit config, and slim CPU-wheel Dockerfiles.

### Changed
- Aligned scientific terminology: "Projected Cloud Cover Map" (reserving "Radar" strictly for RainViewer precipitation radar) and "11 Sky Classes (10 WMO Genera + Sky States)".
- Reorganized technical documentation into `docs/` (`API.md`, `ARCHITECTURE.md`, `DATASETS.md`, `INSTALLATION.md`, `MODEL_CARD.md`, `TRAINING.md`, `TROUBLESHOOTING.md`).
