# Sky Observer — Model Card

All models run 100% locally in PyTorch (`torch >= 2.2`) with automatic hardware scaling from **Low-End CPU-only PCs (4 GB RAM, 256×256)** up to **NVIDIA CUDA / AMD ROCm / DirectML / Apple MPS GPUs**.

---

## Stage 1 — Cloud Segmentation (Pluggable Backends)

**Target classes (5 sky classes + static `Not_Sky = 255` ignore index):**
`0: clear_sky`, `1: thin_cloud`, `2: dense_cloud`, `3: dark_cloud`, `4: sun_glare`.

| Backend | Architecture / License | GPU Required | Intended Use |
| :--- | :--- | :---: | :--- |
| `radiometric-physics-v1` *(default)* | Calibrated CIE-Lab + `(B-R)/(B+R)` + AWB compensation + sun glare guard (MIT) | No (CPU-only) | Zero-GPU baseline for Low-End PCs & out-of-the-box daytime skies |
| `DeepLabV3-MobileNetV3` | Official `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` (**BSD-3-Clause**) | Optional | Fast daytime neural segmentation ready to fine-tune on SWIMSEG/SWINySEG |
| `DINOv3-LinearProbe` | Frozen ViT patch encoder + trainable `1×1 Conv` linear probe (inspired by **LenghuSky-8**, Apache-2.0 methodology; optional DINOv3 weights governed separately by the **DINOv3 License**) | Recommended for ViT backbone; CPU-safe with local fallback | Day + night sky adaptation from **50–100 user-labeled frames** (`scripts/train_dinov3_probe.py`) |
| `CloudSegUNet-ConvNeXt-ASPP` | ConvNeXt-Tiny + ASPP + scSE attention UNet (also supports MobileNetV3-Small, EfficientNet-B0, lightweight depthwise-separable UNet) | Recommended | Multi-scale daytime cloud segmentation fine-tuning |

- **Literature references** (architecture design inspiration; no third-party GPL/unlicensed source code copied):
  - **LenghuSky-8** (Rui et al., CVPR Findings 2025, Apache-2.0) — DINOv3 linear-probe day/night segmentation + optical-flow advection benchmark
  - **SCANet**, **UCloudNet**, **DDUNet**, **ACLNet** — multi-scale attention and lightweight sky/cloud segmentation paradigms

---

## Stage 2 — Cloud Type Classifier (`CloudTypeClassifier`)

- **Task**: Probability distribution over **11 Sky Classes** (10 WMO cloud genera + non-WMO sky states `clear_sky`, `mixed_clouds`, `unknown`; optional grouped WMO mode). Low-confidence predictions (`< 0.35`) return `unknown` rather than forcing an inaccurate genus.
- **Architecture**: Pretrained `EfficientNet-B0` (or `ConvNeXt-Tiny` / compact CNN) fused via a gated multimodal layer with an 18-dimensional radiometric, coverage, texture, and solar-elevation feature vector.

---

## Stage 3 — Short-Term Nowcasting (`Persistence` / `Optical-Flow Advection` / `TemporalSkyConvGRU`)

- **Baselines (always computed alongside predictions)**:
  - **Persistence**: `cover(t + h) = cover(t)`
  - **Optical-Flow Mask Advection**: semi-Lagrangian `cv2.remap` advection of the Stage-1 cloud mask + lightweight perturbed-flow ensemble (`cover_p10`, `cover_p50`, `cover_p90`, `fov_exit_fraction`, pixel-level probability map at `GET /api/forecast/advected-mask.png`)
- **Neural Component (`TemporalSkyConvGRU`)** *(Experimental)*: 1D temporal convolution + 2-layer GRU over 20-dimensional visual/motion feature sequences for camera-only horizons (`+5m`, `+15m`, `+20m`; extended `+30m`/`+60m` only when optional online NWP is enabled).
- **Honest Verification**: All verified forecasts record **MAE, RMSE, Brier score, Persistence MAE, Optical-Flow MAE, and Forecast Skill Score** (`1 - MAE_model / MAE_persistence`). Horizons that do not beat persistence or suffer `>= 70%` FOV exit are flagged `low_reliability: true`.

---

## Stage 4 — Event & Anomaly Detection (`SkyEventEngine` + `SkyEventClassifier-GRU`)

- Hybrid deterministic thresholds + optional sequence GRU over 8 operational/sky event classes (`normal`, `rapid_cloud_development`, `sky_clearing`, `cloud_cover_increasing`, `dark_cloud_development`, `storm_like_structure`, `camera_obstruction`, `unusual_change`).

---

## Anti-Drift Auto-Train Safeguard

When optional Location Auto-Train runs:
1. A permanent **fixed holdout manifest** (`data/datasets/location_auto_reinforce/fixed_holdout.json`) locks human-labeled and stratified site frames strictly in `val` across all future cycles (preventing train/val leakage).
2. New checkpoints are trained in standby and **promoted only if their validation metric meets or exceeds the active model** (within `promotion_tolerance`, default `0.01`). Regressing checkpoints stay in standby with the rejection reason logged in `model_registry.notes`.

---

## Reproducible Benchmarks

Run the unified evaluator locally to measure metrics on your machine and datasets (no hard-coded or invented claims):

```bash
PYTHONPATH=backend python scripts/evaluate.py
PYTHONPATH=backend python scripts/evaluate_nowcast.py --from-db
```

---

## Limitations & Responsible Use

1. Visual cloud classification is a single-camera visual approximation, not an authoritative meteorological observation.
2. Narrow-FOV single cameras cannot observe clouds before they enter the field of view; `+30m/+60m` skill degrades sharply without external NWP/satellite inputs.
3. No camera model directly measures pressure, humidity, or true cloud-base altitude; precipitation/storm cues are experimental visual indicators only.
