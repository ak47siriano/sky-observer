# Sky Observer — Model Card

All models run locally in PyTorch (`torch 2.15.0`) and are sized for **8 GB NVIDIA VRAM** and **16 GB RAM**, with automatic CPU fallback when CUDA is unavailable.

---

## Model 1 — Cloud Segmentation (`CloudSegUNet-Mobile`)
- **Task**: Pixel-level 6-class sky segmentation (`clear_sky`, `thin_cloud`, `dense_cloud`, `dark_cloud`, `haze_fog`, `uncertain`).
- **Architecture**: Depthwise-separable Mobile-UNet encoder-decoder (~185K parameters) ensembled at inference with a calibrated radiometric physical prior `(B-R)/(B+R)` + Lab luminance and Shannon-entropy uncertainty masking.
- **Input**: RGB frame resized to configurable inference resolution (`384×384`, `512×512`, or `640×640`).
- **Baseline Checkpoint (`segmentation-v1.0.0-c1bd45`)**:
  - Dataset: `skyobserver_local_baseline` (144 frames across 8 sky scenarios, group-split without sequence leakage)
  - Device: CPU fallback (4 epochs)
  - Measured validation metrics: `val_miou = 0.6304`, `val_pixel_acc = 0.8756`, `val_loss = 0.5392`

## Model 2 — Cloud Type Classifier (`CloudTypeClassifier-Fusion`)
- **Task**: Probability distribution over 11 visual cloud classes (`clear_sky`, `cirrus`, `cirrostratus`, `cumulus`, `stratocumulus`, `stratus`, `altocumulus`, `altostratus`, `cumulonimbus_like`, `mixed_clouds`, `unknown`). Low-confidence predictions (`< 0.35`) default to `unknown`.
- **Architecture**: Compact depthwise-separable CNN backbone (`224×224` RGB) fused with a 12-dimensional radiometric/coverage/texture descriptor vector.
- **Baseline Checkpoint (`classification-v1.0.0-137e04`)**:
  - Dataset: `skyobserver_local_baseline` (class-balanced cross-entropy loss, 6 epochs)
  - Measured validation metrics: `val_accuracy = 0.8750`, `val_macro_f1 = 0.7179`, `val_precision = 0.7714`, `val_recall = 0.7000`

## Model 3 — Temporal Sky Predictor (`TemporalSkyConvGRU`) [Experimental]
- **Task**: Predicts future cloud coverage (`+5m`, `+15m`, `+30m`, `+60m`) and probabilities for clearing, increasing cloudiness, overcast, darkening, and experimental precipitation visual cues from sequences of 20-dimensional frame features.
- **Architecture**: 1D temporal convolution + 2-layer GRU (`hidden_dim=64`).
- **Baseline Checkpoint (`temporal-v1.0.0-da88f3`)**:
  - Dataset: `skyobserver_local_baseline` (6-frame sequences, 12 epochs)
  - Measured validation metrics: `val_mae_pct = 21.08%`, `val_loss = 0.3857`
  - **Status**: **Experimental.** Should be fine-tuned and chronologically verified on multi-day/multi-week real Android camera sequences before relying on neural horizon predictions alone.

## Model 4 — Event Classifier (`SkyEventClassifier-GRU`)
- **Task**: Sequence classification across 8 operational and sky events (`normal`, `rapid_cloud_development`, `sky_clearing`, `cloud_cover_increasing`, `dark_cloud_development`, `storm_like_structure`, `camera_obstruction`, `unusual_change`).
- **Architecture**: Sequence GRU (`hidden_dim=48`) + MLP head.
- **Baseline Checkpoint (`event-v1.0.0-e46ee3`)**:
  - Dataset: `skyobserver_local_baseline` (12 epochs)
  - Measured validation metrics: `val_accuracy = 0.7500`, `val_macro_f1 = 0.4286`

---

## Limitations & Responsible Meteorological Use
1. Baseline v1.0.0 checkpoints were trained on `skyobserver_local_baseline` (locally generated multi-scenario sky sequences) to validate the end-to-end training/inference pipeline offline. Fine-tune on public datasets (SWIMSEG, SWINySEG, WSISEG, CCSN) and your own Android camera annotations in **Model Lab** for field accuracy.
2. Visual cloud genus classification is an image-based visual approximation, not an authoritative meteorological observation.
3. No model measures temperature, pressure, humidity, or true cloud-base altitude; precipitation/storm signals are experimental visual indicators only.
