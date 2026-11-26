# Sky Observer — Local AI Training & Fine-Tuning Guide

## 1. Overview
Sky Observer supports local PyTorch training and fine-tuning across four lightweight model families designed to stay well below an **8 GB VRAM** and **16 GB RAM** ceiling:
1. **Model 1 (`segmentation`) — `CloudSegUNet`**: Depthwise-separable Mobile-UNet predicting 6 classes (`clear_sky`, `thin_cloud`, `dense_cloud`, `dark_cloud`, `haze_fog`, `uncertain`).
2. **Model 2 (`classification`) — `CloudTypeClassifier`**: CNN + 12-feature radiometric/texture fusion classifier for 11 cloud formations.
3. **Model 3 (`temporal`) — `TemporalSkyConvGRU`**: Experimental 1D-Conv + 2-layer GRU predicting cloud cover and state likelihoods at `+5m, +15m, +30m, +60m` from sequences of frame features.
4. **Model 4 (`event`) — `SkyEventClassifier`**: Sequence GRU classifying operational/weather events (`rapid_cloud_development`, `sky_clearing`, `storm_like_structure`, etc.).

## 2. Resource Controls (8 GB VRAM Budget)
- `ai.max_gpu_memory_fraction: 0.85` enforces a PyTorch per-process CUDA memory cap.
- `ai.mixed_precision: true` uses `torch.amp` FP16 autocast on CUDA.
- `ai.batch_size: 4` (recommended for 512×512 on 8 GB VRAM) and `ai.gradient_accumulation_steps` allow larger effective batches without OOM.
- Automatic fallback to CPU (`batch_size=2` recommended) when CUDA is absent.

## 3. Workflow in the Dashboard (`Model Lab` & `Dataset Labeling`)
1. **Collect or Bootstrap Data**: Use **Prepare Local Baseline Dataset** in Model Lab, import a public dataset directory (`SWIMSEG`, `SWINySEG`, `WSISEG`, `CCSN`), or paint masks and label your own Android camera frames in **Dataset Labeling**.
2. **Launch Training**: Select the model family, dataset, epochs, batch size, and learning rate. Training runs in a non-blocking background thread while camera capture and the dashboard remain responsive.
3. **Checkpoints & Resumption**: Both `_best.pt` and `_last.pt` (including optimizer state and epoch history) are stored in `/data/models`.
4. **Activation & Rollback**: Trained checkpoints are registered in `model_registry`. You can activate any version or roll back to the prior version in one click.

## 4. CLI Training Script
To bootstrap the baseline dataset and train all 4 initial models from the terminal:
```bash
python scripts/train_baseline.py
```
