# Sky Observer — Troubleshooting Guide

## 1. Camera Shows `DISCONNECTED`
- Verify the Android smartphone and PC are on the **same local Wi-Fi/LAN subnet** (avoid guest Wi-Fi client isolation).
- Open the configured snapshot URL (e.g. `http://192.168.1.50:8080/shot.jpg`) in a local browser on the PC to confirm the Android IP camera app is serving frames.
- For RTSP streams, confirm UDP/TCP port (e.g. `8554` or `554`) is reachable through the host firewall.
- Use **Settings → Test** to inspect exact HTTP/RTSP latency or error messages.

## 2. GPU Not Detected / Running in `CPU FALLBACK`
- Confirm NVIDIA drivers and `nvidia-smi` work on the host.
- For Docker on Linux/WSL2, ensure `nvidia-container-toolkit` is installed and rebuild with CUDA wheels:
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`
- Check `/api/system/gpu` to inspect `cuda_available` and VRAM limits.

## 3. Out of VRAM During Training (8 GB GPU)
- Reduce `ai.batch_size` from `4` to `2` or `1` in `config/config.yaml` and increase `ai.gradient_accumulation_steps` to `2` or `4`.
- Lower `vision.inference_resolution` from `640` to `512` or `384`.
- Ensure `ai.mixed_precision: true` and `ai.max_gpu_memory_fraction: 0.85`.

## 4. `docker compose` Fails to Connect to Daemon (Windows)
- Start **Docker Desktop** and ensure the Linux container engine (`dockerDesktopLinuxEngine`) is running before invoking `docker compose up -d`.

## 5. Forecast Shows `unavailable`
- Forecasts intentionally return `unavailable` when no usable frames have been captured or when the latest frame failed image quality checks (e.g. night darkness, lens obstruction, severe glare).
