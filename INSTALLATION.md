# Sky Observer — Installation Guide

## 1. Prerequisites
- **Hardware**: Desktop PC with 16 GB RAM (recommended) and optional NVIDIA GPU (8 GB VRAM, CUDA 12+) — CPU-only fallback works automatically.
- **Camera**: Android smartphone on the same local Wi-Fi / LAN running an IP camera app (e.g., *IP Webcam* or *DroidCam* / RTSP camera server), or connected via **Iriun Webcam** (USB / Wi-Fi).
- **Software**:
  - **Docker deployment**: Docker Engine 24+ and Docker Compose v2+. For NVIDIA GPU acceleration on Linux/WSL2, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  - **Direct host deployment**: Python 3.11+ and Node.js 20+.

## 2. Docker Deployment (Recommended)

### CPU-Only Mode (Linux / Windows Docker Desktop)
```bash
cp .env.example .env
docker compose build
docker compose up -d
```

### NVIDIA GPU Accelerated Mode (8 GB VRAM)
```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Or use the included helper scripts:
- Linux: `bash scripts/setup.sh`
- Windows PowerShell: `.\scripts\setup.ps1`

Dashboard URL: **http://localhost:8080**

## 3. Direct Local Execution (Without Docker)
```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

cd frontend && npm install && npm run build && cd ..
python scripts/train_baseline.py

PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8080
```
