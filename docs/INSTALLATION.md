# Sky Observer — Installation & Hardware Guide

## 1. Prerequisites

| Tier | CPU & RAM | GPU | Notes |
| :--- | :--- | :--- | :--- |
| **Minimum (Low-End / Mini-PC)** | Dual-Core · **4 GB RAM** | None (**100% CPU**) | Runs `radiometric-physics-v1` at `256×256` in real time |
| **Recommended (Balanced)** | Quad-Core · **8 GB RAM** | Optional (4+ GB VRAM) or CPU | Runs `384×384` + optional daytime `DeepLabV3-MobileNetV3` |
| **Workstation (High-End / Ultra)** | 6–12+ Cores · **16–32 GB RAM** | NVIDIA CUDA / AMD ROCm / DirectML / Apple MPS | Runs `512–768px`, fast fine-tuning, and DINOv3 linear probing |

- **Docker deployment**: Docker Engine 24+ and Docker Compose v2+.
- **Native deployment**: Python 3.11+ and Node.js 20+.

---

## 2. Docker Deployment (Recommended, Slim CPU Image)

The default `Dockerfile` installs **CPU-only PyTorch wheels** (`--index-url https://download.pytorch.org/whl/cpu`) and runs as a non-root user (`skyobserver`) with healthcheck on `http://127.0.0.1:8080/api/health`.

### CPU-Only Mode (Linux / macOS / Windows Docker Desktop)
```bash
cp .env.example .env
docker compose up -d --build
```

### NVIDIA GPU Accelerated Mode (Linux / WSL2 + NVIDIA Container Toolkit)
```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Open **`http://localhost:8080`**.

---

## 3. Direct Local Execution (Without Docker)

### Linux / macOS (Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r backend/requirements.txt

cd frontend && npm ci && npm run build && cd ..

PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8080
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r backend/requirements.txt

cd frontend; npm ci; npm run build; cd ..

$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

> Bind `--host 0.0.0.0` only on a trusted private LAN when you want to view the PWA from your phone. See [SECURITY.md](../SECURITY.md).

### Optional Extras & GPU Backends
- **NVIDIA CUDA (Linux / Windows)**: `pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision`
- **AMD ROCm (Linux)**: `pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision`
- **AMD / Intel DirectML (Windows)**: `pip install -e ".[directml]"`
- **DINOv3 / DINOv2 ViT linear probing**: `pip install -e ".[dinov3]"`
- **Home Assistant MQTT Auto-Discovery**: `pip install -e ".[mqtt]"`

---

## 4. Repurposing an Old Android Smartphone as a 24/7 Sky Camera

Using a spare Android phone with **IP Webcam** (`http://<lan-ip>:8080/shot.jpg`) or **Iriun** is the lowest-cost way to deploy Sky Observer. Follow these field practices for safety and image stability:

1. **Battery Swelling & Continuous Power Safety**:
   - Never leave a lithium-ion smartphone permanently sealed in direct sun at 100% charge.
   - Use an Android battery-charge limiter (built-in *Protect Battery* at 80%, or root/AccA limit at 50–70%), OR plug the USB charger into a smart outlet cycling power (e.g., charge 40–70%) so the battery never sits continuously pinned at 4.35 V while hot.
   - Keep screen brightness at minimum (or enable IP Webcam's *Screen Dim / Run in Background*) to cut heat output by >50%.
2. **Weatherproofing & Thermal Management**:
   - For outdoor rooftop mounting, use a ventilated white/reflective IP65 junction box with an **acrylic hemispherical dome** or flat optical glass viewport plus a silica-gel desiccant packet to prevent condensation.
   - Avoid black enclosures in direct summer sunlight.
3. **Shooting Through an Indoor Window (Zero-Cost Setup)**:
   - Place the lens **as close to the glass as possible** (ideally within 1 cm) with a matte black cardboard/foam shroud around the lens to block interior room lights.
   - Sky Observer's QC engine automatically flags `window_reflection` in evening/dim frames and excludes flagged frames from Auto-Train.
4. **Exposure & Orientation Calibration**:
   - In IP Webcam, enable **Sensor Data** (`http://<phone-ip>:8080/sensors.json`) so Sky Observer can read accelerometer tilt and compass heading via `POST /api/camera/geometry/from-sensors`.
   - Disable Android battery optimization (*Unrestricted*) for the camera app so Wi-Fi stays active overnight.
5. **Other Low-Cost Hardware**:
   - **ESP32-CAM**: Use the 1-click **ESP32-CAM** preset in Settings (`http://<ip>/capture`, interval 20s). Enable `QVGA`/`VGA`/`SVGA` with JPEG quality `10–12` to avoid `heavy_jpeg_blocking` flags.
   - **Raspberry Pi Camera (`rpicam-vid` / `mediamtx` RTSP)** or **AllSky (`thomasjacquin/allsky` / `AllSkEye`)**: Select the **Raspberry Pi Cam** or **AllSky Folder** preset in Settings.
