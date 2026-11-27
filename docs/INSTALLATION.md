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

## 2. Docker Deployment (Recommended, Single Unified Dockerfile)

Sky Observer uses **one unified multi-stage root [`Dockerfile`](../Dockerfile)** (stage 1 compiles the React PWA with Node 22; stage 2 runs FastAPI + Uvicorn as non-root user `skyobserver`, serving both the API and `frontend/dist` on port **`8080`**).

- **Offline-first by default**: `.env.example` and `docker-compose.yml` set `SKYOBSERVER_OFFLINE_MODE=true` (zero outbound weather/radar traffic). Change `SKYOBSERVER_OFFLINE_MODE=false` in `.env` (or toggle Online/Hybrid inside the UI) to enable Open-Meteo / RainViewer.
- **Persistent volumes**: `docker-compose.yml` persists `db`, `images`, `models`, `datasets`, `timelapse`, `map_tiles`, `masks`, and `./config`.
- **Loopback by default**: port publishing uses `${SKYOBSERVER_HOST:-127.0.0.1}:${SKYOBSERVER_PORT:-8080}:8080`. Only switch `SKYOBSERVER_HOST=0.0.0.0` on a trusted private LAN **after enabling API token authentication** (see [SECURITY.md](../SECURITY.md)).

### CPU-Only Mode (Default — Linux / macOS / Windows Docker Desktop)
Installs slim **CPU-only PyTorch wheels** (`--index-url https://download.pytorch.org/whl/cpu`):
```bash
cp .env.example .env
docker compose up -d --build
```
Open **`http://localhost:8080`**.

### NVIDIA GPU Accelerated Mode (Linux / WSL2 + NVIDIA Container Toolkit)
Overrides `PYTORCH_INDEX_URL` to install CUDA 12.4 PyTorch wheels and reserves the NVIDIA GPU:
```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### USB Webcams / Iriun in Docker (Linux Only)
The standard `docker-compose.yml` supports **network camera sources only** (`http_snapshot` / Android IP Webcam, `mjpeg_stream`, `rtsp`, `web_stream`, or a mounted `allsky_folder`).

- **Linux hosts**: pass `/dev/video0` (or set `SKYOBSERVER_VIDEO_DEVICE` in `.env`) using `docker-compose.usb.yml`:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.usb.yml up -d
  ```
- **Windows / macOS**: Docker Desktop does not natively pass USB webcams into Linux containers. Use a network camera feed (e.g. Android IP Webcam / RTSP) or run Sky Observer natively (Section 3).

### Installing the PWA Over LAN (HTTPS Requirement)
Browsers (and especially **iOS Safari**) only register Service Workers and allow **Add to Home Screen (PWA)** on `localhost` or **HTTPS** origins — plain `http://192.168.x.x:8080` will NOT install as a PWA.

Choose one of these three HTTPS options for LAN mobile access:

1. **Built-in Caddy HTTPS Override (`docker-compose.https.yml`)**:
   - In `.env`, set `SKYOBSERVER_HOST=0.0.0.0` and `SKYOBSERVER_LAN_HOST=<your-lan-ip-or-hostname>` (enable API auth first).
   - Start with:
     ```bash
     docker compose -f docker-compose.yml -f docker-compose.https.yml up -d
     ```
   - Open `https://<your-lan-ip>:8443` and import Caddy's local root CA certificate (`caddy_data/caddy/pki/authorities/local/root.crt`) on your phone so iOS/Android trusts the local HTTPS certificate.
2. **Tailscale (`tailscale serve`) — Easiest Zero-Certificate Option**:
   - Keep `SKYOBSERVER_HOST=127.0.0.1` (never even exposed on LAN) and run `tailscale serve --bg 8080` on the host. Tailscale provides automatic trusted Let's Encrypt HTTPS (`https://<machine>.<tailnet>.ts.net`) with built-in WireGuard authentication.
3. **`mkcert` Local CA**:
   - Generate locally trusted certificates (`mkcert -install && mkcert skyobserver.local <your-lan-ip>`) and mount them into Caddy or your existing Nginx/Traefik reverse proxy.

### Offline Basemaps with Local PMTiles / Protomaps & Self-Hosted Radar (LibreWXR)
- **Offline Basemaps (`PMTiles`)**: Bulk pre-caching tiles from public `tile.openstreetmap.org` servers violates OSMF Tile Usage Policy and is disabled. For 100% offline regional vector maps, extract your local area with the [Protomaps `pmtiles` CLI](https://docs.protomaps.com/pmtiles/cli) and drop the `.pmtiles` file into `data/map_tiles/` (e.g. `data/map_tiles/local.pmtiles`). Sky Observer automatically exposes it with HTTP `Range` support at `GET /api/map/pmtiles/local.pmtiles` (listed via `GET /api/map/pmtiles`).
- **Precipitation Radar (`RainViewer` vs. Self-Hosted `LibreWXR`)**: RainViewer's free tier serves past radar frames only (2 h history, max zoom `7`, Universal Blue scheme). To replace it with a self-hosted **LibreWXR** tile server (or disable precipitation radar), set `online_weather.radar_provider: "librewxr"` and `online_weather.librewxr_tile_url` in `config/config.yaml`.

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
