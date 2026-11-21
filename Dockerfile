# =============================================================================
# Stage 1: Build Frontend (React + TypeScript + Vite)
# =============================================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Unified Runtime (FastAPI + PyTorch + OpenCV + Static Frontend)
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    SKYOBSERVER_ENV=production \
    SKYOBSERVER_DATA_DIR=/data \
    SKYOBSERVER_CONFIG_PATH=/config/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist
COPY config /config

RUN groupadd -r skyobserver && useradd -r -g skyobserver -u 1000 skyobserver && \
    mkdir -p /data/db /data/images/raw /data/images/processed /data/images/masks \
             /data/models /data/datasets /data/timelapse /data/map_tiles /config && \
    chown -R skyobserver:skyobserver /app /data /config

USER skyobserver

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/system/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
