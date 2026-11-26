FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
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
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY config /config

RUN groupadd -r skyobserver && useradd -r -g skyobserver -u 1000 skyobserver && \
    mkdir -p /data/db /data/images /data/models /data/datasets /data/timelapse /config && \
    chown -R skyobserver:skyobserver /app /data /config

USER skyobserver

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
