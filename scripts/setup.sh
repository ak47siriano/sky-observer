#!/usr/bin/env bash
set -euo pipefail

echo "=== Sky Observer Local Setup ==="
if [ ! -f .env ]; then
  cp .env.example .env
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected; building and starting with GPU support..."
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
else
  echo "No nvidia-smi detected; building and starting in CPU fallback mode..."
  docker compose build
  docker compose up -d
fi

echo "Sky Observer dashboard available at: http://localhost:8080"
