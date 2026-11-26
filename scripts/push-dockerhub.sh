#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dockerhub-username> [image-name] [tag]" >&2
  exit 1
fi

DOCKERHUB_USER="$1"
IMAGE_NAME="${2:-sky-observer}"
TAG="${3:-latest}"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
TS_TAG="${DOCKERHUB_USER}/${IMAGE_NAME}:$(date +%Y%m%d-%H%M)"

echo "=== 1/3 Building unified image (Frontend + Backend CPU): ${FULL_IMAGE} ==="
docker build -t "${FULL_IMAGE}" -t "${TS_TAG}" -f Dockerfile .

echo "=== 2/3 Pushing :${TAG} to Docker Hub ==="
docker push "${FULL_IMAGE}"

echo "=== 3/3 Pushing timestamped tag ${TS_TAG} ==="
docker push "${TS_TAG}"

echo "Done: ${FULL_IMAGE}"
