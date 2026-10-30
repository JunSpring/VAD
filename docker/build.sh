#!/usr/bin/env bash
# Build the VAD training image. Context = repo root (needed for COPY docker/entrypoint.sh).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-vad:cu118-torch2.0.1}

docker build -f "$REPO/docker/Dockerfile" -t "$IMAGE" "$REPO" "$@"
echo "Built $IMAGE"
