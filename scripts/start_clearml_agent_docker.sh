#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="mynotes-gpu"
QUEUE="${1:-ofek}"

echo "Building $IMAGE_NAME from $REPO_ROOT/docker/Dockerfile.gpu..."
docker build -t "$IMAGE_NAME" -f "$REPO_ROOT/docker/Dockerfile.gpu" "$REPO_ROOT"

echo "Starting clearml-agent in docker mode on queue: $QUEUE"
echo "Image: $IMAGE_NAME"
echo "Press Ctrl+C to stop."

clearml-agent daemon \
    --queue "$QUEUE" \
    --docker "$IMAGE_NAME" \
    --docker-args "--gpus all --shm-size=8g" \
    --foreground
