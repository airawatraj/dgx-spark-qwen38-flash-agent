#!/usr/bin/env bash
# docker/stop.sh — Stop and remove Qwen3.8-Flash-Next container
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"

echo "=== Stopping $CONTAINER_NAME ==="

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping container $CONTAINER_NAME ..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "✓ Removed $CONTAINER_NAME."
else
  echo "Container $CONTAINER_NAME is not currently running."
fi

# Clean up legacy container name if still present
if docker ps -a --format '{{.Names}}' | grep -q '^spark-brain-flash$'; then
  docker rm -f spark-brain-flash >/dev/null 2>&1 || true
  echo "✓ Removed legacy spark-brain-flash container."
fi
