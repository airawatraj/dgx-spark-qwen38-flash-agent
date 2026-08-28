#!/usr/bin/env bash
# docker/build.sh — Build the patched vLLM image for Qwen3.8-Flash-Next with PLE disk mmap
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-qwen38-flash-dgx}"

echo "=== Building Docker image: $IMAGE_TAG ==="
echo "  Source Dockerfile: Dockerfile"
echo "  Patch module:      src/vllm_ple_mmap.py"
echo

docker build -t "$IMAGE_TAG" .

echo
echo "✓ Docker image built successfully: $IMAGE_TAG"
echo "Next: bash setup/download_model.sh"
echo "      bash docker/start.sh"
