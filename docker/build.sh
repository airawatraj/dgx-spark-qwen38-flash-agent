#!/usr/bin/env bash
# docker/build.sh — Prepare Docker image and HashK artifact for Cogni-Brain
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"

echo "================================================================="
echo "=== Cogni-Brain Container & Artifact Preparation ==="
echo "================================================================="
echo "  SGLang Image:   $IMAGE"
echo "  Runtime:        Bind-mount patched forward runners & kernels"
echo

echo "[1/2] Pulling official SGLang Qwen3.8-Flash-Next image..."
docker pull "$IMAGE"

echo
echo "[2/2] Checking HashK PLE artifact..."
if [ -f "ple_hashk_R4.pt" ]; then
  echo "✓ HashK artifact ple_hashk_R4.pt exists ($(du -h ple_hashk_R4.pt | cut -f1))."
else
  echo "HashK artifact ple_hashk_R4.pt not found."
  echo "Building now via setup/build_hashk.sh..."
  bash setup/build_hashk.sh
fi

echo
echo "================================================================="
echo "✓ Ready to serve Cogni-Brain!"
echo "Next: bash docker/start.sh"
echo "================================================================="
