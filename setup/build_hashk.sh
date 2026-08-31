#!/usr/bin/env bash
# setup/build_hashk.sh — Build the 12.8 GB HashK PLE artifact for Qwen3.8-Flash-Next
#
# Generates ple_hashk_R4.pt from the RadixArk/Qwen3.8-Flash-Next-NVFP4 checkpoint.
# Runs inside the official SGLang container via GPU (~6 minutes on DGX Spark).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
R="${HASHK_R:-4}"
ARTIFACT_OUT="$REPO_DIR/ple_hashk_R${R}.pt"

echo "================================================================="
echo "=== Building HashK PLE Artifact for Qwen3.8-Flash-Next (R=$R) ==="
echo "================================================================="
echo "  Artifact target: $ARTIFACT_OUT"
echo "  HF Cache:        $HF_CACHE"
echo "  Docker Image:    $IMAGE"
echo

if [ -f "$ARTIFACT_OUT" ]; then
  echo "✓ Artifact already exists: $ARTIFACT_OUT ($(du -h "$ARTIFACT_OUT" | cut -f1))"
  echo "To rebuild, remove it first: rm -f $ARTIFACT_OUT"
  exit 0
fi

# Ensure output directory and permissions
mkdir -p "$REPO_DIR"

echo "Pulling SGLang image if not present..."
docker pull "$IMAGE" || true

echo
echo "Starting HashK PLE builder in Docker..."
echo "  This streams the 51.2 GB FP8 shards, applies SplitMix64 polynomial rehash,"
echo "  computes unbiased mean-pooling, and fits per-head ridge projections."
echo "  Runtime: ~6 min on DGX Spark."
echo

docker run --rm --gpus all \
  --ipc=host \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$REPO_DIR":/out \
  -e "HF_HOME=/root/.cache/huggingface" \
  -e "HASHK_R=$R" \
  -e "HASHK_OUT=/out/ple_hashk_R${R}.pt" \
  --entrypoint python3 \
  "$IMAGE" \
  /out/tools/build_hashk_ple.py

if [ -f "$ARTIFACT_OUT" ]; then
  echo
  echo "================================================================="
  echo "✓ HashK PLE Artifact built successfully!"
  echo "  Location: $ARTIFACT_OUT"
  echo "  Size:     $(du -h "$ARTIFACT_OUT" | cut -f1)"
  echo "  Next:     bash docker/start.sh"
  echo "================================================================="
else
  echo
  echo "ERROR: Failed to build $ARTIFACT_OUT"
  exit 1
fi
