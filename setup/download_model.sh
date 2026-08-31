#!/usr/bin/env bash
# setup/download_model.sh — Download Qwen3.8-Flash-Next NVFP4 weights to local HF cache
#
# Checkpoint size: ~135 GB (resumable)
# Needs ~150 GB free on the filesystem hosting $HOME/.cache/huggingface.
# Tip: Run inside tmux if on a remote SSH session.
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"

echo "================================================================="
echo "=== Downloading Qwen3.8-Flash-Next Checkpoint ==="
echo "================================================================="
echo "  Model ID:     $MODEL"
echo "  Target dir:   $HF_CACHE"
echo "  Expected:     ~135 GB (safetensors + FP8 PLE table shards)"
echo

mkdir -p "$HF_CACHE"

# Check if model snapshot already exists
SNAP_HOST="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | head -1 || true)"
if [[ -n "$SNAP_HOST" && -d "$SNAP_HOST" ]]; then
  FILE_COUNT=$(find "$SNAP_HOST" -type f -name "*.safetensors" 2>/dev/null | wc -l)
  if [[ "$FILE_COUNT" -gt 10 ]]; then
    echo "✓ Found existing snapshot with $FILE_COUNT safetensors files at:"
    echo "  $SNAP_HOST"
    echo "  Weights appear to be downloaded."
    echo
    echo "Next: Build HashK artifact (if not done yet):"
    echo "  bash setup/build_hashk.sh"
    exit 0
  fi
fi

echo "Starting download using Hugging Face CLI container or uvx..."
echo

if command -v huggingface-cli >/dev/null 2>&1; then
  echo "Using host huggingface-cli..."
  huggingface-cli download "$MODEL"
elif command -v uvx >/dev/null 2>&1; then
  echo "Using uvx hf..."
  uvx hf download "$MODEL"
else
  echo "Using Docker container download..."
  docker run --rm --name qwen38-dl \
    -e HF_HOME=/hf \
    -v "$HF_CACHE:/hf" \
    --entrypoint bash \
    "$IMAGE" \
    -c "huggingface-cli download '$MODEL'"
fi

echo
echo "================================================================="
echo "✓ Checkpoint download complete and verified."
echo "  Model cache location: $REPO_DIR"
echo
echo "Next step: Build HashK compressed PLE artifact (~6 min):"
echo "  bash setup/build_hashk.sh"
echo "================================================================="
