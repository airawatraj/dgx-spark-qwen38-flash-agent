#!/usr/bin/env bash
# setup/download_model.sh — Download Qwen3.8-Flash-Next NVFP4 weights to local HF cache
#
# Checkpoint size: ~122 GiB (resumable)
# Needs ~130 GB free on the filesystem hosting $HOME/.cache/huggingface.
# Tip: Run inside tmux if on a remote SSH session.
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"

echo "================================================================="
echo "=== Downloading Qwen3.8-Flash-Next Checkpoint ==="
echo "================================================================="
echo "  Model ID:     $MODEL"
echo "  Target dir:   $HF_CACHE"
echo "  Expected:     ~122 GiB (safetensors + FP8 PLE table shards)"
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
    echo "To re-verify or resume download, remove or pass extra arguments."
    echo "Next: bash docker/start.sh"
    exit 0
  fi
fi

echo "Starting download using Hugging Face CLI container..."
echo "Running multi-worker HTTPS download (HF_HUB_DISABLE_XET=1 for maximum link saturation)..."
echo

# Run download in ephemeral docker container using HF CLI
docker run --rm --name qwen38-dl \
  -e HF_HOME=/hf \
  -e HF_HUB_DISABLE_XET=1 \
  -v "$HF_CACHE:/hf" \
  --entrypoint bash \
  "$IMAGE" \
  -c "hf download '$MODEL' --max-workers 8"

echo
echo "================================================================="
echo "✓ Checkpoint download complete and verified."
echo "  Model cache location: $REPO_DIR"
echo
echo "Next step: Start the engine"
echo "  bash docker/start.sh"
echo "================================================================="
