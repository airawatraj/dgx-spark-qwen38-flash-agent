#!/usr/bin/env bash
# docker/start.sh — Serve Qwen3.8-Flash-Next on a single NVIDIA DGX Spark / GB10
#
# Uses vLLM with the PLE table mmap patch from disk, native MTP speculative
# decoding (k=2), Piecewise CUDA graph capture, and native agentic tool calling.
#
# Usage:
#   bash docker/start.sh                      # default 262k native context, MTP=2
#   PORT=8000 bash docker/start.sh            # custom port
#   YARN=1 CTX=500000 bash docker/start.sh    # 500k context via YaRN
#
# Tunables (env vars):
#   PORT=8000         host port for OpenAI API (default 8000)
#   CTX=262144        max context length (native: 262144; with YARN=1 up to ~500000)
#   YARN=0            1 = YaRN rope scaling (factor 4) for context > 262144
#   SEQS=8            max concurrent sequences
#   GPU_MEM=0.85      fraction of 128 GB pool for weights + KV (~76 GB weights, ~20+ GB KV)
#   MTP=2             speculative tokens from model's MTP head (0 = off)
#   KV_DTYPE=auto     auto (=bf16, required by QSA sparse attention)
#   PREWARM=0         1 = stream the table once at boot to warm page cache
#   WORKERS=32        worker threads for mmap row gather
#   EXTRA=            extra vLLM flags passed verbatim
#   CONTAINER_NAME=   container name (default: spark-brain)
#   IMAGE=            Docker image (default: qwen38-flash-dgx)
#   MODEL=            Hugging Face model ID (default: RadixArk/Qwen3.8-Flash-Next-NVFP4)
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-2}"
KV_DTYPE="${KV_DTYPE:-auto}"
PREWARM="${PREWARM:-0}"
WORKERS="${WORKERS:-32}"
EXTRA="${EXTRA:-}"

# ── Preflight Checks ──────────────────────────────────────────────────────────
echo "=== DGX Spark Qwen3.8-Flash-Next Preflight ==="
echo "  Container:              $CONTAINER_NAME"
echo "  Image:                  $IMAGE"
echo "  Model ID:               $MODEL"
echo "  Served name:            $SERVED_MODEL_NAME"
echo "  Port:                   $PORT"
echo "  Context length:         $CTX tokens (YaRN=$YARN)"
echo "  Concurrent seqs:        $SEQS"
echo "  MTP spec tokens:        $MTP"
echo "  GPU mem util:           $GPU_MEM"
echo "  KV cache dtype:         $KV_DTYPE (QSA native BF16)"
echo "  HF cache:               $HF_CACHE"
echo "  PLE mmap workers:       $WORKERS"
echo

# Verify Docker image exists
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: Docker image '$IMAGE' not found."
  echo "Build it first with: bash docker/build.sh"
  exit 1
fi

# Resolve local snapshot directory
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP_HOST="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | head -1 || true)"
if [ -z "$SNAP_HOST" ]; then
  echo "ERROR: Checkpoint not found under $REPO_DIR"
  echo "Please download the weights first:"
  echo "  bash setup/download_model.sh"
  exit 1
fi
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$(basename "$SNAP_HOST")"
echo "  Found local snapshot:   $SNAP_HOST"
echo

# ── Piecewise CUDA Graph & Splitting Ops Configuration ─────────────────────────
# The PLE gather is a CPU op + pageable host->device copy: it MUST run outside
# CUDA graphs. We declare it a splitting op and use PIECEWISE capture (never FULL*).
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

# ── YaRN Context Scaling Configuration ─────────────────────────────────────────
OVR_ARGS=()
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
ALLOW_LONG=0
if [ "$YARN" != "0" ]; then
  OVR_ARGS=(--hf-overrides "$YARN_OVR")
  ALLOW_LONG=1
fi

# ── Multi-Token Prediction (MTP) Speculative Configuration ─────────────────────
SPEC=()
if [ "$MTP" != "0" ]; then
  if [ "$YARN" != "0" ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"max_model_len\":${CTX}}")
  else
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
  fi
fi

# ── Launch Container ──────────────────────────────────────────────────────────
echo "Cleaning up any existing container named $CONTAINER_NAME ..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting $CONTAINER_NAME on port $PORT ..."
# shellcheck disable=SC2086
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --shm-size 16g \
  -p "${PORT}:8000" \
  -v "$HF_CACHE:/hf" \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 \
  -e VLLM_PLE_MMAP_WORKERS="$WORKERS" \
  -e VLLM_PLE_MMAP_PREWARM="$PREWARM" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG" \
  "$IMAGE" \
  "$SNAP_IN" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --load-format safetensors \
    --max-model-len "$CTX" \
    --max-num-seqs "$SEQS" \
    --gpu-memory-utilization "$GPU_MEM" \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    $CC \
    --no-enable-flashinfer-autotune \
    --kv-cache-dtype "$KV_DTYPE" \
    "${OVR_ARGS[@]}" \
    $EXTRA \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    "${SPEC[@]}"

echo
echo "✓ Container '$CONTAINER_NAME' is starting on port $PORT."
echo "  Model name: '$SERVED_MODEL_NAME' (context: $CTX, yarn=$YARN, mtp=$MTP, seqs=$SEQS)"
echo "  First boot loads ~76 GiB of non-table weights into unified memory (~8 min)."
echo
echo "Follow initialization logs with:"
echo "  docker logs -f $CONTAINER_NAME"
echo
echo "Check health status with:"
echo "  bash docker/status.sh"
echo "  bash benchmark/smoke_test.sh localhost:$PORT"
