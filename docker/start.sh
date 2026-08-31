#!/usr/bin/env bash
# docker/start.sh — Serve Qwen3.8-Flash-Next as "Cogni-Brain" in container "spark-brain" on DGX Spark
#
# Optimized for single NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory):
#   - HashK GPU-Resident PLE Table (12.8 GB, ple_hashk_R4.pt)
#   - SGLang NEXTN Speculative Decoding (3 steps, 4 draft tokens)
#   - Grouped-BMM QSA Gather + FlashInfer Cutlass FP4 GEMM
#   - RadixAttention Prefix Caching (~139k tok/s warm prefill)
#   - FP8 KV Cache (700k+ tokens pool) + Mamba State Checkpoint Rollback
#   - SM121 Grace-Blackwell Kernel Bugfix Patches
#
# Usage:
#   bash docker/start.sh                        # Default Cogni-Brain on port 8000, 262k context
#   PORT=8000 bash docker/start.sh              # Custom port
#   THINKING=medium bash docker/start.sh        # Default reasoning: medium | low | xhigh | off
#   PLE_MODE=packed bash docker/start.sh        # Lossless-ish 28.8 GB packed table without MTP
#
# Tunables (env vars):
#   CONTAINER_NAME     Docker container name                (default: spark-brain)
#   SERVED_MODEL_NAME  Served model name in OpenAI API      (default: Cogni-Brain)
#   PORT               Host port for OpenAI-compatible API  (default: 8000)
#   CTX                Context length                       (default: 262144)
#   MEM_FRACTION       Fraction of static GPU memory        (default: 0.95)
#   THINKING           Reasoning effort: medium|low|xhigh|off (default: medium)
#   PLE_MODE           hashk (fastest, spec-decode) | packed (default: hashk)
#   SEQS               Max concurrent running requests      (default: 8)
#   HF_CACHE           Path to Hugging Face cache           (default: ~/.cache/huggingface)
#   IMAGE              Docker image                         (default: lmsysorg/sglang:qwen38flashnext)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.95}"
THINKING="${THINKING:-medium}"
PLE_MODE="${PLE_MODE:-hashk}"
SEQS="${SEQS:-8}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
MODEL_PATH="${MODEL_PATH:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HASHK_ARTIFACT="$REPO_DIR/ple_hashk_R4.pt"

# ── Reasoning Template Kwargs ────────────────────────────────────────────────
if [ "$THINKING" = "off" ]; then
  KWARGS='{"enable_thinking": false}'
else
  KWARGS="{\"enable_thinking\": true, \"reasoning_effort\": \"$THINKING\"}"
fi

# ── PLE Mode & Speculative Decoding Configuration ───────────────────────────
PLE_ENV=()
SPEC_FLAGS=()
if [ "$PLE_MODE" = "hashk" ]; then
  if [ ! -f "$HASHK_ARTIFACT" ]; then
    echo "================================================================="
    echo "ERROR: HashK artifact not found at $HASHK_ARTIFACT"
    echo "Please build it once before running (~6 min on DGX Spark):"
    echo "  bash setup/build_hashk.sh"
    echo "================================================================="
    exit 1
  fi
  PLE_ENV=(-e "SGLANG_QWEN4_PLE_HASHK=/patches/ple_hashk_R4.pt")
  SPEC_FLAGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
elif [ "$PLE_MODE" = "packed" ]; then
  echo "NOTE: Running in PLE_MODE=packed (28.8 GB packed table on GPU, NEXTN spec-decode disabled)."
  PLE_ENV=(-e "SGLANG_QWEN4_PLE_NVFP4=1")
else
  echo "ERROR: Unknown PLE_MODE='$PLE_MODE'. Must be 'hashk' or 'packed'."
  exit 1
fi

# ── Preflight Summary ────────────────────────────────────────────────────────
echo "================================================================="
echo "=== DGX Spark Qwen3.8-Flash-Next Serving: $SERVED_MODEL_NAME ==="
echo "================================================================="
echo "  Container:          $CONTAINER_NAME"
echo "  Served Model Name:  $SERVED_MODEL_NAME"
echo "  Port:               $PORT"
echo "  Model Path:         $MODEL_PATH"
echo "  PLE Mode:           $PLE_MODE"
echo "  Context Window:     $CTX tokens"
echo "  Memory Fraction:    $MEM_FRACTION"
echo "  Reasoning Default:  $THINKING"
echo "  Max Concurrency:    $SEQS streams (Mamba slots: 24)"
echo "  KV Cache:           FP8 E4M3 (page-size 64)"
echo "  Docker Image:       $IMAGE"
echo "  HF Cache:           $HF_CACHE"
echo

# Verify required patch files exist
for p in qwen4_exp_nvfp4.py flash_fwd.py qwen_sparse_attn_backend.py sparse_attn.py; do
  if [ ! -f "$REPO_DIR/patches/$p" ]; then
    echo "ERROR: Required patch file 'patches/$p' is missing."
    exit 1
  fi
done

# ── Container Launch ─────────────────────────────────────────────────────────
echo "Cleaning up any existing container named '$CONTAINER_NAME'..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting '$CONTAINER_NAME' in background..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --shm-size 32g \
  -p "${PORT}:${PORT}" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$REPO_DIR":/patches \
  -v "$REPO_DIR/patches/qwen4_exp_nvfp4.py":/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py:ro \
  -v "$REPO_DIR/patches/flash_fwd.py":/usr/local/lib/python3.12/dist-packages/flash_attn/cute/flash_fwd.py:ro \
  -v "$REPO_DIR/patches/qwen_sparse_attn_backend.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro \
  -v "$REPO_DIR/patches/sparse_attn.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sparse_attn.py:ro \
  "${PLE_ENV[@]}" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --language-only \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --mamba-track-interval 64 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 32768 \
    --max-running-requests "$SEQS" \
    --max-mamba-cache-size 24 \
    --mamba-ssm-dtype bfloat16 \
    --context-length "$CTX" \
    --mem-fraction-static "$MEM_FRACTION" \
    --default-chat-template-kwargs "$KWARGS" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --strip-thinking-cache \
    "${SPEC_FLAGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"

echo
echo "================================================================="
echo "✓ Container '$CONTAINER_NAME' is booting on port $PORT."
echo "  Served Model: '$SERVED_MODEL_NAME'"
echo "  Boot time:    ~8-9 min warm (~20 min on first run with weight download)"
echo
echo "Commands:"
echo "  Logs:    docker logs -f $CONTAINER_NAME"
echo "  Status:  bash docker/status.sh"
echo "  Health:  curl http://localhost:$PORT/health"
echo "================================================================="
