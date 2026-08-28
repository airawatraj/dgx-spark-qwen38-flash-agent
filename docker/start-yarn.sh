#!/usr/bin/env bash
# docker/start-yarn.sh — Launch Qwen3.8-Flash-Next with 500K YaRN ultra-long context
#
# Scales the context window up to 500,000 tokens using YaRN RoPE factor 4 scaling.
set -euo pipefail

export YARN=1
export CTX=500000
export SEQS="${SEQS:-2}"
export GPU_MEM="${GPU_MEM:-0.85}"

echo "=== Launching Qwen3.8-Flash-Next in 500K YaRN Mode ==="
echo "  Context length:  500,000 tokens"
echo "  YaRN scaling:    factor 4.0"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/start.sh"
