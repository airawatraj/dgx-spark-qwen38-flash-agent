#!/usr/bin/env bash
# docker/status.sh — Inspect Qwen3.8-Flash-Next (spark-brain) container, memory & SGLang metrics
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
PORT="${PORT:-8000}"

echo "================================================================="
echo "=== Cogni-Brain (Container: $CONTAINER_NAME) Status ==="
echo "================================================================="
echo

# ── Container Status ──────────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  STARTED_AT=$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')
  STATUS=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')
  echo "  Container:       RUNNING ($STATUS, started $STARTED_AT)"
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  STATUS=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')
  echo "  Container:       STOPPED ($STATUS)"
else
  echo "  Container:       NOT FOUND"
fi

# ── API Health Check ──────────────────────────────────────────────────────────
if curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "  API Health:      HEALTHY (http://localhost:$PORT)"
else
  echo "  API Health:      NOT READY / NOT REACHABLE on port $PORT"
fi

# ── Models Endpoint Check ─────────────────────────────────────────────────────
MODELS_JSON="$(curl -sf -m 3 "http://localhost:$PORT/v1/models" 2>/dev/null || true)"
if [[ -n "$MODELS_JSON" ]]; then
  SERVED_MODELS="$(printf '%s\n' "$MODELS_JSON" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')"
  echo "  Served Models:   ${SERVED_MODELS:-Cogni-Brain}"
fi

echo
echo "=== System Unified Memory ==="
if command -v free >/dev/null 2>&1; then
  free -h
else
  echo "  'free' utility not available on this platform"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "=== NVIDIA GPU / Unified Pool ==="
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
fi

echo
echo "=== SGLang Engine Metrics ==="
METRICS="$(curl -sf -m 3 "http://localhost:$PORT/metrics" 2>/dev/null || true)"
if [[ -n "$METRICS" ]]; then
  RUNNING="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /sglang:num_running_reqs/ {print $2; exit}')"
  WAITING="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /sglang:num_waiting_reqs/ {print $2; exit}')"
  MEM_USAGE="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /sglang:gpu_memory_usage/ {print $2; exit}')"
  CACHE_HIT="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /sglang:cache_hit_rate/ {printf "%.1f%%", $2 * 100; exit}')"
  
  echo "  Requests running:  ${RUNNING:-0}"
  echo "  Requests waiting:  ${WAITING:-0}"
  if [[ -n "${MEM_USAGE:-}" ]]; then
    echo "  GPU mem usage:     ${MEM_USAGE}"
  fi
  if [[ -n "${CACHE_HIT:-}" ]]; then
    echo "  Radix cache hit:   ${CACHE_HIT}"
  fi
else
  echo "  Metrics endpoint not yet active or reachable"
fi

echo
echo "=== Recent Container Logs (last 10 lines) ==="
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker logs "$CONTAINER_NAME" --tail 10 2>&1 | sed 's/^/  /'
else
  echo "  No container found"
fi
echo "================================================================="
