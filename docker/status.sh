#!/usr/bin/env bash
# docker/status.sh — Inspect Qwen3.8-Flash-Next container, memory & vLLM metrics
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
PORT="${PORT:-8000}"

echo "=== Qwen3.8-Flash-Next Status ==="
echo

# ── Container Status ──────────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  STARTED_AT=$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')
  STATUS=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')
  echo "  Container:       RUNNING ($STATUS, started $STARTED_AT)"
else
  echo "  Container:       NOT RUNNING"
fi

# ── API Health Check ──────────────────────────────────────────────────────────
if curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "  API:             HEALTHY (http://localhost:$PORT)"
else
  echo "  API:             NOT REACHABLE on port $PORT"
fi

echo
echo "=== System Unified Memory ==="
if command -v free >/dev/null 2>&1; then
  free -h
else
  echo "  'free' utility is not available on this platform"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "=== NVIDIA GPU / Unified Pool ==="
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
fi

echo
echo "=== vLLM Engine Metrics ==="
METRICS="$(curl -sf -m 3 "http://localhost:$PORT/metrics" 2>/dev/null || true)"
if [[ -n "$METRICS" ]]; then
  KV_CACHE="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /gpu_cache_usage_perc/ {printf "%.1f%%", $2 * 100; exit}')"
  RUNNING="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /(^|:)num_requests_running([[:space:]]|$)/ {print $2; exit}')"
  WAITING="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /(^|:)num_requests_waiting([[:space:]]|$)/ {print $2; exit}')"
  TOKENS_GEN="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /generation_tokens_total/ {print $2; exit}')"
  echo "  KV cache used:     ${KV_CACHE:-unknown}"
  echo "  Requests running:  ${RUNNING:-0}"
  echo "  Requests waiting:  ${WAITING:-0}"
  if [[ -n "${TOKENS_GEN:-}" ]]; then
    echo "  Total gen tokens:  ${TOKENS_GEN}"
  fi
else
  echo "  Metrics endpoint not reachable"
fi

echo
echo "=== Recent Container Logs (last 10 lines) ==="
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker logs "$CONTAINER_NAME" --tail 10 2>&1 | sed 's/^/  /'
else
  echo "  No container found"
fi
