#!/usr/bin/env bash
# Flash-Next / Cogni-Brain watchdog: revives a dead server AND detects the wedged state.
# Rate-limited so a crash-looping boot cannot spiral (max 4 actions/day).
#
# Wedge mode: after aborted client requests the engine can degrade silently --
# zombie requests grind at ~5 tok/s, speculative decoding rejects every draft
# (accept len: 1.00), state slots leak -- while /health still returns 200.
# Signature: every recent "Decode batch" log line shows accept len: 1.00 with
# running requests. Healthy traffic shows ~3+.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
LOG="${LOG:-$REPO_DIR/watchdog.log}"

limit_ok() {
  recent=$(grep -c "$(date '+%Y-%m-%d')" "$LOG" 2>/dev/null || echo 0)
  [ "$recent" -lt 4 ]
}

st=$(docker ps --filter "name=^${CONTAINER_NAME}$" --format "{{.Status}}")

if ! curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  [ -n "$st" ] && exit 0   # up but unhealthy = likely mid-boot; leave it alone
  limit_ok || exit 0
  docker start "$CONTAINER_NAME" >/dev/null 2>&1 && \
    echo "$(date '+%Y-%m-%d %H:%M:%S') watchdog: restarted $CONTAINER_NAME (down)" >> "$LOG"
  exit 0
fi

# Healthy endpoint -- check for the silent wedge.
lines=$(docker logs "$CONTAINER_NAME" --since 3m 2>&1 | grep -a "Decode batch" | tail -8)
[ -z "$lines" ] && exit 0
total=$(echo "$lines" | wc -l)
wedged=$( (echo "$lines" | grep -c "accept len: 1.00") || true )
running=$( (echo "$lines" | grep -cE "#running-req: [1-9]") || true )
if [ "$total" -ge 6 ] && [ "$wedged" -eq "$total" ] && [ "$running" -eq "$total" ]; then
  limit_ok || exit 0
  echo "$(date '+%Y-%m-%d %H:%M:%S') watchdog: WEDGE detected (accept len 1.00 x$total) - restarting" >> "$LOG"
  docker restart "$CONTAINER_NAME" >/dev/null 2>&1
fi
