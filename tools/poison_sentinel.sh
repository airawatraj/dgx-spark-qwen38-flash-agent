#!/usr/bin/env bash
# Scheduled poisoning probe: run the recipe against production, log verdict
# with timestamp + memory state. If POISONED, restart the server so the box
# heals itself. Tagged POISONPROBE in crontab for easy removal.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
LOG="${LOG:-$REPO_DIR/poison_sentinel.log}"
TS=$(date '+%Y-%m-%d %H:%M')
OFF=$(( ($(date +%s) % 86400) * 10 + 700000 ))

curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
  echo "$TS SKIP server-not-healthy" >> "$LOG"; exit 0; }

MEM=$(free -g 2>/dev/null | awk '/^Mem:/{print "free="$4"G avail="$7"G"}' || echo "free=unknown")
SWAP=$(free -g 2>/dev/null | awk '/^Swap:/{print "swapused="$3"G"}' || echo "swap=unknown")
OUT=$(python3 "$REPO_DIR/tools/bisect_probe.py" "$OFF" 2>&1)
VERDICT=$( (echo "$OUT" | grep -oE 'VERDICT: [A-Z]+ \([0-9]+ corrupt\)' | tail -1) || echo "VERDICT: UNKNOWN (0 corrupt)" )
echo "$TS $MEM $SWAP $VERDICT" >> "$LOG"

if echo "$VERDICT" | grep -q POISONED; then
  echo "$TS restarting server after poisoned verdict" >> "$LOG"
  bash "$REPO_DIR/docker/start.sh" >/dev/null 2>&1
fi
