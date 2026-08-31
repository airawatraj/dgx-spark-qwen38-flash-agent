#!/usr/bin/env bash
# docker/start-yarn.sh — Launch Cogni-Brain with 500K extended context
set -euo pipefail

export CTX="${CTX:-500000}"
export SEQS="${SEQS:-2}"
export MEM_FRACTION="${MEM_FRACTION:-0.95}"

echo "================================================================="
echo "=== Launching Cogni-Brain in 500K Extended Context Mode ==="
echo "================================================================="
echo "  Context length:  $CTX tokens"
echo "  Concurrent seqs: $SEQS"
echo "  Memory fraction: $MEM_FRACTION"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/start.sh"
