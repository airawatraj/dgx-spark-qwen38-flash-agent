#!/usr/bin/env bash
# benchmark/smoke_test.sh — Quick verification of server health, coherence, prefill & decode
#
# Usage:
#   bash benchmark/smoke_test.sh
#   bash benchmark/smoke_test.sh localhost:8000
set -euo pipefail

EP="${1:-localhost:8000}"
BASE="http://$EP"
MODEL="${MODEL:-qwen3.8-flash-next}"

echo "================================================================="
echo "=== DGX Spark Qwen3.8-Flash-Next Smoke Test ($EP) ==="
echo "================================================================="
echo

echo "[1/4] Health Check..."
if curl -sf -m 5 "$BASE/health" >/dev/null; then
  echo "  ✓ Server is healthy"
else
  echo "  ✗ Server is not ready at $BASE/health"
  exit 1
fi

echo
echo "[2/4] Coherence Test..."
COHERENCE_RESP=$(curl -s -m 120 "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0}")
echo "$COHERENCE_RESP" | python3 -c 'import json,sys; r=json.load(sys.stdin); print("  Output:", repr(r["choices"][0]["text"].strip()))'

echo
echo "[3/4] Prefill Test (TTFT on ~8,000 token prompt)..."
python3 - "$BASE" "$MODEL" <<'PY'
import json, sys, time, urllib.request

base = sys.argv[1]
model = sys.argv[2]
prompt = "word " * 8000
t0 = time.time()
payload = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}).encode()
req = urllib.request.Request(f"{base}/v1/completions", data=payload, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.load(resp)
        dt = time.time() - t0
        u = res.get("usage", {})
        prompt_tokens = u.get("prompt_tokens", 8000)
        rate = prompt_tokens / dt if dt > 0 else 0
        print(f"  ✓ {prompt_tokens} tokens prefilled in {dt:.2f}s => {rate:.0f} tok/s prefill speed")
except Exception as e:
    print(f"  ✗ Prefill test failed: {e}")
PY

echo
echo "[4/4] Single-Stream Decode Test (256 tokens)..."
python3 - "$BASE" "$MODEL" <<'PY'
import json, sys, time, urllib.request

base = sys.argv[1]
model = sys.argv[2]
t0 = time.time()
payload = json.dumps({"model": model, "prompt": "Hello", "max_tokens": 256, "temperature": 0, "ignore_eos": True}).encode()
req = urllib.request.Request(f"{base}/v1/completions", data=payload, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.load(resp)
        dt = time.time() - t0
        u = res.get("usage", {})
        completion_tokens = u.get("completion_tokens", 256)
        rate = completion_tokens / dt if dt > 0 else 0
        print(f"  ✓ {completion_tokens} tokens decoded in {dt:.2f}s => {rate:.1f} tok/s decode speed")
except Exception as e:
    print(f"  ✗ Decode test failed: {e}")
PY

echo
echo "================================================================="
echo "✓ Smoke test complete."
echo "================================================================="
