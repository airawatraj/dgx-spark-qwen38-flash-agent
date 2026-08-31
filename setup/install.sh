#!/usr/bin/env bash
# setup/install.sh — Preflight dependency and environment checks for DGX Spark Qwen3.8-Flash-Next (Cogni-Brain)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
HASHK_ARTIFACT="$REPO_DIR/ple_hashk_R4.pt"

echo "================================================================="
echo "=== DGX Spark Qwen3.8-Flash-Next (Cogni-Brain) Preflight ==="
echo "================================================================="
echo

# 1. Docker
echo "[1/6] Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed or not in PATH."
  exit 1
fi
docker version --format '  Docker Server Version: {{.Server.Version}}'

# 2. NVIDIA GPU / Container Runtime
echo
echo "[2/6] Checking NVIDIA Container Runtime..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
  echo "  WARNING: nvidia-smi not found in PATH. Verify GPU driver on host."
fi

# 3. uv / uvx
echo
echo "[3/6] Checking uv & uvx package manager..."
if ! command -v uv >/dev/null 2>&1; then
  echo "  NOTE: uv is not installed. Installing via astral.sh..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "  uv version:  $(uv --version 2>/dev/null || echo 'available in ~/.cargo/bin')"

# 4. Hugging Face Auth
echo
echo "[4/6] Checking Hugging Face authentication..."
if uvx hf auth whoami >/dev/null 2>&1; then
  echo "  Hugging Face authenticated: OK"
else
  echo "  NOTE: Not authenticated with Hugging Face (or public model access)."
  echo "  If needed, run: uvx hf auth login"
fi

# 5. SGLang Docker Image
echo
echo "[5/6] Checking SGLang Docker image ($IMAGE)..."
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  Docker image '$IMAGE' is available locally."
else
  echo "  Pulling Docker image '$IMAGE'..."
  docker pull "$IMAGE" || echo "  Will be pulled automatically on first launch."
fi

# 6. HashK PLE Artifact
echo
echo "[6/6] Checking HashK PLE artifact (ple_hashk_R4.pt)..."
if [ -f "$HASHK_ARTIFACT" ]; then
  echo "  HashK artifact found: $HASHK_ARTIFACT ($(du -h "$HASHK_ARTIFACT" | cut -f1))"
else
  echo "  HashK artifact not found yet. Build it with:"
  echo "    bash setup/build_hashk.sh"
fi

echo
echo "================================================================="
echo "✓ Preflight check complete."
echo
echo "Next steps:"
echo "  1. Download model weights (~135 GB checkpoint, run inside tmux):"
echo "     bash setup/download_model.sh"
echo "  2. Build HashK compressed PLE artifact (~6 min GPU step):"
echo "     bash setup/build_hashk.sh"
echo "  3. Start serving container (spark-brain):"
echo "     bash docker/start.sh"
echo "================================================================="
