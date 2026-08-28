#!/usr/bin/env bash
# setup/install.sh — Preflight dependency and environment checks for DGX Spark Qwen3.8-Flash-Next
set -euo pipefail

echo "================================================================="
echo "=== DGX Spark Qwen3.8-Flash-Next Preflight & Setup Check ==="
echo "================================================================="
echo

# 1. Docker
echo "[1/5] Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed or not in PATH."
  exit 1
fi
docker version --format '  Docker Server Version: {{.Server.Version}}'

# 2. NVIDIA GPU / Container Runtime
echo
echo "[2/5] Checking NVIDIA Container Runtime..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
  echo "  WARNING: nvidia-smi not found in PATH. Verify GPU driver on host."
fi

# 3. uv / uvx
echo
echo "[3/5] Checking uv & uvx package manager..."
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
echo "  uv version:  $(uv --version)"
echo "  uvx version: $(uvx --version)"

# 4. Hugging Face Auth
echo
echo "[4/5] Checking Hugging Face authentication..."
if uvx hf auth whoami >/dev/null 2>&1; then
  echo "  Hugging Face authenticated: OK"
else
  echo "  NOTE: Not authenticated with Hugging Face (or public model access)."
  echo "  If needed, run: uvx hf auth login"
fi

# 5. Docker Image
echo
echo "[5/5] Checking Docker image..."
IMAGE="${IMAGE:-qwen38-flash-dgx}"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  Docker image '$IMAGE' is already built and available."
else
  echo "  Docker image '$IMAGE' not found locally."
  echo "  Building image with PLE mmap patch now..."
  bash docker/build.sh
fi

echo
echo "================================================================="
echo "✓ Preflight check complete."
echo
echo "Next steps:"
echo "  1. Download model weights (~122 GB, run inside tmux):"
echo "     bash setup/download_model.sh"
echo "  2. Start serving container:"
echo "     bash docker/start.sh"
echo "================================================================="
