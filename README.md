# Running Cogni-Brain on DGX Spark · Qwen3.8-Flash-Next NVFP4 + PLE Disk MMAP

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Base Model](https://img.shields.io/badge/base%20model-Qwen3.8--Flash--Next--NVFP4-limegreen)
![Speculative](https://img.shields.io/badge/speculative-MTP%20(K%3D2)-purple)
![Runtime](https://img.shields.io/badge/runtime-vLLM%20%2B%20PLE%20MMAP-orange)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Context](https://img.shields.io/badge/context-262K%20%7C%20500K%20YaRN-blue)
![Tool Eval](https://img.shields.io/badge/tool--eval-100%2F100-success)
![Reasoning](https://img.shields.io/badge/reasoning-qwen3-black)
![Quantization](https://img.shields.io/badge/quantization-NVFP4-purple)

This repo documents inference tuning and serving of [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (~176B total: 125B MoE backbone + 51B n-gram PLE table, 6B active) on a single NVIDIA DGX Spark / GB10 (128 GB unified memory).

> ⚠️ **Personal workstation setup. Not for enterprise use. Use at your own risk.**

---

## What is Qwen3.8-Flash-Next?

[Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) is served here as **Cogni-Brain** — a consistent model alias across agent frameworks (Claude Code, Continue, Open WebUI, etc.) regardless of which model is running underneath.

Key characteristics:
- **Zero-Copy PLE Disk MMAP** — Serves the 44 GiB FP8 n-gram PLE table directly from NVMe via `mmap` (`src/vllm_ple_mmap.py`), dropping resident memory from 122 GiB to ~76 GiB and unlocking ~52 GiB of unified pool for the KV cache (~720k–790k tokens).
- **176B Total / 6B Active MoE** — Deep sparse Mixture-of-Experts architecture with extreme parameter efficiency and high throughput.
- **Multi-Token Prediction (MTP, $k=2$)** — Native built-in speculative draft head yielding 25–35+ tok/s decode.
- **QSA Sparse Attention + GDN Linear Attention** — Fast, memory-efficient attention preserving linear $O(N)$ prefill scaling.
- **262K Native Context / 500K YaRN** — Native 262,144 token context window, expandable up to 500,000 tokens via YaRN RoPE scaling (`docker/start-yarn.sh`).
- **Native Tool Calling & Reasoning** — Built-in `qwen3_coder` tool parser and `qwen3` reasoning parser (100/100 tool-eval benchmark score).

---

## Quick Start

> ⚠️ **Note:** Run `download_model.sh` once before first launch (~122 GB total). Run in `tmux` if on SSH.

```bash
# 1. Verify prerequisites (Docker, GPU runtime, uv, HF auth)
bash setup/install.sh

# 2. Download model weights — one-time, ~122 GB (run in tmux on SSH)
bash setup/download_model.sh

# 3. Build patched vLLM image
bash docker/build.sh

# 4. Launch serving container (vLLM PLE MMAP, 262K context, MTP=2)
bash docker/start.sh

# 5. Follow initialization logs (wait for "Application startup complete", ~8 min on first boot)
docker logs -f spark-brain

# 6. Check container status & health
bash docker/status.sh
bash benchmark/smoke_test.sh localhost:8000

# 7. Run benchmarks
uv run benchmark/benchmark_speed.py
uv run benchmark/benchmark_smarts.py
# Optional: full spark-arena-style overnight sweep
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv
```

### 500K Long-Context Launch (YaRN Mode)

To launch with YaRN RoPE factor 4 scaling for up to 500,000 tokens of context:

```bash
bash docker/start-yarn.sh
```

---

## OpenAI-Compatible API Usage

```bash
# Basic completion / reasoning request
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Cogni-Brain",
    "messages": [
      {"role": "user", "content": "Explain how speculative decoding works in sparse MoE models."}
    ],
    "max_tokens": 512
  }'
```

---

## Benchmarks

### Speed Benchmark (custom script)

```bash
# Full run (TPS, TTFT, concurrent sessions, context window)
uv run benchmark/benchmark_speed.py

# Skip slower subtests for a quicker check
uv run benchmark/benchmark_speed.py --skip-context
uv run benchmark/benchmark_speed.py --skip-context --skip-concurrent

# Custom endpoint or model alias
uv run benchmark/benchmark_speed.py --host localhost --port 8000 --model Cogni-Brain
```

> Tests: baseline TPS, TPS vs output length, concurrent sessions (1–8), context window (up to 262K), health & KV stats.

### Smarts Benchmark (tool-eval-bench)

```bash
# Quick smoke test (recommended first run)
uv run benchmark/benchmark_smarts.py

# Throughput sweep
uv run benchmark/benchmark_smarts.py --mode perf

# Deterministic multi-trial evaluation
uv run benchmark/benchmark_smarts.py --mode trials --seed 42 --trials 3
```

> Evaluates: tool selection, parameter precision, multi-step chains, refusal behaviour, error recovery (15 scenarios).

### spark-arena Benchmark (llama-benchy)

```bash
# 1. Standard spark-arena sweep (tests depths 0 to 128K across concurrencies 1, 2, 4, 8)
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv

# 2. Custom endpoint or single depth
uv run benchmark/benchmark_speed_arena.py \
  --base-url http://localhost:8000/v1 \
  --depth 131072 \
  --concurrency 1
```

---

## Benchmark Results Summary

> Benchmarks run on DGX Spark GB10 · August 2026  
> vLLM + PLE MMAP Patch · Native MTP ($k=2$) · tool-eval-bench (15/15 PASS)

### Speed & Context Performance

![Speed benchmark results](assets/benchmark_speed_qwen38_flash.png)

| Test | Metric | Result (MTP $k=2$) | Notes |
|---|---|---|---|
| Single-stream Decode ($c=1$) | Steady-state TPS | **23–26 tok/s** | Invariant across context depths up to 131K |
| Single-stream Prefill ($c=1$) | Throughput | **~1,000 tok/s** | Linear $O(N)$ prefill scaling |
| Multi-stream Decode ($c=8, d=0$) | Peak Aggregate TPS | **92.7 tok/s** | Concurrency scaling on low context |
| Context Depth | Max Tested | **131K / 262K native** | Scalable to 500K with YaRN |
| Tool-Use Evaluation | Pass Rate | **100/100 (15/15 PASS)** | Native `qwen3_coder` tool calling |

### Tool-Use Evaluation (Smarts)

![Smarts benchmark scenarios](assets/benchmark_smarts_qwen38_flash_1.png)
![Smarts benchmark score](assets/benchmark_smarts_qwen38_flash_2.png)

---

## Environment Overrides & Tuning

You can override defaults with environment variables before running `docker/start.sh`:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Exposed host port for the OpenAI-compatible API |
| `CTX` | `262144` | Context length (native 262,144; with `YARN=1` up to 500,000) |
| `YARN` | `0` | Set to `1` for YaRN RoPE context expansion |
| `MTP` | `2` | Number of speculative tokens from model's MTP head (`0` to disable) |
| `SEQS` | `8` | Maximum concurrent sequences |
| `GPU_MEM` | `0.85` | Fraction of 128 GB pool for weights + KV (~76 GB weights, ~52 GB KV) |
| `WORKERS` | `32` | Worker threads for multithreaded mmap row gather |
| `PREWARM` | `0` | Set to `1` to stream table at boot to warm the OS page cache |
| `CONTAINER_NAME` | `spark-brain` | Name of the Docker container |

---

## Repository Structure

```text
dgx-spark-qwen38-flash-agent/
├── README.md                      ← this document
├── HOW_IT_WORKS.md                ← In-depth breakdown of PLE mmap & GB10 fixes
├── Dockerfile                     ← Patched vLLM image serving PLE table via mmap
├── src/
│   ├── vllm_ple_mmap.py           ← Complete PLE disk mmap patch module
│   └── test_ple_mmap_cpu.py       ← CPU unit test for safetensors gather validation
├── docker/
│   ├── build.sh                   ← Docker image builder (qwen38-flash-dgx)
│   ├── start.sh                   ← Production launch script (262K native, MTP=2)
│   ├── start-yarn.sh              ← Ultra-long context launch (500K YaRN)
│   ├── status.sh                  ← Container, health, memory & vLLM metrics inspector
│   └── stop.sh                    ← Safe container teardown
├── setup/
│   ├── install.sh                 ← Dependency and environment preflight checks
│   └── download_model.sh          ← High-speed model downloader (~122 GB)
├── benchmark/
│   ├── benchmark_speed.py         ← TPS, TTFT, concurrency & context benchmark
│   ├── benchmark_smarts.py        ← tool-eval-bench wrapper (15 agentic scenarios)
│   ├── benchmark_speed_arena.py   ← Spark-arena compatible throughput sweep
│   └── smoke_test.sh              ← Health, coherence, prefill & decode verification
└── assets/                        ← Architecture diagrams and benchmark visualizations
```
