# Running Cogni-Brain on DGX Spark · Qwen3.8-Flash-Next NVFP4 + HashK GPU PLE & SGLang NEXTN

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Base Model](https://img.shields.io/badge/base%20model-Qwen3.8--Flash--Next--NVFP4-limegreen)
![Speculative](https://img.shields.io/badge/speculative-NEXTN%20(3--step)-purple)
![Runtime](https://img.shields.io/badge/runtime-SGLang%20%2B%20HashK%20GPU--PLE-orange)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Context](https://img.shields.io/badge/context-262K%20Native-blue)
[![Spark Arena](https://img.shields.io/badge/spark--arena-verified-darkgreen)](https://spark-arena.com/benchmark/a5682a93-73d1-4a65-a486-e71cbe4ba950)
![Tool Eval](https://img.shields.io/badge/tool--eval-86%2F100%20(151%2F176)-success)
![Reasoning](https://img.shields.io/badge/reasoning-qwen3-black)
![Quantization](https://img.shields.io/badge/quantization-NVFP4-purple)

This repository documents running [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (~176B total params: 125B MoE backbone + 51B n-gram PLE table, 6B active) on a **single NVIDIA DGX Spark / GB10** (128 GB unified memory), served as **Cogni-Brain** in container **`spark-brain`**.

By compressing the 51.2 GB n-gram PLE table 4× into a **12.8 GB GPU-resident HashK artifact** (`ple_hashk_R4.pt`), the entire model fits directly in GPU VRAM (~97 GB resident), freeing memory for the **8 GB MTP draft head** (NEXTN speculative decoding), an **FP8 KV cache** (~700k+ tokens pool), and **RadixAttention** (up to ~139,000 tok/s warm prefill).

> ⚠️ **Personal workstation setup. Not for enterprise use. Use at your own risk.**

---

## Performance Overview

| Dimension | Specification & Results |
|---|---|
| **Served Model Name** | `Cogni-Brain` |
| **Docker Container** | `spark-brain` |
| **Single-Stream Decode** | **~36 tok/s (code)** / **~21–27 tok/s (free-form)** |
| **Multi-Stream Concurrency** | **~57 to 157 tok/s aggregate** across 4–8 streams |
| **Cold Prefill** | **~2,000–2,500 tok/s** |
| **Warm Prefill (Radix Cache)** | **up to ~139,000 tok/s** (56× acceleration) |
| **Context Window** | **262,144 tokens** native context (exact needle recall at 222k tokens) |
| **Coding & Agentic Quality** | **12/12 executed-code pass**, **86/100 tool-eval-bench quality** (151/176 points) |

---

## Quick Start

### 1. Verify Environment & Dependencies
```bash
bash setup/install.sh
```

### 2. Download Checkpoint (~135 GB, one-time)
```bash
# Run inside tmux on remote SSH:
bash setup/download_model.sh
```

### 3. Build HashK Compressed PLE Table (~6 min on GPU, one-time)
```bash
bash setup/build_hashk.sh
# Creates ple_hashk_R4.pt (12.8 GB) in repo root
```

### 4. Launch Container (`spark-brain` serving `Cogni-Brain`)
```bash
# Default launch on port 8000 (HashK + NEXTN spec decode + FP8 KV cache)
bash docker/start.sh

# Or with custom port / reasoning level:
PORT=8000 THINKING=medium bash docker/start.sh
```

### 5. Check Health & Logs
```bash
# Follow boot logs (~8-9 min warm boot):
docker logs -f spark-brain

# Inspect status & memory:
bash docker/status.sh

# Run smoke test:
bash benchmark/smoke_test.sh localhost:8000
```

### 6. Run Benchmarks
```bash
# Speed, TTFT & Concurrency suite:
uv run benchmark/benchmark_speed.py

# Multi-stream & depth speed probe:
python3 tools/speed_probe.py

# Agentic tool calling benchmark:
uv run benchmark/benchmark_smarts.py
```

---

## Visual Benchmark Evidence (Release v1.0.0 Testing)

The visual benchmark artifacts from our baseline **v1.0.0 release testing** on DGX Spark:

### 1. Spark Arena Verified Sweep
[![Spark Arena Benchmark Results](assets/benchmark_spark-arena_qwen38_flash.png)](https://spark-arena.com/benchmark/a5682a93-73d1-4a65-a486-e71cbe4ba950)
> 🔗 [spark-arena.com/benchmark/a5682a93-73d1-4a65-a486-e71cbe4ba950](https://spark-arena.com/benchmark/a5682a93-73d1-4a65-a486-e71cbe4ba950)

### 2. Speed, TTFT & Context Scaling
![Speed benchmark results](assets/benchmark_speed_qwen38_flash.png)

### 3. Tool-Eval Agentic Benchmark (93/100, 14/15 PASS)
![Smarts benchmark scenarios](assets/benchmark_smarts_qwen38_flash_1.png)
![Smarts benchmark score](assets/benchmark_smarts_qwen38_flash_2.png)

*For detailed comparative analysis between v1.0.0 and v1.1.0, see [`EXPERIMENTS.md`](EXPERIMENTS.md).*

---

## OpenAI-Compatible API Usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Cogni-Brain",
    "messages": [
      {"role": "user", "content": "Explain how speculative decoding works in sparse MoE architectures."}
    ],
    "max_tokens": 512
  }'
```

---

## Architecture & Upstream Blackwell (SM121) Patches

The container mounts 4 critical patches over `lmsysorg/sglang:qwen38flashnext`:

1. [`patches/qwen4_exp_nvfp4.py`](patches/qwen4_exp_nvfp4.py): Enables GPU-resident HashK table loading (`SGLANG_QWEN4_PLE_HASHK`) and packed NVFP4 mode.
2. [`patches/flash_fwd.py`](patches/flash_fwd.py): Fixes variable-length TMA-O epilogue MLIR crash in FlashAttention-4.
3. [`patches/qwen_sparse_attn_backend.py`](patches/qwen_sparse_attn_backend.py): Bypasses buggy SM100-only TRTLLM-gen decode on SM121 (preventing silent `!!!!` NaN tokens) and implements hole-tolerant QSA gather with masked SDPA.
4. [`patches/sparse_attn.py`](patches/sparse_attn.py): Fixes long-context Triton FP8 RHS dot product compilation error.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) and [`docs/LANDMINES.md`](docs/LANDMINES.md) for full architectural deep dive, ablations, and runtime ledger.

---

## Operational & Reliability Tooling

- [`tools/poison_sentinel.sh`](tools/poison_sentinel.sh): Cron canary running deep-context probes to detect and auto-recover from any silent corruption.
- [`tools/watchdog.sh`](tools/watchdog.sh): Watchdog script detecting silent wedges (accept-len 1.00 while `/health` is green).
- [`tools/bench_cc3.py`](tools/bench_cc3.py): Wall-clock window aggregate concurrency benchmark with per-request output validity verification.

---

## License & Citations

- Code & scripts: MIT License.
- Upstream patches: subject to original licenses (SGLang Apache-2.0 / FlashAttention BSD-3).
- See [`CITATION.cff`](CITATION.cff) for academic and technical attribution.
