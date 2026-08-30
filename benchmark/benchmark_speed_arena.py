#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""
DGX Spark / Qwen3.8-Flash-Next Full Spark-Arena Benchmark
Runs the long-form llama-benchy sweep used for spark-arena-style measurements.

Usage:
  uv run benchmark/benchmark_speed_arena.py
  uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_arena.csv
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
    "dim": "\033[2m",
}


def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def header(title):
    line = "─" * 60
    print(f"\n{c(line, 'cyan')}")
    print(f"{c('  ' + title, 'bold')}")
    print(f"{c(line, 'cyan')}")


def result_line(label, value, color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)}")


def build_command(args):
    command = [
        "uv",
        "tool",
        "run",
        "--from",
        "llama-benchy",
        "llama-benchy",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer,
        "--pp",
        str(args.pp),
        "--tg",
        str(args.tg),
        "--extra-body",
        "return_token_ids=false",
        "--save-result",
        args.save_result,
        "--depth",
        *[str(depth) for depth in args.depth],
        "--concurrency",
        *[str(concurrency) for concurrency in args.concurrency],
    ]
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run the full spark-arena-style llama-benchy sweep against the local endpoint.",
        epilog="Example: uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_arena.csv",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--served-model-name", default="Cogni-Brain")
    parser.add_argument("--tokenizer", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--pp", type=int, default=2048)
    parser.add_argument("--tg", type=int, default=128)
    parser.add_argument(
        "--depth",
        nargs="+",
        type=int,
        default=[0, 4096, 8192, 16384, 32768, 65535, 131072, 259968],
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--save-result", default="benchmark/results_arena.csv")
    args = parser.parse_args()

    header("FULL ARENA BENCHMARK")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Base URL", args.base_url)
    result_line("Model", args.model)
    result_line("Served model name", args.served_model_name)
    result_line("Tokenizer", args.tokenizer)
    result_line("Prompt prefill", args.pp)
    result_line("Token generation", args.tg)
    result_line("Depths", ", ".join(str(depth) for depth in args.depth))
    result_line("Concurrency", ", ".join(str(concurrency) for concurrency in args.concurrency))
    result_line("Output CSV", args.save_result)
    print()
    print(f"  {c('This sweep can take several hours. Run it with only spark-brain active.', 'yellow')}")

    if shutil.which("uv") is None:
        print(f"\n{c('uv is not installed or not on PATH', 'red')}")
        sys.exit(1)

    output_path = Path(args.save_result)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(args)

    header("COMMAND")
    print("  " + " ".join(command))

    header("RUNNING")
    print(f"  {c('Streaming llama-benchy output below...', 'dim')}")

    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print(f"\n{c('Benchmark interrupted by user', 'yellow')}")
        sys.exit(130)
    except FileNotFoundError:
        print(f"\n{c('Failed to start uv', 'red')}")
        sys.exit(1)

    if completed.returncode == 0:
        header("DONE")
        result_line("Status", "Success")
        result_line("Saved results", args.save_result)
    else:
        header("FAILED")
        result_line("Exit code", completed.returncode, color="red")
        print(f"  {c('llama-benchy did not complete successfully.', 'red')}")
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
