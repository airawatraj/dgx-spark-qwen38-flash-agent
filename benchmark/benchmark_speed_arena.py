#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""
DGX Spark / Qwen3.8-Flash-Next Full Spark-Arena Benchmark
Runs llama-benchy sweep compatible with spark-arena leaderboard formatting.

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
        args.pp,
        "--tg",
        args.tg,
        "--save-result",
        args.save_result,
    ]

    for depth in args.depth:
        command.extend(["--depth", str(depth)])

    for concurrency in args.concurrency:
        command.extend(["--concurrency", str(concurrency)])

    return command


def main():
    parser = argparse.ArgumentParser(
        description="DGX Spark / Qwen3.8-Flash-Next Full Arena Benchmark Sweep"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Target base URL",
    )
    parser.add_argument(
        "--model",
        default="Cogni-Brain",
        help="Target model identifier",
    )
    parser.add_argument(
        "--served-model-name",
        default="Cogni-Brain",
        help="Served model name exposed by the API",
    )
    parser.add_argument(
        "--tokenizer",
        default="RadixArk/Qwen3.8-Flash-Next-NVFP4",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--pp",
        default="512,2048,8192",
        help="Prompt prefill sizes to test",
    )
    parser.add_argument(
        "--tg",
        default="128,512",
        help="Token generation lengths to test",
    )
    parser.add_argument(
        "--depth",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Queue depths to test",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--save-result",
        default="benchmark/results_arena.csv",
        help="Path to save output CSV results",
    )
    args = parser.parse_args()

    header("SPARK-ARENA BENCHMARK SWEEP")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Base URL", args.base_url)
    result_line("Model", args.model)
    result_line("Served model name", args.served_model_name)
    result_line("Tokenizer", args.tokenizer)
    result_line("Prompt prefill", args.pp)
    result_line("Token generation", args.tg)
    result_line("Depths", ", ".join(str(d) for d in args.depth))
    result_line("Concurrency", ", ".join(str(c_val) for c_val in args.concurrency))
    result_line("Output CSV", args.save_result)
    print()

    if shutil.which("uv") is None:
        print(f"\n{c('ERROR: uv is not installed or not on PATH', 'red')}")
        sys.exit(1)

    output_path = Path(args.save_result)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(args)

    header("COMMAND")
    print("  " + " ".join(command))

    header("RUNNING")
    print(f"  {c('Streaming llama-benchy arena sweep output below...', 'dim')}\n")

    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print(f"\n{c('Benchmark sweep interrupted by user', 'yellow')}")
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
        print(f"  {c('llama-benchy sweep did not complete successfully.', 'red')}")
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
