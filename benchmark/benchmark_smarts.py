#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""
DGX Spark / Qwen3.8-Flash-Next Smarts & Tool-Calling Benchmark
Runs tool-eval-bench against the local OpenAI-compatible endpoint.

Usage:
  uv run benchmark/benchmark_smarts.py
  uv run benchmark/benchmark_smarts.py --base-url http://localhost:8000/v1
  uv run benchmark/benchmark_smarts.py --mode perf
  uv run benchmark/benchmark_smarts.py --mode trials --seed 42 --trials 3
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime


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
        "tool-eval-bench",
        "tool-eval-bench",
        "--base-url",
        args.base_url,
    ]

    if args.mode == "short":
        command.append("--short")
    elif args.mode == "perf":
        command.append("--perf")
    elif args.mode == "trials":
        command.extend(["--seed", str(args.seed), "--trials", str(args.trials)])

    return command


def mode_description(mode):
    descriptions = {
        "short": "Runs short scenario subset for a quick sanity check",
        "perf": "Runs the tool-eval throughput sweep",
        "trials": "Runs deterministic multi-trial evaluation",
    }
    return descriptions.get(mode, "Unknown mode")


def main():
    parser = argparse.ArgumentParser(
        description="DGX Spark / Qwen3.8-Flash-Next Tool-Use Evaluation"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Target OpenAI base URL",
    )
    parser.add_argument(
        "--mode",
        choices=["short", "perf", "trials"],
        default="short",
        help="Evaluation mode (short, perf, trials)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trials")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    args = parser.parse_args()

    header("SMARTS / TOOL EVAL BENCHMARK")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Base URL", args.base_url)
    result_line("Mode", f"{args.mode} ({mode_description(args.mode)})")
    if args.mode == "trials":
        result_line("Seed", args.seed)
        result_line("Trials", args.trials)

    if shutil.which("uv") is None:
        print(f"\n{c('ERROR: uv is not installed or not on PATH', 'red')}")
        sys.exit(1)

    command = build_command(args)

    header("COMMAND")
    print("  " + " ".join(command))

    header("RUNNING")
    print(f"  {c('Streaming tool-eval-bench output below...', 'dim')}\n")

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
        result_line("Status", "Success (Completed)")
    else:
        header("FAILED")
        result_line("Exit code", completed.returncode, color="red")
        print(f"  {c('tool-eval-bench did not complete successfully.', 'red')}")
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
