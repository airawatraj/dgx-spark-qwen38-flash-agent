#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Qwen3.8-Flash-Next Speed & Context Benchmark
Tests TPS, TTFT, concurrency scaling, and context window.
Usage: uv run benchmark/benchmark_speed.py [--host localhost] [--port 8000] [--model Cogni-Brain]
"""

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime

import requests


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


def result_line(label, value, unit="", color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)} {unit}")


def make_prompt(target_tokens):
    base = ("The quick brown fox jumps over the lazy dog. " * 50).split()
    # 1 word ~ 1.33 tokens, so target_tokens / 1.33 words gives approximately target_tokens
    n_words = max(10, int(target_tokens / 1.33))
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the above text in one sentence."


def count_tokens_approx(text):
    return int(len(text.split()) * 1.33)


def stream_completion(
    host, port, model, prompt, max_tokens=200, timeout=120, debug=False
):
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    t_first_token = None
    generated_tokens = 0

    try:
        with requests.post(
            url, json=payload, headers=headers, stream=True, timeout=timeout
        ) as r:
            if r.status_code != 200:
                return None, None, 0, f"HTTP {r.status_code}: {r.text[:100]}"

            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        text = choices[0].get("text", "")
                        if text:
                            generated_tokens += 1
                        if debug:
                            print(text, end="", flush=True)
                except json.JSONDecodeError:
                    continue

        t_end = time.perf_counter()
        if debug:
            print()

        ttft = (t_first_token - t_start) * 1000 if t_first_token else None
        gen_time = (t_end - t_first_token) if t_first_token else (t_end - t_start)
        tps = (generated_tokens / gen_time) if gen_time > 0 else 0.0

        return ttft, tps, generated_tokens, None

    except Exception as exc:
        return None, None, 0, str(exc)


def test_baseline_tps(host, port, model, n_runs=3, max_tokens=256, debug=False):
    header("1. BASELINE SPEED (SINGLE SESSION)")
    prompt = "Explain why the sky is blue in three detailed paragraphs."
    tps_runs = []
    ttft_runs = []

    for i in range(1, n_runs + 1):
        ttft, tps, n_tok, err = stream_completion(
            host, port, model, prompt, max_tokens=max_tokens, debug=debug
        )
        if err:
            print(f"  Run {i}: {c('FAILED - ' + err, 'red')}")
            continue
        tps_runs.append(tps)
        if ttft is not None:
            ttft_runs.append(ttft)
        print(
            f"  Run {i}: {c(f'{tps:.1f} tok/s', 'green')}  "
            f"(TTFT: {c(f'{ttft:.0f} ms' if ttft else 'N/A', 'dim')}, {n_tok} tokens)"
        )

    if not tps_runs:
        print(f"\n  {c('All baseline runs failed', 'red')}")
        return 0, 0

    avg_tps = statistics.mean(tps_runs)
    peak_tps = max(tps_runs)
    avg_ttft = statistics.mean(ttft_runs) if ttft_runs else 0

    print()
    result_line("Average TPS", f"{avg_tps:.1f}", "tok/s")
    result_line("Peak TPS", f"{peak_tps:.1f}", "tok/s")
    result_line("Average TTFT", f"{avg_ttft:.0f}", "ms")

    return avg_tps, peak_tps


def test_concurrent(host, port, model, n_streams_list=(2, 4, 8), max_tokens=200):
    header("2. CONCURRENCY SCALING TEST")
    print(f"  {c('Testing aggregate generation throughput across concurrent streams', 'dim')}\n")

    for n_streams in n_streams_list:
        results = []
        threads = []

        def worker():
            prompt = "Write a comprehensive essay describing the history of computer architecture."
            ttft, tps, n_tok, err = stream_completion(
                host, port, model, prompt, max_tokens=max_tokens, timeout=180
            )
            results.append((ttft, tps, n_tok, err))

        t0 = time.perf_counter()
        for _ in range(n_streams):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        t_total = time.perf_counter() - t0
        successful = [r for r in results if r[3] is None]
        total_tokens = sum(r[2] for r in successful)
        agg_tps = total_tokens / t_total if t_total > 0 else 0

        status_str = f"{len(successful)}/{n_streams} OK"
        color = "green" if len(successful) == n_streams else "yellow"
        print(
            f"  {c(f'{n_streams} streams:', 'bold')} "
            f"{c(f'{agg_tps:.1f} agg tok/s', color)} "
            f"({total_tokens} total tokens in {t_total:.1f}s, {status_str})"
        )


def test_context_window(host, port, model, max_ctx=260000):
    header("3. CONTEXT WINDOW SCALING TEST")
    test_token_targets = [1000, 4000, 16000, 32000, 64000, 131000, 260000]
    if max_ctx > 260000:
        test_token_targets = [t for t in test_token_targets if t < max_ctx] + [max_ctx]
    else:
        test_token_targets = [t for t in test_token_targets if t <= max_ctx]

    max_ok = 0
    for target_tok in test_token_targets:
        prompt = make_prompt(target_tok)
        approx_tok = count_tokens_approx(prompt)

        print(f"  Testing ~{approx_tok:,} tokens context... ", end="", flush=True)
        t0 = time.perf_counter()
        ttft, tps, n_tok, err = stream_completion(
            host, port, model, prompt, max_tokens=32, timeout=600
        )
        dt = time.perf_counter() - t0

        if err:
            print(f"{c('FAILED (' + err[:50] + ')', 'red')}")
            break
        else:
            max_ok = approx_tok
            print(f"{c('PASS', 'green')} (TTFT: {ttft:.0f}ms, {tps:.1f} tok/s)")

    print()
    result_line("Max Verified Context", f"{max_ok:,}", "tokens")
    return max_ok


def main():
    parser = argparse.ArgumentParser(
        description="DGX Spark / Qwen3.8-Flash-Next Benchmark Suite"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--skip-concurrent", action="store_true")
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=260000,
        help="Maximum context tokens to test (default: 260000; up to 500000 in YaRN mode)",
    )
    args = parser.parse_args()

    print(f"\n{c('=====================================================', 'cyan')}")
    print(f"{c(' DGX Spark / Qwen3.8-Flash-Next Performance Suite', 'bold')}")
    print(f"{c('=====================================================', 'cyan')}")
    print(f"Target: http://{args.host}:{args.port} (Model: {args.model})")

    # Verify endpoint is up
    try:
        r = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if r.status_code != 200:
            print(c(f"Endpoint returned HTTP {r.status_code}", "red"))
            sys.exit(1)
    except Exception as exc:
        print(c(f"Cannot reach server: {exc}", "red"))
        print(c("Make sure the container is started with: bash docker/start.sh", "dim"))
        sys.exit(1)

    print(f"{c('Server is online and healthy.', 'green')}\n")

    avg_tps, peak_tps = test_baseline_tps(
        args.host, args.port, args.model, debug=args.debug
    )

    if not args.skip_concurrent:
        test_concurrent(args.host, args.port, args.model)

    max_ctx = 0
    if not args.skip_context:
        max_ctx = test_context_window(
            args.host, args.port, args.model, max_ctx=args.max_ctx
        )

    header("SUMMARY")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Target Model", args.model)
    result_line("Average Baseline TPS", f"{avg_tps:.1f}", "tok/s")
    result_line("Peak Single-Stream TPS", f"{peak_tps:.1f}", "tok/s")
    if not args.skip_context:
        result_line("Max Context Tested", f"{max_ctx:,}", "tokens")
    print(f"\n{c('Benchmark complete.', 'bold')}\n")


if __name__ == "__main__":
    main()
