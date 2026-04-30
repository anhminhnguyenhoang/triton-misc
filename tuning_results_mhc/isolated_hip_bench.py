#!/usr/bin/env python3
"""Per-shape isolated Triton vs HIP bench.

The default ``bench_mhc.py --with-hip`` runs all shapes in the same Python
process, which makes earlier shapes pay the JIT/cudagraph warm-up cost while
later shapes inherit a polluted Triton autotune cache. For final reporting we
want per-shape isolation.

Usage (inside mhc-xdit container):
    python tuning_results_mhc/isolated_hip_bench.py --reps 5
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = "op_tests/op_benchmarks/triton/bench_mhc.py"

DEFAULT_M = [1024, 2048, 4096, 8192, 16384]
DEFAULT_C = [512, 1024, 4096, 32768]
N = 4


def run_one(M, C, reps):
    """Run bench in a fresh process; return (triton_ms, hip_ms) median over reps."""
    triton_times = []
    hip_times = []
    for _ in range(reps):
        out = subprocess.run(
            [
                sys.executable,
                BENCH,
                "-M", str(M),
                "-n", str(N),
                "-C", str(C),
                "--with-hip",
                "-metric", "time",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if out.returncode != 0:
            print(f"  [error] M={M} C={C} returncode={out.returncode}")
            print(out.stderr[-500:])
            continue
        last = out.stdout.strip().splitlines()[-1]
        parts = last.split()
        try:
            triton_ms = float(parts[-2])
            hip_ms = float(parts[-1])
        except (ValueError, IndexError):
            print(f"  [parse-error] {last!r}")
            continue
        triton_times.append(triton_ms)
        hip_times.append(hip_ms)
    if not triton_times:
        return None, None
    return min(triton_times), min(hip_times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("-M", type=int, nargs="+", default=DEFAULT_M)
    p.add_argument("-C", type=int, nargs="+", default=DEFAULT_C)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    rows = []
    print(f"{'C':>6} {'M':>6} {'Triton':>9} {'HIP':>9} {'T/H':>7}")
    print("-" * 45)
    for C in args.C:
        for M in args.M:
            t, h = run_one(M, C, args.reps)
            if t is None:
                continue
            ratio = t / h if h > 0 else 0.0
            rows.append({"C": C, "M": M, "triton_ms": t, "hip_ms": h, "ratio": ratio})
            print(f"{C:>6} {M:>6} {t*1000:>8.2f}u {h*1000:>8.2f}u {ratio:>6.2f}x")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nSaved -> {args.out}")

    wins = sum(1 for r in rows if r["ratio"] < 0.98)
    ties = sum(1 for r in rows if 0.98 <= r["ratio"] <= 1.02)
    losses = sum(1 for r in rows if r["ratio"] > 1.02)
    print(f"\nTotal {len(rows)}: {wins} wins (Triton < HIP), {ties} ties (±2%), {losses} losses")


if __name__ == "__main__":
    main()
