#!/usr/bin/env python3
"""Format min-of-reps timings (from `isolated_hip_bench.py`) as the same table
that `bench_mhc.py --with-hip -metric all` produces.

This avoids the JIT cold-start spikes that appear in single in-process bench runs
when you want a canonical reproducible table for a PR description.

Usage:
    python tuning_results_mhc/format_metric_all.py \
        --json tuning_results_mhc/benchmarks/triton36_tuned_vs_hip_min.json \
        --sinkhorn-iters 20
"""
import argparse
import json
from pathlib import Path


def metrics_for(M, n, C, sinkhorn_iters):
    nC = n * C
    n_squared = n * n
    N = n_squared + 2 * n

    flops_matmul = 2.0 * M * nC * n + 2.0 * M * nC * n + 2.0 * M * nC * n_squared
    flops_rms = 4.0 * M * nC
    flops_apply_pre = 2.0 * M * n * C
    flops_sinkhorn = 10.0 * M * n_squared * sinkhorn_iters
    total_flops = flops_matmul + flops_rms + flops_apply_pre + flops_sinkhorn

    elem_size = 2
    bias_size = 4
    mem_read = (
        M * nC * elem_size
        + M * nC * elem_size
        + nC * n * elem_size
        + nC * n * elem_size
        + nC * n_squared * elem_size
        + N * bias_size
    )
    mem_write = M * n * elem_size + M * n_squared * elem_size + M * C * elem_size
    total_mem = mem_read + mem_write
    return total_flops, total_mem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    rows = json.loads(Path(args.json).read_text())
    rows.sort(key=lambda r: (r["C"], r["M"]))

    header = (
        "          M    n        C  triton_time(ms)  hip_time(ms)  "
        "triton_throughput(TFLOPS)  hip_throughput(TFLOPS)  "
        "triton_bandwidth(GB/s)  hip_bandwidth(GB/s)  "
        "triton_arithmetic_intensity(FLOP/byte)  hip_arithmetic_intensity(FLOP/byte)"
    )
    print(header)
    for i, r in enumerate(rows):
        M = r["M"]
        n = args.n
        C = r["C"]
        t_ms = r["triton_ms"]
        h_ms = r["hip_ms"]
        flops, mem = metrics_for(M, n, C, args.sinkhorn_iters)
        ai = flops / mem
        t_tf = flops / (t_ms * 1e-3) / 1e12
        h_tf = flops / (h_ms * 1e-3) / 1e12
        t_bw = mem / (t_ms * 1e-3) / 1e9
        h_bw = mem / (h_ms * 1e-3) / 1e9
        print(
            f"{i:<3} {float(M):>7.1f} {float(n):>4.1f} {float(C):>8.1f}"
            f"        {t_ms:>9.6f}      {h_ms:>8.6f}"
            f"                  {t_tf:>9.6f}              {h_tf:>9.6f}"
            f"             {t_bw:>11.6f}          {h_bw:>11.6f}"
            f"                              {ai:>11.6f}                         {ai:>11.6f}"
        )


if __name__ == "__main__":
    main()
