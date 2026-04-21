"""Benchmark sinkhorn_knopp with default configs across all test shapes."""

import json
import sys
import torch
import triton

from aiter.ops.triton.fusions.mhc import sinkhorn_knopp
from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config


def bench_sinkhorn(M, N, C, num_iters=20, dtype=torch.bfloat16):
    logits = torch.randn((M, N, N), device="cuda", dtype=dtype)

    # Warm up to compile the kernel
    _ = sinkhorn_knopp(logits, C, num_iters=num_iters)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: sinkhorn_knopp(logits, C, num_iters=num_iters),
        warmup=25,
        rep=100,
    )
    return ms


def main():
    C = 1024
    num_iters = 20

    # Clear LRU cache so config changes are picked up
    get_mhc_config.cache_clear()
    if hasattr(get_mhc_config, "_config_cache"):
        del get_mhc_config._config_cache

    M_values = [1, 4, 16, 64, 256]
    N_values = [2, 4, 8, 16, 32]

    # Show what config is being used
    cfg, _ = get_mhc_config("MHC_SINKHORN", M_values[0], C)
    print(f"Active config: {cfg}")
    print()

    results = {}
    print(f"{'M':>6} {'N':>4} {'time_ms':>10}")
    print("-" * 24)

    for N in N_values:
        for M in M_values:
            ms = bench_sinkhorn(M, N, C, num_iters)
            results[f"M{M}_N{N}"] = ms
            print(f"{M:>6} {N:>4} {ms:>10.4f}")
        print()

    # Save results to file for comparison
    out_file = sys.argv[1] if len(sys.argv) > 1 else "sk_bench_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
