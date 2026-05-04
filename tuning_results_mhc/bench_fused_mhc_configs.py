"""Benchmark fused_mhc (sinkhorn mode) with default configs across key shapes."""

import json
import sys
import torch
import triton

from aiter.ops.triton.fusions.mhc import fused_mhc
from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs


def bench_fused_mhc(M, n, C, dtype=torch.bfloat16):
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val = generate_mhc_inputs(
        M, n, C, dtype
    )

    # Warm up
    _ = fused_mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: fused_mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val),
        warmup=25,
        rep=100,
    )
    return ms


def main():
    n = 4

    # Clear LRU cache so config changes are picked up
    get_mhc_config.cache_clear()
    if hasattr(get_mhc_config, "_config_cache"):
        del get_mhc_config._config_cache

    M_values = [1, 16, 64, 256, 1024]
    C_values = [512, 1024, 4096]

    # Show configs being used
    for C in C_values:
        cfg, specialized = get_mhc_config("MHC_FUSED", M_values[0], C, mode="sinkhorn")
        print(f"C={C} (specialized={specialized}): {cfg}")
    print()

    results = {}
    print(f"{'C':>6} {'M':>6} {'time_ms':>10}")
    print("-" * 26)

    for C in C_values:
        for M in M_values:
            ms = bench_fused_mhc(M, n, C)
            key = f"C{C}_M{M}"
            results[key] = ms
            print(f"{C:>6} {M:>6} {ms:>10.4f}")
        print()

    out_file = sys.argv[1] if len(sys.argv) > 1 else "fused_mhc_bench_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
