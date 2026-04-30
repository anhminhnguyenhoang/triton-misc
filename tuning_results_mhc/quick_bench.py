#!/usr/bin/env python3
"""Direct bench of two configs on a single (M, n, C) shape, alternating runs.

Useful for sorting out whether a "regression" reported by ``bench_mhc.py`` is
real or just system noise.

Usage (from /workspace/aiter):
    python tuning_results_mhc/quick_bench.py -M 1024 -C 512 --reps 10
"""
import argparse
import statistics

import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs

OLD = {
    "BLOCK_M": 64, "BLOCK_K": 256, "NUM_KSPLIT": 1,
    "num_warps": 2, "num_stages": 1, "waves_per_eu": 0,
}
NEW = {
    "BLOCK_M": 32, "BLOCK_K": 64, "NUM_KSPLIT": 1,
    "num_warps": 8, "num_stages": 1, "waves_per_eu": 2,
    "USE_REDUCE_SPLITC": False,
}


def bench(M, n, C, config):
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val = generate_mhc_inputs(
        M, n, C, torch.bfloat16
    )
    cfg_local = dict(config)
    def run():
        mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val,
            sinkhorn_iters=20, config=dict(cfg_local))
    run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(run, warmup=50, rep=200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=1024)
    p.add_argument("-n", type=int, default=4)
    p.add_argument("-C", type=int, default=512)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--old-config", type=str, help="JSON for OLD config (overrides default)")
    p.add_argument("--new-config", type=str, help="JSON for NEW config (overrides default)")
    args = p.parse_args()

    import json
    old_cfg = json.loads(args.old_config) if args.old_config else OLD
    new_cfg = json.loads(args.new_config) if args.new_config else NEW

    print(f"M={args.M}, n={args.n}, C={args.C}")
    print(f"OLD: {old_cfg}")
    print(f"NEW: {new_cfg}\n")

    old_times = []
    new_times = []
    for i in range(args.reps):
        # Alternate to balance any drift in GPU state
        old_times.append(bench(args.M, args.n, args.C, old_cfg))
        new_times.append(bench(args.M, args.n, args.C, new_cfg))
        print(f"  Rep {i+1:2d}: OLD={old_times[-1]*1000:.2f}us  NEW={new_times[-1]*1000:.2f}us")

    print()
    print(f"OLD: median={statistics.median(old_times)*1000:.2f}us  min={min(old_times)*1000:.2f}us  max={max(old_times)*1000:.2f}us")
    print(f"NEW: median={statistics.median(new_times)*1000:.2f}us  min={min(new_times)*1000:.2f}us  max={max(new_times)*1000:.2f}us")
    sp = statistics.median(old_times) / statistics.median(new_times)
    print(f"NEW vs OLD median speedup: {sp:.3f}x")


if __name__ == "__main__":
    main()
