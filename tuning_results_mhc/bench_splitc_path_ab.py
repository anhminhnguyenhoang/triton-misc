#!/usr/bin/env python3
"""A/B bench: inline-apply vs dedicated split-C apply path for the MHC kernel.

For each `(M, C)` in the requested grid, force `USE_REDUCE_SPLITC=False` then
`=True` (with the rest of the config coming from the production JSON via
`get_mhc_config`) and report which path is faster. Useful as a quick
diagnostic when:

- Bringing up a new GPU arch (decide initial `USE_REDUCE_SPLITC` defaults
  before running the full tuner sweep).
- Sanity-checking that a per-shape JSON entry is on the right side of the
  inline/dedicated trade-off.
- Picking / validating the wrapper's `DEFAULT_REDUCE_SPLITC_THRESHOLD`.

Usage (from /home/anguyenh/aiter):
    python tuning_results/bench_splitc_path_ab.py
    python tuning_results/bench_splitc_path_ab.py -M 4096 8192 -C 512 1024 2048
    python tuning_results/bench_splitc_path_ab.py -n 4 --dtype bf16 --hres-mode sinkhorn
"""
import argparse
import sys
from itertools import product

import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc
from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def bench_one(M, n, C, *, dtype, sinkhorn_iters, use_reduce_splitc):
    """Run `do_bench` on `mhc()` with the given path forced."""
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val = generate_mhc_inputs(
        M, n, C, dtype
    )
    base_cfg, _ = get_mhc_config("MHC_FUSED", M, C, mode="sinkhorn")
    cfg = dict(base_cfg)
    cfg["USE_REDUCE_SPLITC"] = use_reduce_splitc
    # Split-C block sizes only matter when USE_REDUCE_SPLITC is True; the wrapper
    # falls back to sensible defaults when the keys are missing.
    if not use_reduce_splitc:
        cfg.pop("BLOCK_M_SPLITC", None)
        cfg.pop("BLOCK_C_SPLITC", None)

    def run():
        mhc(
            x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val,
            sinkhorn_iters=sinkhorn_iters, config=dict(cfg),
        )

    return triton.testing.do_bench(run, warmup=25, rep=100)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "-M", nargs="+", type=int,
        default=[1024, 2048, 4096, 8192, 16384],
        help="M values to sweep (default: 1024..16384).",
    )
    p.add_argument(
        "-C", nargs="+", type=int,
        default=[128, 512, 1024, 2048, 4096, 32768],
        help="C values to sweep (default: 128..32768).",
    )
    p.add_argument("-n", type=int, default=4, help="Stream parameter (default 4).")
    p.add_argument(
        "--dtype", choices=DTYPE_MAP.keys(), default="bf16",
        help="Element dtype (default bf16).",
    )
    p.add_argument(
        "--hres-mode", choices=["sinkhorn", "lite"], default="sinkhorn",
        help="H_res mode (sinkhorn -> 20 iters, lite -> 0 iters).",
    )
    p.add_argument(
        "--sort", choices=["c-then-m", "m-then-c"], default="c-then-m",
        help="Row order in the output table.",
    )
    args = p.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    sinkhorn_iters = 20 if args.hres_mode == "sinkhorn" else 0

    if args.sort == "c-then-m":
        shapes = sorted(product(args.M, args.C), key=lambda mc: (mc[1], mc[0]))
    else:
        shapes = sorted(product(args.M, args.C), key=lambda mc: (mc[0], mc[1]))

    print(
        f"\nA/B sweep: dtype={args.dtype} n={args.n} hres={args.hres_mode}"
        f"  ({sinkhorn_iters} SK iters)\n"
    )
    print(f"{'M':>6} {'C':>6}  {'inline(ms)':>11} {'dedi(ms)':>10} {'speedup':>8}  best")
    print("-" * 56)

    last_C = None
    crossovers = {}  # M -> smallest C where dedi beats inline
    prev_winner_per_M = {}
    for M, C in shapes:
        if args.sort == "c-then-m" and last_C is not None and C != last_C:
            print()
        last_C = C

        t_inline = bench_one(
            M, args.n, C, dtype=dtype, sinkhorn_iters=sinkhorn_iters,
            use_reduce_splitc=False,
        )
        t_dedi = bench_one(
            M, args.n, C, dtype=dtype, sinkhorn_iters=sinkhorn_iters,
            use_reduce_splitc=True,
        )
        speedup = max(t_inline, t_dedi) / min(t_inline, t_dedi)
        winner = "inline" if t_inline < t_dedi else "dedi"
        print(
            f"{M:>6} {C:>6}  {t_inline:>11.4f} {t_dedi:>10.4f} {speedup:>7.2f}x  {winner}"
        )

        prev = prev_winner_per_M.get(M)
        if prev == "inline" and winner == "dedi" and M not in crossovers:
            crossovers[M] = C
        prev_winner_per_M[M] = winner

    if crossovers:
        print("\n=== Crossover (smallest tested C where dedicated beats inline) ===")
        for M in sorted(crossovers):
            print(f"  M={M:>6}: C={crossovers[M]}")
    else:
        print(
            "\n[note] no inline -> dedicated crossover detected within the sweep range"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
