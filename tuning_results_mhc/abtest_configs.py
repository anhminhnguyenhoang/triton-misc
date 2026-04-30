#!/usr/bin/env python3
"""Robust per-bucket A/B comparison with statistically-stable timing.

Same idea as ``smart_merge_configs.py`` but performs many ``do_bench`` reps in
the same process (after a single JIT warm-up per config) and uses the median
to filter GPU/system noise. Writes only configs that show a real
(``> margin``) speedup.

Usage (from /workspace/aiter, inside container):
    python tuning_results_mhc/abtest_configs.py --reps 5 --margin 0.02
    python tuning_results_mhc/abtest_configs.py --dry-run
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "aiter" / "ops" / "triton" / "configs"
BACKUP_DIR = REPO_ROOT / "tuning_results_mhc" / "benchmarks"

ARCH = "gfx942"
HRES = "SINKHORN"
N = 4
M_TUNED = [1024, 2048, 4096, 8192, 16384]
C_BUCKETS = [128, 512, 1024, 4096, 32768]
SK_ITERS = 20


def bench_config(M, n, C, config, reps):
    """Build the inputs once, JIT-warm once, then ``do_bench`` ``reps`` times."""
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val = generate_mhc_inputs(
        M, n, C, torch.bfloat16
    )

    cfg_local = dict(config)

    def run():
        mhc(
            x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val,
            sinkhorn_iters=SK_ITERS, config=dict(cfg_local),
        )

    run()
    torch.cuda.synchronize()
    times = [
        triton.testing.do_bench(run, warmup=50, rep=200) for _ in range(reps)
    ]
    return statistics.median(times), min(times), max(times)


def fallback_old(old_dict, M):
    """When the bak file lacks a specific M_LEQ bucket, use 'any' as the OLD reference."""
    key = f"M_LEQ_{M}"
    if key in old_dict:
        return old_dict[key]
    return old_dict.get("any")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="Don't write files.")
    p.add_argument("--reps", type=int, default=5, help="do_bench reps per config (median is taken).")
    p.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Minimum relative speedup of NEW over OLD to keep NEW (default 2%%).",
    )
    args = p.parse_args()

    summary = []

    for C in C_BUCKETS:
        prod_file = CONFIG_DIR / f"{ARCH}-MHC_FUSED_{HRES}-C={C}.json"
        bak_file = BACKUP_DIR / f"{ARCH}-MHC_FUSED_{HRES}-C={C}.json.bak.before"
        if not prod_file.exists() or not bak_file.exists():
            print(f"[skip] missing files for C={C}")
            continue

        new = json.loads(prod_file.read_text())
        old = json.loads(bak_file.read_text())
        result = dict(new)

        print(f"\n=== C={C} ===")
        for M in M_TUNED:
            key = f"M_LEQ_{M}"
            new_cfg = new.get(key)
            old_cfg = fallback_old(old, M)
            if new_cfg is None:
                print(f"  {key}: missing in NEW, skip")
                continue
            if old_cfg is None:
                print(f"  {key}: missing in OLD even after 'any' fallback, skip")
                continue
            if old_cfg == new_cfg:
                print(f"  {key}: identical, skip")
                continue

            old_med, old_min, old_max = bench_config(M, N, C, old_cfg, args.reps)
            new_med, new_min, new_max = bench_config(M, N, C, new_cfg, args.reps)

            speedup = old_med / new_med if new_med > 0 else 0.0
            keep_new = speedup > 1.0 + args.margin
            decision = "NEW" if keep_new else "OLD"
            if not keep_new:
                result[key] = old_cfg

            print(
                f"  {key}: OLD={old_med*1000:>7.1f}us [{old_min*1000:>5.1f}-{old_max*1000:>5.1f}]"
                f"  NEW={new_med*1000:>7.1f}us [{new_min*1000:>5.1f}-{new_max*1000:>5.1f}]"
                f"  ({speedup:>4.2f}x) -> {decision}"
            )
            summary.append((C, M, key, old_med, new_med, speedup, decision))

        # 'any' should mirror the largest tuned M result we kept.
        max_key = f"M_LEQ_{max(M_TUNED)}"
        if max_key in result:
            result["any"] = dict(result[max_key])

        if not args.dry_run:
            prod_file.write_text(json.dumps(result, indent=4) + "\n")
            print(f"  -> wrote {prod_file.name}")
        else:
            print(f"  [dry-run] {prod_file.name}")

    print("\n=== SUMMARY ===")
    print(f"{'C':>6} {'M':>6} {'OLD ms':>8} {'NEW ms':>8} {'speedup':>8} pick")
    for C, M, key, t_old, t_new, sp, dec in summary:
        print(f"{C:>6} {M:>6} {t_old*1000:>8.1f} {t_new*1000:>8.1f} {sp:>7.2f}x {dec}")
    n_new = sum(1 for *_, d in summary if d == "NEW")
    n_old = sum(1 for *_, d in summary if d == "OLD")
    print(f"\nKept: NEW={n_new}, restored OLD={n_old} ({n_new + n_old} compared)")


if __name__ == "__main__":
    main()
