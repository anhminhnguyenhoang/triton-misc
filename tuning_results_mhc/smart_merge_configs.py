#!/usr/bin/env python3
"""Per-bucket A/B merge: pick whichever config (old or new) is faster.

Compares each (M, C) bucket's old config (from .bak.before backup) against the
current tuner-winner in the production JSON. Whichever is faster on a fresh
``do_bench`` is the one we keep. This guarantees zero regressions vs the
pre-tuning baseline while still adopting any wins the tuner produced.

Usage (from /workspace/aiter, inside container):
    python tuning_results_mhc/smart_merge_configs.py
    python tuning_results_mhc/smart_merge_configs.py --dry-run
    python tuning_results_mhc/smart_merge_configs.py --reps 3
"""
import argparse
import json
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


def bench_one(M, n, C, config, reps=1):
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val = generate_mhc_inputs(
        M, n, C, torch.bfloat16
    )

    def run():
        mhc(
            x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val,
            sinkhorn_iters=SK_ITERS, config=dict(config),
        )

    # Warm up once to JIT-compile.
    run()
    torch.cuda.synchronize()

    times = [
        triton.testing.do_bench(run, warmup=25, rep=100) for _ in range(reps)
    ]
    return min(times)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="Print decisions, don't write.")
    p.add_argument("--reps", type=int, default=2, help="do_bench repetitions per config; we keep the min.")
    p.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Required relative speedup of NEW over OLD to pick NEW (default 1%%, i.e. NEW must be at least 1%% faster).",
    )
    args = p.parse_args()

    summary = []  # (C, M, key, t_old, t_new, decision)

    for C in C_BUCKETS:
        prod_file = CONFIG_DIR / f"{ARCH}-MHC_FUSED_{HRES}-C={C}.json"
        bak_file = BACKUP_DIR / f"{ARCH}-MHC_FUSED_{HRES}-C={C}.json.bak.before"
        if not prod_file.exists() or not bak_file.exists():
            print(f"[skip] missing files for C={C}")
            continue

        new = json.loads(prod_file.read_text())  # currently has tuner winners
        old = json.loads(bak_file.read_text())  # pre-tuning baseline

        result = dict(new)  # start with new, restore old where it wins

        print(f"\n=== C={C} ===")
        for M in M_TUNED:
            key = f"M_LEQ_{M}"
            old_cfg = old.get(key)
            new_cfg = new.get(key)
            if old_cfg is None or new_cfg is None:
                print(f"  {key}: missing in one side, skip")
                continue
            if old_cfg == new_cfg:
                print(f"  {key}: identical, skip")
                continue

            t_old = bench_one(M, N, C, old_cfg, reps=args.reps)
            t_new = bench_one(M, N, C, new_cfg, reps=args.reps)

            # Require NEW to be at least (1 + margin)x faster than OLD to pick NEW;
            # otherwise restore OLD. Treats sub-margin differences as noise.
            if t_new * (1.0 + args.margin) < t_old:
                decision = "NEW"
                # keep new (already in result)
            else:
                decision = "OLD"
                result[key] = old_cfg

            speedup = t_old / t_new if t_new > 0 else 0
            print(
                f"  {key}: OLD={t_old*1000:>7.1f}us  NEW={t_new*1000:>7.1f}us  "
                f"({speedup:>4.2f}x) -> {decision}"
            )
            summary.append((C, M, key, t_old, t_new, decision))

        # Refresh "any" from the largest tuned M result we ended up keeping.
        max_key = f"M_LEQ_{max(M_TUNED)}"
        if max_key in result:
            result["any"] = dict(result[max_key])

        if not args.dry_run:
            prod_file.write_text(json.dumps(result, indent=4) + "\n")
            print(f"  -> wrote {prod_file.name}")
        else:
            print(f"  [dry-run] would write {prod_file.name}")

    print("\n=== SUMMARY ===")
    print(f"{'C':>6} {'M':>6} {'OLD ms':>8} {'NEW ms':>8} {'speedup':>8} pick")
    for C, M, key, t_old, t_new, dec in summary:
        speedup = t_old / t_new if t_new > 0 else 0
        print(f"{C:>6} {M:>6} {t_old*1000:>8.1f} {t_new*1000:>8.1f} {speedup:>7.2f}x {dec}")
    n_new = sum(1 for *_, d in summary if d == "NEW")
    n_old = sum(1 for *_, d in summary if d == "OLD")
    print(f"\nKept: NEW={n_new}, restored OLD={n_old} ({n_new + n_old} compared)")


if __name__ == "__main__":
    main()
