#!/usr/bin/env python3
"""Run paired OLD vs FINAL benchmark across all (M, C) shapes in the same
process, alternating per shape, to filter out cross-process JIT/contention
noise. Reads OLD configs from the .bak.before files and FINAL configs from
the production JSONs.
"""
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
M_LIST = [1024, 2048, 4096, 8192, 16384]
C_LIST = [128, 512, 1024, 4096, 32768]
SK_ITERS = 20
REPS = 5


def _config_for_M(d, M):
    """Production lookup: pick the smallest M_LEQ_X with X >= M, else 'any'."""
    candidates = []
    for k in d:
        if k.startswith("M_LEQ_"):
            x = int(k[len("M_LEQ_"):])
            if x >= M:
                candidates.append((x, d[k]))
    if candidates:
        candidates.sort(key=lambda kv: kv[0])
        return candidates[0][1]
    return d.get("any")


def bench(M, n, C, config):
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val = generate_mhc_inputs(
        M, n, C, torch.bfloat16
    )
    cfg_local = dict(config)
    def run():
        mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_val,
            sinkhorn_iters=SK_ITERS, config=dict(cfg_local))
    run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(run, warmup=50, rep=200)


def main():
    print(f"{'C':>6} {'M':>6} {'OLD med':>9} {'NEW med':>9} {'speedup':>9}")
    print("-" * 55)
    rows = []
    for C in C_LIST:
        old_d = json.loads((BACKUP_DIR / f"{ARCH}-MHC_FUSED_{HRES}-C={C}.json.bak.before").read_text())
        new_d = json.loads((CONFIG_DIR / f"{ARCH}-MHC_FUSED_{HRES}-C={C}.json").read_text())
        for M in M_LIST:
            old_cfg = _config_for_M(old_d, M)
            new_cfg = _config_for_M(new_d, M)
            if old_cfg is None or new_cfg is None:
                continue
            old_t, new_t = [], []
            for _ in range(REPS):
                old_t.append(bench(M, N, C, old_cfg))
                new_t.append(bench(M, N, C, new_cfg))
            old_med = statistics.median(old_t)
            new_med = statistics.median(new_t)
            sp = old_med / new_med
            rows.append((C, M, old_med, new_med, sp))
            print(f"{C:>6} {M:>6} {old_med*1000:>8.2f}u {new_med*1000:>8.2f}u {sp:>7.3f}x")

    wins = sum(1 for *_, sp in rows if sp > 1.02)
    ties = sum(1 for *_, sp in rows if 0.98 <= sp <= 1.02)
    losses = sum(1 for *_, sp in rows if sp < 0.98)
    print(f"\nTotal {len(rows)}: {wins} wins (>2%), {ties} ties, {losses} regressions")


if __name__ == "__main__":
    main()
