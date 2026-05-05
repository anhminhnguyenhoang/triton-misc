# MHC Triton Tuning — Session Log (Rounds 3 + 4 + Consolidation)

**Date:** 2026-05-05
**Scope:** Continue tuning the `mhc` Triton kernel on top of the prior session's R1 (Triton 3.6.0 retune) + R2 (post-refactor sweep + focused retune).

Detailed engineering memo: [`triton_3_6_0_retune_report.md`](./triton_3_6_0_retune_report.md). This file is a chronological session log; numbers below are condensed from that report.

---

## Inherited state (start of session)

- Triton 3.6.0 / ROCm 7.12.0 / PyTorch 2.9.1, gfx950, dev container `anguyenh-dev-2`.
- Round-2 final score: **13/15 wins vs HIP**; only HIP-winning shapes were `C=512/M=1024` (HIP 1.39×) and `C=512/M=2048` (HIP 1.13×).
- Per-shape `tested_configs_*.json` cache held ~25K configs across 25 shapes — only **17%** of the 144,000-config search space.
- 6 raw `best_configs_*.json` files at workspace root (from R2 sweep + 1 R2-focused retune).

---

## Round 3 — "Resume sweep on the unexplored space"

### User instruction

> "take into account the best configs we found, do another round of retuning to explore the remaining unchecked configs to find even better performing ones. Append to the report upon finish"

### Strategy

1. **Don't clear `tested_configs_*.json`** — preserve what's been tested.
2. **Bump `PATIENCE` to 2000** (R2 used 500, R2-focused used 1000).
3. Run full sweep on all 5 C buckets via `run_tune_all.sh` on GPUs 4-7 inside the dev container.
4. Trust `refresh_mhc_configs.py` to pick the global best per-shape across all `best_configs_*.json` files (R2 + R3).

### Execution

- Launched via `docker exec -d` (background); polled with `AwaitShell` and tail of `tune_C*.log`.
- **Wall time: 1:51:08** across 5 C buckets (12-29 min/bucket).

### Results

`refresh_mhc_configs.py` reported **15 of 25 shapes got new winners**. Biggest gains at C=1024 (M=1024 -9.5%, M=16384 -8.7%); C=32768 came back `[unchanged]`.

**Bench (warm cache, post-refresh)** vs R2 final, all 315 pytest passing:

| Shape | R2 final | R3 | Δ | vs HIP |
|---|---:|---:|---:|---|
| C=512/M=1024 | 0.0223 | **0.0202** | -10% | HIP 1.39× → **1.25×** |
| C=512/M=2048 | 0.0229 | **0.0209** | -9%  | HIP 1.13× → **1.02× (tie)** |
| C=512/M=4096 | 0.0265 | **0.0225** | -15% | Triton 1.17× → **1.37×** |
| C=512/M=8192 | 0.0267 | **0.0255** | -4%  | Triton 1.62× → **1.71×** |
| C=4096/M=2048 | 0.0410 | **0.0381** | -7% | Triton 1.32× → **1.42×** |

7 shapes faster by 1-15%, 8 within ±1% noise, no regressions. Wins-vs-HIP still 13/15 but the gap narrowed materially on the two HIP-leading shapes.

### Key insight

The most consequential discovery: **`BLOCK_M=32, BLOCK_K=256, BLOCK_C=256, num_stages=2`** at `C=512/M={1024,2048}`. R2 sweep had landed on `BLOCK_C=32`; R2-focused had landed on `BLOCK_C=128`. Neither random walk had reached `BLOCK_C=256` with `num_stages=2` — but it's ~10% faster. **Resume-sweep with preserved cache > focused retune with cleared cache** for finding configs in unsampled corners.

---

## Round 4 — "Clean up artefacts then squeeze a final pass"

### User instruction

> "clear triton cache and dead processes and do another round of tuning on the remaining configs to squeeze any better performance"

### Pre-flight cleanup

1. Killed orphaned `tmux new-session -d -s mhc-tune ...` from R2's launch attempt (sleeping for 24h on host).
2. `rm -rf /root/.triton/cache` inside container — freed **38 GB / 88,966 cached kernel binaries** (took 13 s).
3. Left `tested_configs_*.json` in place — resume cache shows **72% explored / 28% (40,193 configs) remaining**.

### Strategy

- **`PATIENCE=3000`** — bumped from R3's 2000. Max remaining per shape was 3192, so this effectively guarantees full exhaustion of the remaining space for every shape.
- Same cleared-cache + resume-sweep pattern as R3, just with the cache wipe up front.

### Execution

- Wall time: **1:24:00** (faster than R3 despite cold compile cache, because remaining-config counts per shape were small enough that most shapes hit end-of-list before patience triggered).

### Results

7 of 25 shapes got new winners. **The big surprise: `C=32768`** — which R3 had declared `[unchanged]` for *all* 5 M values — produced strict wins on 4 of 5 M values:

```
[update] gfx950-MHC_FUSED_SINKHORN-C=32768.json
    M_LEQ_2048:   306.9us   (was 311.12us)        ← -1.4%
    M_LEQ_4096:   514.7us   (was 530.91us)        ← -3.0%
    M_LEQ_8192:   949.3us   (was 961.14us)        ← -1.2%
    M_LEQ_16384: 1795.6us   (was 1825.67us)       ← -1.6%
```

Plus minor wins at C=128/M_LEQ_2048 and C=4096/M_LEQ_{1024, 8192}.

**Bench (warm cache, 3rd sample after refresh)** — all 315 pytest passing:

| Shape | R3 | R4 | Δ |
|---|---:|---:|---:|
| C=32768/M=4096  | 0.5330 | **0.5183** | -3% |
| C=32768/M=8192  | 0.9663 | **0.9542** | -1% |
| C=32768/M=16384 | 1.8295 | **1.8047** | -1% |
| C=32768/M=2048  | 0.3124 | **0.3089** | -1% |
| C=4096/M=4096   | 0.0688 | **0.0684** | -1% |

13/15 wins vs HIP retained; leads on the C=32768 buckets all widened.

### Noise on `C=512/M=4096`

Bench showed `0.0254 / 0.0295 / 0.0287 ms` across three back-to-back R4 runs (config byte-identical to R3). The R4 tuner's own `do_bench` measured **0.024 ms** for this same config — so **R3's 0.0225 ms was an optimistic-side outlier of a noisy distribution**, not a regression. Documented in the report's noise note.

### Key insight

A bucket reported `[unchanged]` after one sweep is **not** at the global optimum — it just means the random walk hadn't yet stumbled into a better basin. Round 4's preserved-cache + bumped-patience strategy provably exhausted the remaining space (max-remaining ≤ patience for every shape) and unlocked wins that R3 couldn't see.

---

## Final consolidation

### User instruction

> "clean up all the artefacts and copy all the updated tuning automations, tools, reports, markdowns, etc. to /home/anguyenh/aiter/triton-misc/tuning_results_mhc"

### Layout produced (29 MB total)

```
triton-misc/tuning_results_mhc/
├── triton_3_6_0_retune_report.md   # full R0-R4 engineering memo (47 KB)
├── PR_description.md               # PR write-up (13 KB)
├── tuning_report.md                # original Triton 3.5.1 tuning memo (67 KB)
├── session_round3_round4_chat.md   # this file
├── tune_mhc.py                     # multi-GPU tuner driver
├── refresh_mhc_configs.py          # picks global best across all best_configs_* files
├── bench_fused_mhc_configs.py      # standalone bench helper
├── bench_sk_configs.py             # ditto for sinkhorn
├── run_tune_all.sh                 # full-sweep orchestrator (PATIENCE env override)
├── run_tune_problem_shapes.sh      # focused-retune orchestrator
├── tuning_space_mhc.json           # 5760-config search space (num_stages=[1,2])
├── benchmarks/                     # legacy bench scripts
├── backup_triton351_20260430/      # pre-3.6.0 (R0) prod configs
├── backup_gfx950_pre_round{2,3,4}_*/  # per-round backups
├── logs/                           # 26 logs: tune_C*.log, bench_*.log, pytest_*.log, run_tune_all_round*.log
└── raw_tuner_outputs/              # 41 files
    ├── best_configs_mhc_*.json (16)    # R2/R2-focused/R3/R4 raw winners
    └── tested_configs_mhc_*.json (25)  # per-shape tuner cache (resume state)
```

### Cleanup performed

- 16 `best_configs_mhc_*.json` and 25 `tested_configs_mhc_*.json` (28 MB) moved off workspace root → `raw_tuner_outputs/`.
- `__pycache__/` removed from both source and destination.
- Workspace root verified clean (no stray `*.json` or `*.log`).

### Untouched (intentional)

- Repo's `tuning_results/` kept intact — still the live working location for these tools.
- Production configs at `aiter/ops/triton/configs/gfx950-MHC_FUSED_SINKHORN-C=*.json` — part of the codebase, owned by the PR.

---

## Cumulative score across rounds

| Metric | R0 (Triton 3.5.1) | R2 final | R3 | R4 final |
|---|---:|---:|---:|---:|
| Wins vs HIP | 7/15 | 13/15 | 13/15 | **13/15** |
| Worst HIP gap | 1.43× (C=512/M=1024) | 1.39× | **1.25×** | 1.26× (noise-equivalent) |
| Best Triton lead | 1.31× (C=32768/M=8192) | 1.43× | 1.42× | **1.44×** (C=32768/M=16384) |
| Search-space coverage | <5% | ~17% | ~72% | **~100%** of remaining → effectively exhausted |

The only HIP wins remaining are the two smallest-M shapes at C=512:

- `C=512/M=1024`: HIP 1.25-1.26×. Structural launch-overhead bound (per `triton_3_6_0_retune_report.md` §8).
- `C=512/M=2048`: HIP 1.02× — tied within run-to-run noise.

Closing those gaps requires **kernel-level changes** (eliminate split-K dispatch at small M, or fuse epilogue back into main kernel for small-M regime), out of scope for this PR. Flagged as a known follow-up.

---

## Lessons captured (R3/R4 specific)

1. **Resume-sweep > focused retune** for finding configs in unsampled corners. Preserving the `tested_configs_*` cache and re-running the full sweep (with bumped patience) is more thorough than clearing the cache for a few "problem" shapes.
2. **`refresh_mhc_configs.py`'s "global-best across all files" semantics is the load-bearing piece** of the iterative-resume strategy. Across 16 timestamped tuner output files, refresh deterministically picks the per-shape minimum without any manual merging.
3. **`max remaining ≤ patience` ≈ exhausting the bucket.** The right setting for the *finishing* round.
4. **Cleared Triton cache adds startup cost, not steady-state cost.** First-run pytest after cache wipe took 60 s vs 6-10 s warm. Always run a warm-up bench before quoting numbers when the Triton cache has just been wiped.
5. **`[unchanged]` after one sweep is a soft signal, not proof of global optimality.** R3's `[unchanged]` on `C=32768` was overturned by R4's deeper coverage of the remaining 50% of that bucket's space.

---

## Next steps (not done)

- Update `PR_description.md` with the R3/R4 final numbers (still reflects the R2-final state).
- Consider expanding the search space (`BLOCK_M ≥ 256`, `BLOCK_K=512`, etc.) for a R5 if a future Triton/ROCm bump warrants re-tuning.
- Kernel-level work to close the C=512 small-M gap (out of scope for this PR).
