# Triton 3.6.0 + retuned MHC vs HIP `aiter.mhc_pre`

GPU: AMD MI300X (gfx942) · dtype: bf16 · `n=4`, `sinkhorn_iters=20`
Container: `amdsiloai/pytorch-xdit:v26.4` (torch `2.9.1+rocm`, **triton `3.6.0`**)
Triton kernel: retuned configs in `aiter/ops/triton/configs/gfx942-MHC_FUSED_SINKHORN-C=*.json`
HIP kernel: `aiter.mhc_pre` (`mhc_pre_gemm_sqrsum` + `mhc_pre_big_fuse`)
Bench: `triton.testing.do_bench` from a fresh process per shape, 5 reps, **min** of reps.

## Time (μs) — lower is better

| C / M     | 1024 | 2048 | 4096 | 8192 | 16384 |
|----------:|-----:|-----:|-----:|-----:|------:|
| **512** Triton  | 39.7  | 38.6  | 46.5  | 50.4  | **59.6**  |
| **512** HIP     | 27.9  | 29.1  | 30.1  | 45.1  | 73.9      |
| ratio T/H       | 1.42× | 1.33× | 1.55× | 1.12× | **0.81×** |
| **1024** Triton | 42.5  | 42.1  | 48.3  | **61.1** | **118.7** |
| **1024** HIP    | 29.9  | 30.2  | 41.8  | 67.7     | 168.9     |
| ratio T/H       | 1.42× | 1.39× | 1.15× | **0.90×** | **0.70×** |
| **4096** Triton | 68.3  | 77.5  | **119.6** | **246.8** | **510.8** |
| **4096** HIP    | 45.8  | 66.7  | 125.5     | 293.0     | 543.2     |
| ratio T/H       | 1.49× | 1.16× | **0.95×** | **0.84×** | **0.94×** |
| **32768** Triton| **271.8** | 535.6 | 1060.2 | 2027.4 | 4308.8 |
| **32768** HIP   | 287.5     | 533.4 | 1071.1 | 2053.5 | 4068.7 |
| ratio T/H       | **0.95×** | 1.00× | 0.99× | 0.99× | 1.06× |

**Bold** = Triton beats HIP, _italic_ candidates = parity (≤ 1% gap).

## Summary (20 shapes, default suite + C=1024)

| | Triton/HIP |
|---|---|
| Triton wins (T/H < 0.98) | **7** |
| Ties (±2%)               | **3** |
| Losses                   | 10    |
| Best                     | **0.70×** (C=1024, M=16384) |
| Worst                    | 1.55×   (C=512, M=4096)    |
| Geomean                  | ~1.07×                      |

## Progress across iterations

| Stage | Container | Triton | wins | ties | losses | worst | geomean |
|------|-----------|--------|------|------|--------|-------|---------|
| Pre-retune (initial tune) | `rocm/pytorch-private:triton_internal_issue_1801` | 3.5.1 | 0 | 4 | 16 | 2.92× | ~1.30× |
| Pre-retune              | `amdsiloai/pytorch-xdit:v26.4` | 3.5.1 | 2 | 7 | 11 | 2.49× | ~1.27× |
| Pre-retune              | xdit-v26.4 | **3.6.0** | 5 | 2 | 13 | 1.89× | ~1.18× |
| **Post-retune** (this report) | xdit-v26.4 | **3.6.0** | **7** | **3** | **10** | **1.55×** | **~1.07×** |

## A/B test against the Triton 3.6 baseline (current commit's pre-retune configs)

Run with `tuning_results_mhc/abtest_configs.py --reps 7 --margin 0.05`,
each (M, C) bucket comparing OLD = Triton 3.6 baseline vs NEW = retuned, in-process.
**Result: 13 wins, 12 ties (kept old), 0 regressions.** Highlights:
- C=4096 `M_LEQ_4096`: 210μs → 125μs (1.68×)
- C=4096 `M_LEQ_8192`: 362μs → 256μs (1.42×)
- C=4096 `M_LEQ_16384`: 835μs → 529μs (1.58×)
- C=32768 `M_LEQ_16384`: 8090μs → 4695μs (1.72×)
- C=1024 `M_LEQ_16384`: 143μs → 120μs (1.19×)
- C=512 `M_LEQ_1024,2048`: 1.20×, 1.29×

## Notes

- HIP path: `n==4`, `C ≥ 512` divisible by 128. `C=128` cannot be benchmarked against HIP.
- Each row of the bench is run in a separate Python process (`isolated_hip_bench.py`)
  to remove autotune-cache pollution; min-of-reps is used because cold-start of `do_bench`
  can over-estimate later-warm shapes.
- Correctness: all rows pass `atol=4e-2/8e-2`; a few rows show 0.0–2.0% of elements
  drifting beyond `rtol`, consistent with bf16 accumulation order in the sinkhorn
  iteration loop.
- Logs: `triton36_baseline_default.log`, `triton36_baseline_C1024.log`,
  `triton36_tuned_vs_hip_min.{log,json}`, `abtest_t36_vs_new.log`.
