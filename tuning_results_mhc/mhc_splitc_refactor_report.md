# MHC Apply-Step Split-C Kernel: Refactor, Tuning, and C=7168 Analysis

A compact summary of the work that introduced a dedicated `(M, C)`-parallel
apply kernel for the MHC pre-stream, generalized the dispatch via a tunable
`USE_REDUCE_SPLITC` flag, ran a multi-GPU autotune sweep on gfx950, and
analyzed how the resulting configs serve C=7168 (DeepSeek-V3 hidden dim per
stream).

---

## 1. Background

`mhc()` (Manifold-constrained Hyper Connection) does three streams:

- **Pre**: GEMM `(M, K) → (M, N_total)`, slice the pre-block `(M, n)`, sigmoid,
  then **apply step** `layer_input[m, c] = Σ_i pre_mix[m, i] * x[m, i*C + c]`.
- **Post**: sigmoid on the post-block `(M, n)`, scale, write to `comb_mix`.
- **Res**: log-domain Sinkhorn-Knopp on the `(M, n²)` res-block, write to
  `comb_mix`.

The bottleneck addressed here was the **apply step** in the pre stream. Before
this work it ran inline inside the fused kernel as a 3D-broadcast reduction
over `(BLOCK_M, N_POW2, BLOCK_C)`, with **no C-axis parallelism** beyond the
single inner loop in each fused-kernel program. At large C (>=4 K), that
inline pattern leaves CUs idle.

---

## 2. Tier-2 split: introduce `_mhc_reduce_splitc_kernel`

### 2.1 The split

Added a dedicated apply kernel gridded as `(cdiv(M, BLOCK_M), cdiv(C, BLOCK_C))`,
parallel over both `M` and `C` axes:

```python
@triton.jit
def _mhc_reduce_splitc_kernel(
    x_ptr, pre_mix_ptr, out_ptr,
    M, C: tl.constexpr, n: tl.constexpr,
    stride_xm, stride_xk, stride_pm, stride_pn, stride_om, stride_oc,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, N_POW2: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_c = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    i_n = tl.arange(0, N_POW2)
    pre_mix_2d = tl.load(...)            # (BLOCK_M, N_POW2)
    _mhc_apply_pre_mix_tile(             # shared helper: see §5
        x_ptr, out_ptr, pre_mix_2d, rm, rc, i_n,
        M, C, n, stride_xm, stride_xk, stride_om, stride_oc,
    )
```

The fused kernel writes `pre_mix` to a small `(M, n)` fp32 intermediate
buffer; the splitc kernel consumes it and produces `layer_input`.

### 2.2 First-cut perf

The dedicated kernel won at large C, but **regressed at small C** because the
extra launch overhead (~10–20 us) dominated the kernel runtime. That motivated
the hybrid dispatch in §3.

---

## 3. Hybrid dispatch: `USE_REDUCE_SPLITC`

### 3.1 Mechanism

A `tl.constexpr` boolean gate inside both `_mhc_fused_kernel` and
`_mhc_fused_reduce_kernel`:

```python
if is_pre_program:
    pre_mix = tl.sigmoid(out) + hc_pre_eps
    if USE_REDUCE_SPLITC:
        # Stash pre_mix to the (M, n) buffer for the dedicated kernel.
        tl.store(pre_mix_ptr + ..., pre_mix, mask=...)
    else:
        # Apply step runs inline via 3D broadcast - skips a kernel launch.
        ...for c0 in range(0, C, BLOCK_C): _mhc_apply_pre_mix_tile(...)
```

The wrapper (`aiter/ops/triton/fusions/mhc.py`) reads `USE_REDUCE_SPLITC` from
the config JSON, with a C-based fallback when the JSON entry doesn't specify:

```python
DEFAULT_REDUCE_SPLITC_THRESHOLD = 4096
use_reduce_splitc = bool(
    config.pop("USE_REDUCE_SPLITC", C >= DEFAULT_REDUCE_SPLITC_THRESHOLD)
)
```

### 3.2 Threshold derivation

A/B sweep on gfx950 across `C ∈ {128, 512, 1024, 2048, 4096, 32768}`,
`M ∈ {1024..16384}`, n=4, sinkhorn=20:

| C bucket | inline winner | dedicated winner | call |
|---:|---:|---:|---|
| 128 | 1.10x – 1.40x | – | inline uniformly |
| 512 | 1.15x – 1.35x | – | inline uniformly |
| 1024 | 1.10x – 1.52x | – | inline uniformly |
| 2048 | 1.05x – 1.52x at M extremes | up to 1.07x at mid-M | **inline (asymmetric risk)** |
| 4096 | – | 1.06x – 1.74x | **dedicated uniformly** |
| 32768 | – | 1.10x – 1.30x | **dedicated uniformly** |

→ `DEFAULT_REDUCE_SPLITC_THRESHOLD = 4096`. Per-shape JSON entries override
this; the fallback only fires when callers skip the JSON (custom configs,
out-of-bucket shapes). C=7168 hits the fallback (see §7).

---

## 4. Tuning infrastructure

### 4.1 Tuning space

`tuning_results/tuning_space_mhc.json` (`fused_mhc` section):

```json
{
  "BLOCK_M":          [32, 64, 128, 256],
  "BLOCK_N":          [16],
  "BLOCK_K":          [32, 64, 128, 256],
  "NUM_KSPLIT":       [1, 2, 4, 8],
  "waves_per_eu":     [0, 1, 2],
  "num_stages":       [1],
  "num_warps":        [2, 4, 8],
  "USE_REDUCE_SPLITC":[false, true],
  "BLOCK_M_SPLITC":   [16, 32, 64, 128],
  "BLOCK_C_SPLITC":   [32, 64, 128, 256]
}
```

`tune_mhc.py` prunes the cross product: when `USE_REDUCE_SPLITC=False`, the
splitc block sizes are skipped (one row per GEMM tuple instead of N×N).

### 4.2 Multi-GPU run

Launched on GPUs 4–7 of the dev container with `setsid nohup` (tmux is
unavailable in `rocm/pytorch:rocm7.1_ubuntu22.04` due to libtinfo6 conflict):

```bash
setsid nohup bash tuning_results/run_tune_all.sh > tuning_results/logs/run_all.log 2>&1 &
```

`run_tune_all.sh` shards C buckets across `--workers 4`. Per-(M, C) winners
land in `tuning_results/best_configs_mhc_fused_mhc_<hres>_M*_n*_C*.json`.

### 4.3 Merging winners into production

`tuning_results/refresh_mhc_configs.py`:

- Loads tuner outputs for all `(M, C)` buckets.
- Strips `BLOCK_N` (always derived from `n_squared` at launch) and any
  `None`-valued split-C block entries.
- Defaults `--arches` to the active GPU arch only (originally was `gfx942 +
  gfx950`, which silently overwrote pre-existing gfx942 winners with gfx950
  results - reverted that and gated to active arch).
- Writes back to `aiter/ops/triton/configs/{arch}-MHC_FUSED_SINKHORN-C={C}.json`.

---

## 5. Code organization refactor

After tuning settled, the apply-tile body was duplicated in three places
(two inline branches + the standalone splitc kernel). Extracted a single
`@triton.jit` device helper - pure code motion:

```python
@triton.jit
def _mhc_apply_pre_mix_tile(
    x_ptr, out_ptr, pre_mix_2d, rm, rc, i_n,
    M, C: tl.constexpr, n: tl.constexpr,
    stride_xm, stride_xk, stride_om, stride_oc,
):
    """One (M-tile, C-tile) of: out[rm, rc] = Σ_i pre_mix_2d[rm, i] * x[rm, i*C+rc]."""
    x_tile = tl.load(...).to(tl.float32)
    li_acc = tl.sum(pre_mix_2d[:, :, None] * x_tile, axis=1)
    tl.store(out_ptr + ..., li_acc.to(out_ptr.dtype.element_ty), mask=...)
```

Both inline branches and `_mhc_reduce_splitc_kernel` collapse to a one-liner
helper call inside their respective tile loops. ~60 lines of duplication
removed across 3 callers.

### 5.1 Naming evolution

To make the dispatch self-documenting (this is the C-axis split, not "the
apply kernel"):

| Old | New |
|---|---|
| `USE_APPLY_KERNEL` | `USE_REDUCE_SPLITC` |
| `BLOCK_M_APPLY` | `BLOCK_M_SPLITC` |
| `BLOCK_C_APPLY` | `BLOCK_C_SPLITC` |
| `_mhc_apply_kernel` | `_mhc_reduce_splitc_kernel` |
| `use_apply_kernel` (wrapper local) | `use_reduce_splitc` |
| `DEFAULT_APPLY_KERNEL_THRESHOLD` | `DEFAULT_REDUCE_SPLITC_THRESHOLD` |

Rename was applied across kernel module, wrapper, `__init__.py`, tuning
space JSON, `tune_mhc.py`, `refresh_mhc_configs.py`,
`bench_apply_path_ab.py → bench_splitc_path_ab.py`, and 76 keys across the
5 `gfx950-MHC_FUSED_SINKHORN-C=*.json` configs (40+ buckets).

### 5.2 Verification

- `pytest op_tests/triton_tests/fusions/test_mhc.py -x -q` → **315 / 315
  passed** (covers both `USE_REDUCE_SPLITC` paths).
- A/B benches (helper-extracted vs helper-inlined back, identical dispatch
  + JSON):
  - >50 us workloads: <0.5% delta (e.g. 16384/4096: 0.3398 vs 0.3404 ms;
    16384/32768: 2.5109 vs 2.5202 ms).
  - <50 us workloads: <5% delta - within `do_bench` noise floor at
    sub-50 us scale.
  - Conclusion: pure code motion, value-preserving.

### 5.3 BLOCK_M_SPLITC heuristic note

Across the 13 splitc-using config entries, only 4 / 13 have
`BLOCK_M_SPLITC == BLOCK_M`. The tuner consistently picks
`BLOCK_M_SPLITC ≤ BLOCK_M`, often 1/4 to 1/16 smaller at large C - because
the splitc kernel's per-block working set is `BLOCK_M × N_POW2 × BLOCK_C` fp32
in registers, and large `BLOCK_C` (128–256 at C ≥ 4 K) forces `BLOCK_M` down
to avoid VGPR spills. The fused-kernel `BLOCK_M` is sized for the GEMM where
that pressure doesn't apply, so the two have **different optima**.

→ Keep the current default `32 if M >= 32 else next_power_of_2(M)`. A modest
improvement for the C ≥ 4096 fallback path would be:
`16 if C >= 4096 else (32 if M >= 32 else next_power_of_2(M))`. Not blocking;
the fallback only fires for shapes outside the tuned C buckets.

---

## 6. Final perf at tuned C buckets (gfx950, n=4, sinkhorn=20)

| Shape (M, C) | Triton (ms) | HIP (ms) | Winner |
|---|---:|---:|:---:|
| 16384 / 512 | 0.0465 | 0.0718 | **Triton +54%** |
| 16384 / 4096 | 0.3405 | 0.3398 | tie |
| 16384 / 32768 | 2.5109 | 2.5906 | **Triton +3%** |
| 8192 / 4096 | 0.1833 | 0.1940 | **Triton +6%** |
| 8192 / 32768 | 1.2307 | 1.3516 | **Triton +10%** |
| 4096 / 32768 | 0.6559 | 0.7122 | **Triton +9%** |

Triton matches or beats HIP everywhere we directly tuned. Untuned C buckets
are addressed in §7.

---

## 7. C = 7168 analysis (DeepSeek-V3 hidden dim per stream)

### 7.1 Config selection flow

C=7168 is **not** in the tuned bucket set `{128, 512, 1024, 4096, 32768}`.
`get_mhc_config` falls back to the **largest C-bucket ≤ 7168 → C=4096
config**. Loaded values per M:

| M | BLOCK_M | BLOCK_K | NUM_KSPLIT | num_warps | BLOCK_M_SPLITC | BLOCK_C_SPLITC | USE_REDUCE_SPLITC |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1024 | 64 | 128 | 8 | 4 | 64 | 64 | True |
| 2048 | 32 | 128 | 8 | 4 | 16 | 256 | True |
| 4096 | 64 | 64 | 8 | 4 | 16 | 128 | True |
| 8192 | 64 | 256 | 2 | 4 | 16 | 256 | True |
| 16384 | 128 | 128 | 2 | 8 | 16 | 256 | True |

All entries dispatch to splitc. The `DEFAULT_REDUCE_SPLITC_THRESHOLD = 4096`
fallback would also dispatch to splitc (7168 ≥ 4096), so default and JSON
agree.

### 7.2 Inline vs dedicated A/B at C=7168

```
     M    inline(ms)   dedi(ms)  speedup  best
  1024      0.1520     0.0647    2.35x   dedi
  2048      0.1321     0.0919    1.44x   dedi
  4096      0.2879     0.1652    1.74x   dedi
  8192      0.3603     0.3143    1.15x   dedi
 16384      0.6052     0.5721    1.06x   dedi
```

Dedicated wins on every M (1.06x – 2.35x). Threshold is correctly placed.

### 7.3 Triton vs HIP at C=7168

| M | Triton (ms) | HIP (ms) | Gap |
|---:|---:|---:|---:|
| 1024 | 0.0647 | 0.0518 | **HIP +25%** |
| 2048 | 0.0930 | 0.0799 | **HIP +16%** |
| 4096 | 0.1622 | 0.1564 | HIP +3.7% |
| 8192 | 0.3101 | 0.2916 | HIP +6.3% |
| 16384 | 0.5718 | 0.5677 | HIP +0.7% |

Triton lags HIP at every M, with the gap widening at small M. C=7168 is the
**only** measured C where Triton consistently lags - at every directly-tuned
C (512, 4096, 32768) Triton matches or beats HIP.

### 7.4 Root cause

The C=4096 winners were tuned for `K = n*C = 16384`. At C=7168, the GEMM has
`K = 28672` (1.75x larger), but inherits:

- `BLOCK_K = 128` → 224 K-iterations per tile at C=7168 vs 128 at C=4096.
  A larger `BLOCK_K = 256` would halve the loop count; at K=16384 the tuner
  picked 128 because of register pressure trade-offs that change at K=28672.
- `NUM_KSPLIT = 8` → 8 chunks of 3584 at C=7168 (awkward, not power-of-2)
  vs 8 chunks of 2048 at C=4096 (clean).
- Splitc-side block sizes (`BLOCK_M_SPLITC=16`, `BLOCK_C_SPLITC=256`) carry
  over fine - per-block work is independent of K.

The gap is **smallest at large M** (16384: 0.7%) where the GEMM amortizes,
and **widest at small M** (1024: 25%) where launch + setup overhead
dominates and the 1.75x-longer K-loop hits hardest.

### 7.5 Recommendation

If C=7168 is a production shape (e.g., DeepSeek-V3 / V3-LITE), add a tuned
config:

```bash
# From /home/anguyenh/aiter
HIP_VISIBLE_DEVICES=4,5,6,7 setsid nohup python tuning_results/tune_mhc.py \
  --kernel fused_mhc --hres-mode sinkhorn \
  --M 1024 2048 4096 8192 16384 --n 4 --C 7168 \
  --workers 4 > tuning_results/logs/tune_C7168.log 2>&1 &

# After it completes:
python tuning_results/refresh_mhc_configs.py
```

Expected gains:

- Fused-kernel side: `BLOCK_K`, `NUM_KSPLIT` re-tuned for K=28672 should
  close most of the small-M gap (~10–20%).
- Splitc-kernel side: `BLOCK_C_SPLITC = 224` (a divisor of 7168 → 32
  perfectly-aligned C-blocks vs 28 with masking) plus a possibly larger
  `BLOCK_M_SPLITC` could pick up another few %.

Estimated post-tune state at C=7168: Triton ≈ HIP at small M, +5–10% over
HIP at M ≥ 4096 (consistent with what was achieved at C=4096 and C=32768).

---

## 8. Files of record

- `aiter/ops/triton/_triton_kernels/fusions/mhc.py` - kernels and helper.
- `aiter/ops/triton/fusions/mhc.py` - wrapper + dispatch threshold.
- `aiter/ops/triton/configs/gfx950-MHC_FUSED_SINKHORN-C=*.json` - 5 tuned
  per-C configs (C ∈ {128, 512, 1024, 4096, 32768}).
- `tuning_results/tuning_space_mhc.json` - autotune search space.
- `tuning_results/tune_mhc.py` - autotune driver.
- `tuning_results/refresh_mhc_configs.py` - merge tuner winners into prod
  config JSONs.
- `tuning_results/bench_splitc_path_ab.py` - inline-vs-dedicated A/B
  utility (also useful for arch bring-up).
- `tuning_results/run_tune_all.sh` - multi-GPU shard launcher.
