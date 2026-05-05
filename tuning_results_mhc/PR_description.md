Co-authors: @waqahmed-amd-fi @anhminhnguyenhoang

## Motivation

Implements **mHC (manifold-constrained Hyper Connection)** ([arXiv:2512.24880v2](https://arxiv.org/abs/2512.24880v2)) as fused Triton kernels for AMD Instinct GPUs (MI300X / MI355X).

mHC constrains dynamic residual matrices to the Birkhoff polytope via iterative Sinkhorn-Knopp (SK) normalization to prevent gradient explosions in multi-stream residual connections.

## Technical Details

`mhc()` fuses one mHC layer's three streams (`x -> phi_pre, phi_post, phi_res`) into a Triton entry point with two paths chosen at config-load time by the tuned `NUM_KSPLIT`:

1. **`NUM_KSPLIT == 1`** -- single fused kernel `_mhc_fused_kernel` runs RMSNorm, the 3-stream phi projection, per-stream activation (sigmoid+bias for pre/post, in-kernel log-domain Sinkhorn-Knopp for res), and the pre-stream apply step `layer_input = pre_mix * x` inline.
2. **`NUM_KSPLIT > 1`** -- split-K GEMM `_mhc_fused_split_kernel` writes per-K partials into a unified `(NUM_KSPLIT, M, n+n+n^2)` accumulator (one MFMA pass over `x` covers all 3 streams); `_mhc_reduce_apply_kernel` over `(M_blocks, C_blocks)` then reduces partials, runs all 3 stream activations (with sinkhorn), and the apply step in one launch. A `RES_PID_C` constexpr keeps post/res streams pinned to a single workgroup (one CDNA4 CU) when `C` fits one tile.

`sinkhorn_iters` controls the in-kernel Sinkhorn-Knopp iteration count (default 20; `0` returns raw `H^res` logits, skipping SK).

**Layout**:

- Kernels: `aiter/ops/triton/_triton_kernels/fusions/mhc.py` (`_mhc_fused_kernel`, `_mhc_fused_split_kernel`, `_mhc_reduce_apply_kernel`).
- API: `aiter/ops/triton/fusions/mhc.py` -- `mhc()`.
- Configs: `aiter/ops/triton/configs/{arch}-MHC_FUSED_SINKHORN-C=*.json` (gfx942/gfx950 buckets: C in {128, 512, 1024, 4096, 32768}; each M-bucket encodes `BLOCK_M/N/K/C`, `NUM_KSPLIT`, `num_warps`, `num_stages`, `waves_per_eu`).
- Reference + tests: `op_tests/triton_tests/utils/mhc_ref.py`, `op_tests/triton_tests/fusions/test_mhc.py`.
- Benchmark: `op_tests/op_benchmarks/triton/bench_mhc.py` (`--with-hip` for side-by-side; needs `--dtype bf16`, n=4, C >= 512).

**Further docs:** https://amd.atlassian.net/wiki/spaces/AIG/pages/1388423373/Triton+Kernels+mHC

## Test Plan

`op_tests/triton_tests/fusions/test_mhc.py` (**315 tests**): correctness vs PyTorch reference (bf16/fp16) for both inline (`NUM_KSPLIT==1`) and split-K (`NUM_KSPLIT>1`) branches, in-kernel Sinkhorn-Knopp convergence + doubly-stochastic check, edge cases (zero input, large values, epsilon / alpha / sinkhorn_iters sweeps).

## Test Setup

Replace 'your_NTID' as needed:

```
docker run -it --rm --name mhc-dev --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/your_NTID/aiter:/workspace/aiter -w /workspace/aiter \
  amdsiloai/pytorch-xdit:v26.4 \
  bash

# inside docker, remove the preinstalled amd_aiter repository and navigate to the one in your workspace
python setup.py develop
pytest op_tests/triton_tests/fusions/test_mhc.py -v -s
python op_tests/op_benchmarks/triton/bench_mhc.py --with-hip -metric all
```

## Test Result

315 / 315 pytest pass. Tolerances: 1e-2 small C, 5e-2 large C (bf16 accumulation).

### gfx950 (MI355X) -- retuned for Triton 3.6.0

Environment: Ubuntu 24.04.4, ROCm 7.12.0, PyTorch 2.9.1 (built against ROCm 7.12.60610), **Triton 3.6.0**, image `amdsiloai/pytorch-xdit:v26.4`.

Numbers below are `op_tests/op_benchmarks/triton/bench_mhc.py --with-hip -metric all` after three rounds of tuning (full sweep with `num_stages in [1,2]`, patience=500, plus a resume-sweep with patience=2000 on the partially-explored cache to cover the unexplored ~83% of the 5760-config search space):

```
          M    n        C  triton_time(ms)  hip_time(ms)  triton_throughput(TFLOPS)  hip_throughput(TFLOPS)  triton_bandwidth(GB/s)  hip_bandwidth(GB/s)  triton_arithmetic_intensity(FLOP/byte)  hip_arithmetic_intensity(FLOP/byte)
0    1024.0  4.0    512.0         0.020153      0.016057                   5.748840                7.376707              472.821411           607.809761                               12.167543                            12.167543
1    2048.0  4.0    512.0         0.020913      0.020476                  11.225536               11.388369              918.262915           931.745368                               12.230377                            12.230377
2    4096.0  4.0    512.0         0.022511      0.030922                  19.822169               15.115359             1571.649548          1231.534404                               12.262038                            12.262038
3    8192.0  4.0    512.0         0.025538      0.043609                  35.857254               21.368988             2978.986125          1738.765696                               12.277930                            12.277930
4   16384.0  4.0    512.0         0.037371      0.071635                  49.857708               26.015790             4059.743501          2116.212260                               12.285892                            12.285892
5    1024.0  4.0   4096.0         0.028297      0.038519                  32.040165               23.546478             2687.046045          1976.189271                               11.912832                            11.912832
6    2048.0  4.0   4096.0         0.038138      0.054308                  47.708503               33.447786             3969.589265          2790.668317                               11.974531                            11.974531
7    4096.0  4.0   4096.0         0.068753      0.097658                  52.974802               37.270346             4405.168496          3102.860855                               12.005621                            12.005621
8    8192.0  4.0   4096.0         0.148486      0.193705                  48.966068               37.544728             4066.082908          3121.020240                               12.021226                            12.021226
9   16384.0  4.0   4096.0         0.282109      0.342154                  51.144884               42.508251             4252.078138          3534.889843                               12.029044                            12.029044
10   1024.0  4.0  32768.0         0.176853      0.187133                  41.122033               38.760471             3460.866442          3261.827141                               11.880859                            11.880859
11   2048.0  4.0  32768.0         0.312351      0.356332                  46.400215               40.776458             3895.796188          3408.317590                               11.942414                            11.942414
12   4096.0  4.0  32768.0         0.532992      0.716751                  54.531960               40.536664             4554.474250          3378.381000                               11.973432                            11.973432
13   8192.0  4.0  32768.0         0.966253      1.352579                  59.923825               42.831586             4996.379837          3563.350601                               11.989001                            11.989001
14  16384.0  4.0  32768.0         1.829504      2.601146                  63.400499               44.736749             5277.363764          3713.545441                               11.996801                            11.996801
```

vs HIP (`aiter.mhc_pre`): **13 Triton wins, 1 tie, 1 HIP win** across the 15-shape default suite. Biggest wins at high-M / large-C: `16384,512` **1.92x**, `8192,512` **1.71x**, `16384,32768` **1.42x**, `8192,32768` **1.40x**, `4096,32768` **1.34x**. Triton parity-or-faster on every shape with `C >= 4096` and on every shape with `M >= 4096`. The HIP-favored / tie shapes are both at small M / small C (`1024,512` HIP 1.25x; `2048,512` HIP 1.02x -- within run-to-run noise, effectively a tie), where launch overhead dominates the wall clock and HIP's hand-fused single-kernel path edges out the multi-launch Triton dispatch.

### gfx942 (MI300X) -- Triton 3.6.0 (configs from prior co-author capture)

Environment: Ubuntu 22.04, ROCm 7.2.2, PyTorch 2.9.1+rocm7.2.2 (`amdsiloai/pytorch-xdit:v26.4`), **Triton 3.6.0**.

Numbers below are min-of-5 with each shape run in a fresh process (isolated bench harness) to remove autotune-cache pollution; throughput / bandwidth / arithmetic-intensity columns are computed from the same FLOP/byte formulas `bench_mhc.py` uses, so the table is comparable to gfx950 above. **Note**: the post-refactor kernels (current PR head) have not been re-measured on MI300X from this MI355X box; gfx942 configs are the prior co-author's tune. A fresh gfx942 sweep on the post-refactor kernels is a planned follow-up.

```
          M    n        C  triton_time(ms)  hip_time(ms)  triton_throughput(TFLOPS)  hip_throughput(TFLOPS)  triton_bandwidth(GB/s)  hip_bandwidth(GB/s)  triton_arithmetic_intensity(FLOP/byte)  hip_arithmetic_intensity(FLOP/byte)
0    1024.0  4.0    512.0         0.039681      0.027936                   2.936494                4.171070              241.338273           342.802978                               12.167543                            12.167543
1    2048.0  4.0    512.0         0.038639      0.029071                   6.031368                8.016443              493.146510           655.453476                               12.230377                            12.230377
2    4096.0  4.0    512.0         0.046535      0.030047                  10.015946               15.512099              816.825529          1265.050621                               12.262038                            12.262038
3    8192.0  4.0    512.0         0.050351      0.045053                  18.513715               20.690832             1507.885683          1685.205247                               12.277930                            12.277930
4   16384.0  4.0    512.0         0.059619      0.073868                  31.271375               25.239185             2545.307771          2054.322630                               12.285892                            12.285892
5    1024.0  4.0   4096.0         0.068254      0.045774                  13.321512               19.863819             1118.248894          1667.430419                               11.912832                            11.912832
6    2048.0  4.0   4096.0         0.077456      0.066726                  23.477754               27.253139             1960.640777          2275.925306                               11.974531                            11.974531
7    4096.0  4.0   4096.0         0.119564      0.125503                  30.418737               28.979274             2533.707939          2413.808881                               12.005621                            12.005621
8    8192.0  4.0   4096.0         0.246793      0.292962                  29.473979               24.829062             2451.827985          2065.435053                               12.021226                            12.021226
9   16384.0  4.0   4096.0         0.510828      0.543159                  28.479143               26.783950             2367.531615          2226.606647                               12.029044                            12.029044
10   1024.0  4.0  32768.0         0.271851      0.287498                  26.672825               25.221164             2245.024988          2122.840117                               11.880859                            11.880859
11   2048.0  4.0  32768.0         0.535561      0.533420                  27.078275               27.186960             2267.403758          2276.504488                               11.942414                            11.942414
12   4096.0  4.0  32768.0         1.060194      1.071146                  27.357386               27.077669             2284.840790          2261.479290                               11.973432                            11.973432
13   8192.0  4.0  32768.0         2.027440      2.053475                  28.611585               28.248833             2386.486130          2356.229046                               11.989001                            11.989001
14  16384.0  4.0  32768.0         4.308759      4.068728                  26.925745               28.514205             2244.410358          2376.817356                               11.996801                            11.996801
```

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
