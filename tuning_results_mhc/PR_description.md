Co-authors: @waqahmed-amd-fi @anhminhnguyenhoang

## Motivation

Implements **mHC (manifold-constrained Hyper Connection)** ([arXiv:2512.24880v2](https://arxiv.org/abs/2512.24880v2)) as fused Triton kernels for AMD Instinct GPUs (MI300X / MI355X).

mHC constrains dynamic residual matrices to the Birkhoff polytope via iterative Sinkhorn-Knopp (SK) normalization to prevent gradient explosions in multi-stream residual connections.

## Technical Details

`mhc()` fuses one mHC layer's three streams into a single Triton kernel:

1. **RMSNorm + 3-stream projection matmul** (`x → φ_pre, φ_post, φ_res`).
2. **Per-stream activation**: sigmoid for pre/post; in-kernel log-domain Sinkhorn-Knopp for res. `sinkhorn_iters` controls the iteration count (default 20; `0` returns raw `H^res` logits, skipping SK).
3. **Input apply step** for the pre stream: `layer_input = pre_mix · x`.

**Input apply 2-step dispatch.** The pre-stream apply `layer_input[m, c] = Σ_i pre_mix[m, i] · x[m, i·C + c]` runs one of two paths, picked by the `USE_REDUCE_SPLITC` constexpr:

- **Inline** (small C): 3D-broadcast reduction inside the fused kernel — avoids a ~10-20 us extra launch.
- **SplitC** (large C): fused kernel stages `pre_mix` to a small `(M, n)` fp32 buffer; a dedicated `_mhc_reduce_splitc_kernel` parallelizes over `(M, C)`.

**Layout**:

- Kernels: `aiter/ops/triton/_triton_kernels/fusions/mhc.py` (`_mhc_fused_kernel`, split-K `_mhc_fused_split_kernel` + `_mhc_fused_reduce_kernel`, `_mhc_reduce_splitc_kernel`, `_sinkhorn_knopp_log_domain_kernel`).
- API: `aiter/ops/triton/fusions/mhc.py` — `mhc()`, `sinkhorn_knopp()`.
- Configs: `aiter/ops/triton/configs/{arch}-MHC_FUSED_SINKHORN-C=*.json` (gfx942/gfx950 buckets: C ∈ {128, 512, 1024, 4096, 32768}; each M-bucket carries its tuned dispatch decision).
- Reference + tests: `op_tests/triton_tests/utils/mhc_ref.py`, `op_tests/triton_tests/fusions/test_mhc.py`.
- Benchmark: `op_tests/op_benchmarks/triton/bench_mhc.py` (`--with-hip` for side-by-side; needs `--dtype bf16`, n=4, C ≥ 512).

**Further docs:** https://amd.atlassian.net/wiki/spaces/AIG/pages/1388423373/Triton+Kernels+mHC

## Test Plan

`op_tests/triton_tests/fusions/test_mhc.py` (**315 tests**): correctness vs PyTorch reference (bf16/fp16) for both `USE_REDUCE_SPLITC` paths, split-K variant, standalone Sinkhorn-Knopp convergence + doubly-stochastic check, edge cases (zero input, large values, epsilon / alpha / sinkhorn_iters sweeps).

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

### gfx950 (MI355X)

Environment: Ubuntu 24.04.4, ROCm 7.12.0, PyTorch 2.9.1 (built against ROCm 7.12.60610), Triton 3.5.1.

```
          M    n        C  triton_time(ms)  hip_time(ms)  triton_throughput(TFLOPS)  hip_throughput(TFLOPS)  triton_bandwidth(GB/s)  hip_bandwidth(GB/s)  triton_arithmetic_intensity(FLOP/byte)  hip_arithmetic_intensity(FLOP/byte)
0    1024.0  4.0    512.0         0.023141      0.016236                   5.080602                7.215024              426.161823           594.962388                               12.167543                            12.167543
1    2048.0  4.0    512.0         0.023770      0.020421                  10.140817               11.436739              853.656722           936.341321                               12.230377                            12.230377
2    4096.0  4.0    512.0         0.026981      0.030849                  17.635786               15.109527             1440.409072          1231.300235                               12.262038                            12.262038
3    8192.0  4.0    512.0         0.028849      0.043888                  32.606566               21.264619             2676.737725          1730.896665                               12.277930                            12.277930
4   16384.0  4.0    512.0         0.050032      0.071866                  37.277101               25.946292             3036.042721          2109.725972                               12.285892                            12.285892
5    1024.0  4.0   4096.0         0.045398      0.038589                  20.159888               23.515742             1682.748869          1974.250932                               11.912832                            11.912832
6    2048.0  4.0   4096.0         0.052582      0.054525                  34.372032               33.322591             2870.414785          2785.386753                               11.974531                            11.974531
7    4096.0  4.0   4096.0         0.086092      0.097770                  42.252214               37.221031             3517.944614          3100.052950                               12.005621                            12.005621
8    8192.0  4.0   4096.0         0.172444      0.193736                  42.235693               37.560050             3516.661271          3124.930616                               12.021226                            12.021226
9   16384.0  4.0   4096.0         0.319925      0.340096                  45.391594               42.661316             3778.527019          3543.162422                               12.029044                            12.029044
10   1024.0  4.0  32768.0         0.206985      0.191060                  34.987029               37.940477             2942.724219          3194.729984                               11.880859                            11.880859
11   2048.0  4.0  32768.0         0.368481      0.355948                  39.506189               40.740945             3319.476022          3411.866389                               11.942414                            11.942414
12   4096.0  4.0  32768.0         0.656073      0.712326                  44.363220               40.704987             3700.435064          3402.195815                               11.973432                            11.973432
13   8192.0  4.0  32768.0         1.220723      1.339598                  47.391410               43.280550             3951.142510          3620.307034                               11.989001                            11.989001
14  16384.0  4.0  32768.0         2.339178      2.586346                  49.469254               44.995169             4120.452084          3740.062112                               11.996801                            11.996801
```

### gfx942 (MI300X) — retuned for Triton 3.6.0

Environment: Ubuntu 22.04, ROCm 7.2.2, PyTorch 2.9.1+rocm7.2.2 (`amdsiloai/pytorch-xdit:v26.4`), **Triton 3.6.0**.

Numbers below are min-of-5 with each shape run in a fresh process (`tuning_results_mhc/isolated_hip_bench.py`) to remove autotune-cache pollution; throughput / bandwidth / arithmetic-intensity columns are computed from the same FLOP/byte formulas `bench_mhc.py` uses, so the table is comparable to gfx950 above and to the in-process `bench_mhc.py --with-hip -metric all` output.

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

vs HIP (`aiter.mhc_pre`): **7 wins, 3 ties, 10 losses** across the default suite + C=1024 (20 shapes); Triton parity-or-faster on every shape with `C ≥ 4096` from `M=4096` upward, biggest wins on `C=1024 M=16384` (0.70×) and `C=4096 M=8192` (0.84×). Remaining HIP-favored region is small-C (512, 1024) at low-mid M, dominated by HIP's hand-fused `mhc_pre_big_fuse` apply step.

Tuning artefacts (search space, run scripts, A/B harness, isolated bench) live under `tuning_results_mhc/` in the [triton-misc](https://github.com/ROCm/triton-misc) repo.

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
