# Tuned Triton MHC vs HIP `aiter.mhc_pre` — `amdsiloai/pytorch-xdit:v26.4`

GPU: AMD MI300X (gfx942) · dtype: bf16 · `n=4`, `sinkhorn_iters=20`
Container: `amdsiloai/pytorch-xdit:v26.4` (torch `2.9.1+git20a8ad0`)
Triton kernel: tuned configs in `aiter/ops/triton/configs/gfx942-MHC_FUSED_SINKHORN-C=*.json`
HIP kernel: `aiter.mhc_pre` (`mhc_pre_gemm_sqrsum` + `mhc_pre_big_fuse`)
Benchmark: `triton.testing.do_bench` median, single process per row.

> Container prep: removed pre-installed `amd-aiter` (`pip uninstall -y amd-aiter`,
> `rm -rf /app/external/aiter`), then `pip install -e .` from `/workspace/aiter`,
> `pip install matplotlib`, and rebuilt `module_mhc.so` (the cached one was ABI-bound
> to torch 2.10).

> HIP path requires `n==4`, `C ∈ {512, 1024, 2048, 4096, …}` divisible by 128, and `C ≥ 512`.
> So `C=128` is excluded from the comparison.

## Time (ms) — lower is better

### C = 512
| M     | Triton | HIP   | Triton / HIP |
|------:|-------:|------:|-------------:|
|  1024 | 0.123  | 0.090 | 1.37×        |
|  2048 | 0.119  | 0.088 | 1.35×        |
|  4096 | 0.116  | 0.084 | 1.37×        |
|  8192 | 0.111  | 0.045 | 2.43×        |
| 16384 | 0.101  | 0.075 | 1.34×        |

### C = 1024
| M     | Triton | HIP   | Triton / HIP |
|------:|-------:|------:|-------------:|
|  1024 | 0.043  | 0.039 | 1.10×        |
|  2048 | 0.112  | 0.087 | 1.29×        |
|  4096 | 0.108  | 0.084 | 1.28×        |
|  8192 | 0.077  | 0.068 | 1.13×        |
| 16384 | **0.124** | 0.169 | **0.73× (Triton wins)** |

### C = 4096
| M     | Triton | HIP   | Triton / HIP |
|------:|-------:|------:|-------------:|
|  1024 | 0.185  | 0.079 | 2.34×        |
|  2048 | 0.176  | 0.071 | 2.49×        |
|  4096 | 0.239  | 0.126 | 1.89×        |
|  8192 | **0.273** | 0.294 | **0.93× (Triton wins)** |
| 16384 | 0.539  | 0.545 | 0.99× (≈parity) |

### C = 32768
| M     | Triton | HIP   | Triton / HIP |
|------:|-------:|------:|-------------:|
|  1024 | 0.303  | 0.307 | 0.99× (≈parity) |
|  2048 | 0.543  | 0.538 | 1.01×        |
|  4096 | 1.118  | 1.076 | 1.04×        |
|  8192 | 2.116  | 2.072 | 1.02×        |
| 16384 | 4.641  | 4.072 | 1.14×        |

## Summary (20 shapes)

| | Triton/HIP |
|---|---|
| Best (Triton wins) | **0.73×** (C=1024, M=16384) |
| Worst              | 2.49×   (C=4096, M=2048)    |
| Geomean            | ~1.27×                      |
| Triton wins / parity / loss | 2 / 4 / 14 |

## vs. previous container (`rocm/pytorch-private:triton_internal_issue_1801`)

The xdit `v26.4` image moves results meaningfully closer to HIP:
- Large-C upper-M region used to be the worst (C=32768/M=16384: 8.08ms → **4.64ms**, ~2× Triton speedup).
- Triton now beats HIP on `C=1024, M=16384` and `C=4096, M=8192`.
- Worst-case slowdown improves from 2.92× to 2.49×.
- Same gap remains at moderate C (512/1024/4096) for low-mid M, where HIP's hand-fused
  big_fuse still dominates the bandwidth-bound apply step.

## Notes

- Default suite: `M ∈ {1024, 2048, 4096, 8192, 16384}`, `C ∈ {512, 4096, 32768}`
  (run via `bench_mhc.py --with-hip -metric time`).
- C=1024 sweep is captured separately; C=128 cannot be benchmarked against HIP.
- Correctness: all rows pass `atol=4e-2/8e-2`; a few rows show 0.0–1.7% of elements
  drifting beyond `rtol`, consistent with bf16 sinkhorn-iteration accumulation order.
- Logs: `tuning_results_mhc/benchmarks/triton_vs_hip_xdit_default.log` and
  `triton_vs_hip_xdit_C1024.log`.
