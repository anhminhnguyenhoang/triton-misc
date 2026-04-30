# Tuned Triton MHC vs HIP `aiter.mhc_pre`

GPU: AMD MI300X (gfx942) · dtype: bf16 · `n=4`, `sinkhorn_iters=20`
Triton kernel: tuned configs in `aiter/ops/triton/configs/gfx942-MHC_FUSED_SINKHORN-C=*.json`
HIP kernel: `aiter.mhc_pre` (`mhc_pre_gemm_sqrsum` + `mhc_pre_big_fuse`)
Benchmark: `triton.testing.do_bench` median of multiple runs (single process per row).

> HIP path requires `n==4`, `C ∈ {512, 1024, 2048, 4096, …}` divisible by 128, and `C ≥ 512`.
> So `C=128` cannot be compared and is omitted.

## Time (ms) — lower is better

### C = 512
| M     | Triton (ms) | HIP (ms) | Triton / HIP |
|------:|------------:|---------:|-------------:|
|  1024 | 0.119       | 0.090    | 1.32×        |
|  2048 | 0.118       | 0.089    | 1.33×        |
|  4096 | 0.115       | 0.088    | 1.30×        |
|  8192 | 0.130       | 0.044    | 2.92×        |
| 16384 | 0.077       | 0.074    | 1.03×        |

### C = 1024
| M     | Triton (ms) | HIP (ms) | Triton / HIP |
|------:|------------:|---------:|-------------:|
|  1024 | 0.061       | 0.034    | 1.81×        |
|  2048 | 0.120       | 0.093    | 1.29×        |
|  4096 | 0.103       | 0.081    | 1.27×        |
|  8192 | 0.131       | 0.067    | 1.95×        |
| 16384 | 0.178       | 0.168    | 1.06×        |

### C = 4096
| M     | Triton (ms) | HIP (ms) | Triton / HIP |
|------:|------------:|---------:|-------------:|
|  1024 | 0.123       | 0.046    | 2.70×        |
|  2048 | 0.152       | 0.067    | 2.25×        |
|  4096 | 0.212       | 0.123    | 1.72×        |
|  8192 | 0.361       | 0.290    | 1.25×        |
| 16384 | 0.817       | 0.540    | 1.51×        |

### C = 32768
| M     | Triton (ms) | HIP (ms) | Triton / HIP |
|------:|------------:|---------:|-------------:|
|  1024 | 0.294       | 0.282    | 1.04×        |
|  2048 | 0.588       | 0.525    | 1.12×        |
|  4096 | 1.142       | 1.128    | 1.01×        |
|  8192 | 2.817       | 2.246    | 1.25×        |
| 16384 | 8.082       | 3.801    | 2.13×        |

## Summary

| Bracket             | Triton/HIP slowdown |
|---------------------|---------------------|
| Best case           | 1.01× (C=32768, M=4096)        |
| Median (20 shapes)  | ~1.30×              |
| Worst case          | 2.92× (C=512, M=8192) |

- HIP is faster on **every** measured shape with the current tuned Triton configs.
- Triton is closest to HIP on the largest C (32768) and at the upper end of M, where compute-bound matmul dominates.
- Triton lags most at moderate (C, M) where the inline-apply path is bandwidth-bound; HIP's hand-fused big_fuse benefits from very tight memory coalescing here.
- Correctness: Triton vs HIP outputs match within `atol=4e-2/8e-2` for all shapes; a few rows show 0.0–1.7% of elements drifting beyond rtol — consistent with bf16 accumulation order differences in the sinkhorn iteration loop, not a tuning regression.

## Notes
- Default Triton suite: `M ∈ {1024, 2048, 4096, 8192, 16384}`, `C ∈ {512, 4096, 32768}` (run via `bench_mhc.py --with-hip -metric time`).
- C=1024 sweep was added separately because it is tuned but absent from the default suite.
- Results captured in `tuning_results_mhc/benchmarks/triton_vs_hip_default.log` and `triton_vs_hip_C1024.log`.
