import argparse
import itertools
import json
import pandas as pd

MI350X_BW_MEM = 8e12  # TB/s
MI350X_BW_MEM_EMPIRICAL = 6e12  # TB/s
MI350X_PERF_FP8 = 4.6e15  # TFLOPS
MI350X_PERF_FP8_EMPIRICAL = 3.8e15  # TFLOPS
KERNEL_LAUNCH_OVERHEAD = 4e-6  # 3.5-4us


# def calc_arithmetic_intensity(m, n, k, dtype="fp16"):
#     if dtype == "fp16":
#         factor = 2
#     elif dtype == "fp8":
#         factor = 1
#     else:
#         raise ValueError("Unsupported dtype. Use 'fp16' or 'fp8'.")

#     return (2 * m * n * k) / (factor * (m * n + n * k + m * k))


# def calc_roofline_time(m, n, k, bound, in_us=False):
#     if bound == "compute":
#         roofline_time = (2 * m * n * k) / MI350X_PERF_FP8_EMPIRICAL  # in seconds
#         roofline_time = max(roofline_time, KERNEL_LAUNCH_OVERHEAD)
#     else:
#         bytes_moved = m * n + n * k + m * k  # in bytes
#         roofline_time = bytes_moved / MI350X_BW_MEM_EMPIRICAL  # in seconds
#         roofline_time = max(roofline_time, KERNEL_LAUNCH_OVERHEAD)

#     if in_us:
#         return roofline_time / 1e-6  # in us

#     return roofline_time  # in seconds


# def calc_roofline_throughput(m, n, k, bound, roofline_time_s=None):
#     if roofline_time_s is None:
#         roofline_time_s = calc_roofline_time(m, n, k, bound)
#     flops = 2 * m * n * k  # in FLOPS
#     return flops / roofline_time_s * 1e-12  # in TFLOPS


# def calc_roofline_bandwidth(m, n, k, bound, roofline_time_s=None):
#     if roofline_time_s is None:
#         roofline_time_s = calc_roofline_time(m, n, k, bound)

#     mem_read = m * k + n * k  # in bytes
#     mem_write = (m * n) * 2  # in bytes
#     mem = mem_read + mem_write  # in bytes
#     return mem / roofline_time_s * 1e-9  # in GB/s



def calc_arithmetic_intensity(m, n, k, dtype="fp16", block_shape=(128, 128)):
    """
    Calculate arithmetic intensity (FLOPS per byte of memory traffic).
    
    For FP8 blockscale GEMM:
    - FLOPS: 2 * M * N * K
    - Memory: includes FP8 matrices + FP32 scales + output
    """
    if dtype == "fp16":
        # For FP16 GEMM (no blockscale):
        # x: M*K*2, weight: N*K*2, y: M*N*2
        factor = 2
        bytes_moved = factor * (m * n + n * k + m * k)
    elif dtype == "fp8":
        # For FP8 blockscale GEMM:
        block_shape_n, block_shape_k = block_shape
        scale_n = (n + block_shape_n - 1) // block_shape_n
        scale_k = (k + block_shape_k - 1) // block_shape_k
        
        bytes_read = (m * k * 1 +  # x (FP8)
                     n * k * 1 +   # weight (FP8)
                     m * scale_k * 4 +  # x_scale (FP32)
                     scale_n * scale_k * 4)  # w_scale (FP32)
        bytes_write = m * n * 2  # y (bf16)
        bytes_moved = bytes_read + bytes_write
    else:
        raise ValueError("Unsupported dtype. Use 'fp16' or 'fp8'.")

    return (2 * m * n * k) / bytes_moved


def calc_roofline_time(m, n, k, bound, in_us=False, block_shape=(128, 128)):
    """
    Calculate roofline time for FP8 blockscale GEMM.
    
    Memory includes:
    - x: M x K (FP8 e4m3, 1 byte per element)
    - weight: N x K (FP8 e4m3, 1 byte per element)
    - y: M x N (output, 2 bytes per element for bf16)
    - x_scale: M x scale_k (FP32, 4 bytes per element)
    - w_scale: scale_n x scale_k (FP32, 4 bytes per element)
    """
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    
    if bound == "compute":
        roofline_time = (2 * m * n * k) / MI350X_PERF_FP8_EMPIRICAL  # in seconds
        roofline_time = max(roofline_time, KERNEL_LAUNCH_OVERHEAD)
    else:
        # Input bytes (read):
        # - x: M * K * 1 byte (FP8 e4m3)
        # - weight: N * K * 1 byte (FP8 e4m3)
        # - x_scale: M * scale_k * 4 bytes (FP32)
        # - w_scale: scale_n * scale_k * 4 bytes (FP32)
        bytes_read = (m * k * 1 +  # x (FP8)
                     n * k * 1 +   # weight (FP8)
                     m * scale_k * 4 +  # x_scale (FP32)
                     scale_n * scale_k * 4)  # w_scale (FP32)
        
        # Output bytes (write):
        # - y: M * N * 2 bytes (bf16)
        bytes_write = m * n * 2
        
        bytes_moved = bytes_read + bytes_write
        roofline_time = bytes_moved / MI350X_BW_MEM_EMPIRICAL  # in seconds
        roofline_time = max(roofline_time, KERNEL_LAUNCH_OVERHEAD)

    if in_us:
        return roofline_time / 1e-6  # in us

    return roofline_time  # in seconds


def calc_roofline_throughput(m, n, k, bound, roofline_time_s=None):
    if roofline_time_s is None:
        roofline_time_s = calc_roofline_time(m, n, k, bound)
    flops = 2 * m * n * k  # in FLOPS
    return flops / roofline_time_s * 1e-12  # in TFLOPS


def calc_roofline_bandwidth(m, n, k, bound, roofline_time_s=None, block_shape=(128, 128)):
    """
    Calculate memory bandwidth for FP8 blockscale GEMM.
    
    Memory includes:
    - x: M x K (FP8 e4m3, 1 byte per element) 
    - weight: N x K (FP8 e4m3, 1 byte per element)
    - y: M x N (output, 2 bytes per element for bf16)
    - x_scale: M x scale_k (FP32, 4 bytes per element)
    - w_scale: scale_n x scale_k (FP32, 4 bytes per element)
    """
    if roofline_time_s is None:
        roofline_time_s = calc_roofline_time(m, n, k, bound, block_shape=block_shape)

    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    # Input bytes (read):
    mem_read = (m * k * 1 +  # x (FP8)
                n * k * 1 +   # weight (FP8)
                m * scale_k * 4 +  # x_scale (FP32)
                scale_n * scale_k * 4)  # w_scale (FP32)
    
    # Output bytes (write):
    mem_write = m * n * 2  # y (bf16)
    
    mem = mem_read + mem_write  # in bytes
    return mem / roofline_time_s * 1e-9  # in GB/s


def load_dims_from_json(json_path):
    """Load M, N, K dimensions from a JSON file."""
    with open(json_path, 'r') as f:
        dims = json.load(f)
    
    m_vals = dims.get('M', [])
    n_vals = dims.get('N', [])
    k_vals = dims.get('K', [])
    
    if not m_vals or not n_vals or not k_vals:
        raise ValueError("JSON file must contain 'M', 'N', and 'K' arrays")
    
    return m_vals, n_vals, k_vals


def main():
    parser = argparse.ArgumentParser(
        description='Calculate roofline performance metrics for FP8 GEMM operations'
    )
    parser.add_argument(
        '--dims-json',
        type=str,
        default="unique_dims_test.json",
        help='Path to JSON file containing M, N, K dimensions (format: {"M": [...], "N": [...], "K": [...]})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='mnk_theoretical_peaked.csv',
        help='Output CSV file path (default: mnk_theoretical_peaked.csv)'
    )
    
    args = parser.parse_args()
    
    # Load dimensions from JSON file or use defaults
    if args.dims_json:
        print(f"Loading dimensions from {args.dims_json}")
        m_vals, n_vals, k_vals = load_dims_from_json(args.dims_json)
    else:
        # Default values
        print("Using default dimensions")
        k_vals = [7168]
        m_vals = [128]
        n_vals = [512]
    
    print(f"M values: {m_vals}")
    print(f"N values: {n_vals}")
    print(f"K values: {k_vals}")

    combinations = list(itertools.product(m_vals, n_vals, k_vals))
    y_vals_fp8 = [calc_arithmetic_intensity(m, n, k, "fp8") for m, n, k in combinations]
    ops_byte_ratio_fp8 = MI350X_PERF_FP8 / MI350X_BW_MEM
    bounds_fp8 = ["memory" if y < ops_byte_ratio_fp8 else "compute" for y in y_vals_fp8]
    roofline_us_fp8 = [
        calc_roofline_time(m, n, k, b, in_us=True)
        for (m, n, k), b in zip(combinations, bounds_fp8)
    ]
    roofline_tflops_fp8 = [
        calc_roofline_throughput(m, n, k, b)
        for (m, n, k), b in zip(combinations, bounds_fp8)
    ]
    roofline_bw_fp8 = [
        calc_roofline_bandwidth(m, n, k, b)
        for (m, n, k), b in zip(combinations, bounds_fp8)
    ]
    df = pd.DataFrame(
        {
            "M": m,
            "N": n,
            "K": k,
            "AI FP8": y_val_fp8,
            "Bound FP8": bound_fp8,
            "RL time (us)": rl_us_fp8,
            "RL TFLOPS": rl_tflops_fp8,
            "RL BW (GB/s)": rl_bw_fp8,
        }
        for (m, n, k), y_val_fp8, bound_fp8, rl_us_fp8, rl_tflops_fp8, rl_bw_fp8 in zip(
            itertools.product(m_vals, n_vals, k_vals),
            y_vals_fp8,
            bounds_fp8,
            roofline_us_fp8,
            roofline_tflops_fp8,
            roofline_bw_fp8,
        )
    )
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print("Ops/Byte Ratio FP8:", ops_byte_ratio_fp8)
    print(df)


if __name__ == "__main__":
    main()
