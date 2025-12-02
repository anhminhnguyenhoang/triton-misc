import pandas as pd
import argparse
import json
import re
import subprocess
import itertools

from pathlib import Path
import os, re, glob, math
from typing import List, Tuple

parser = argparse.ArgumentParser(
    description="Benchmark GEMM shapes from unique dims JSON"
)
parser.add_argument(
    "--dims_json",
    type=str,
    default="unique_dims.json",
    help="Path to the JSON file with unique dims",
)
args = parser.parse_args()

dims_file_path = args.dims_json
cols = ["M", "N", "K", "time (ms)", "throughput (TFLOPs)", "bandwidth (GB/s)"]


try:
    # Load unique dims from JSON
    with open(dims_file_path, "r") as f:
        unique_dims = json.load(f)
    m_list = unique_dims["M"]
    n_list = unique_dims["N"]
    k_list = unique_dims["K"]
    print(
        f"\nLoaded unique dims from '{dims_file_path}': M={m_list}, N={n_list}, K={k_list}"
    )

    # Generate unique shapes combinations
    unique_shapes = list(itertools.product(m_list, n_list, k_list))
    print(f"Generated {len(unique_shapes)} unique shape combinations")

    print("\nRunning benchmarks for all M, N, K combinations...")
    res_bench = []

    for m, n, k in unique_shapes:
        cmd = (
            "TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1"
            " TRITON_HIP_USE_ASYNC_COPY=1"
            " AMDGCN_USE_BUFFER_OPS=1"
            " TRITON_HIP_USE_BLOCK_PINGPONG=1 "
        )
        cmd += f"python op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py --shape {m} {n} {k} --metric all"
        # print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # print(result.stdout.split())
        res = [v for v in result.stdout.split()[-3:]]  # Get last 3 float values from output
        pair = m, n, k, *res
        res_bench.append(pair)
        pair_str = " ".join([f"{cname.split()[0]}={v}" for cname, v in zip(cols, pair)])
        print("Result for", pair_str)

    # Create DataFrame
    bench_df = pd.DataFrame(
        res_bench,
        columns=cols,
    )

    # Print the DataFrame
    print("\nBenchmark results:")
    print(bench_df)

    # Save to CSV
    bench_df.to_csv("bench_res.csv", index=False)
    print("\nSaved benchmark results to 'bench_res.csv'")

except FileNotFoundError:
    print(f"Error: The file '{dims_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("traceback:")
    import traceback

    traceback.print_exc()
