"""
Multi-threaded tuning script for MHC (manifold-constrained Hyper Connection) kernels.

This script performs systematic exploration of kernel configurations for fused_mhc
and sinkhorn_knopp kernels using multi-GPU parallel workers.

Usage:
    python tune_mhc.py --kernel fused_mhc -M 1024 2048 4096 -n 4 -C 1024 --hres-mode lite --workers 0 1 2 3
    python tune_mhc.py --kernel fused_mhc -M 1024 2048 4096 -n 4 -C 1024 --hres-mode sinkhorn --workers 0 1 2 3
    python tune_mhc.py --kernel sinkhorn_knopp -M 1024 2048 4096 -n 8 --sinkhorn-iters 20
"""

import argparse
import glob
import os
from datetime import datetime
from multiprocessing import Process, Queue, set_start_method, cpu_count, Event
import json
import sys
import torch
import triton
from queue import Empty
import fcntl  # For file locking
import itertools
import random

from aiter.ops.triton.fusions.mhc import fused_mhc, sinkhorn_knopp
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_autotune_config(config_file, kernel):
    """
    Load tuning space configuration from JSON file for the specified kernel.

    Args:
        config_file: Path to JSON file containing tuning parameter ranges
        kernel: Kernel name ('fused_mhc' or 'sinkhorn_knopp')

    Returns:
        List of config dictionaries to test
    """
    try:
        with open(config_file, "r") as f:
            tuning_space = json.load(f)
        print(f"Loaded tuning space from '{config_file}'")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tuning space configuration file '{config_file}' not found. "
            f"Please provide a valid JSON file with tuning parameters."
        )

    # Get kernel-specific tuning space
    if kernel not in tuning_space:
        raise KeyError(
            f"Kernel '{kernel}' not found in tuning space. "
            f"Available kernels: {list(tuning_space.keys())}"
        )

    kernel_space = tuning_space[kernel]

    if kernel == "fused_mhc":
        # Extract parameter ranges for fused_mhc
        block_m = kernel_space.get("BLOCK_M", [64])
        block_n = kernel_space.get("BLOCK_N", [16])
        block_k = kernel_space.get("BLOCK_K", [64])
        num_ksplit = kernel_space.get("NUM_KSPLIT", [1])
        num_warps = kernel_space.get("num_warps", [4])
        num_stages = kernel_space.get("num_stages", [1])
        waves_per_eu = kernel_space.get("waves_per_eu", [2])
        hres_op = kernel_space.get("HRES_OP", [0])

        configs = []
        for bm, bn, bk, nks, nw, ns, wpe, hop in itertools.product(
            block_m,
            block_n,
            block_k,
            num_ksplit,
            num_warps,
            num_stages,
            waves_per_eu,
            hres_op,
        ):
            configs.append(
                dict(
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                    NUM_KSPLIT=nks,
                    num_warps=nw,
                    num_stages=ns,
                    waves_per_eu=wpe,
                    HRES_OP=hop,
                )
            )
    elif kernel == "sinkhorn_knopp":
        # Extract parameter ranges for sinkhorn_knopp
        block_m = kernel_space.get("BLOCK_M", [1])
        num_warps = kernel_space.get("num_warps", [4])
        num_stages = kernel_space.get("num_stages", [1])
        waves_per_eu = kernel_space.get("waves_per_eu", [2])

        configs = []
        for bm, nw, ns, wpe in itertools.product(
            block_m,
            num_warps,
            num_stages,
            waves_per_eu,
        ):
            configs.append(
                dict(
                    BLOCK_M=bm,
                    num_warps=nw,
                    num_stages=ns,
                    waves_per_eu=wpe,
                )
            )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return configs


def get_tensors_fused_mhc(M, n, C, dtype, hres_mode="lite"):
    """
    Generate input tensors for fused_mhc benchmarking.

    Args:
        M: Batch/sequence dimension
        n: Stream parameter (manifold dimension)
        C: Hidden dimension per stream
        dtype: Data type
        hres_mode: H_res computation mode ("lite" or "sinkhorn")

    Returns:
        Tuple of tensors for fused_mhc
    """
    x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n_streams = \
        generate_mhc_inputs(M, n, C, dtype, hres_mode=hres_mode)
    return x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n_streams


def get_tensors_sinkhorn(M, n, dtype):
    """
    Generate input tensors for sinkhorn_knopp benchmarking.

    Args:
        M: Batch dimension
        n: Matrix size (n x n matrices)
        dtype: Data type

    Returns:
        3D tensor of shape (M, n, n) for sinkhorn_knopp
    """
    # Generate random logits for Sinkhorn-Knopp
    logits = torch.randn((M, n, n), device='cuda', dtype=dtype)
    return logits


def get_config_cache_filename(kernel, M, n, C, sinkhorn_iters, hres_mode=None):
    """Generate cache filename for tracking tested configs."""
    if kernel == "fused_mhc":
        return f"tested_configs_mhc_{kernel}_{hres_mode}_M{M}_n{n}_C{C}.json"
    else:  # sinkhorn_knopp
        return f"tested_configs_mhc_{kernel}_M{M}_n{n}_iters{sinkhorn_iters}.json"


def load_tested_configs(kernel, M, n, C, sinkhorn_iters, config_file, hres_mode=None):
    """
    Load set of already tested config indices from disk.
    Returns empty set if config_file doesn't match (different tuning space).
    """
    cache_file = get_config_cache_filename(kernel, M, n, C, sinkhorn_iters, hres_mode)
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)

            # Check if the same config file was used
            saved_config_file = data.get("config_file", None)
            if saved_config_file != config_file:
                print(
                    f"[Load] Config file mismatch: saved='{saved_config_file}', current='{config_file}'"
                )
                print(
                    f"[Load] Starting fresh - previous results from different tuning space"
                )
                return set()

            # Combine successful and failed indices
            successful_indices = set(data.get("tested_configs_successful", {}).keys())
            failed_indices = set(data.get("tested_configs_failed", {}).keys())
            return successful_indices | failed_indices
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_tested_config(
    kernel, M, n, C, sinkhorn_iters, config_idx, config, config_file, success=True, error_msg=None, hres_mode=None
):
    """
    Save a tested config to disk with file locking to prevent race conditions.
    Separates successful and failed configs for easy inspection.
    """
    cache_file = get_config_cache_filename(kernel, M, n, C, sinkhorn_iters, hres_mode)

    # Use file locking to ensure atomic read-modify-write
    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Open/create file for reading and writing
            with open(cache_file, "a+") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                try:
                    # Move to beginning and read existing data
                    f.seek(0)
                    content = f.read()

                    if content:
                        data = json.loads(content)
                    else:
                        data = {
                            "kernel": kernel,
                            "M": M,
                            "n": n,
                            "C": C,
                            "sinkhorn_iters": sinkhorn_iters,
                            "hres_mode": hres_mode,
                            "config_file": config_file,
                            "tested_configs_successful": {},
                            "tested_configs_failed": {},
                        }

                    # Ensure config_file is always set (for old files)
                    if "config_file" not in data:
                        data["config_file"] = config_file

                    # Add config to appropriate section
                    config_key = str(config_idx)
                    if success:
                        data["tested_configs_successful"][config_key] = config
                    else:
                        data["tested_configs_failed"][config_key] = {
                            "config": config,
                            "error": error_msg,
                        }

                    # Update totals
                    data["total_successful"] = len(data["tested_configs_successful"])
                    data["total_failed"] = len(data["tested_configs_failed"])
                    data["total_tested"] = (
                        data["total_successful"] + data["total_failed"]
                    )

                    # Truncate file and write updated data
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure written to disk

                    return  # Success!

                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except (IOError, OSError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(
                    f"[WARNING] Failed to save tested config after {max_retries} retries: {e}"
                )
                raise
            import time

            time.sleep(0.1)  # Wait a bit before retry


def bench_worker_fused_mhc(
    results_queue, M, n, C, dtype, config, config_idx, config_file, hres_mode, device_id=None
):
    """Worker function to run fused_mhc benchmark and put result in queue immediately."""
    try:
        # Set device if specified
        if device_id is not None:
            torch.cuda.set_device(device_id)

        # Clean up GPU core dump files before running
        for core_file in glob.glob("gpucore.*"):
            try:
                os.remove(core_file)
            except:
                pass

        # Generate inputs
        x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n_streams = \
            get_tensors_fused_mhc(M, n, C, dtype, hres_mode)

        # Benchmark with config
        ms = triton.testing.do_bench(
            lambda: fused_mhc(
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res,
                bias, n_streams,
                hres_mode=hres_mode,
                config=config,
            ),
            warmup=25,
            rep=100,
        )

        # Clean up any core dumps created during benchmark
        for core_file in glob.glob("gpucore.*"):
            try:
                os.remove(core_file)
            except:
                pass

        # Save that this config was tested successfully
        save_tested_config("fused_mhc", M, n, C, 0, config_idx, config, config_file, success=True, hres_mode=hres_mode)

        # Put result in queue immediately
        results_queue.put(
            {"config_idx": config_idx, "config": config, "time_ms": ms, "success": True}
        )
    except (RuntimeError, OSError, MemoryError, Exception) as e:
        # Catch GPU memory access faults, runtime errors, and other exceptions
        error_msg = str(e)

        # Check for GPU core dump files which indicate memory access faults
        core_files = glob.glob("gpucore.*")
        if core_files:
            error_msg = (
                f"GPU memory access fault (core dumps: {len(core_files)}): {error_msg}"
            )
            # Clean up core dump files
            for core_file in core_files:
                try:
                    os.remove(core_file)
                except:
                    pass

        # Save failed config with error message
        save_tested_config(
            "fused_mhc", M, n, C, 0, config_idx, config, config_file, success=False, error_msg=error_msg, hres_mode=hres_mode
        )

        # Put failure result in queue
        results_queue.put(
            {
                "config_idx": config_idx,
                "config": config,
                "time_ms": None,
                "success": False,
                "error": error_msg,
            }
        )


def bench_worker_sinkhorn(
    results_queue, M, n, sinkhorn_iters, dtype, config, config_idx, config_file, device_id=None
):
    """Worker function to run sinkhorn_knopp benchmark and put result in queue immediately."""
    try:
        # Set device if specified
        if device_id is not None:
            torch.cuda.set_device(device_id)

        # Clean up GPU core dump files before running
        for core_file in glob.glob("gpucore.*"):
            try:
                os.remove(core_file)
            except:
                pass

        # Generate inputs
        logits = get_tensors_sinkhorn(M, n, dtype)

        # Benchmark with config
        ms = triton.testing.do_bench(
            lambda: sinkhorn_knopp(logits, num_iters=sinkhorn_iters, config=config),
            warmup=25,
            rep=100,
        )

        # Clean up any core dumps created during benchmark
        for core_file in glob.glob("gpucore.*"):
            try:
                os.remove(core_file)
            except:
                pass

        # Save that this config was tested successfully
        save_tested_config("sinkhorn_knopp", M, n, 0, sinkhorn_iters, config_idx, config, config_file, success=True)

        # Put result in queue immediately
        results_queue.put(
            {"config_idx": config_idx, "config": config, "time_ms": ms, "success": True}
        )
    except (RuntimeError, OSError, MemoryError, Exception) as e:
        # Catch GPU memory access faults, runtime errors, and other exceptions
        error_msg = str(e)

        # Check for GPU core dump files which indicate memory access faults
        core_files = glob.glob("gpucore.*")
        if core_files:
            error_msg = (
                f"GPU memory access fault (core dumps: {len(core_files)}): {error_msg}"
            )
            # Clean up core dump files
            for core_file in core_files:
                try:
                    os.remove(core_file)
                except:
                    pass

        # Save failed config with error message
        save_tested_config(
            "sinkhorn_knopp", M, n, 0, sinkhorn_iters, config_idx, config, config_file, success=False, error_msg=error_msg
        )

        # Put failure result in queue
        results_queue.put(
            {
                "config_idx": config_idx,
                "config": config,
                "time_ms": None,
                "success": False,
                "error": error_msg,
            }
        )


def result_monitor(
    results_queue,
    total_configs,
    kernel,
    M,
    n,
    C,
    sinkhorn_iters,
    hres_mode,
    fname,
    stop_event,
    tested_configs_file,
    max_configs_without_improvement,
):
    """
    Continuously monitor results queue and update best config.
    This runs in a separate process to ensure results are saved immediately.
    Early stops after specified configs without improvement.
    """
    best_time = float("inf")
    best_config = None
    best_config_idx = None
    valid_count = 0
    failed_count = 0
    configs_since_improvement = 0
    last_progress_print = 0  # Track when we last printed progress

    # Load existing best configs if available
    try:
        with open(fname, "r") as f:
            best_configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_configs = {}

    if kernel == "fused_mhc":
        shape_str = f"M={M}, n={n}, C={C}, mode={hres_mode}"
        config_key = f"M{M}_n{n}_C{C}_{hres_mode}"
    else:
        shape_str = f"M={M}, n={n}, iters={sinkhorn_iters}"
        config_key = f"M{M}_n{n}_iters{sinkhorn_iters}"

    print(f"\n[Monitor] Started monitoring results for {kernel}: {shape_str}")
    print(f"[Monitor] Waiting for {total_configs} benchmark results...")
    print(
        f"[Monitor] Will stop early if no improvement after {max_configs_without_improvement} configs"
    )
    sys.stdout.flush()

    while valid_count + failed_count < total_configs:
        try:
            # Wait for result with timeout to allow checking stop_event
            result = results_queue.get(timeout=1.0)

            config_idx = result["config_idx"]
            config = result["config"]
            total_processed = valid_count + failed_count + 1  # +1 for current result

            if result["success"]:
                time_ms = result["time_ms"]
                valid_count += 1

                # Check if this is the best so far
                if time_ms < best_time:
                    best_time = time_ms
                    best_config = config
                    best_config_idx = config_idx
                    configs_since_improvement = 0  # Reset counter

                    # Update best configs dictionary immediately
                    best_configs[config_key] = {
                        "config": best_config,
                        "time_ms": best_time,
                        "kernel": kernel,
                        "M": M,
                        "n": n,
                        "C": C,
                        "sinkhorn_iters": sinkhorn_iters,
                        "hres_mode": hres_mode,
                    }

                    # Write to file immediately
                    with open(fname, "w") as f:
                        json.dump(best_configs, f, indent=2)

                    print(
                        f"[Monitor] ✓ New best! Config {config_idx+1}/{total_configs}: {time_ms:.3f} ms "
                        f"(Progress: {total_processed}/{total_configs})"
                    )
                    sys.stdout.flush()
                    last_progress_print = total_processed
                else:
                    configs_since_improvement += 1

                    # Check for early stopping
                    if configs_since_improvement >= max_configs_without_improvement:
                        print(
                            f"\n[Monitor] ⚠️  Early stopping: No improvement for {max_configs_without_improvement} configs"
                        )
                        print(
                            f"[Monitor] Best config found: {best_time:.3f} ms at index {best_config_idx + 1}"
                        )
                        sys.stdout.flush()
                        stop_event.set()  # Signal workers to stop
                        break
            else:
                failed_count += 1
                configs_since_improvement += 1

                # Check for early stopping even on failures
                if configs_since_improvement >= max_configs_without_improvement:
                    print(
                        f"\n[Monitor] ⚠️  Early stopping: No improvement for {max_configs_without_improvement} configs"
                    )
                    if best_config is not None:
                        print(
                            f"[Monitor] Best config found: {best_time:.3f} ms at index {best_config_idx + 1}"
                        )
                    sys.stdout.flush()
                    stop_event.set()  # Signal workers to stop
                    break

                if failed_count <= 5:  # Only show first few errors
                    print(
                        f"[Monitor] ✗ Config {config_idx+1}/{total_configs} failed: {result.get('error', 'Unknown error')}"
                    )
                    sys.stdout.flush()

            # Print progress every 50 configs (regardless of success/failure)
            if total_processed - last_progress_print >= 50:
                best_str = f"{best_time:.3f} ms" if best_config is not None else "N/A"
                print(
                    f"[Monitor] Progress: {total_processed}/{total_configs} "
                    f"(Valid: {valid_count}, Failed: {failed_count}, Best: {best_str}, "
                    f"No improvement: {configs_since_improvement}/{max_configs_without_improvement})"
                )
                sys.stdout.flush()
                last_progress_print = total_processed

        except Empty:
            # Timeout - check if we should stop
            if stop_event and stop_event.is_set():
                print(f"[Monitor] Stop signal received, exiting...")
                sys.stdout.flush()
                break
            continue

    # Final summary
    print(f"\n[Monitor] {'='*80}")
    print(f"[Monitor] RESULTS SUMMARY for {kernel}: {shape_str}")
    print(f"[Monitor] {'='*80}")
    print(
        f"[Monitor] Total configs tested: {valid_count + failed_count}/{total_configs}"
    )
    print(f"[Monitor] Valid results: {valid_count}")
    print(f"[Monitor] Failed configs: {failed_count}")

    if best_config is not None:
        print(f"[Monitor] Best time: {best_time:.3f} ms")
        print(f"[Monitor] Best config index: {best_config_idx + 1}")
        print(f"[Monitor] Best config: {best_config}")
        print(f"[Monitor] Results saved to: {fname}")
    else:
        print(f"[Monitor] WARNING: No valid configuration found!")

    print(f"[Monitor] {'='*80}\n")
    sys.stdout.flush()

    return best_config, best_time


def worker_batch_fused_mhc(
    results_queue,
    M,
    n,
    C,
    dtype,
    configs_indexed,
    stop_event,
    config_file,
    hres_mode,
    device_id=None,
):
    """
    Worker process that handles a batch of configs for fused_mhc.
    Each config result is immediately put in the queue.
    Respects stop_event for early termination.
    """
    # Set device at the start of worker process
    if device_id is not None:
        torch.cuda.set_device(device_id)

    for item in configs_indexed:
        original_idx, config = item[0], item[1]
        # Check if we should stop
        if stop_event.is_set():
            print(f"[Worker GPU {device_id}] Received stop signal, terminating...")
            break

        bench_worker_fused_mhc(
            results_queue, M, n, C, dtype, config, original_idx, config_file, hres_mode, device_id
        )


def worker_batch_sinkhorn(
    results_queue,
    M,
    n,
    sinkhorn_iters,
    dtype,
    configs_indexed,
    stop_event,
    config_file,
    device_id=None,
):
    """
    Worker process that handles a batch of configs for sinkhorn_knopp.
    Each config result is immediately put in the queue.
    Respects stop_event for early termination.
    """
    # Set device at the start of worker process
    if device_id is not None:
        torch.cuda.set_device(device_id)

    for item in configs_indexed:
        original_idx, config = item[0], item[1]
        # Check if we should stop
        if stop_event.is_set():
            print(f"[Worker GPU {device_id}] Received stop signal, terminating...")
            break

        bench_worker_sinkhorn(
            results_queue, M, n, sinkhorn_iters, dtype, config, original_idx, config_file, device_id
        )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Tune MHC Kernels",
        description="Multi-threaded tuning for fused_mhc and sinkhorn_knopp kernels",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Kernel selection
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        choices=["fused_mhc", "sinkhorn_knopp"],
        help="Kernel to tune",
    )

    # Shape parameters
    parser.add_argument(
        "-M",
        nargs="+",
        type=int,
        required=True,
        help="List of M dimensions (batch/sequence sizes)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=4,
        help="Stream parameter (manifold dimension)",
    )
    parser.add_argument(
        "-C",
        nargs="+",
        type=int,
        default=[1024],
        help="List of hidden dimensions per stream (each C runs as separate tuning)",
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        help="Number of Sinkhorn-Knopp iterations (for sinkhorn_knopp)",
    )
    parser.add_argument(
        "--hres-mode",
        type=str,
        default="lite",
        choices=["lite", "sinkhorn"],
        help="H_res computation mode for fused_mhc (lite or sinkhorn)",
    )

    # Data type
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Data type for computation",
    )

    # Worker configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect based on available GPUs/CPUs). Cannot be used with --workers.",
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=None,
        help="Specific GPU device IDs to use (e.g., --workers 0 1 2). Cannot be used with --num-workers.",
    )

    # Tuning configuration
    parser.add_argument(
        "--config-file",
        type=str,
        default="tuning_space_mhc.json",
        help="Path to JSON file containing tuning space configuration",
    )
    parser.add_argument(
        "--max-configs-without-improvement",
        type=int,
        default=1000,
        help="Stop early if no improvement after this many configs",
    )

    return parser.parse_args()


# Set multiprocessing start method to spawn
try:
    set_start_method("spawn")
except RuntimeError:
    pass  # Already set


def main():
    args = parse_args()

    # Validate that both --num-workers and --workers are not provided
    if args.num_workers is not None and args.workers is not None:
        raise ValueError("Cannot specify both --num-workers and --workers. Use only one.")

    overall_start_time = datetime.now()

    kernel = args.kernel
    dtype = arg_to_torch_dtype[args.dtype]
    M_list = args.M
    n = args.n
    C_list = args.C
    sinkhorn_iters = args.sinkhorn_iters
    hres_mode = args.hres_mode

    print(f"Tuning kernel: {kernel}")
    print(f"M values: {M_list}")
    print(f"n: {n}")
    if kernel == "fused_mhc":
        print(f"C values: {C_list}")
        print(f"H_res mode: {hres_mode}")
    else:
        print(f"Sinkhorn iterations: {sinkhorn_iters}")
    print(f"Data type: {args.dtype}")
    sys.stdout.flush()

    configs = get_autotune_config(args.config_file, kernel)

    # Create indices and shuffle them for random exploration
    num_configs = len(configs)
    config_indices = list(range(num_configs))
    random.shuffle(config_indices)

    print(f"Total configurations to tune: {len(configs)} (shuffled randomly)")
    sys.stdout.flush()

    # Determine number of workers and device IDs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.workers is not None:
        # Use specific device IDs
        device_ids = args.workers
        num_workers = len(device_ids)
        print(f"Using {num_workers} worker(s) on GPU(s): {device_ids}")
    elif args.num_workers is not None:
        # Use specified number of workers with round-robin GPU assignment
        num_workers = args.num_workers
        device_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        if num_gpus > 0:
            print(f"Detected {num_gpus} GPU(s)")
            print(f"Using {num_workers} parallel workers with round-robin GPU assignment")
        else:
            print(f"No GPUs detected, using {num_workers} CPU workers")
    else:
        # Auto-detect: use all available GPUs or quarter of CPU cores
        try:
            num_workers = num_gpus if num_gpus > 0 else max(1, cpu_count() // 4)
        except:
            num_workers = 4  # Default fallback
        device_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        if num_gpus > 0:
            print(f"Detected {num_gpus} GPU(s)")
            print(f"Using {num_workers} parallel workers (auto-detected)")
        else:
            print(f"No GPUs detected, using {num_workers} CPU workers")

    # Track output files for each C value
    output_files = []

    # Process each C value as a completely separate tuning run
    for C in C_list:
        c_start_time = datetime.now()

        print(f"\n{'#'*80}")
        print(f"# TUNING C={C}")
        print(f"{'#'*80}")
        sys.stdout.flush()

        # Generate C-specific output filename
        if kernel == "fused_mhc":
            M_str = '_'.join(str(m) for m in M_list)
            fname = f"best_configs_mhc_{kernel}_{hres_mode}_M{M_str}_n{n}_C{C}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            M_str = '_'.join(str(m) for m in M_list)
            fname = f"best_configs_mhc_{kernel}_M{M_str}_n{n}_iters{sinkhorn_iters}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_files.append(fname)

        # Process each M value for this C
        for M in M_list:
            print(f"\n{'='*80}")
            if kernel == "fused_mhc":
                print(f"Starting tuning for {kernel}: M={M}, n={n}, C={C}, mode={hres_mode}")
            else:
                print(f"Starting tuning for {kernel}: M={M}, n={n}, iters={sinkhorn_iters}")
            print(f"{'='*80}")
            sys.stdout.flush()

            # Load already tested config indices for this shape
            tested_indices = load_tested_configs(kernel, M, n, C, sinkhorn_iters, args.config_file, hres_mode)
            if tested_indices:
                # Convert string keys back to integers
                tested_indices = {int(idx) for idx in tested_indices}
                print(
                    f"[Main] Found {len(tested_indices)} previously tested configs, will skip them"
                )
                sys.stdout.flush()

            # Filter out already-tested configs using shuffled indices
            configs_to_test = []
            for shuffled_idx in config_indices:
                if shuffled_idx not in tested_indices:
                    configs_to_test.append((shuffled_idx, configs[shuffled_idx]))

            if not configs_to_test:
                print(
                    f"[Main] All {len(configs)} configs already tested for this shape, skipping..."
                )
                sys.stdout.flush()
                continue

            print(
                f"[Main] Testing {len(configs_to_test)}/{len(configs)} configs (skipping {len(tested_indices)} already tested)"
            )
            sys.stdout.flush()

            # Create a multiprocessing Queue for results
            results_queue = Queue()

            # Generate the tested configs filename for this shape
            tested_configs_file = get_config_cache_filename(kernel, M, n, C, sinkhorn_iters, hres_mode)

            # Start the result monitor process
            stop_event = Event()

            monitor_process = Process(
                target=result_monitor,
                args=(
                    results_queue,
                    len(configs_to_test),
                    kernel,
                    M,
                    n,
                    C,
                    sinkhorn_iters,
                    hres_mode,
                    fname,
                    stop_event,
                    tested_configs_file,
                    args.max_configs_without_improvement,
                ),
            )
            monitor_process.start()

            # Launch worker processes
            worker_processes = []
            configs_per_worker = len(configs_to_test) // num_workers
            remainder = len(configs_to_test) % num_workers

            config_start = 0
            for worker_id in range(num_workers):
                # Distribute configs evenly, with remainder distributed to first workers
                num_configs_for_worker = configs_per_worker + (
                    1 if worker_id < remainder else 0
                )
                config_end = config_start + num_configs_for_worker

                worker_configs_indexed = configs_to_test[config_start:config_end]

                # Assign GPU device - use specific device_id or round-robin
                device_id = device_ids[worker_id % len(device_ids)]

                # Create worker process based on kernel type
                if kernel == "fused_mhc":
                    worker_process = Process(
                        target=worker_batch_fused_mhc,
                        args=(
                            results_queue,
                            M,
                            n,
                            C,
                            dtype,
                            worker_configs_indexed,
                            stop_event,
                            args.config_file,
                            hres_mode,
                            device_id,
                        ),
                    )
                else:  # sinkhorn_knopp
                    worker_process = Process(
                        target=worker_batch_sinkhorn,
                        args=(
                            results_queue,
                            M,
                            n,
                            sinkhorn_iters,
                            dtype,
                            worker_configs_indexed,
                            stop_event,
                            args.config_file,
                            device_id,
                        ),
                    )

                worker_processes.append(worker_process)
                worker_process.start()

                config_start = config_end

            # Wait for all workers to complete
            for worker_process in worker_processes:
                worker_process.join()

            # Wait for monitor to finish processing all results
            monitor_process.join()

            if kernel == "fused_mhc":
                print(f"[Main] Completed tuning for {kernel}: M={M}, n={n}, C={C}, mode={hres_mode}\n")
            else:
                print(f"[Main] Completed tuning for {kernel}: M={M}, n={n}, iters={sinkhorn_iters}\n")
            sys.stdout.flush()

        # Per-C summary
        c_end_time = datetime.now()
        c_total_time = c_end_time - c_start_time

        print(f"\n{'-'*80}")
        print(f"COMPLETED C={C}")
        print(f"  Time: {c_total_time}")
        print(f"  M values tuned: {len(M_list)}")
        print(f"  Results saved to: {fname}")
        print(f"{'-'*80}")
        sys.stdout.flush()

    overall_end_time = datetime.now()
    overall_total_time = overall_end_time - overall_start_time

    print(f"\n{'='*80}")
    print(f"TUNING COMPLETE")
    print(f"  Kernel: {kernel}")
    if kernel == "fused_mhc":
        print(f"  H_res mode: {hres_mode}")
        print(f"  C values tuned: {C_list}")
    print(f"  Total time: {overall_total_time}")
    print(f"  M values per C: {len(M_list)}")
    print(f"  Configs per M value: {len(configs)}")
    print(f"  Total benchmarks: {len(C_list) * len(M_list) * len(configs)}")
    print(f"  Output files:")
    for f in output_files:
        print(f"    - {f}")
    print(f"{'='*80}")
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
