import argparse
import glob
import os
from datetime import datetime
from multiprocessing import Process, Queue, set_start_method, cpu_count
import json
import sys
import torch
import triton
from queue import Empty
import fcntl  # For file locking
from op_tests.triton_tests.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
)
import math
import itertools
import random

block_shape = (128, 128)


def get_autotune_config(config_file):
    """
    Load tuning space configuration from JSON file.

    Args:
        config_file: Path to JSON file containing tuning parameter ranges

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

    # Extract parameter ranges from JSON
    block_size_m = tuning_space.get("block_size_m", [1])
    block_size_n = tuning_space.get("block_size_n", [16])
    block_size_k = tuning_space.get("block_size_k", [128])
    group_size_m = tuning_space.get("group_size_m", [1])
    num_warps = tuning_space.get("num_warps", [1])
    num_stages = tuning_space.get("num_stages", [1])
    waves_per_eu = tuning_space.get("waves_per_eu", [0])
    matrix_instr_nonkdim = tuning_space.get("matrix_instr_nonkdim", [16])
    cache_modifier = tuning_space.get("cache_modifier", [None])
    num_ksplit = tuning_space.get("num_ksplit", [1])

    configs = []
    for bm, bn, bk, gm, nw, ns, wpe, mi, cm, nks in itertools.product(
        block_size_m,
        block_size_n,
        block_size_k,
        group_size_m,
        num_warps,
        num_stages,
        waves_per_eu,
        matrix_instr_nonkdim,
        cache_modifier,
        num_ksplit,
    ):
        configs.append(
            dict(
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=gm,
                matrix_instr_nonkdim=mi,
                num_warps=nw,
                num_stages=ns,
                waves_per_eu=wpe,
                cache_modifier=cm,
                NUM_KSPLIT=nks,
            )
        )

    return configs


def get_config_cache_filename(M, N, K):
    """Generate cache filename for tracking tested configs."""
    return f"tested_configs_M{M}_N{N}_K{K}.json"


def load_tested_configs(M, N, K, config_file):
    """
    Load set of already tested config indices from disk.
    Returns empty set if config_file doesn't match (different tuning space).
    """
    cache_file = get_config_cache_filename(M, N, K)
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
    M, N, K, config_idx, config, config_file, success=True, error_msg=None
):
    """
    Save a tested config to disk with file locking to prevent race conditions.
    Separates successful and failed configs for easy inspection.

    Args:
        M, N, K: Matrix dimensions
        config_idx: Index in the shuffled configs list
        config: The configuration dict
        config_file: Path to the tuning space config file
        success: Whether the config ran successfully
        error_msg: Error message if failed
    """
    cache_file = get_config_cache_filename(M, N, K)

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
                            "M": M,
                            "N": N,
                            "K": K,
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


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    impl: callable,
    config: dict = None,
):
    block_shape_n, block_shape_k = block_shape
    if config is None:
        config = {}
    c_dtype = torch.bfloat16

    x, weight, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
        M, N, K, block_shape_n, block_shape_k, layout=layout, output=True
    )
    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * weight.element_size()
    mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: impl(x, weight, x_scale, w_scale, c_dtype, y, config),  # noqa: E731
        warmup=25,
        rep=100,
    )

    # Return exactly one scalar depending on which metric is active
    if metric == "time":
        return ms
    elif metric == "throughput":
        tflops = flops / ms * 1e-9
        return tflops


def bench_worker(
    results_queue, M, N, K, layout, config, config_idx, config_file, device_id=None
):
    """Worker function to run benchmark and put result in queue immediately."""
    try:
        # Set device if specified
        if device_id is not None:
            torch.cuda.set_device(device_id)

        block_shape_n, block_shape_k = (128, 128)
        c_dtype = torch.bfloat16
        from op_tests.triton_tests.test_gemm_a8w8_blockscale import (
            generate_gemm_a8w8_blockscale_inputs,
        )
        from aiter.ops.triton.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
        )

        # Clean up GPU core dump files before running
        for core_file in glob.glob("gpucore.*"):
            try:
                os.remove(core_file)
            except:
                pass

        x, weight, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
            M, N, K, block_shape_n, block_shape_k, layout=layout, output=True
        )
        # print the benching config
        # print(f"[Worker GPU {device_id}] Benchmarking config idx {config_idx}: {config}")
        ms = triton.testing.do_bench(
            lambda: triton_gemm_a8w8_blockscale(
                x, weight, x_scale, w_scale, c_dtype, y, config
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
        save_tested_config(M, N, K, config_idx, config, config_file, success=True)

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
            M, N, K, config_idx, config, config_file, success=False, error_msg=error_msg
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
    M,
    N,
    K,
    fname,
    stop_event,
    tested_configs_file,
    max_configs_without_improvement,
):
    """
    Continuously monitor results queue and update best config.
    This runs in a separate process to ensure results are saved immediately.
    Early stops after specified configs without improvement.
    Tracks tested configs to disk for resume capability.
    """
    best_time = float("inf")
    best_config = None
    best_config_idx = None
    valid_count = 0
    failed_count = 0
    configs_since_improvement = 0

    # Load existing best configs if available
    try:
        with open(fname, "r") as f:
            best_configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_configs = {}

    print(f"\n[Monitor] Started monitoring results for M={M}, N={N}, K={K}")
    print(f"[Monitor] Waiting for {total_configs} benchmark results...")
    print(
        f"[Monitor] Will stop early if no improvement after {max_configs_without_improvement} configs"
    )

    while valid_count + failed_count < total_configs:
        try:
            # Wait for result with timeout to allow checking stop_event
            result = results_queue.get(timeout=1.0)

            config_idx = result["config_idx"]
            config = result["config"]

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
                    best_configs[str(M)] = {
                        "config": best_config,
                        "time_ms": best_time,
                        "N": N,
                        "K": K,
                    }

                    # Write to file immediately
                    with open(fname, "w") as f:
                        json.dump(best_configs, f, indent=2)

                    print(
                        f"[Monitor] ✓ New best! Config {config_idx+1}/{total_configs}: {time_ms:.3f} ms"
                    )
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
                        stop_event.set()  # Signal workers to stop
                        break

                    if (valid_count + failed_count) % 50 == 0:
                        print(
                            f"[Monitor] Progress: {valid_count + failed_count}/{total_configs} "
                            f"(Valid: {valid_count}, Failed: {failed_count}, Best: {best_time:.3f} ms, "
                            f"No improvement: {configs_since_improvement}/{max_configs_without_improvement})"
                        )
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
                    stop_event.set()  # Signal workers to stop
                    break

                if failed_count <= 5:  # Only show first few errors
                    print(
                        f"[Monitor] ✗ Config {config_idx+1}/{total_configs} failed: {result.get('error', 'Unknown error')}"
                    )
                elif failed_count % 50 == 0:
                    print(
                        f"[Monitor] Progress: {valid_count + failed_count}/{total_configs} "
                        f"(Valid: {valid_count}, Failed: {failed_count})"
                    )

        except Empty:
            # Timeout - check if we should stop
            if stop_event and stop_event.is_set():
                print(f"[Monitor] Stop signal received, exiting...")
                break
            continue

    # Final summary
    print(f"\n[Monitor] {'='*80}")
    print(f"[Monitor] RESULTS SUMMARY for M={M}, N={N}, K={K}")
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

    return best_config, best_time


def filter_untested_configs(configs, tested_configs_file):
    """
    Filter out configs that have already been tested.
    Returns only untested configs and count of skipped configs.
    """
    try:
        with open(tested_configs_file, "r") as f:
            tested_data = json.load(f)
            tested_hashes = set(tested_data.get("tested_hashes", []))

        untested_configs = []
        skipped_count = 0

        for config in configs:
            config_hash = hash(frozenset(config.items()))
            if config_hash not in tested_hashes:
                untested_configs.append(config)
            else:
                skipped_count += 1

        return untested_configs, skipped_count
    except (FileNotFoundError, json.JSONDecodeError):
        # No previous run, all configs are untested
        return configs, 0


def get_ff_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--M",
        nargs="+",
        type=int,
        required=True,
        help="List of M values separated by space",
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Single N value",
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Single K value",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect based on available GPUs/CPUs)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="tuning_space_gemm_a8w8.json",
        help="Path to JSON file containing tuning space configuration (default: tuning_space_gemm_a8w8.json)",
    )
    parser.add_argument(
        "--max-configs-without-improvement",
        type=int,
        default=1000,
        help="Stop early if no improvement after this many configs (default: 1000)",
    )
    return parser.parse_args()


def parse_args():
    parser = get_parser(kernel_name="A8W8 GEMM Blockscale")
    parser = add_argparse_ff(parser)
    return get_ff_args(parser)


# Set multiprocessing start method to spawn
try:
    set_start_method("spawn")
except RuntimeError:
    pass  # Already set


def main():
    args = parse_args()

    if args.metric != "time":
        print("This tuning script only supports --metric time")
        return

    start_time = datetime.now()

    # Create shapes from M list, N, K
    shapes = [(m, args.N, args.K) for m in args.M]
    print(f"Tuning for shapes: {shapes}")

    if not shapes:
        print("No shapes to tune. Exiting.")
        return

    configs = get_autotune_config(args.config_file)

    # Create indices and shuffle them for random exploration
    # Keep both original configs list and shuffled indices for mapping
    num_configs = len(configs)
    config_indices = list(range(num_configs))
    random.shuffle(config_indices)

    print(f"Total configurations to tune: {len(configs)} (shuffled randomly)")

    # Generate output filename
    fname = f"best_configs_M{'_'.join(str(m) for m in args.M)}_N{args.N}_K{args.K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Determine number of workers and available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        try:
            # Use number of GPUs if available, otherwise quarter of CPU cores
            num_workers = num_gpus if num_gpus > 0 else max(1, cpu_count() // 4)
        except:
            num_workers = 4  # Default fallback

    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s)")
        print(f"Using {num_workers} parallel workers with round-robin GPU assignment")
    else:
        print(f"No GPUs detected, using {num_workers} CPU workers")
        num_gpus = 1  # Avoid division by zero, will use None for device_id

    # Process each shape
    for M, N, K in shapes:
        print(f"\n{'='*80}")
        print(f"Starting tuning for M={M}, N={N}, K={K}")
        print(f"{'='*80}")

        # Load already tested config indices for this shape
        tested_indices = load_tested_configs(M, N, K, args.config_file)
        if tested_indices:
            # Convert string keys back to integers
            tested_indices = {int(idx) for idx in tested_indices}
            print(
                f"[Main] Found {len(tested_indices)} previously tested configs, will skip them"
            )

        # Filter out already-tested configs using shuffled indices
        # configs_to_test contains tuples of (original_config_idx, config)
        configs_to_test = []
        for shuffled_idx in config_indices:
            if shuffled_idx not in tested_indices:
                configs_to_test.append((shuffled_idx, configs[shuffled_idx]))

        if not configs_to_test:
            print(
                f"[Main] All {len(configs)} configs already tested for this shape, skipping..."
            )
            continue

        print(
            f"[Main] Testing {len(configs_to_test)}/{len(configs)} configs (skipping {len(tested_indices)} already tested)"
        )

        # Create a multiprocessing Queue for results
        results_queue = Queue()

        # Generate the tested configs filename for this shape
        tested_configs_file = get_config_cache_filename(M, N, K)

        # Start the result monitor process
        from multiprocessing import Event

        stop_event = Event()

        monitor_process = Process(
            target=result_monitor,
            args=(
                results_queue,
                len(configs_to_test),
                M,
                N,
                K,
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

            # Assign GPU device in round-robin fashion
            device_id = worker_id % num_gpus if num_gpus > 0 else None

            # Create worker process that will handle multiple configs
            worker_process = Process(
                target=worker_batch,
                args=(
                    results_queue,
                    M,
                    N,
                    K,
                    args.layout,
                    worker_configs_indexed,
                    stop_event,
                    args.config_file,
                    worker_id % num_gpus if num_gpus > 0 else None,
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

        print(f"[Main] Completed tuning for M={M}, N={N}, K={K}\n")

    end_time = datetime.now()
    total_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"TUNING COMPLETE")
    print(f"  Total time: {total_time}")
    print(f"  Shapes tuned: {len(shapes)}")
    print(f"  Configs per shape: {len(configs)}")
    print(f"  Total benchmarks: {len(shapes) * len(configs)}")
    print(f"  Results saved to: {fname}")
    print(f"{'='*80}")


def worker_batch(
    results_queue,
    M,
    N,
    K,
    layout,
    configs_indexed,
    stop_event,
    config_file,
    device_id=None,
):
    """
    Worker process that handles a batch of configs.
    Each config result is immediately put in the queue.
    Respects stop_event for early termination.
    configs_indexed: list of (original_idx, config) tuples
    device_id: GPU device ID to use for this worker (round-robin assignment)
    """
    # Set device at the start of worker process
    if device_id is not None:
        torch.cuda.set_device(device_id)
        # print(f"[Worker] Process assigned to GPU {device_id}")

    for item in configs_indexed:
        original_idx, config = item[0], item[1]
        # Check if we should stop
        if stop_event.is_set():
            print(f"[Worker GPU {device_id}] Received stop signal, terminating...")
            break

        bench_worker(
            results_queue, M, N, K, layout, config, original_idx, config_file, device_id
        )


if __name__ == "__main__":
    sys.exit(main())
