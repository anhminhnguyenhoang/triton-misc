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
from op_tests.op_benchmarks.triton.bench_attn_qk_int8_per_block import (
    get_tensors,
)
from aiter.ops.triton.attn_qk_int8_per_block import (
    attn_qk_int8_per_block,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
)
import itertools
import random


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
    block_size_m = tuning_space.get("block_size_m", [128])
    block_size_n = tuning_space.get("block_size_n", [32])
    num_warps = tuning_space.get("num_warps", [4])
    num_stages = tuning_space.get("num_stages", [3])
    waves_per_eu = tuning_space.get("waves_per_eu", [2])
    stage = tuning_space.get("stage", [1])

    configs = []
    for bm, bn, nw, ns, wpe, st in itertools.product(
        block_size_m,
        block_size_n,
        num_warps,
        num_stages,
        waves_per_eu,
        stage,
    ):
        configs.append(
            dict(
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                num_warps=nw,
                num_stages=ns,
                waves_per_eu=wpe,
                stage=st,
            )
        )

    return configs


def get_config_cache_filename(batch_size, num_heads, seq_len, head_dim):
    """Generate cache filename for tracking tested configs."""
    return f"tested_configs_B{batch_size}_H{num_heads}_S{seq_len}_D{head_dim}.json"


def load_tested_configs(batch_size, num_heads, seq_len, head_dim, config_file):
    """
    Load set of already tested config indices from disk.
    Returns empty set if config_file doesn't match (different tuning space).
    """
    cache_file = get_config_cache_filename(batch_size, num_heads, seq_len, head_dim)
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
    batch_size, num_heads, seq_len, head_dim, config_idx, config, config_file, success=True, error_msg=None
):
    """
    Save a tested config to disk with file locking to prevent race conditions.
    Separates successful and failed configs for easy inspection.

    Args:
        batch_size, num_heads, seq_len, head_dim: Shape dimensions
        config_idx: Index in the shuffled configs list
        config: The configuration dict
        config_file: Path to the tuning space config file
        success: Whether the config ran successfully
        error_msg: Error message if failed
    """
    cache_file = get_config_cache_filename(batch_size, num_heads, seq_len, head_dim)

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
                            "batch_size": batch_size,
                            "num_heads": num_heads,
                            "seq_len": seq_len,
                            "head_dim": head_dim,
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


def bench_worker(
    results_queue, batch_size, num_heads, seq_len, head_dim, config, config_idx, config_file, device_id=None
):
    """Worker function to run benchmark and put result in queue immediately."""
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
        q, k, v, q_scale, k_scale = get_tensors(batch_size, num_heads, seq_len, head_dim)
        
        # Benchmark with config
        ms = triton.testing.do_bench(
            lambda: attn_qk_int8_per_block(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, config=config),
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
        save_tested_config(batch_size, num_heads, seq_len, head_dim, config_idx, config, config_file, success=True)

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
            batch_size, num_heads, seq_len, head_dim, config_idx, config, config_file, success=False, error_msg=error_msg
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
    batch_size,
    num_heads,
    seq_len,
    head_dim,
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

    print(f"\n[Monitor] Started monitoring results for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
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
                    best_configs[str(seq_len)] = {
                        "config": best_config,
                        "time_ms": best_time,
                        "batch_size": batch_size,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
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
    print(f"[Monitor] RESULTS SUMMARY for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
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


def get_ff_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--seq-len",
        nargs="+",
        type=int,
        required=True,
        help="List of sequence length values separated by space",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=5,
        help="Number of attention heads (default: 5)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension (default: 128)",
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
        default="tuning_space_attn_qk_int8.json",
        help="Path to JSON file containing tuning space configuration (default: tuning_space_attn_qk_int8.json)",
    )
    parser.add_argument(
        "--max-configs-without-improvement",
        type=int,
        default=1000,
        help="Stop early if no improvement after this many configs (default: 1000)",
    )
    return parser.parse_args()


def parse_args():
    parser = get_parser(kernel_name="Attention QK INT8 Per Block")
    parser = add_argparse_ff(parser)
    return get_ff_args(parser)


# Set multiprocessing start method to spawn
try:
    set_start_method("spawn")
except RuntimeError:
    pass  # Already set


def main():
    args = parse_args()

    start_time = datetime.now()

    # Create shapes from seq_len list, batch_size, num_heads, head_dim
    shapes = [(args.batch_size, args.num_heads, seq_len, args.head_dim) for seq_len in args.seq_len]
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
    fname = f"best_configs_B{args.batch_size}_H{args.num_heads}_S{'_'.join(str(s) for s in args.seq_len)}_D{args.head_dim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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
    for batch_size, num_heads, seq_len, head_dim in shapes:
        print(f"\n{'='*80}")
        print(f"Starting tuning for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
        print(f"{'='*80}")

        # Load already tested config indices for this shape
        tested_indices = load_tested_configs(batch_size, num_heads, seq_len, head_dim, args.config_file)
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
        tested_configs_file = get_config_cache_filename(batch_size, num_heads, seq_len, head_dim)

        # Start the result monitor process
        from multiprocessing import Event

        stop_event = Event()

        monitor_process = Process(
            target=result_monitor,
            args=(
                results_queue,
                len(configs_to_test),
                batch_size,
                num_heads,
                seq_len,
                head_dim,
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
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
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

        print(f"[Main] Completed tuning for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}\n")

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
    batch_size,
    num_heads,
    seq_len,
    head_dim,
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
            results_queue, batch_size, num_heads, seq_len, head_dim, config, original_idx, config_file, device_id
        )


if __name__ == "__main__":
    sys.exit(main())

