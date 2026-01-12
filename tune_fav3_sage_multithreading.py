"""
Multi-threaded tuning script for fav3_sage triton kernel.

This script performs systematic exploration of kernel configurations for the fav3_sage
attention kernel using multi-GPU parallel workers.

Usage:
    python tune_fav3_sage_multithreading.py --seq-len 4096 8192 --batch-size 1 --num-heads 5 --head-dim 128

Example with specific workers:
    python tune_fav3_sage_multithreading.py --seq-len 4096 --workers 0 1 2 3
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

from aiter.ops.triton.fav3_sage import (
    fav3_sage_wrapper_func,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
)


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
    block_m = tuning_space.get("BLOCK_M", [256])
    block_n = tuning_space.get("BLOCK_N", [128])
    num_warps = tuning_space.get("num_warps", [8])
    num_stages = tuning_space.get("num_stages", [2])
    waves_per_eu = tuning_space.get("waves_per_eu", [2])
    pre_load_v = tuning_space.get("PRE_LOAD_V", [False])

    configs = []
    for bm, bn, nw, ns, wpe, plv in itertools.product(
        block_m,
        block_n,
        num_warps,
        num_stages,
        waves_per_eu,
        pre_load_v,
    ):
        configs.append(
            dict(
                BLOCK_M=bm,
                BLOCK_N=bn,
                num_warps=nw,
                num_stages=ns,
                waves_per_eu=wpe,
                PRE_LOAD_V=plv,
            )
        )

    return configs


def get_tensors(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout="bshd"):
    """
    Generate high-precision tensors for fav3_sage_wrapper_func benchmarking.
    
    The wrapper function handles quantization internally.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads  
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        head_dim: Head dimension
        layout: Tensor layout - "bshd" or "bhsd"
        
    Returns:
        Tuple of (q, k, v) in bfloat16
    """
    # Generate random tensors in the specified layout
    if layout == "bshd":
        q = torch.randn((batch_size, seq_len_q, num_heads, head_dim), 
                       device='cuda', dtype=torch.bfloat16)
        k = torch.randn((batch_size, seq_len_k, num_heads, head_dim), 
                       device='cuda', dtype=torch.bfloat16)
        v = torch.randn((batch_size, seq_len_k, num_heads, head_dim), 
                       device='cuda', dtype=torch.bfloat16)
    else:  # bhsd
        q = torch.randn((batch_size, num_heads, seq_len_q, head_dim), 
                       device='cuda', dtype=torch.bfloat16)
        k = torch.randn((batch_size, num_heads, seq_len_k, head_dim), 
                       device='cuda', dtype=torch.bfloat16)
        v = torch.randn((batch_size, num_heads, seq_len_k, head_dim), 
                       device='cuda', dtype=torch.bfloat16)
    
    return q, k, v


def get_config_cache_filename(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout):
    """Generate cache filename for tracking tested configs."""
    return f"tested_configs_fav3_sage_B{batch_size}_H{num_heads}_Sq{seq_len_q}_Sk{seq_len_k}_D{head_dim}_{layout}.json"


def load_tested_configs(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout, config_file):
    """
    Load set of already tested config indices from disk.
    Returns empty set if config_file doesn't match (different tuning space).
    """
    cache_file = get_config_cache_filename(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout)
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
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout, config_idx, config, config_file, success=True, error_msg=None
):
    """
    Save a tested config to disk with file locking to prevent race conditions.
    Separates successful and failed configs for easy inspection.
    """
    cache_file = get_config_cache_filename(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout)

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
                            "seq_len_q": seq_len_q,
                            "seq_len_k": seq_len_k,
                            "head_dim": head_dim,
                            "layout": layout,
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
    results_queue, batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout, config, config_idx, config_file, device_id=None
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

        # Generate high-precision inputs (wrapper handles quantization internally)
        q, k, v = get_tensors(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout)
        
        # Benchmark with config
        ms = triton.testing.do_bench(
            lambda: fav3_sage_wrapper_func(
                q, k, v,
                causal=False,
                inference_mode=True,
                layout=layout,
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
        save_tested_config(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout, config_idx, config, config_file, success=True)

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
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout, config_idx, config, config_file, success=False, error_msg=error_msg
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
    seq_len_q,
    seq_len_k,
    head_dim,
    layout,
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

    # Load existing best configs if available
    try:
        with open(fname, "r") as f:
            best_configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_configs = {}

    print(f"\n[Monitor] Started monitoring results for B={batch_size}, H={num_heads}, Sq={seq_len_q}, Sk={seq_len_k}, D={head_dim}, layout={layout}")
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
                    config_key = f"{seq_len_q}x{seq_len_k}"
                    best_configs[config_key] = {
                        "config": best_config,
                        "time_ms": best_time,
                        "batch_size": batch_size,
                        "num_heads": num_heads,
                        "seq_len_q": seq_len_q,
                        "seq_len_k": seq_len_k,
                        "head_dim": head_dim,
                        "layout": layout,
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
    print(f"[Monitor] RESULTS SUMMARY for B={batch_size}, H={num_heads}, Sq={seq_len_q}, Sk={seq_len_k}, D={head_dim}, layout={layout}")
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


def worker_batch(
    results_queue,
    batch_size,
    num_heads,
    seq_len_q,
    seq_len_k,
    head_dim,
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

    for item in configs_indexed:
        original_idx, config = item[0], item[1]
        # Check if we should stop
        if stop_event.is_set():
            print(f"[Worker GPU {device_id}] Received stop signal, terminating...")
            break

        bench_worker(
            results_queue, batch_size, num_heads, seq_len_q, seq_len_k, head_dim, layout, config, original_idx, config_file, device_id
        )


def parse_seq_len(value):
    """Parse sequence length as 'seq_q,seq_k' tuple or single int (same for both)."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Expected format 'seq_q,seq_k', got '{value}'")
        return (int(parts[0]), int(parts[1]))
    else:
        seq = int(value)
        return (seq, seq)


def get_ff_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--seq-len",
        nargs="+",
        type=parse_seq_len,
        required=True,
        help="List of sequence lengths as 'seq_q,seq_k' tuples or single int. Examples: '4096' '4096,8192' '1024,2048'",
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
        "--tensor-layout",
        type=str,
        default="bshd",
        choices=["bshd", "bhsd"],
        help="Tensor layout: bshd or bhsd (default: bshd)",
    )
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
    parser.add_argument(
        "--config-file",
        type=str,
        default="tuning_space_fav3_sage.json",
        help="Path to JSON file containing tuning space configuration (default: tuning_space_fav3_sage.json)",
    )
    parser.add_argument(
        "--max-configs-without-improvement",
        type=int,
        default=1000,
        help="Stop early if no improvement after this many configs (default: 1000)",
    )
    return parser.parse_args()


def parse_args():
    parser = get_parser(kernel_name="FAv3 Sage Attention")
    return get_ff_args(parser)


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

    start_time = datetime.now()

    # Create shapes from seq_len list (tuples of seq_q,seq_k), batch_size, num_heads, head_dim
    shapes = [(args.batch_size, args.num_heads, seq_q, seq_k, args.head_dim) for seq_q, seq_k in args.seq_len]
    print(f"Tuning for shapes (B, H, Sq, Sk, D): {shapes}")
    print(f"Layout: {args.tensor_layout}")

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
    seq_str = '_'.join(f"{sq}x{sk}" for sq, sk in args.seq_len)
    fname = f"best_configs_fav3_sage_B{args.batch_size}_H{args.num_heads}_S{seq_str}_D{args.head_dim}_{args.tensor_layout}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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

    # Process each shape
    for batch_size, num_heads, seq_len_q, seq_len_k, head_dim in shapes:
        print(f"\n{'='*80}")
        print(f"Starting tuning for B={batch_size}, H={num_heads}, Sq={seq_len_q}, Sk={seq_len_k}, D={head_dim}, layout={args.tensor_layout}")
        print(f"{'='*80}")

        # Load already tested config indices for this shape
        tested_indices = load_tested_configs(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, args.tensor_layout, args.config_file)
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
        tested_configs_file = get_config_cache_filename(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, args.tensor_layout)

        # Start the result monitor process
        stop_event = Event()

        monitor_process = Process(
            target=result_monitor,
            args=(
                results_queue,
                len(configs_to_test),
                batch_size,
                num_heads,
                seq_len_q,
                seq_len_k,
                head_dim,
                args.tensor_layout,
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

            # Create worker process that will handle multiple configs
            worker_process = Process(
                target=worker_batch,
                args=(
                    results_queue,
                    batch_size,
                    num_heads,
                    seq_len_q,
                    seq_len_k,
                    head_dim,
                    args.tensor_layout,
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

        print(f"[Main] Completed tuning for B={batch_size}, H={num_heads}, Sq={seq_len_q}, Sk={seq_len_k}, D={head_dim}, layout={args.tensor_layout}\n")

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


if __name__ == "__main__":
    sys.exit(main())
