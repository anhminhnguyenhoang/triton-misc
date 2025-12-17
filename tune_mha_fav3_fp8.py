"""
Tuning script for FlashAttention v3 FP8 (FAv3 FP8)

This script tunes the block configurations for FAv3 FP8 attention kernel.
Inputs are generated as float16 (same as FAv2) and then quantized to FP8 using the
_quantize_bshd function from aiter.ops.triton.mha_v3.
"""

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
import fcntl
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
from aiter.ops.triton.mha_v3 import _quantize_bshd
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
)
import itertools
import random


def get_tensors(batch_size, num_heads, seq_len, head_dim, device="cuda"):
    """
    Generate input tensors for FAv3 FP8 attention.
    
    Generates high-precision tensors (fp16 to match benchmark default) and quantizes them to FP8.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Device to create tensors on (default: cuda)
    
    Returns:
        Tuple of (q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale)
    """
    # Generate high precision tensors (fp16 to match benchmark default)
    q = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float16, device=device)
    k = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float16, device=device)
    v = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float16, device=device)
    
    # Quantize to FP8
    fp8_dtype = torch.float8_e4m3fnuz
    q_fp8, q_descale = _quantize_bshd(q, fp8_dtype)
    k_fp8, k_descale = _quantize_bshd(k, fp8_dtype)
    v_fp8, v_descale = _quantize_bshd(v, fp8_dtype)
    
    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


def get_autotune_config(config_file):
    """Load tuning space configuration from JSON file."""
    try:
        with open(config_file, "r") as f:
            tuning_space = json.load(f)
        print(f"Loaded FAv3 FP8 tuning space from '{config_file}'")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tuning space configuration file '{config_file}' not found."
        )

    # Extract parameter ranges from JSON
    block_m = tuning_space.get("BLOCK_M", [128])
    block_n = tuning_space.get("BLOCK_N", [128])
    pre_load_v = tuning_space.get("PRE_LOAD_V", [False])
    waves_per_eu = tuning_space.get("waves_per_eu", [2])
    num_warps = tuning_space.get("num_warps", [4])
    num_stages = tuning_space.get("num_stages", [1])

    configs = []
    for bm, bn, pv, wpe, nw, ns in itertools.product(
        block_m,
        block_n,
        pre_load_v,
        waves_per_eu,
        num_warps,
        num_stages,
    ):
        configs.append(
            dict(
                BLOCK_M=bm,
                BLOCK_N=bn,
                PRE_LOAD_V=pv,
                waves_per_eu=wpe,
                num_warps=nw,
                num_stages=ns,
            )
        )

    return configs


def get_config_cache_filename(batch_size, num_heads, seq_len, head_dim):
    """Generate cache filename for tracking tested configs."""
    return f"tested_configs_fav3_fp8_B{batch_size}_H{num_heads}_S{seq_len}_D{head_dim}.json"


def load_tested_configs(batch_size, num_heads, seq_len, head_dim, config_file):
    """Load set of already tested config indices from disk."""
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
    """Save a tested config to disk with file locking."""
    cache_file = get_config_cache_filename(batch_size, num_heads, seq_len, head_dim)

    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            with open(cache_file, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                try:
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

                    if "config_file" not in data:
                        data["config_file"] = config_file

                    config_key = str(config_idx)
                    if success:
                        data["tested_configs_successful"][config_key] = config
                    else:
                        data["tested_configs_failed"][config_key] = {
                            "config": config,
                            "error": error_msg,
                        }

                    data["total_successful"] = len(data["tested_configs_successful"])
                    data["total_failed"] = len(data["tested_configs_failed"])
                    data["total_tested"] = (
                        data["total_successful"] + data["total_failed"]
                    )

                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                    return

                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except (IOError, OSError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(
                    f"[WARNING] Failed to save tested config after {max_retries} retries: {e}"
                )
                raise
            import time

            time.sleep(0.1)


def bench_worker(
    results_queue, batch_size, num_heads, seq_len, head_dim, config, config_idx, config_file, device_id=None
):
    """Worker function to run FAv3 FP8 benchmark and put result in queue."""
    try:
        if device_id is not None:
            torch.cuda.set_device(device_id)

        # Clean up GPU core dump files
        for core_file in glob.glob("gpucore.*"):
            try:
                os.remove(core_file)
            except:
                pass

        # Generate inputs
        q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = get_tensors(batch_size, num_heads, seq_len, head_dim)
        
        # Benchmark with config
        # For FAv3 FP8, we need to temporarily replace the autotune configs
        from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import fwd_prefill
        
        # Save original autotune configs
        original_configs = fwd_prefill.fwd_prefill_autotune_configs
        
        # Create a single test config
        test_config = triton.Config(
            {
                "BLOCK_M": config["BLOCK_M"],
                "BLOCK_N": config["BLOCK_N"],
                "PRE_LOAD_V": config["PRE_LOAD_V"],
                "waves_per_eu": config["waves_per_eu"],
            },
            num_stages=config["num_stages"],
            num_warps=config["num_warps"],
        )
        
        # Temporarily replace autotune configs with our test config
        fwd_prefill.fwd_prefill_autotune_configs = [test_config]
        
        try:
            # Clear kernel cache to force recompilation with new config
            if hasattr(fwd_prefill.attn_fwd, 'cache'):
                fwd_prefill.attn_fwd.cache.clear()
            
            softmax_scale = head_dim ** (-0.5)
            
            ms = triton.testing.do_bench(
                lambda: flash_attn_3.fwd(
                    q_fp8, k_fp8, v_fp8,
                    None, None, None, None,  # k_new, v_new, qv, out
                    None, None, None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
                    None, None, None, None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
                    None, None, None,  # page_table, kv_batch_idx, leftpad_k
                    None, None, None,  # rotary_cos, rotary_sin, seqlens_rotary
                    q_descale, k_descale, v_descale,
                    softmax_scale,
                    False,  # causal
                    -1, -1,  # window_size
                    0,  # attention_chunk
                    0.0,  # softcap
                    False,  # rotary_interleaved
                    None, 1, None, 0,  # scheduler_metadata, num_splits, pack_gqa, sm_margin
                ),
                warmup=25,
                rep=100,
            )
        finally:
            # Restore original autotune configs
            fwd_prefill.fwd_prefill_autotune_configs = original_configs

        # Clean up core dumps
        for core_file in glob.glob("gpucore.*"):
            try:
                os.remove(core_file)
            except:
                pass

        save_tested_config(batch_size, num_heads, seq_len, head_dim, config_idx, config, config_file, success=True)

        results_queue.put(
            {"config_idx": config_idx, "config": config, "time_ms": ms, "success": True}
        )
    except (RuntimeError, OSError, MemoryError, Exception) as e:
        error_msg = str(e)

        core_files = glob.glob("gpucore.*")
        if core_files:
            error_msg = f"GPU memory access fault (core dumps: {len(core_files)}): {error_msg}"
            for core_file in core_files:
                try:
                    os.remove(core_file)
                except:
                    pass

        save_tested_config(
            batch_size, num_heads, seq_len, head_dim, config_idx, config, config_file, success=False, error_msg=error_msg
        )

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
    """Monitor results queue and update best config."""
    best_time = float("inf")
    best_config = None
    best_config_idx = None
    valid_count = 0
    failed_count = 0
    configs_since_improvement = 0

    try:
        with open(fname, "r") as f:
            best_configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_configs = {}

    print(f"\n[Monitor] Started monitoring FAv3 FP8 results for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    print(f"[Monitor] Waiting for {total_configs} benchmark results...")
    print(
        f"[Monitor] Will stop early if no improvement after {max_configs_without_improvement} configs"
    )

    while valid_count + failed_count < total_configs:
        try:
            result = results_queue.get(timeout=1.0)

            config_idx = result["config_idx"]
            config = result["config"]

            if result["success"]:
                time_ms = result["time_ms"]
                valid_count += 1

                if time_ms < best_time:
                    best_time = time_ms
                    best_config = config
                    best_config_idx = config_idx
                    configs_since_improvement = 0

                    best_configs[str(seq_len)] = {
                        "config": best_config,
                        "time_ms": best_time,
                        "batch_size": batch_size,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
                    }

                    with open(fname, "w") as f:
                        json.dump(best_configs, f, indent=2)

                    print(
                        f"[Monitor] ✓ New best! Config {config_idx+1}/{total_configs}: {time_ms:.3f} ms"
                    )
                else:
                    configs_since_improvement += 1

                    if configs_since_improvement >= max_configs_without_improvement:
                        print(
                            f"\n[Monitor] ⚠️  Early stopping: No improvement for {max_configs_without_improvement} configs"
                        )
                        print(
                            f"[Monitor] Best config found: {best_time:.3f} ms at index {best_config_idx + 1}"
                        )
                        stop_event.set()
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

                if configs_since_improvement >= max_configs_without_improvement:
                    print(
                        f"\n[Monitor] ⚠️  Early stopping: No improvement for {max_configs_without_improvement} configs"
                    )
                    if best_config is not None:
                        print(
                            f"[Monitor] Best config found: {best_time:.3f} ms at index {best_config_idx + 1}"
                        )
                    stop_event.set()
                    break

                if failed_count <= 5:
                    print(
                        f"[Monitor] ✗ Config {config_idx+1}/{total_configs} failed: {result.get('error', 'Unknown error')}"
                    )
                elif failed_count % 50 == 0:
                    print(
                        f"[Monitor] Progress: {valid_count + failed_count}/{total_configs} "
                        f"(Valid: {valid_count}, Failed: {failed_count})"
                    )

        except Empty:
            if stop_event and stop_event.is_set():
                print(f"[Monitor] Stop signal received, exiting...")
                break
            continue

    print(f"\n[Monitor] {'='*80}")
    print(f"[Monitor] FAv3 FP8 RESULTS SUMMARY for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
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
        default=16,
        help="Number of attention heads (default: 16)",
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
        default="tuning_space_mha_fav3_fp8.json",
        help="Path to JSON file containing tuning space configuration (default: tuning_space_mha_fav3_fp8.json)",
    )
    parser.add_argument(
        "--max-configs-without-improvement",
        type=int,
        default=1000,
        help="Stop early if no improvement after this many configs (default: 1000)",
    )
    return parser.parse_args()


def parse_args():
    parser = get_parser(kernel_name="FlashAttention v3 FP8 (FAv3 FP8)")
    parser = add_argparse_ff(parser)
    return get_ff_args(parser)


# Set multiprocessing start method to spawn
try:
    set_start_method("spawn")
except RuntimeError:
    pass


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
    """Worker process that handles a batch of configs."""
    if device_id is not None:
        torch.cuda.set_device(device_id)

    for item in configs_indexed:
        original_idx, config = item[0], item[1]
        if stop_event.is_set():
            print(f"[Worker GPU {device_id}] Received stop signal, terminating...")
            break

        bench_worker(
            results_queue, batch_size, num_heads, seq_len, head_dim, config, original_idx, config_file, device_id
        )


def main():
    args = parse_args()

    # Validate mutually exclusive arguments
    if args.num_workers is not None and args.workers is not None:
        raise ValueError("--num-workers and --workers are mutually exclusive. Please provide only one.")

    start_time = datetime.now()

    shapes = [(args.batch_size, args.num_heads, seq_len, args.head_dim) for seq_len in args.seq_len]
    print(f"Tuning FAv3 FP8 for shapes: {shapes}")

    if not shapes:
        print("No shapes to tune. Exiting.")
        return

    configs = get_autotune_config(args.config_file)

    num_configs = len(configs)
    config_indices = list(range(num_configs))
    random.shuffle(config_indices)

    print(f"Total configurations to tune: {len(configs)} (shuffled randomly)")

    fname = f"best_configs_fav3_fp8_B{args.batch_size}_H{args.num_heads}_S{'_'.join(str(s) for s in args.seq_len)}_D{args.head_dim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Determine number of workers and device IDs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.workers is not None:
        device_ids = args.workers
        num_workers = len(device_ids)
        print(f"Using {num_workers} worker(s) on GPU(s): {device_ids}")
    elif args.num_workers is not None:
        num_workers = args.num_workers
        device_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        if num_gpus > 0:
            print(f"Detected {num_gpus} GPU(s)")
            print(f"Using {num_workers} parallel workers with round-robin GPU assignment")
        else:
            print(f"No GPUs detected, using {num_workers} CPU workers")
    else:
        try:
            num_workers = num_gpus if num_gpus > 0 else max(1, cpu_count() // 4)
        except:
            num_workers = 4
        device_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        if num_gpus > 0:
            print(f"Detected {num_gpus} GPU(s)")
            print(f"Using {num_workers} parallel workers (auto-detected)")
        else:
            print(f"No GPUs detected, using {num_workers} CPU workers")

    for batch_size, num_heads, seq_len, head_dim in shapes:
        print(f"\n{'='*80}")
        print(f"Starting FAv3 FP8 tuning for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
        print(f"{'='*80}")

        tested_indices = load_tested_configs(batch_size, num_heads, seq_len, head_dim, args.config_file)
        if tested_indices:
            tested_indices = {int(idx) for idx in tested_indices}
            print(
                f"[Main] Found {len(tested_indices)} previously tested configs, will skip them"
            )

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

        results_queue = Queue()
        tested_configs_file = get_config_cache_filename(batch_size, num_heads, seq_len, head_dim)

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

        worker_processes = []
        configs_per_worker = len(configs_to_test) // num_workers
        remainder = len(configs_to_test) % num_workers

        config_start = 0
        for worker_id in range(num_workers):
            num_configs_for_worker = configs_per_worker + (
                1 if worker_id < remainder else 0
            )
            config_end = config_start + num_configs_for_worker

            worker_configs_indexed = configs_to_test[config_start:config_end]

            # Assign device ID based on --workers or round-robin
            if device_ids is not None:
                device_id = device_ids[worker_id % len(device_ids)]
            else:
                device_id = None

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
                    device_id,
                ),
            )
            worker_processes.append(worker_process)
            worker_process.start()

            config_start = config_end

        for worker_process in worker_processes:
            worker_process.join()

        monitor_process.join()

        print(f"[Main] Completed FAv3 FP8 tuning for B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}\n")

    end_time = datetime.now()
    total_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"FAv3 FP8 TUNING COMPLETE")
    print(f"  Total time: {total_time}")
    print(f"  Shapes tuned: {len(shapes)}")
    print(f"  Configs per shape: {len(configs)}")
    print(f"  Total benchmarks: {len(shapes) * len(configs)}")
    print(f"  Results saved to: {fname}")
    print(f"{'='*80}")


if __name__ == "__main__":
    sys.exit(main())

