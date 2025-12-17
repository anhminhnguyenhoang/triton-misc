#!/usr/bin/env python3
"""
Script to analyze captured attention input tensors (q, k, v) from diffusion model inference.

Auto-discovers captured input files and generates distribution analysis plots.

Captured inputs are in BHSD format (batch, heads, seqlen, dim) as saved by
op_tests/sagev1_tests/sageattn_cogvideo.py InputCaptureWrapper.

Usage:
    python analyze_captured_inputs.py
    python analyze_captured_inputs.py --input_dir ./my_inputs
    python analyze_captured_inputs.py --max_files 100
"""

import os
import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


# Default input directory
DEFAULT_INPUT_DIR = Path("./captured_inputs")


def parse_input_filename(filename: str) -> Optional[Dict]:
    """
    Parse captured input filename to extract call index.
    
    Pattern: {kernel_name}_input_{call_idx:06d}.pt
    Example: sagev1_input_000042.pt
    """
    pattern = r"(.+)_input_(\d+)\.pt"
    match = re.match(pattern, filename)
    if match:
        return {
            "kernel_name": match.group(1),
            "call_idx": int(match.group(2)),
        }
    return None


def discover_input_files(input_dir: Path) -> List[Path]:
    """
    Discover all captured input .pt files in the directory.
    
    Returns:
        Sorted list of file paths
    """
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return []
    
    # Find all *_input_*.pt files
    files = list(input_dir.glob("*_input_*.pt"))
    
    # Sort by call index
    def get_call_idx(path: Path) -> int:
        parsed = parse_input_filename(path.name)
        return parsed["call_idx"] if parsed else 0
    
    return sorted(files, key=get_call_idx)


def load_captured_input(file_path: Path) -> Dict[str, Any]:
    """
    Load a captured input file.
    
    Returns dict with:
        q, k, v: tensors
        q_shape, k_shape, v_shape: shape lists
        dtype: string
        call_idx: int
        kwargs: dict
    """
    return torch.load(file_path, weights_only=False)


def load_metadata(input_dir: Path) -> Optional[Dict]:
    """Load metadata file if it exists."""
    metadata_files = list(input_dir.glob("*_metadata.pt"))
    if metadata_files:
        return torch.load(metadata_files[0], weights_only=False)
    return None


def get_shape_key(data: Dict) -> Tuple:
    """Generate a hashable key from shapes."""
    return (
        tuple(data["q_shape"]),
        tuple(data["k_shape"]),
        tuple(data["v_shape"]),
        data.get("dtype", "unknown")
    )


def print_tensor_statistics(name: str, tensor: torch.Tensor, indent: int = 2):
    """Print summary statistics for a tensor."""
    flat = tensor.float().flatten().numpy()
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  Shape: {tuple(tensor.shape)}")
    print(f"{prefix}  Dtype: {tensor.dtype}")
    print(f"{prefix}  Mean:  {flat.mean():.6f}")
    print(f"{prefix}  Std:   {flat.std():.6f}")
    print(f"{prefix}  Min:   {flat.min():.6f}")
    print(f"{prefix}  Max:   {flat.max():.6f}")
    print(f"{prefix}  P1:    {np.percentile(flat, 1):.6f}")
    print(f"{prefix}  P50:   {np.percentile(flat, 50):.6f}")
    print(f"{prefix}  P99:   {np.percentile(flat, 99):.6f}")


def collect_statistics(inputs: List[Dict]) -> Dict[str, List]:
    """
    Collect statistics across all inputs.
    
    Returns dict with lists of statistics per input.
    """
    stats = {
        "call_idx": [],
        "q_mean": [], "q_std": [], "q_min": [], "q_max": [],
        "k_mean": [], "k_std": [], "k_min": [], "k_max": [],
        "v_mean": [], "v_std": [], "v_min": [], "v_max": [],
        "batch": [], "heads": [], "seq_q": [], "seq_k": [], "dim": [],
    }
    
    for data in inputs:
        q = data["q"].float()
        k = data["k"].float()
        v = data["v"].float()
        
        stats["call_idx"].append(data.get("call_idx", 0))
        
        # Q stats
        stats["q_mean"].append(q.mean().item())
        stats["q_std"].append(q.std().item())
        stats["q_min"].append(q.min().item())
        stats["q_max"].append(q.max().item())
        
        # K stats
        stats["k_mean"].append(k.mean().item())
        stats["k_std"].append(k.std().item())
        stats["k_min"].append(k.min().item())
        stats["k_max"].append(k.max().item())
        
        # V stats
        stats["v_mean"].append(v.mean().item())
        stats["v_std"].append(v.std().item())
        stats["v_min"].append(v.min().item())
        stats["v_max"].append(v.max().item())
        
        # Shape info (BHSD format)
        q_shape = data["q_shape"]
        k_shape = data["k_shape"]
        stats["batch"].append(q_shape[0])
        stats["heads"].append(q_shape[1])
        stats["seq_q"].append(q_shape[2])
        stats["seq_k"].append(k_shape[2])
        stats["dim"].append(q_shape[3])
    
    return stats


def analyze_shapes(inputs: List[Dict]) -> Dict[Tuple, List[int]]:
    """
    Analyze shape distribution across inputs.
    
    Returns dict mapping shape_key -> list of call indices
    """
    shape_groups = defaultdict(list)
    
    for data in inputs:
        key = get_shape_key(data)
        shape_groups[key].append(data.get("call_idx", 0))
    
    return dict(shape_groups)


def plot_value_histograms(inputs: List[Dict], plots_dir: Path):
    """Plot overlaid histograms of q, k, v value distributions."""
    # Subsample values for efficiency
    max_samples = 100000
    
    q_vals = []
    k_vals = []
    v_vals = []
    
    for data in inputs:
        q_flat = data["q"].float().flatten().numpy()
        k_flat = data["k"].float().flatten().numpy()
        v_flat = data["v"].float().flatten().numpy()
        
        # Random subsample if too large
        if len(q_flat) > max_samples // len(inputs):
            idx = np.random.choice(len(q_flat), max_samples // len(inputs), replace=False)
            q_flat = q_flat[idx]
            k_flat = k_flat[idx]
            v_flat = v_flat[idx]
        
        q_vals.append(q_flat)
        k_vals.append(k_flat)
        v_vals.append(v_flat)
    
    q_all = np.concatenate(q_vals)
    k_all = np.concatenate(k_vals)
    v_all = np.concatenate(v_vals)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Captured Input Value Distributions", fontsize=12)
    
    # Q histogram
    ax = axes[0]
    ax.hist(q_all, bins=100, alpha=0.7, color="blue", density=True)
    ax.set_title(f"Q Values (n={len(q_all):,})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.axvline(q_all.mean(), color='red', linestyle='--', label=f'mean={q_all.mean():.3f}')
    ax.legend()
    
    # K histogram
    ax = axes[1]
    ax.hist(k_all, bins=100, alpha=0.7, color="green", density=True)
    ax.set_title(f"K Values (n={len(k_all):,})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.axvline(k_all.mean(), color='red', linestyle='--', label=f'mean={k_all.mean():.3f}')
    ax.legend()
    
    # V histogram
    ax = axes[2]
    ax.hist(v_all, bins=100, alpha=0.7, color="orange", density=True)
    ax.set_title(f"V Values (n={len(v_all):,})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.axvline(v_all.mean(), color='red', linestyle='--', label=f'mean={v_all.mean():.3f}')
    ax.legend()
    
    plt.tight_layout()
    save_path = plots_dir / "histogram_qkv_values.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def plot_qkv_comparison_boxplots(inputs: List[Dict], plots_dir: Path):
    """Plot box plots comparing q, k, v distributions."""
    # Subsample for efficiency
    max_samples = 50000
    
    q_vals = []
    k_vals = []
    v_vals = []
    
    for data in inputs:
        q_flat = data["q"].float().flatten().numpy()
        k_flat = data["k"].float().flatten().numpy()
        v_flat = data["v"].float().flatten().numpy()
        
        n_sample = min(max_samples // len(inputs), len(q_flat))
        idx = np.random.choice(len(q_flat), n_sample, replace=False)
        q_vals.append(q_flat[idx])
        k_vals.append(k_flat[idx])
        v_vals.append(v_flat[idx])
    
    q_all = np.concatenate(q_vals)
    k_all = np.concatenate(k_vals)
    v_all = np.concatenate(v_vals)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([q_all, k_all, v_all], labels=["Q", "K", "V"])
    ax.set_title("Q, K, V Value Distributions")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / "boxplot_qkv_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def plot_per_head_statistics(inputs: List[Dict], plots_dir: Path):
    """Plot per-head mean/std distribution using a representative input."""
    # Use first input as representative
    if not inputs:
        return None
    
    data = inputs[0]
    q = data["q"].float()  # BHSD: (batch, heads, seq, dim)
    k = data["k"].float()
    v = data["v"].float()
    
    n_heads = q.shape[1]
    
    # Compute per-head statistics
    q_means = [q[:, h, :, :].mean().item() for h in range(n_heads)]
    k_means = [k[:, h, :, :].mean().item() for h in range(n_heads)]
    v_means = [v[:, h, :, :].mean().item() for h in range(n_heads)]
    
    q_stds = [q[:, h, :, :].std().item() for h in range(n_heads)]
    k_stds = [k[:, h, :, :].std().item() for h in range(n_heads)]
    v_stds = [v[:, h, :, :].std().item() for h in range(n_heads)]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Per-Head Statistics (Input 0, {n_heads} heads)", fontsize=12)
    
    x = np.arange(n_heads)
    width = 0.25
    
    # Means
    ax = axes[0]
    ax.bar(x - width, q_means, width, label='Q', color='blue', alpha=0.7)
    ax.bar(x, k_means, width, label='K', color='green', alpha=0.7)
    ax.bar(x + width, v_means, width, label='V', color='orange', alpha=0.7)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Mean")
    ax.set_title("Per-Head Mean Values")
    ax.legend()
    ax.set_xticks(x[::max(1, n_heads//20)])  # Show subset of ticks
    ax.grid(True, alpha=0.3)
    
    # Stds
    ax = axes[1]
    ax.bar(x - width, q_stds, width, label='Q', color='blue', alpha=0.7)
    ax.bar(x, k_stds, width, label='K', color='green', alpha=0.7)
    ax.bar(x + width, v_stds, width, label='V', color='orange', alpha=0.7)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Std Dev")
    ax.set_title("Per-Head Standard Deviation")
    ax.legend()
    ax.set_xticks(x[::max(1, n_heads//20)])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / "per_head_statistics.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def plot_statistics_over_time(stats: Dict[str, List], plots_dir: Path):
    """Plot how statistics evolve across call indices."""
    if not stats["call_idx"]:
        return None
    
    call_idx = np.array(stats["call_idx"])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Statistics Over Call Index", fontsize=12)
    
    # Mean over time
    ax = axes[0, 0]
    ax.plot(call_idx, stats["q_mean"], 'b-', label='Q', alpha=0.7)
    ax.plot(call_idx, stats["k_mean"], 'g-', label='K', alpha=0.7)
    ax.plot(call_idx, stats["v_mean"], 'orange', label='V', alpha=0.7)
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Mean")
    ax.set_title("Mean Value Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Std over time
    ax = axes[0, 1]
    ax.plot(call_idx, stats["q_std"], 'b-', label='Q', alpha=0.7)
    ax.plot(call_idx, stats["k_std"], 'g-', label='K', alpha=0.7)
    ax.plot(call_idx, stats["v_std"], 'orange', label='V', alpha=0.7)
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Std Dev")
    ax.set_title("Standard Deviation Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Min/Max range over time
    ax = axes[1, 0]
    ax.fill_between(call_idx, stats["q_min"], stats["q_max"], alpha=0.3, color='blue', label='Q range')
    ax.fill_between(call_idx, stats["k_min"], stats["k_max"], alpha=0.3, color='green', label='K range')
    ax.fill_between(call_idx, stats["v_min"], stats["v_max"], alpha=0.3, color='orange', label='V range')
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Value")
    ax.set_title("Min/Max Range Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sequence lengths over time
    ax = axes[1, 1]
    ax.plot(call_idx, stats["seq_q"], 'b-', label='Seq Q', marker='o', markersize=3)
    ax.plot(call_idx, stats["seq_k"], 'g-', label='Seq K', marker='x', markersize=3)
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Sequence Length")
    ax.set_title("Sequence Lengths Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / "statistics_over_time.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def plot_shape_timeline(stats: Dict[str, List], plots_dir: Path):
    """Visualize how shapes change during inference."""
    if not stats["call_idx"]:
        return None
    
    call_idx = np.array(stats["call_idx"])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Shape Dimensions Over Time", fontsize=12)
    
    # Batch size
    ax = axes[0, 0]
    ax.plot(call_idx, stats["batch"], 'b-', marker='o', markersize=3)
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Batch Size")
    ax.set_title("Batch Size")
    ax.grid(True, alpha=0.3)
    
    # Number of heads
    ax = axes[0, 1]
    ax.plot(call_idx, stats["heads"], 'g-', marker='o', markersize=3)
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Num Heads")
    ax.set_title("Number of Heads")
    ax.grid(True, alpha=0.3)
    
    # Sequence lengths
    ax = axes[1, 0]
    ax.plot(call_idx, stats["seq_q"], 'b-', label='Seq Q', marker='o', markersize=3)
    ax.plot(call_idx, stats["seq_k"], 'r-', label='Seq K', marker='x', markersize=3)
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Sequence Length")
    ax.set_title("Sequence Lengths")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Head dimension
    ax = axes[1, 1]
    ax.plot(call_idx, stats["dim"], 'purple', marker='o', markersize=3)
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Head Dim")
    ax.set_title("Head Dimension")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / "shape_timeline.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def generate_random_inputs_like_captured(inputs: List[Dict], seed: int = 20) -> List[Dict]:
    """
    Generate random inputs matching the shapes of captured inputs.
    
    Uses the same generation method as bench_diffusion_attention.py:
    - torch.manual_seed(seed)
    - torch.randn in BSHD format (batch, seq, heads, dim)
    - Then transpose to BHSD format (batch, heads, seq, dim) for comparison
    
    Args:
        inputs: List of captured input dicts with q_shape, k_shape, v_shape, dtype
        seed: Random seed (default 20 to match benchmark)
    
    Returns:
        List of dicts with generated q, k, v tensors in BHSD format
    """
    torch.manual_seed(seed)
    
    generated = []
    for data in inputs:
        # Parse shapes (BHSD format: batch, heads, seq, dim)
        q_shape = data["q_shape"]  # [batch, heads, seq_q, dim]
        k_shape = data["k_shape"]  # [batch, heads, seq_k, dim]
        v_shape = data["v_shape"]  # [batch, heads, seq_k, dim_v]
        
        batch = q_shape[0]
        heads_q = q_shape[1]
        heads_k = k_shape[1]
        seq_q = q_shape[2]
        seq_k = k_shape[2]
        dim = q_shape[3]
        dim_v = v_shape[3]
        
        # Determine dtype
        dtype_str = data.get("dtype", "torch.float16")
        if "float16" in dtype_str:
            dtype = torch.float16
        elif "bfloat16" in dtype_str:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Generate in BSHD format (like benchmark does)
        # q = torch.randn((BATCH, N_CTX_Q, HQ, D_HEAD), device=device, dtype=dtype)
        q_bshd = torch.randn((batch, seq_q, heads_q, dim), dtype=dtype)
        k_bshd = torch.randn((batch, seq_k, heads_k, dim), dtype=dtype)
        v_bshd = torch.randn((batch, seq_k, heads_k, dim_v), dtype=dtype)
        
        # Transpose to BHSD format for comparison with captured inputs
        q_bhsd = q_bshd.transpose(1, 2).contiguous()
        k_bhsd = k_bshd.transpose(1, 2).contiguous()
        v_bhsd = v_bshd.transpose(1, 2).contiguous()
        
        generated.append({
            "q": q_bhsd,
            "k": k_bhsd,
            "v": v_bhsd,
            "q_shape": list(q_bhsd.shape),
            "k_shape": list(k_bhsd.shape),
            "v_shape": list(v_bhsd.shape),
            "dtype": dtype_str,
            "call_idx": data.get("call_idx", 0),
        })
    
    return generated


def print_comparison_statistics(captured_stats: Dict[str, List], generated_stats: Dict[str, List]):
    """Print side-by-side comparison of captured vs generated input statistics."""
    
    def fmt(val):
        return f"{val:+.6f}" if val >= 0 else f"{val:.6f}"
    
    print("\n  Comparison Table (Captured vs Generated with seed=20):")
    print("  " + "-" * 70)
    print(f"  {'Metric':<25} {'Captured':>15} {'Generated':>15} {'Diff':>12}")
    print("  " + "-" * 70)
    
    # Q tensor
    cap_q_mean = np.mean(captured_stats['q_mean'])
    gen_q_mean = np.mean(generated_stats['q_mean'])
    cap_q_std = np.mean(captured_stats['q_std'])
    gen_q_std = np.mean(generated_stats['q_std'])
    
    print(f"  {'Q mean of means':<25} {cap_q_mean:>15.6f} {gen_q_mean:>15.6f} {cap_q_mean - gen_q_mean:>+12.6f}")
    print(f"  {'Q mean of stds':<25} {cap_q_std:>15.6f} {gen_q_std:>15.6f} {cap_q_std - gen_q_std:>+12.6f}")
    print(f"  {'Q global min':<25} {np.min(captured_stats['q_min']):>15.6f} {np.min(generated_stats['q_min']):>15.6f} {np.min(captured_stats['q_min']) - np.min(generated_stats['q_min']):>+12.6f}")
    print(f"  {'Q global max':<25} {np.max(captured_stats['q_max']):>15.6f} {np.max(generated_stats['q_max']):>15.6f} {np.max(captured_stats['q_max']) - np.max(generated_stats['q_max']):>+12.6f}")
    
    print("  " + "-" * 70)
    
    # K tensor
    cap_k_mean = np.mean(captured_stats['k_mean'])
    gen_k_mean = np.mean(generated_stats['k_mean'])
    cap_k_std = np.mean(captured_stats['k_std'])
    gen_k_std = np.mean(generated_stats['k_std'])
    
    print(f"  {'K mean of means':<25} {cap_k_mean:>15.6f} {gen_k_mean:>15.6f} {cap_k_mean - gen_k_mean:>+12.6f}")
    print(f"  {'K mean of stds':<25} {cap_k_std:>15.6f} {gen_k_std:>15.6f} {cap_k_std - gen_k_std:>+12.6f}")
    print(f"  {'K global min':<25} {np.min(captured_stats['k_min']):>15.6f} {np.min(generated_stats['k_min']):>15.6f} {np.min(captured_stats['k_min']) - np.min(generated_stats['k_min']):>+12.6f}")
    print(f"  {'K global max':<25} {np.max(captured_stats['k_max']):>15.6f} {np.max(generated_stats['k_max']):>15.6f} {np.max(captured_stats['k_max']) - np.max(generated_stats['k_max']):>+12.6f}")
    
    print("  " + "-" * 70)
    
    # V tensor
    cap_v_mean = np.mean(captured_stats['v_mean'])
    gen_v_mean = np.mean(generated_stats['v_mean'])
    cap_v_std = np.mean(captured_stats['v_std'])
    gen_v_std = np.mean(generated_stats['v_std'])
    
    print(f"  {'V mean of means':<25} {cap_v_mean:>15.6f} {gen_v_mean:>15.6f} {cap_v_mean - gen_v_mean:>+12.6f}")
    print(f"  {'V mean of stds':<25} {cap_v_std:>15.6f} {gen_v_std:>15.6f} {cap_v_std - gen_v_std:>+12.6f}")
    print(f"  {'V global min':<25} {np.min(captured_stats['v_min']):>15.6f} {np.min(generated_stats['v_min']):>15.6f} {np.min(captured_stats['v_min']) - np.min(generated_stats['v_min']):>+12.6f}")
    print(f"  {'V global max':<25} {np.max(captured_stats['v_max']):>15.6f} {np.max(generated_stats['v_max']):>15.6f} {np.max(captured_stats['v_max']) - np.max(generated_stats['v_max']):>+12.6f}")
    
    print("  " + "-" * 70)


def plot_captured_vs_generated_histograms(
    captured_inputs: List[Dict],
    generated_inputs: List[Dict],
    plots_dir: Path
):
    """Plot overlaid histograms comparing captured vs generated input distributions."""
    max_samples = 100000
    
    def extract_values(inputs: List[Dict], tensor_key: str) -> np.ndarray:
        vals = []
        for data in inputs:
            flat = data[tensor_key].float().flatten().numpy()
            n_sample = min(max_samples // len(inputs), len(flat))
            if n_sample < len(flat):
                idx = np.random.choice(len(flat), n_sample, replace=False)
                flat = flat[idx]
            vals.append(flat)
        return np.concatenate(vals)
    
    cap_q = extract_values(captured_inputs, "q")
    cap_k = extract_values(captured_inputs, "k")
    cap_v = extract_values(captured_inputs, "v")
    
    gen_q = extract_values(generated_inputs, "q")
    gen_k = extract_values(generated_inputs, "k")
    gen_v = extract_values(generated_inputs, "v")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Captured vs Generated (seed=20) Value Distributions", fontsize=12)
    
    # Q histogram
    ax = axes[0]
    ax.hist(cap_q, bins=100, alpha=0.5, color="blue", density=True, label=f'Captured (n={len(cap_q):,})')
    ax.hist(gen_q, bins=100, alpha=0.5, color="red", density=True, label=f'Generated (n={len(gen_q):,})')
    ax.set_title("Q Values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
    # K histogram
    ax = axes[1]
    ax.hist(cap_k, bins=100, alpha=0.5, color="blue", density=True, label='Captured')
    ax.hist(gen_k, bins=100, alpha=0.5, color="red", density=True, label='Generated')
    ax.set_title("K Values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
    # V histogram
    ax = axes[2]
    ax.hist(cap_v, bins=100, alpha=0.5, color="blue", density=True, label='Captured')
    ax.hist(gen_v, bins=100, alpha=0.5, color="red", density=True, label='Generated')
    ax.set_title("V Values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.tight_layout()
    save_path = plots_dir / "histogram_captured_vs_generated.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def plot_captured_vs_generated_boxplots(
    captured_inputs: List[Dict],
    generated_inputs: List[Dict],
    plots_dir: Path
):
    """Plot box plots comparing captured vs generated distributions."""
    max_samples = 50000
    
    def extract_values(inputs: List[Dict], tensor_key: str) -> np.ndarray:
        vals = []
        for data in inputs:
            flat = data[tensor_key].float().flatten().numpy()
            n_sample = min(max_samples // len(inputs), len(flat))
            if n_sample < len(flat):
                idx = np.random.choice(len(flat), n_sample, replace=False)
                flat = flat[idx]
            vals.append(flat)
        return np.concatenate(vals)
    
    cap_q = extract_values(captured_inputs, "q")
    cap_k = extract_values(captured_inputs, "k")
    cap_v = extract_values(captured_inputs, "v")
    
    gen_q = extract_values(generated_inputs, "q")
    gen_k = extract_values(generated_inputs, "k")
    gen_v = extract_values(generated_inputs, "v")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = [cap_q, gen_q, cap_k, gen_k, cap_v, gen_v]
    labels = ["Q\n(Captured)", "Q\n(Generated)", "K\n(Captured)", "K\n(Generated)", "V\n(Captured)", "V\n(Generated)"]
    colors = ['blue', 'red', 'blue', 'red', 'blue', 'red']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax.set_title("Captured vs Generated (seed=20) Value Distributions")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label='Captured'),
        Patch(facecolor='red', alpha=0.5, label='Generated (seed=20)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    save_path = plots_dir / "boxplot_captured_vs_generated.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def plot_stats_comparison_over_time(
    captured_stats: Dict[str, List],
    generated_stats: Dict[str, List],
    plots_dir: Path
):
    """Plot captured vs generated statistics over call index."""
    if not captured_stats["call_idx"]:
        return None
    
    call_idx = np.array(captured_stats["call_idx"])
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Captured vs Generated Statistics Over Call Index", fontsize=12)
    
    tensor_names = ['q', 'k', 'v']
    colors = ['blue', 'green', 'orange']
    
    for i, (tensor, color) in enumerate(zip(tensor_names, colors)):
        # Mean comparison
        ax = axes[0, i]
        ax.plot(call_idx, captured_stats[f'{tensor}_mean'], f'{color}', label='Captured', alpha=0.7)
        ax.plot(call_idx, generated_stats[f'{tensor}_mean'], 'r--', label='Generated', alpha=0.7)
        ax.set_xlabel("Call Index")
        ax.set_ylabel("Mean")
        ax.set_title(f"{tensor.upper()} Mean")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Std comparison
        ax = axes[1, i]
        ax.plot(call_idx, captured_stats[f'{tensor}_std'], f'{color}', label='Captured', alpha=0.7)
        ax.plot(call_idx, generated_stats[f'{tensor}_std'], 'r--', label='Generated', alpha=0.7)
        ax.set_xlabel("Call Index")
        ax.set_ylabel("Std Dev")
        ax.set_title(f"{tensor.upper()} Std Dev")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plots_dir / "stats_captured_vs_generated_over_time.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze captured attention input tensors (q, k, v) from diffusion model inference."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing captured input .pt files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of input files to analyze (default: all)"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--compare_generated",
        action="store_true",
        help="Generate random inputs (seed=20) matching captured shapes and compare statistics"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20,
        help="Random seed for generating comparison inputs (default: 20 to match benchmark)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    
    print("=" * 60)
    print("Captured Input Tensor Analysis")
    print("=" * 60)
    print(f"\nInput directory: {input_dir}")
    
    # Discover input files
    input_files = discover_input_files(input_dir)
    
    if not input_files:
        print("\nNo captured input files found!")
        print("Run the capture script first:")
        print("  python op_tests/sagev1_tests/sageattn_cogvideo.py --save_inputs --input_dir ./captured_inputs")
        return 1
    
    print(f"Found {len(input_files)} input files")
    
    # Apply max_files limit
    if args.max_files and args.max_files < len(input_files):
        print(f"Limiting analysis to first {args.max_files} files")
        input_files = input_files[:args.max_files]
    
    # Load metadata if available
    metadata = load_metadata(input_dir)
    if metadata:
        print(f"\nMetadata found:")
        print(f"  Total calls: {metadata.get('total_calls', 'N/A')}")
        print(f"  Saved count: {metadata.get('saved_count', 'N/A')}")
        print(f"  Kernel name: {metadata.get('kernel_name', 'N/A')}")
    
    # Load all inputs
    print(f"\nLoading {len(input_files)} input files...")
    inputs = []
    for i, fpath in enumerate(input_files):
        data = load_captured_input(fpath)
        inputs.append(data)
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(input_files)} files...")
    print(f"  Loaded all {len(inputs)} files")
    
    # Shape analysis
    print(f"\n{'=' * 60}")
    print("Shape Analysis")
    print("=" * 60)
    shape_groups = analyze_shapes(inputs)
    print(f"\nUnique shapes: {len(shape_groups)}")
    print("\nShape distribution (BHSD format):")
    for shape_key, call_indices in sorted(shape_groups.items(), key=lambda x: -len(x[1])):
        q_shape, k_shape, v_shape, dtype = shape_key
        count = len(call_indices)
        print(f"  Count: {count:4d} | Q={q_shape}, K={k_shape}, V={v_shape}, dtype={dtype}")
    
    # Collect statistics
    print(f"\n{'=' * 60}")
    print("Value Statistics")
    print("=" * 60)
    stats = collect_statistics(inputs)
    
    # Print aggregate statistics
    print("\nAggregate statistics across all inputs:")
    print(f"\n  Q tensor:")
    print(f"    Mean of means:  {np.mean(stats['q_mean']):.6f}")
    print(f"    Std of means:   {np.std(stats['q_mean']):.6f}")
    print(f"    Mean of stds:   {np.mean(stats['q_std']):.6f}")
    print(f"    Global min:     {np.min(stats['q_min']):.6f}")
    print(f"    Global max:     {np.max(stats['q_max']):.6f}")
    
    print(f"\n  K tensor:")
    print(f"    Mean of means:  {np.mean(stats['k_mean']):.6f}")
    print(f"    Std of means:   {np.std(stats['k_mean']):.6f}")
    print(f"    Mean of stds:   {np.mean(stats['k_std']):.6f}")
    print(f"    Global min:     {np.min(stats['k_min']):.6f}")
    print(f"    Global max:     {np.max(stats['k_max']):.6f}")
    
    print(f"\n  V tensor:")
    print(f"    Mean of means:  {np.mean(stats['v_mean']):.6f}")
    print(f"    Std of means:   {np.std(stats['v_mean']):.6f}")
    print(f"    Mean of stds:   {np.mean(stats['v_std']):.6f}")
    print(f"    Global min:     {np.min(stats['v_min']):.6f}")
    print(f"    Global max:     {np.max(stats['v_max']):.6f}")
    
    # Print sample input details
    if inputs:
        print(f"\n{'=' * 60}")
        print("Sample Input Details (first file)")
        print("=" * 60)
        sample = inputs[0]
        print(f"\nFile: {input_files[0].name}")
        print(f"Call index: {sample.get('call_idx', 'N/A')}")
        print_tensor_statistics("Q", sample["q"])
        print_tensor_statistics("K", sample["k"])
        print_tensor_statistics("V", sample["v"])
    
    # Compare with generated random inputs
    generated_inputs = None
    generated_stats = None
    if args.compare_generated:
        print(f"\n{'=' * 60}")
        print(f"Comparison with Generated Random Inputs (seed={args.seed})")
        print("=" * 60)
        print(f"\nGenerating random inputs matching {len(inputs)} captured shapes...")
        print(f"(Using same method as bench_diffusion_attention.py: torch.randn in BSHD format)")
        
        generated_inputs = generate_random_inputs_like_captured(inputs, seed=args.seed)
        generated_stats = collect_statistics(generated_inputs)
        
        print_comparison_statistics(stats, generated_stats)
        
        # Print sample generated input details
        if generated_inputs:
            print(f"\n  Sample Generated Input Details (first):")
            gen_sample = generated_inputs[0]
            print_tensor_statistics("Q (generated)", gen_sample["q"], indent=4)
            print_tensor_statistics("K (generated)", gen_sample["k"], indent=4)
            print_tensor_statistics("V (generated)", gen_sample["v"], indent=4)
    
    # Generate plots
    if not args.no_plots:
        plots_dir = input_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print("Generating Plots")
        print("=" * 60)
        print(f"Plots will be saved to: {plots_dir}")
        
        print("\nGenerating value histograms...")
        plot_value_histograms(inputs, plots_dir)
        
        print("Generating Q/K/V comparison boxplots...")
        plot_qkv_comparison_boxplots(inputs, plots_dir)
        
        print("Generating per-head statistics...")
        plot_per_head_statistics(inputs, plots_dir)
        
        print("Generating statistics over time...")
        plot_statistics_over_time(stats, plots_dir)
        
        print("Generating shape timeline...")
        plot_shape_timeline(stats, plots_dir)
        
        # Generate comparison plots if we have generated inputs
        if generated_inputs is not None:
            print("\nGenerating captured vs generated comparison plots...")
            plot_captured_vs_generated_histograms(inputs, generated_inputs, plots_dir)
            plot_captured_vs_generated_boxplots(inputs, generated_inputs, plots_dir)
            plot_stats_comparison_over_time(stats, generated_stats, plots_dir)
    
    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())

