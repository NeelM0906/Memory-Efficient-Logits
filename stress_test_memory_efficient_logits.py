#!/usr/bin/env python3
"""
Stress Test for Memory-Efficient Logits Implementation

This script progressively increases problem size until hitting memory limits
to benchmark and compare standard vs memory-efficient logits computation.

Author: Neelanjan Mitra
License: MIT
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
import time
import gc
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
import json


class MemoryEfficientLogits(Function):
    """
    Memory-efficient implementation of logits computation using chunked processing.

    This custom autograd function processes input in chunks to avoid creating
    large intermediate tensors that could cause OOM errors.
    """

    @staticmethod
    def forward(ctx, input, weight, transform_fn, target=None, batch_size=None):
        """
        Forward pass with chunked processing.

        Args:
            input: Input tensor of shape (bsz_times_qlen, hidden_dim)
            weight: Weight tensor of shape (hidden_dim, vocab_size)
            transform_fn: Function to apply to logits (e.g., cross_entropy)
            target: Target tensor for supervised learning (optional)
            batch_size: Chunk size for processing (optional)
        """
        bsz_times_qlen, hidden_dim = input.shape
        hidden_dim, vocab_size = weight.shape

        if batch_size is None:
            batch_size = max(1, min(bsz_times_qlen, 4096))

        ctx.save_for_backward(input, weight)
        ctx.transform_fn = transform_fn
        ctx.batch_size = batch_size
        ctx.target = target
        ctx.bsz_times_qlen = bsz_times_qlen

        outputs = []
        for i in range(0, bsz_times_qlen, batch_size):
            end_idx = min(i + batch_size, bsz_times_qlen)
            input_chunk = input[i:end_idx]

            logits_chunk = input_chunk @ weight

            if target is not None:
                target_chunk = target[i:end_idx]
                output_chunk = transform_fn(logits_chunk, target_chunk)
            else:
                output_chunk = transform_fn(logits_chunk)

            outputs.append(output_chunk)

        if len(outputs) == 1:
            output = outputs[0]
        elif outputs[0].dim() == 0:
            output = sum(outputs) / len(outputs) if len(outputs) > 0 else outputs[0]
        else:
            output = torch.cat(outputs, dim=0)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with chunked processing.
        """
        input, weight = ctx.saved_tensors
        transform_fn = ctx.transform_fn
        batch_size = ctx.batch_size
        target = ctx.target
        bsz_times_qlen = ctx.bsz_times_qlen

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        for i in range(0, bsz_times_qlen, batch_size):
            end_idx = min(i + batch_size, bsz_times_qlen)
            input_chunk = input[i:end_idx].detach().requires_grad_(True)

            with torch.enable_grad():
                logits_chunk = input_chunk @ weight
                logits_chunk.requires_grad_(True)

                if target is not None:
                    target_chunk = target[i:end_idx]
                    output_chunk = transform_fn(logits_chunk, target_chunk)
                else:
                    output_chunk = transform_fn(logits_chunk)

                if grad_output.dim() == 0:
                    grad_out_chunk = grad_output
                else:
                    grad_out_chunk = grad_output[i:end_idx]

                grad_logits, = torch.autograd.grad(
                    outputs=output_chunk,
                    inputs=logits_chunk,
                    grad_outputs=grad_out_chunk,
                    retain_graph=False
                )

            grad_input[i:end_idx] = grad_logits @ weight.t()
            grad_weight += input_chunk.t() @ grad_logits

        return grad_input, grad_weight, None, None, None


def memory_efficient_cross_entropy(input, weight, target, batch_size=None):
    """
    Memory-efficient cross entropy computation.

    Args:
        input: Input tensor of shape (bsz_times_qlen, hidden_dim)
        weight: Weight tensor of shape (hidden_dim, vocab_size)
        target: Target labels
        batch_size: Chunk size for processing

    Returns:
        Cross entropy loss
    """
    def ce_transform(logits, target_chunk):
        return F.cross_entropy(logits, target_chunk, reduction='mean')

    return MemoryEfficientLogits.apply(input, weight, ce_transform, target, batch_size)


def standard_cross_entropy(input, weight, target):
    """
    Standard cross entropy computation (baseline).

    Args:
        input: Input tensor of shape (bsz_times_qlen, hidden_dim)
        weight: Weight tensor of shape (hidden_dim, vocab_size)
        target: Target labels

    Returns:
        Cross entropy loss
    """
    logits = input @ weight
    return F.cross_entropy(logits, target, reduction='mean')


# ============ UTILITY FUNCTIONS ============

def get_gpu_memory_detailed():
    """Get detailed GPU memory usage statistics."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory

        return {
            'allocated_bytes': allocated,
            'reserved_bytes': reserved,
            'total_bytes': total,
            'allocated_gb': allocated / 1024**3,
            'reserved_gb': reserved / 1024**3,
            'total_gb': total / 1024**3,
            'free_gb': (total - allocated) / 1024**3
        }
    return None


def measure_forward_backward_memory_time(func, *args, **kwargs) -> Tuple[torch.Tensor, float, float, float]:
    """
    Measure total GPU memory and time for forward + backward pass.

    Returns:
        (loss, total_time, total_memory_gb, baseline_memory_gb)
    """
    if not torch.cuda.is_available():
        start_time = time.time()
        loss = func(*args, **kwargs)
        if loss.requires_grad:
            loss.backward()
        total_time = time.time() - start_time
        return loss, total_time, 0.0, 0.0

    # Clean up and synchronize
    cleanup()

    # Get baseline memory (model tensors already loaded)
    baseline_memory = torch.cuda.memory_allocated()

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Start timing just before the operation
    torch.cuda.synchronize()
    start_time = time.time()

    # Run forward pass
    loss = func(*args, **kwargs)

    # Run backward pass if gradients are enabled
    if loss.requires_grad:
        loss.backward()

    # Stop timing and sync
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    # Get peak memory during operation
    peak_memory = torch.cuda.max_memory_allocated()

    # Calculate total memory footprint and additional memory used
    total_memory_gb = peak_memory / 1024**3
    baseline_memory_gb = baseline_memory / 1024**3
    additional_memory_gb = (peak_memory - baseline_memory) / 1024**3

    return loss, total_time, total_memory_gb, additional_memory_gb


def cleanup():
    """Clean up memory and cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def format_size(num_bytes):
    """Format byte size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


# ============ VISUALIZATION FUNCTIONS ============

def create_performance_visualizations(results: List[Dict], output_dir: str = "results"):
    """Create comprehensive performance visualization charts."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        print("No results to visualize")
        return

    # Extract data for plotting - handle None values properly
    vocab_sizes = [r['vocab_size'] for r in results]
    logits_sizes = [r['logits_size_gb'] for r in results]

    # Create separate arrays for successful tests only
    std_success_indices = [i for i, r in enumerate(results) if r.get('std_success', False)]
    eff_success_indices = [i for i, r in enumerate(results) if r.get('eff_success', False)]

    # Standard implementation data (only successful tests)
    std_vocab_sizes = [vocab_sizes[i] for i in std_success_indices]
    std_times = [results[i]['std_time'] for i in std_success_indices]
    std_total_memory = [results[i]['std_total_memory'] for i in std_success_indices]
    std_additional_memory = [results[i]['std_additional_memory'] for i in std_success_indices]

    # Efficient implementation data (only successful tests)
    eff_vocab_sizes = [vocab_sizes[i] for i in eff_success_indices]
    eff_times = [results[i]['eff_time'] for i in eff_success_indices]
    eff_total_memory = [results[i]['eff_total_memory'] for i in eff_success_indices]
    eff_additional_memory = [results[i]['eff_additional_memory'] for i in eff_success_indices]

    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    # 1. Execution Time Comparison (Forward + Backward)
    ax1 = plt.subplot(2, 3, 1)
    if std_vocab_sizes and std_times:
        plt.plot(std_vocab_sizes, std_times, 'ro-', label='Standard (Forward+Backward)', linewidth=2, markersize=8)
    if eff_vocab_sizes and eff_times:
        plt.plot(eff_vocab_sizes, eff_times, 'bo-', label='Memory-Efficient (Forward+Backward)', linewidth=2, markersize=8)

    plt.xlabel('Vocabulary Size')
    plt.ylabel('Total Time (seconds)')
    plt.title('Forward + Backward Time vs Vocabulary Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if std_times or eff_times:
        plt.yscale('log')

    # 2. Total Memory Usage Comparison
    ax2 = plt.subplot(2, 3, 2)
    if std_vocab_sizes and std_total_memory:
        plt.plot(std_vocab_sizes, std_total_memory, 'ro-', label='Standard Total Memory', linewidth=2, markersize=8)
    if eff_vocab_sizes and eff_total_memory:
        plt.plot(eff_vocab_sizes, eff_total_memory, 'bo-', label='Efficient Total Memory', linewidth=2, markersize=8)

    plt.xlabel('Vocabulary Size')
    plt.ylabel('Total GPU Memory (GB)')
    plt.title('Total GPU Memory Usage vs Vocabulary Size')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Additional Memory Usage (Operation-Specific)
    ax3 = plt.subplot(2, 3, 3)
    if vocab_sizes and logits_sizes:
        plt.plot(vocab_sizes, logits_sizes, 'g--', label='Theoretical Logits Size', linewidth=2)

    if std_vocab_sizes and std_additional_memory:
        plt.plot(std_vocab_sizes, std_additional_memory, 'ro-', label='Standard Additional', linewidth=2, markersize=8)
    if eff_vocab_sizes and eff_additional_memory:
        plt.plot(eff_vocab_sizes, eff_additional_memory, 'bo-', label='Efficient Additional', linewidth=2, markersize=8)

    plt.xlabel('Vocabulary Size')
    plt.ylabel('Additional Memory (GB)')
    plt.title('Additional Memory vs Theoretical Logits Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if logits_sizes or std_additional_memory or eff_additional_memory:
        plt.yscale('log')

    # 4. Speedup Analysis - only for vocab sizes where both methods succeeded
    ax4 = plt.subplot(2, 3, 4)
    speedup_vocab_sizes = []
    speedup_values = []

    for i, r in enumerate(results):
        if r.get('std_success', False) and r.get('eff_success', False):
            if r['std_time'] and r['eff_time'] and r['eff_time'] > 0:
                speedup_vocab_sizes.append(r['vocab_size'])
                speedup_values.append(r['std_time'] / r['eff_time'])

    if speedup_vocab_sizes and speedup_values:
        plt.plot(speedup_vocab_sizes, speedup_values, 'go-', linewidth=2, markersize=8)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No speedup')
        plt.legend()

    plt.xlabel('Vocabulary Size')
    plt.ylabel('Speedup Factor (Standard/Efficient)')
    plt.title('Forward+Backward Speedup Analysis')
    plt.grid(True, alpha=0.3)

    # 5. Memory Efficiency (Total Memory)
    ax5 = plt.subplot(2, 3, 5)
    memory_efficiency_vocab_sizes = []
    memory_efficiency_values = []

    for i, r in enumerate(results):
        if r.get('std_success', False) and r.get('eff_success', False):
            if r['std_total_memory'] and r['eff_total_memory'] and r['eff_total_memory'] > 0:
                memory_efficiency_vocab_sizes.append(r['vocab_size'])
                memory_efficiency_values.append(r['std_total_memory'] / r['eff_total_memory'])

    if memory_efficiency_vocab_sizes and memory_efficiency_values:
        plt.plot(memory_efficiency_vocab_sizes, memory_efficiency_values, 'mo-', linewidth=2, markersize=8)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No improvement')
        plt.legend()

    plt.xlabel('Vocabulary Size')
    plt.ylabel('Memory Efficiency (Standard/Efficient)')
    plt.title('Total Memory Efficiency Analysis')
    plt.grid(True, alpha=0.3)

    # 6. Memory Breakdown Comparison
    ax6 = plt.subplot(2, 3, 6)

    if results and std_additional_memory and eff_additional_memory:
        # Get baseline memory (should be same for both)
        baseline_mem = results[0].get('baseline_memory', 0)

        # Only plot for vocab sizes where we have data for both methods
        common_indices = []
        common_vocab_sizes = []
        common_std_additional = []
        common_eff_additional = []

        for i, r in enumerate(results):
            if r.get('std_success', False) and r.get('eff_success', False):
                common_indices.append(i)
                common_vocab_sizes.append(r['vocab_size'])
                common_std_additional.append(r['std_additional_memory'])
                common_eff_additional.append(r['eff_additional_memory'])

        if common_vocab_sizes:
            width = 0.35
            x = np.arange(len(common_vocab_sizes))

            # Stack bars showing baseline + additional
            plt.bar(x - width/2, [baseline_mem] * len(common_std_additional), width,
                    label='Baseline (Model)', alpha=0.7, color='gray')
            plt.bar(x - width/2, common_std_additional, width, bottom=[baseline_mem] * len(common_std_additional),
                    label='Standard Additional', alpha=0.7, color='red')
            plt.bar(x + width/2, [baseline_mem] * len(common_eff_additional), width,
                    alpha=0.7, color='gray')
            plt.bar(x + width/2, common_eff_additional, width, bottom=[baseline_mem] * len(common_eff_additional),
                    label='Efficient Additional', alpha=0.7, color='blue')

            plt.xlabel('Test Configuration')
            plt.ylabel('Memory Usage (GB)')
            plt.title('Memory Usage Breakdown')
            plt.xticks(x, [f'{v//1000}K' for v in common_vocab_sizes], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create detailed summary chart
    create_summary_chart(results, output_dir)


def create_summary_chart(results: List[Dict], output_dir: str):
    """Create a detailed summary comparison chart."""

    if not results:
        print("No results to create summary chart")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get data for successful tests only
    successful_results = [r for r in results if r.get('std_success', False) and r.get('eff_success', False)]

    if not successful_results:
        print("No successful comparisons to chart")
        plt.close(fig)
        return

    vocab_sizes = [r['vocab_size'] for r in successful_results]

    # Chart 1: Time and Memory Combined
    ax1_twin = ax1.twinx()

    std_times = [r['std_time'] for r in successful_results]
    eff_times = [r['eff_time'] for r in successful_results]
    std_memory = [r['std_total_memory'] for r in successful_results]
    eff_memory = [r['eff_total_memory'] for r in successful_results]

    # Time on left axis
    line1 = ax1.plot(vocab_sizes, std_times, 'r-o', label='Standard Time', linewidth=2)
    line2 = ax1.plot(vocab_sizes, eff_times, 'b-o', label='Efficient Time', linewidth=2)
    ax1.set_xlabel('Vocabulary Size')
    ax1.set_ylabel('Time (seconds)', color='black')
    ax1.set_yscale('log')

    # Memory on right axis
    line3 = ax1_twin.plot(vocab_sizes, std_memory, 'r--s', label='Standard Memory', linewidth=2, alpha=0.7)
    line4 = ax1_twin.plot(vocab_sizes, eff_memory, 'b--s', label='Efficient Memory', linewidth=2, alpha=0.7)
    ax1_twin.set_ylabel('Total Memory (GB)', color='gray')

    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title('Forward+Backward Performance and Memory')
    ax1.grid(True, alpha=0.3)

    # Chart 2: Improvement Factors
    speedup = []
    memory_reduction = []

    for r in successful_results:
        if r['eff_time'] > 0:
            speedup.append(r['std_time'] / r['eff_time'])
        else:
            speedup.append(1.0)

        if r['eff_total_memory'] > 0:
            memory_reduction.append(r['std_total_memory'] / r['eff_total_memory'])
        else:
            memory_reduction.append(1.0)

    x = np.arange(len(vocab_sizes))
    width = 0.35

    bars1 = ax2.bar(x - width/2, speedup, width, label='Speed Improvement', alpha=0.7, color='green')
    bars2 = ax2.bar(x + width/2, memory_reduction, width, label='Memory Reduction', alpha=0.7, color='orange')

    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No improvement')
    ax2.set_xlabel('Vocabulary Size')
    ax2.set_ylabel('Improvement Factor')
    ax2.set_title('Total Efficiency Improvements')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{v//1000}K' for v in vocab_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============ STRESS TESTS ============

def stress_test_vocabulary_scaling():
    """
    Test with progressively larger vocabulary sizes.

    Returns:
        List of test results with timing and memory usage data
    """
    print("\n" + "="*60)
    print("STRESS TEST: Vocabulary Size Scaling (Forward + Backward)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fixed dimensions
    bsz = 4
    seq_len = 2048
    hidden_dim = 2048

    # Test configurations
    vocab_sizes = [10_000, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 200_000]

    results = []

    for vocab_size in vocab_sizes:
        print(f"\n{'='*40}")
        print(f"Testing vocab_size = {vocab_size:,}")

        # Calculate theoretical memory requirement for logits only
        logits_size = bsz * seq_len * vocab_size * 2  # fp16
        print(f"Theoretical logits size: {format_size(logits_size)}")

        # Create test data
        input_data = torch.randn(bsz * seq_len, hidden_dim, device=device, dtype=torch.float16, requires_grad=True)
        weight = torch.randn(hidden_dim, vocab_size, device=device, dtype=torch.float16, requires_grad=True)
        target = torch.randint(0, vocab_size, (bsz * seq_len,), device=device)

        # Get baseline memory after loading model tensors
        cleanup()
        baseline_memory_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        # Test standard implementation
        print("\n--- Standard Implementation (Forward + Backward) ---")
        try:
            # Create fresh tensors for standard test
            input_std = input_data.clone().detach().requires_grad_(True)
            weight_std = weight.clone().detach().requires_grad_(True)

            def run_standard():
                with torch.amp.autocast('cuda'):
                    return standard_cross_entropy(input_std, weight_std, target)

            loss_std, std_time, std_total_memory, std_additional_memory = measure_forward_backward_memory_time(run_standard)

            print(f"✓ Success")
            print(f"  Total memory: {std_total_memory:.3f} GB")
            print(f"  Additional memory: {std_additional_memory:.3f} GB")
            print(f"  Forward+Backward time: {std_time:.3f}s")
            print(f"  Loss: {loss_std.item():.6f}")

            std_success = True
            del loss_std, input_std, weight_std

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("✗ OUT OF MEMORY!")
                std_success = False
                std_time = None
                std_total_memory = None
                std_additional_memory = None
            else:
                raise e

        cleanup()

        # Test memory-efficient implementation
        print("\n--- Memory-Efficient Implementation (Forward + Backward) ---")
        try:
            # Test different batch sizes
            batch_sizes = [512, 1024, 2048, 4096]
            best_batch_size = None
            best_time = float('inf')
            best_total_memory = None
            best_additional_memory = None

            for batch_size in batch_sizes:
                cleanup()

                # Create fresh tensors for efficient test
                input_eff = input_data.clone().detach().requires_grad_(True)
                weight_eff = weight.clone().detach().requires_grad_(True)

                def run_efficient():
                    with torch.amp.autocast('cuda'):
                        return memory_efficient_cross_entropy(
                            input_eff, weight_eff, target, batch_size=batch_size
                        )

                loss_eff, eff_time, eff_total_memory, eff_additional_memory = measure_forward_backward_memory_time(run_efficient)

                if eff_time < best_time:
                    best_time = eff_time
                    best_batch_size = batch_size
                    best_total_memory = eff_total_memory
                    best_additional_memory = eff_additional_memory
                    best_loss = loss_eff.item()

                del input_eff, weight_eff, loss_eff

            print(f"✓ Success")
            print(f"  Best batch size: {best_batch_size}")
            print(f"  Total memory: {best_total_memory:.3f} GB")
            print(f"  Additional memory: {best_additional_memory:.3f} GB")
            print(f"  Forward+Backward time: {best_time:.3f}s")
            print(f"  Loss: {best_loss:.6f}")

            eff_success = True

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("✗ OUT OF MEMORY!")
                eff_success = False
                best_time = None
                best_total_memory = None
                best_additional_memory = None
                best_batch_size = None
            else:
                raise e

        # Store results
        results.append({
            'vocab_size': vocab_size,
            'logits_size_gb': logits_size / 1024**3,
            'baseline_memory': baseline_memory_gb,
            'std_success': std_success,
            'std_time': std_time,
            'std_total_memory': std_total_memory,
            'std_additional_memory': std_additional_memory,
            'eff_success': eff_success,
            'eff_time': best_time,
            'eff_total_memory': best_total_memory,
            'eff_additional_memory': best_additional_memory,
            'eff_batch_size': best_batch_size if eff_success else None
        })

        # Show immediate improvement stats
        if std_success and eff_success and std_time and best_time:
            speedup = std_time / best_time
            if std_total_memory and best_total_memory and best_total_memory > 0:
                memory_reduction = std_total_memory / best_total_memory
                additional_memory_reduction = std_additional_memory / best_additional_memory if best_additional_memory > 0 else float('inf')
                print(f"  🚀 Speedup: {speedup:.2f}x")
                print(f"  💾 Total memory reduction: {memory_reduction:.2f}x")
                print(f"  🔧 Additional memory reduction: {additional_memory_reduction:.2f}x")
            else:
                print(f"  🚀 Speedup: {speedup:.2f}x")

        # Stop if standard method fails
        if not std_success and eff_success:
            print(f"\n🎯 Found the breaking point at vocab_size={vocab_size:,}")
            print("Standard method failed but efficient method succeeded!")

        cleanup()

    return results


def stress_test_sequence_length():
    """Test with progressively longer sequences (Forward + Backward)."""
    print("\n" + "="*60)
    print("STRESS TEST: Sequence Length Scaling (Forward + Backward)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fixed dimensions
    bsz = 8
    hidden_dim = 4096
    vocab_size = 50000  # Reduced from 256000 to make it more manageable

    # Test configurations
    seq_lengths = [1024, 2048, 4096, 8192]  # Reduced range for more realistic testing

    for seq_len in seq_lengths:
        print(f"\n{'='*40}")
        print(f"Testing seq_len = {seq_len:,}")

        # Calculate theoretical memory requirement
        logits_size = bsz * seq_len * vocab_size * 2  # fp16
        print(f"Theoretical logits size: {format_size(logits_size)}")

        if logits_size > 8 * 1024**3:  # Skip if > 8GB
            print("Skipping - too large for typical GPU")
            continue

        # Create test data
        input_data = torch.randn(bsz * seq_len, hidden_dim, device=device, dtype=torch.float16, requires_grad=True)
        weight = torch.randn(hidden_dim, vocab_size, device=device, dtype=torch.float16, requires_grad=True)
        target = torch.randint(0, vocab_size, (bsz * seq_len,), device=device)

        cleanup()

        # Only test memory-efficient version for longer sequences
        print("\n--- Memory-Efficient Implementation (Forward + Backward) ---")
        try:
            def run_efficient():
                with torch.amp.autocast('cuda'):
                    return memory_efficient_cross_entropy(input_data, weight, target)

            loss, eff_time, total_memory, additional_memory = measure_forward_backward_memory_time(run_efficient)

            print(f"✓ Success")
            print(f"  Total memory: {total_memory:.3f} GB")
            print(f"  Additional memory: {additional_memory:.3f} GB")
            print(f"  Forward+Backward time: {eff_time:.3f}s")
            print(f"  Throughput: {(bsz * seq_len) / eff_time:.0f} tokens/s")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("✗ OUT OF MEMORY!")
            else:
                raise e

        cleanup()


def print_summary(results):
    """Print a formatted summary table of results."""
    print("\n" + "="*60)
    print("SUMMARY - Forward + Backward Pass Results")
    print("="*60)

    print(f"\n{'Vocab Size':>12} | {'Logits GB':>10} | {'Standard (Total/Additional)':^35} | {'Efficient (Total/Additional)':^35}")
    print(f"{'-'*12} | {'-'*10} | {'-'*35} | {'-'*35}")

    for r in results:
        std_status = "✓" if r['std_success'] else "✗ OOM"
        eff_status = "✓" if r['eff_success'] else "✗ OOM"

        if r['std_success']:
            std_info = f"{std_status} ({r['std_time']:.2f}s, {r['std_total_memory']:.2f}/{r['std_additional_memory']:.2f}GB)"
        else:
            std_info = std_status

        if r['eff_success']:
            eff_info = f"{eff_status} ({r['eff_time']:.2f}s, {r['eff_total_memory']:.2f}/{r['eff_additional_memory']:.2f}GB)"
        else:
            eff_info = eff_status

        print(f"{r['vocab_size']:>12,} | {r['logits_size_gb']:>10.2f} | {std_info:^35} | {eff_info:^35}")

    # Calculate detailed statistics
    successful_comparisons = []
    total_memory_comparisons = []
    additional_memory_comparisons = []

    for i, r in enumerate(results):
        if r['std_success'] and r['eff_success']:
            # Time comparison
            speedup = r['std_time'] / r['eff_time'] if r['eff_time'] > 0 else 1
            successful_comparisons.append(speedup)

            # Total memory comparison
            if r['std_total_memory'] > 0 and r['eff_total_memory'] > 0:
                total_memory_ratio = r['std_total_memory'] / r['eff_total_memory']
                total_memory_comparisons.append(total_memory_ratio)

            # Additional memory comparison
            if r['std_additional_memory'] > 0 and r['eff_additional_memory'] > 0:
                additional_memory_ratio = r['std_additional_memory'] / r['eff_additional_memory']
                additional_memory_comparisons.append(additional_memory_ratio)

        # Check for breaking point
        if not r['std_success'] and r['eff_success']:
            print(f"\n🎯 Breaking point: vocab_size = {r['vocab_size']:,}")
            print(f"   Standard method fails at {r['logits_size_gb']:.2f} GB theoretical logits")
            print(f"   Efficient method still works!")
            break

    # Print comprehensive statistics
    if successful_comparisons:
        print(f"\n📊  Performance Analysis:")
        print(f"   ⏱️  Average speedup (Forward+Backward): {np.mean(successful_comparisons):.2f}x")
        print(f"   📈 Best speedup: {max(successful_comparisons):.2f}x")

        if total_memory_comparisons:
            print(f"   💾 Average total memory reduction: {np.mean(total_memory_comparisons):.2f}x")
            print(f"   🏆 Best total memory reduction: {max(total_memory_comparisons):.2f}x")

        if additional_memory_comparisons:
            print(f"   🔧 Average additional memory reduction: {np.mean(additional_memory_comparisons):.2f}x")
            print(f"   ⭐ Best additional memory reduction: {max(additional_memory_comparisons):.2f}x")

    # Memory overhead analysis
    if results:
        print(f"\n🔍 Memory Overhead Analysis:")
        baseline = results[0]['baseline_memory']
        print(f"   📊 Model baseline memory: {baseline:.2f} GB")

        for r in results:
            if r['std_success']:
                theoretical = r['logits_size_gb']
                actual_additional = r['std_additional_memory']
                overhead_factor = actual_additional / theoretical if theoretical > 0 else 0
                print(f"   🔢 {r['vocab_size']//1000}K vocab: {theoretical:.2f}GB theoretical → {actual_additional:.2f}GB actual ({overhead_factor:.1f}x overhead)")
                break  # Just show one example


def save_results_json(results: List[Dict], filename: str = "results/benchmark_results.json"):
    """Save results to JSON file for later analysis."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Add metadata
    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
        'total_gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__,
        'num_tests': len(results),
        'measurement_type': 'forward_backward_pass',
        'description': 'Measurements including full forward+backward pass with memory accounting'
    }

    output_data = {
        'metadata': metadata,
        'results': results
    }

    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to {filename}")


# ============ MAIN EXECUTION ============

def main():
    """Main execution function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU memory: {total_memory:.2f} GB")
    else:
        print("⚠️  CUDA not available - memory measurements will be limited")

    # Run stress tests
    try:
        print("\n🚀 Starting stress testing with forward+backward passes...")
        print("📋 Key improvements:")
        print("Measures total GPU memory footprint (not just incremental)")
        print("Includes full forward + backward pass")
        print("Proper timing around actual operations")
        print("Baseline memory subtraction for additional memory calculation")

        # 1. Vocabulary scaling test
        results = stress_test_vocabulary_scaling()
        print_summary(results)

        # 2. Save results to JSON
        save_results_json(results)

        # 3. Create visualizations
        if len(results) > 0:
            print("\n📊 Creating  performance visualizations...")
            try:
                create_performance_visualizations(results)
                print("✓ Visualizations saved to 'results/' directory")
            except ImportError:
                print("⚠️  matplotlib not available - skipping visualizations")
                print("   Install with: pip install matplotlib")
            except Exception as e:
                print(f"⚠️  Error creating visualizations: {e}")

        # 4. Sequence length test
        stress_test_sequence_length()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Stress testing complete!")
    print("="*60)

    if torch.cuda.is_available():
        final_memory = get_gpu_memory_detailed()
        print(f"Final GPU memory: {final_memory['allocated_gb']:.2f} GB allocated / {final_memory['total_gb']:.2f} GB total")
        print(f"Memory efficiency: {(final_memory['total_gb'] - final_memory['allocated_gb']) / final_memory['total_gb'] * 100:.1f}% free")


if __name__ == "__main__":
    main()
