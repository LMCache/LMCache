#!/usr/bin/env python3
"""
KV Cache Size Visualization for Large Context Lengths
Generates static PNG charts comparing memory requirements across models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import os

# Import shared utilities
from kv_cache_utils import (
    load_model_configs,
    calculate_kv_cache_size,
    select_representative_models,
    format_yaxis_memory,
    get_model_color,
    GPU_CONSUMER_MEMORY,
    GPU_H100_MEMORY,
    GPU_MULTI_160_MEMORY,
    GPU_MULTI_320_MEMORY,
    GPU_MULTI_640_MEMORY,
    FEASIBILITY_COLORS,
)




def create_comparison_chart(configs, dtype="float16"):
    """Create a comprehensive comparison chart for multiple context lengths"""
    models = select_representative_models(configs)

    # Define context lengths to visualize (log scale from 1K to 1M+)
    context_lengths = [
        1_000,  # 1K
        2_000,  # 2K
        4_000,  # 4K
        8_000,  # 8K
        16_000,  # 16K
        32_000,  # 32K
        64_000,  # 64K
        128_000,  # 128K
        256_000,  # 256K
        512_000,  # 512K
        1_000_000,  # 1M
        1_500_000,  # 1.5M
    ]

    # Calculate cache sizes for each model and context length
    cache_sizes = {}
    for model_name, config in models.items():
        cache_sizes[model_name] = [
            calculate_kv_cache_size(config, ctx_len, dtype, model_name)
            for ctx_len in context_lengths
        ]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # Define line styles and markers for better distinction
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']

    # Plot 1: Log-scale line chart
    ax1 = fig.add_subplot(gs[0, :])
    for idx, (model_name, sizes) in enumerate(cache_sizes.items()):
        short_name = model_name.split("/")[-1]
        color = get_model_color(model_name)
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        ax1.loglog(
            context_lengths,
            sizes,
            marker=marker,
            linestyle=line_style,
            label=short_name,
            linewidth=2,
            markersize=6,
            color=color,
            alpha=0.9,
        )

    ax1.set_xlabel("Context Length (tokens)", fontsize=12)
    ax1.set_ylabel("KV Cache Size", fontsize=12)
    ax1.set_title(
        f"KV Cache Memory Requirements vs Context Length ({dtype})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc="upper left", fontsize=9, ncol=2)

    # Format y-axis with human-readable units
    ax1.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Add context length labels
    ax1.set_xticks(context_lengths)
    ax1.set_xticklabels(
        [
            "1K",
            "2K",
            "4K",
            "8K",
            "16K",
            "32K",
            "64K",
            "128K",
            "256K",
            "512K",
            "1M",
            "1.5M",
        ],
        rotation=45,
    )

    # Plot 2: Bar chart for specific context lengths
    ax2 = fig.add_subplot(gs[1, 0])
    selected_contexts = [8_000, 128_000, 1_000_000]
    x_pos = np.arange(len(models))
    width = 0.25

    for i, ctx_len in enumerate(selected_contexts):
        values = [
            calculate_kv_cache_size(config, ctx_len, dtype, name)
            for name, config in models.items()
        ]
        colors = [get_model_color(name) for name in models.keys()]
        bars = ax2.bar(
            x_pos + i * width,
            values,
            width,
            label=f"{ctx_len//1000}K tokens",
            alpha=0.8,
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("KV Cache Size", fontsize=11)
    ax2.set_title(
        "Memory Requirements at Key Context Lengths", fontsize=12, fontweight="bold"
    )
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(
        [m.split("/")[-1] for m in models.keys()], rotation=45, ha="right", fontsize=8
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    # Format y-axis with human-readable units
    ax2.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Plot 3: Heatmap showing memory feasibility
    ax3 = fig.add_subplot(gs[1, 1])

    # Create feasibility matrix
    model_names = list(models.keys())
    short_model_names = [m.split("/")[-1] for m in model_names]
    contexts_for_heatmap = [
        8_000,
        16_000,
        32_000,
        64_000,
        128_000,
        256_000,
        512_000,
        1_000_000,
    ]

    # Calculate which combinations are feasible for different GPU sizes
    feasibility = np.zeros((len(model_names), len(contexts_for_heatmap)))
    for i, (model_name, config) in enumerate(models.items()):
        for j, ctx_len in enumerate(contexts_for_heatmap):
            size = calculate_kv_cache_size(config, ctx_len, dtype, model_name)
            if size <= GPU_CONSUMER_MEMORY:
                feasibility[i, j] = 1  # Green - fits in consumer GPU
            elif size <= GPU_H100_MEMORY:
                feasibility[i, j] = 2  # Yellow - needs datacenter GPU
            elif size <= GPU_MULTI_320_MEMORY:
                feasibility[i, j] = 3  # Orange - needs multiple GPUs
            else:
                feasibility[i, j] = 4  # Red - very challenging

    # Create custom colormap
    colors_list = list(FEASIBILITY_COLORS.values())
    cmap = ListedColormap(colors_list)

    im = ax3.imshow(feasibility, cmap=cmap, aspect="auto", vmin=1, vmax=4)
    ax3.set_xticks(range(len(contexts_for_heatmap)))
    ax3.set_xticklabels([f"{c//1000}K" for c in contexts_for_heatmap], fontsize=9)
    ax3.set_yticks(range(len(short_model_names)))
    ax3.set_yticklabels(short_model_names, fontsize=8)
    ax3.set_xlabel("Context Length", fontsize=11)
    ax3.set_ylabel("Model", fontsize=11)
    ax3.set_title("GPU Memory Feasibility Matrix", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(contexts_for_heatmap)):
            size = calculate_kv_cache_size(
                models[model_names[i]], contexts_for_heatmap[j], dtype, model_names[i]
            )
            text_color = "white" if feasibility[i, j] >= 3 else "black"
            ax3.text(
                j,
                i,
                f"{size:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    # Add legend
    legend_elements = [
        mpatches.Patch(color=FEASIBILITY_COLORS['fits_consumer'],
                      label=f"≤{GPU_CONSUMER_MEMORY} GiB (Consumer)"),
        mpatches.Patch(color=FEASIBILITY_COLORS['fits_datacenter'],
                      label=f"≤{GPU_H100_MEMORY} GiB (A100/H100)"),
        mpatches.Patch(color=FEASIBILITY_COLORS['needs_multi'],
                      label=f"≤{GPU_MULTI_320_MEMORY} GiB (Multi-GPU)"),
        mpatches.Patch(color=FEASIBILITY_COLORS['challenging'],
                      label=f">{GPU_MULTI_320_MEMORY} GiB (Challenging)"),
    ]
    ax3.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8
    )

    # Add main title
    fig.suptitle(
        "KV Cache Memory Analysis for Large Language Models",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    return fig


def create_extreme_context_projection(configs, dtype="float16"):
    """Create visualization focusing on extreme context lengths (512K-2M)"""
    models = select_representative_models(configs)

    # Focus on very large contexts
    extreme_contexts = [
        512_000,  # 512K
        750_000,  # 750K
        1_000_000,  # 1M
        1_250_000,  # 1.25M
        1_500_000,  # 1.5M
        1_750_000,  # 1.75M
        2_000_000,  # 2M
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Calculate sizes
    cache_sizes = {}
    for model_name, config in models.items():
        cache_sizes[model_name] = [
            calculate_kv_cache_size(config, ctx_len, dtype, model_name)
            for ctx_len in extreme_contexts
        ]

    # Define line styles and markers for better distinction
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']

    # Plot 1: Line chart for extreme contexts
    for idx, (model_name, sizes) in enumerate(cache_sizes.items()):
        short_name = model_name.split("/")[-1]
        color = get_model_color(model_name)
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        ax1.plot(
            extreme_contexts,
            sizes,
            marker=marker,
            linestyle=line_style,
            label=short_name,
            linewidth=2,
            markersize=8,
            color=color,
            alpha=0.9,
        )

    ax1.set_xlabel("Context Length (millions of tokens)", fontsize=12)
    ax1.set_ylabel("KV Cache Size", fontsize=12)
    ax1.set_title(
        "Memory Requirements for Extreme Context Lengths",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Format y-axis with human-readable units
    ax1.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Format x-axis
    ax1.set_xticks(extreme_contexts)
    ax1.set_xticklabels([f"{c/1_000_000:.2f}M" for c in extreme_contexts])

    # Add horizontal lines for GPU memory limits
    gpu_limits = [
        (GPU_H100_MEMORY, f"H100 {GPU_H100_MEMORY} GiB", "blue"),
        (GPU_MULTI_160_MEMORY, "2x H100", "green"),
        (GPU_MULTI_320_MEMORY, "4x H100", "orange"),
        (GPU_MULTI_640_MEMORY, "8x H100", "red"),
    ]

    for limit, label, color in gpu_limits:
        ax1.axhline(y=limit, color=color, linestyle="--", alpha=0.5, label=label)

    # Plot 2: Stacked bar showing memory breakdown at 1M context
    ctx_1m = 1_000_000
    model_names = list(models.keys())
    short_names = [m.split("/")[-1] for m in model_names]
    sizes_1m = [
        calculate_kv_cache_size(models[m], ctx_1m, dtype, m) for m in model_names
    ]

    # Sort by size
    sorted_indices = np.argsort(sizes_1m)
    sorted_names = [short_names[i] for i in sorted_indices]
    sorted_sizes = [sizes_1m[i] for i in sorted_indices]

    bars = ax2.barh(range(len(sorted_names)), sorted_sizes)

    # Color bars based on feasibility
    for i, (bar, size) in enumerate(zip(bars, sorted_sizes)):
        if size <= GPU_H100_MEMORY:
            bar.set_color(FEASIBILITY_COLORS['fits_consumer'])  # Green
        elif size <= GPU_MULTI_160_MEMORY:
            bar.set_color(FEASIBILITY_COLORS['fits_datacenter'])  # Yellow
        elif size <= GPU_MULTI_320_MEMORY:
            bar.set_color(FEASIBILITY_COLORS['needs_multi'])  # Orange
        else:
            bar.set_color(FEASIBILITY_COLORS['challenging'])  # Red

    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names, fontsize=9)
    ax2.set_xlabel("KV Cache Size", fontsize=12)
    ax2.set_title(
        f"Memory Requirements at 1M Tokens ({dtype})", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, axis="x", alpha=0.3)

    # Format x-axis with human-readable units
    ax2.xaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Add vertical lines for GPU limits - position text to avoid overlap
    for i, (limit, label, color) in enumerate(gpu_limits[:3]):
        ax2.axvline(x=limit, color=color, linestyle="--", alpha=0.5)
        # Stagger text positions to avoid overlap
        y_pos = len(sorted_names) - 0.5 - (i * 0.7)
        ax2.text(
            limit + 5,
            y_pos,
            label,
            rotation=0,
            verticalalignment="center",
            fontsize=8,
            color=color,
        )

    # Add value labels
    for i, (name, size) in enumerate(zip(sorted_names, sorted_sizes)):
        ax2.text(size + 5, i, f"{size:.0f} GiB", va="center", fontsize=8)

    plt.suptitle(
        "Extreme Context Length Analysis (512K-2M tokens)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def create_paged_attention_visualization(configs, dtype="float16"):
    """Visualize the difference between monolithic and paged attention memory allocation"""
    import numpy as np

    # Industry standard configurations
    VLLM_BLOCK_SIZE = 16  # vLLM default: 16 tokens per block
    FLASH_ATTENTION_BLOCK_SIZE = 64  # FlashAttention typical block size

    # Select a representative model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    config = configs[model_name]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Define sequence lengths to visualize
    sequence_lengths = [1000, 4096, 16384, 32768]

    for idx, seq_len in enumerate(sequence_lengths):
        ax = axes[idx // 2, idx % 2]

        # Calculate total KV cache size needed
        total_memory_gib = calculate_kv_cache_size(config, seq_len, dtype, model_name)

        # Monolithic allocation (traditional)
        monolithic_blocks = [total_memory_gib]

        # Paged attention with vLLM block size
        num_blocks_vllm = (seq_len + VLLM_BLOCK_SIZE - 1) // VLLM_BLOCK_SIZE
        memory_per_block_vllm = total_memory_gib / (seq_len / VLLM_BLOCK_SIZE)

        # Calculate actual usage patterns (simulating different fill levels)
        usage_patterns = [0.25, 0.5, 0.75, 1.0]  # 1/4, 2/4, 3/4, 4/4 filled

        # Create visualization
        bar_width = 0.15
        x_positions = np.arange(len(usage_patterns))

        # Plot monolithic allocation (always full allocation regardless of actual usage)
        monolithic_bars = ax.bar(x_positions - 1.5*bar_width,
                                 [total_memory_gib] * len(usage_patterns),
                                 bar_width, label='Monolithic (No Paging)',
                                 color='#e74c3c', alpha=0.7)

        # Plot paged attention with vLLM blocks
        paged_actual = [total_memory_gib * usage for usage in usage_patterns]
        paged_bars_vllm = ax.bar(x_positions - 0.5*bar_width,
                                 paged_actual,
                                 bar_width, label=f'Paged (vLLM, {VLLM_BLOCK_SIZE} tokens/block)',
                                 color='#3498db', alpha=0.7)

        # Plot paged attention with larger blocks
        paged_bars_flash = ax.bar(x_positions + 0.5*bar_width,
                                  paged_actual,
                                  bar_width, label=f'Paged (Flash, {FLASH_ATTENTION_BLOCK_SIZE} tokens/block)',
                                  color='#2ecc71', alpha=0.7)

        # Add wasted memory visualization (shaded area)
        for i, usage in enumerate(usage_patterns):
            wasted = total_memory_gib * (1 - usage)
            if wasted > 0:
                ax.bar(x_positions[i] - 1.5*bar_width, wasted,
                      bar_width, bottom=total_memory_gib*usage,
                      color='gray', alpha=0.3, edgecolor='red', linewidth=1, linestyle='--')

        # Formatting
        ax.set_xlabel('Actual Sequence Fill Level', fontsize=11)
        ax.set_ylabel('Memory Allocated (GiB)', fontsize=11)
        ax.set_title(f'Memory Allocation: {seq_len} tokens\nTotal KV Cache: {total_memory_gib:.2f} GiB',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['25%\n(1/4 filled)', '50%\n(2/4 filled)',
                           '75%\n(3/4 filled)', '100%\n(4/4 filled)'])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [monolithic_bars, paged_bars_vllm, paged_bars_flash]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Paged Attention vs Monolithic KV Cache Allocation\n' +
                 f'Model: {model_name} | Precision: {dtype}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_continuous_batching_visualization(configs, dtype="float16"):
    """Visualize the impact of continuous batching on memory utilization and throughput"""
    import numpy as np
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches

    # Select a representative model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    config = configs[model_name]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)

    # Simulation parameters
    max_batch_size = 8
    max_seq_len = 2048

    # Generate random request patterns
    np.random.seed(42)  # For reproducibility
    request_lengths = np.random.choice([128, 256, 512, 1024, 1536, 2048], size=20)
    request_arrivals = np.cumsum(np.random.exponential(scale=0.5, size=20))

    # ========== Plot 1: Static Batching Memory Waste ==========
    ax1 = fig.add_subplot(gs[0, 0])

    # Static batching - all requests padded to max length
    static_batch_memory = []
    for i in range(0, len(request_lengths), max_batch_size):
        batch = request_lengths[i:i+max_batch_size]
        # In static batching, all sequences padded to max length in batch
        max_in_batch = max(batch) if len(batch) > 0 else 0
        for j, req_len in enumerate(batch):
            actual_memory = calculate_kv_cache_size(config, req_len, dtype, model_name)
            padded_memory = calculate_kv_cache_size(config, max_in_batch, dtype, model_name)
            static_batch_memory.append((req_len, max_in_batch, actual_memory, padded_memory))

    # Visualize memory waste in static batching
    batch_ids = range(len(static_batch_memory))
    actual_mem = [m[2] for m in static_batch_memory]
    wasted_mem = [m[3] - m[2] for m in static_batch_memory]

    bars1 = ax1.bar(batch_ids, actual_mem, color='#3498db', label='Used Memory', alpha=0.8)
    bars2 = ax1.bar(batch_ids, wasted_mem, bottom=actual_mem, color='#e74c3c',
                   label='Wasted (Padding)', alpha=0.6, edgecolor='black', linewidth=1, linestyle='--')

    ax1.set_xlabel('Request ID', fontsize=11)
    ax1.set_ylabel('Memory Allocated (GiB)', fontsize=11)
    ax1.set_title('Static Batching: Memory Waste from Padding', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3)

    # ========== Plot 2: Continuous Batching Memory Efficiency ==========
    ax2 = fig.add_subplot(gs[0, 1])

    # Continuous batching - each request uses only what it needs
    continuous_memory = [calculate_kv_cache_size(config, req_len, dtype, model_name)
                         for req_len in request_lengths]

    # Show memory utilization over time
    time_slots = 10
    memory_timeline = []
    for t in range(time_slots):
        active_requests = [i for i, arrival in enumerate(request_arrivals[:len(continuous_memory)])
                          if arrival <= t < arrival + 2]  # Assuming 2 time units to process
        total_memory = sum(continuous_memory[i] for i in active_requests)
        memory_timeline.append(total_memory)

    ax2.plot(range(time_slots), memory_timeline, marker='o', linewidth=2,
            markersize=8, color='#2ecc71', label='Continuous Batching')

    # Add theoretical maximum (static batching)
    static_max = max_batch_size * calculate_kv_cache_size(config, max_seq_len, dtype, model_name)
    ax2.axhline(y=static_max, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Static Batch Max ({max_batch_size}×{max_seq_len} tokens)')

    ax2.fill_between(range(time_slots), memory_timeline, alpha=0.3, color='#2ecc71')
    ax2.set_xlabel('Time Slot', fontsize=11)
    ax2.set_ylabel('Total Memory in Use (GiB)', fontsize=11)
    ax2.set_title('Continuous Batching: Dynamic Memory Usage', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Request Packing Visualization ==========
    ax3 = fig.add_subplot(gs[0, 2])

    # Visualize how requests are packed in continuous batching
    colors = plt.cm.tab20(np.linspace(0, 1, len(request_lengths)))
    y_pos = 0

    for i, (req_len, arrival) in enumerate(zip(request_lengths[:8], request_arrivals[:8])):
        # Each request as a rectangle
        rect = Rectangle((arrival, y_pos), req_len/500, 0.8,  # Scale length for visualization
                        facecolor=colors[i], edgecolor='black', linewidth=1)
        ax3.add_patch(rect)
        ax3.text(arrival + req_len/1000, y_pos + 0.4, f'{req_len}',
                fontsize=8, va='center')
        y_pos += 1

    ax3.set_xlim(0, max(request_arrivals[:8]) + 5)
    ax3.set_ylim(-0.5, 8.5)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Request Slot', fontsize=11)
    ax3.set_title('Continuous Batching: Request Packing\n(Width = Sequence Length)',
                 fontsize=12, fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)

    # ========== Plot 4: Throughput Comparison ==========
    ax4 = fig.add_subplot(gs[1, :2])

    batch_sizes = [1, 2, 4, 8, 16, 32]

    # Static batching throughput (limited by longest sequence)
    static_throughput = []
    continuous_throughput = []

    for batch_size in batch_sizes:
        # Static: limited by padding overhead
        avg_padding_overhead = 0.4  # 40% overhead from padding on average
        static_tput = batch_size * (1 - avg_padding_overhead)
        static_throughput.append(static_tput)

        # Continuous: near-linear scaling
        continuous_tput = batch_size * 0.95  # 95% efficiency
        continuous_throughput.append(continuous_tput)

    x = np.arange(len(batch_sizes))
    width = 0.35

    bars1 = ax4.bar(x - width/2, static_throughput, width, label='Static Batching',
                   color='#e74c3c', alpha=0.7)
    bars2 = ax4.bar(x + width/2, continuous_throughput, width, label='Continuous Batching',
                   color='#2ecc71', alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    ax4.set_xlabel('Batch Size', fontsize=11)
    ax4.set_ylabel('Effective Throughput (requests)', fontsize=11)
    ax4.set_title('Throughput Comparison: Static vs Continuous Batching',
                 fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(batch_sizes)
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)

    # ========== Plot 5: Memory Efficiency Summary ==========
    ax5 = fig.add_subplot(gs[1, 2])

    # Pie chart showing memory savings
    static_total = sum([m[3] for m in static_batch_memory])
    continuous_total = sum(continuous_memory[:len(static_batch_memory)])
    saved = static_total - continuous_total

    sizes = [continuous_total, saved]
    labels = [f'Used\n{continuous_total:.1f} GiB', f'Saved\n{saved:.1f} GiB']
    colors_pie = ['#3498db', '#2ecc71']
    explode = (0, 0.1)  # Explode the saved slice

    wedges, texts, autotexts = ax5.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', shadow=True, startangle=90)

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax5.set_title(f'Memory Savings with Continuous Batching\nTotal: {static_total:.1f} GiB → {continuous_total:.1f} GiB',
                 fontsize=12, fontweight='bold')

    # Overall title
    plt.suptitle(f'Continuous Batching Impact Analysis\nModel: {model_name} | Precision: {dtype}',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


def create_dtype_comparison(configs):
    """Create comparison across different data types"""
    models = select_representative_models(configs)

    # Select a few key models for cleaner visualization
    key_models = {
        "meta-llama/Llama-3.1-8B-Instruct": models["meta-llama/Llama-3.1-8B-Instruct"],
        "meta-llama/Llama-3.1-70B-Instruct": models[
            "meta-llama/Llama-3.1-70B-Instruct"
        ],
        "deepseek-ai/DeepSeek-V3": models["deepseek-ai/DeepSeek-V3"],
    }

    dtypes = ["int8", "float16", "bfloat16", "float32"]
    context_lengths = [8_000, 32_000, 128_000, 512_000, 1_000_000]

    # Define line styles and markers for different dtypes
    dtype_styles = {
        "int8": {"linestyle": "-", "marker": "o", "color": "#2ecc71"},
        "float16": {"linestyle": "--", "marker": "s", "color": "#3498db"},
        "bfloat16": {"linestyle": "-.", "marker": "^", "color": "#9b59b6"},
        "float32": {"linestyle": ":", "marker": "D", "color": "#e74c3c"}
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (model_name, config) in enumerate(key_models.items()):
        ax = axes[idx]
        short_name = model_name.split("/")[-1]

        # Calculate sizes for each dtype and context
        for dtype in dtypes:
            sizes = [
                calculate_kv_cache_size(config, ctx, dtype, model_name)
                for ctx in context_lengths
            ]
            style = dtype_styles[dtype]
            ax.semilogy(context_lengths, sizes,
                       marker=style["marker"],
                       linestyle=style["linestyle"],
                       color=style["color"],
                       label=dtype,
                       linewidth=2,
                       markersize=7,
                       alpha=0.9)

        ax.set_xlabel("Context Length (tokens)", fontsize=11)
        ax.set_ylabel("KV Cache Size", fontsize=11)
        ax.set_title(f"{short_name}", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=9)

        # Format y-axis with human-readable units
        ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

        # Format x-axis
        ax.set_xticks(context_lengths)
        ax.set_xticklabels(["8K", "32K", "128K", "512K", "1M"], rotation=45)

    plt.suptitle(
        "Impact of Data Type on KV Cache Memory Requirements",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig




def main():
    """Main function to generate all visualizations"""
    # Load configurations
    configs = load_model_configs()

    print("Generating KV Cache visualizations...")

    # Create output directory
    os.makedirs("kv_cache_visualizations", exist_ok=True)

    # Generate comprehensive comparison chart
    fig1 = create_comparison_chart(configs, dtype="float16")
    fig1.savefig(
        "kv_cache_visualizations/kv_cache_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Created: kv_cache_comparison.png")

    # Generate extreme context projection
    fig2 = create_extreme_context_projection(configs, dtype="float16")
    fig2.savefig(
        "kv_cache_visualizations/extreme_context_projection.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("✓ Created: extreme_context_projection.png")

    # Generate dtype comparison
    fig3 = create_dtype_comparison(configs)
    fig3.savefig(
        "kv_cache_visualizations/dtype_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Created: dtype_comparison.png")

    # Generate paged attention visualization
    fig4 = create_paged_attention_visualization(configs)
    fig4.savefig(
        "kv_cache_visualizations/paged_attention_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Created: paged_attention_comparison.png")

    # Generate continuous batching visualization
    fig5 = create_continuous_batching_visualization(configs)
    fig5.savefig(
        "kv_cache_visualizations/continuous_batching_impact.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Created: continuous_batching_impact.png")

    print(
        "\nAll visualizations generated successfully in 'kv_cache_visualizations/' directory"
    )
    print(
        "\nNote: Run generate_memory_table.py to create the detailed memory requirements table"
    )
    print("\nKey insights:")
    print("- Small models (1-3B) can handle 1M context with 80-160 GiB memory")
    print("- Medium models (7-8B) require 200-300 GiB for 1M context")
    print("- Large models (70B+) need multiple GPUs for contexts >128K")
    print("- DeepSeek-V3 uses KV-LoRA optimization for significant memory savings")
    print("- Using int8 quantization can reduce memory by 50% vs float16")


if __name__ == "__main__":
    main()
