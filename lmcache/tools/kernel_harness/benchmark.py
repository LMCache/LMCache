# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Callable

# Third Party
import torch

# Local
from .config import Direction, TestConfig
from .reference import reference_multi_layer_block_kv_transfer
from .tensor_factory import (
    create_block_ids,
    create_memory_objects,
    create_vllm_tensors,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config_name: str
    dtype: str
    d2h_latency_ms: float
    h2d_latency_ms: float
    d2h_throughput_gbps: float
    h2d_throughput_gbps: float
    total_bytes: int


def _compute_total_bytes(config: TestConfig) -> int:
    """Compute total bytes transferred per kernel call."""
    element_size = 1 if config.dtype == torch.float8_e4m3fn else config.dtype.itemsize
    return (
        config.num_memory_objects
        * config.tokens_per_object
        * config.num_layers
        * config.hidden_dim
        * config.kv_dim
        * element_size
    )


def run_benchmark(
    config: TestConfig,
    kernel_fn: Callable = reference_multi_layer_block_kv_transfer,
) -> BenchmarkResult:
    """Benchmark D2H and H2D transfers separately.

    Uses CUDA events for precise GPU-side timing.
    """
    device = torch.device("cuda")

    # Set up tensors
    vllm_tensors = create_vllm_tensors(config, device)
    mem_objects = create_memory_objects(config)
    block_ids = create_block_ids(config, seed=42)

    total_bytes = _compute_total_bytes(config)
    dtype_name = "bf16" if config.dtype == torch.bfloat16 else "fp8"

    # Warmup
    for _ in range(config.num_warmup_iters):
        kernel_fn(vllm_tensors, mem_objects, block_ids, config, Direction.D2H)
        torch.cuda.synchronize()
        kernel_fn(vllm_tensors, mem_objects, block_ids, config, Direction.H2D)
        torch.cuda.synchronize()

    # Benchmark D2H
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(config.num_bench_iters):
        kernel_fn(vllm_tensors, mem_objects, block_ids, config, Direction.D2H)
    end.record()
    torch.cuda.synchronize()
    d2h_ms = start.elapsed_time(end) / config.num_bench_iters

    # Benchmark H2D
    start.record()
    for _ in range(config.num_bench_iters):
        kernel_fn(vllm_tensors, mem_objects, block_ids, config, Direction.H2D)
    end.record()
    torch.cuda.synchronize()
    h2d_ms = start.elapsed_time(end) / config.num_bench_iters

    # Compute throughput
    d2h_gbps = total_bytes / (d2h_ms * 1e-3) / 1e9 if d2h_ms > 0 else 0.0
    h2d_gbps = total_bytes / (h2d_ms * 1e-3) / 1e9 if h2d_ms > 0 else 0.0

    return BenchmarkResult(
        config_name=config.name,
        dtype=dtype_name,
        d2h_latency_ms=d2h_ms,
        h2d_latency_ms=h2d_ms,
        d2h_throughput_gbps=d2h_gbps,
        h2d_throughput_gbps=h2d_gbps,
        total_bytes=total_bytes,
    )


def print_benchmark_table(results: list) -> None:
    """Print benchmark results as a formatted table."""
    header = (
        f"{'Config':<20} {'Dtype':<6} "
        f"{'D2H ms':>10} {'H2D ms':>10} "
        f"{'D2H GB/s':>10} {'H2D GB/s':>10} "
        f"{'Bytes':>14}"
    )
    separator = "=" * len(header)

    print()
    print(separator)
    print("LMCache Block Transfer Kernel Benchmark Results")
    print(separator)
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.config_name:<20} {r.dtype:<6} "
            f"{r.d2h_latency_ms:>10.3f} {r.h2d_latency_ms:>10.3f} "
            f"{r.d2h_throughput_gbps:>10.2f} {r.h2d_throughput_gbps:>10.2f} "
            f"{r.total_bytes:>14,}"
        )

    print(separator)
    print()
