#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark Q capture planning and metrics snapshot overhead.

The benchmark prints one JSON document to stdout and never writes result files.
It can also run against an older LMCache checkout: metrics fields are reported
as unavailable when that checkout does not expose ``metrics_snapshot()``.

Example:
    python benchmarks/microbenchmark/q_capture_observability_benchmark.py \
        --device cuda --requests 8 --tokens-per-request 256 --iterations 100
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any
import argparse
import json
import platform
import statistics
import time

# Third Party
import torch

# First Party
from lmcache.sdk.qringbuffer import QRingBuffer, QRingBufferCapture


@dataclass
class _Op:
    """Minimal aligned store operation consumed by query capture."""

    block_ids: list[list[int]]
    token_ids: list[int]
    start: int
    end: int


@dataclass
class _Request:
    """Minimal STORE metadata consumed by query capture."""

    request_id: str
    direction: str
    op: _Op
    cache_salt: str = ""


def _percentile(values: list[int], quantile: float) -> int:
    """Return a nearest-rank percentile from nonempty integer samples."""
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA work when the selected device is asynchronous."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_fixture(
    device: torch.device,
    requests: int,
    tokens_per_request: int,
    block_size: int,
    hidden_dim: int,
) -> tuple[QRingBufferCapture, QRingBuffer, torch.Tensor, Any, Any]:
    """Build one stable continuous-batching capture input."""
    blocks_per_request = tokens_per_request // block_size
    total_blocks = requests * blocks_per_request
    ring = QRingBuffer(
        num_layers=1,
        num_blocks=total_blocks,
        block_size=block_size,
        hidden_dim=hidden_dim,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device=device,
    )
    ring_adapter = SimpleNamespace(q_ring=ring)
    worker_adapter = SimpleNamespace(is_kv_writer=True)
    capture = QRingBufferCapture(worker_adapter, ring_adapter)  # type: ignore[arg-type]

    row_slots: list[int] = []
    request_metadata: list[_Request] = []
    for request_index in range(requests):
        first_block = request_index * blocks_per_request
        gpu_blocks = list(range(first_block, first_block + blocks_per_request))
        for block_id in gpu_blocks:
            row_slots.extend(
                block_id * block_size + offset for offset in range(block_size)
            )
        request_metadata.append(
            _Request(
                request_id=f"request-{request_index}",
                direction="STORE",
                op=_Op(
                    block_ids=[gpu_blocks],
                    token_ids=list(range(tokens_per_request)),
                    start=0,
                    end=tokens_per_request,
                ),
            )
        )

    total_tokens = requests * tokens_per_request
    query = torch.zeros(
        (total_tokens, hidden_dim),
        dtype=ring.dtype,
        device=device,
    )
    metadata = SimpleNamespace(requests=request_metadata)
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor(row_slots, dtype=torch.int64, device=device)
    )
    return capture, ring, query, metadata, attn_metadata


def _run_plan_once(
    capture: QRingBufferCapture,
    ring: QRingBuffer,
    query: torch.Tensor,
    metadata: Any,
    attn_metadata: Any,
    expected_requests: int,
) -> None:
    """Build and immediately reclaim one plan, checking allocator invariants."""
    state = capture._build_q_step_state(query, metadata, attn_metadata)
    if state is None or len(state.stores) != expected_requests:
        raise RuntimeError("capture plan did not include every benchmark request")
    for store in state.stores:
        ring.free(store.ring_block_ids)
    if ring.num_free_blocks() != ring.num_blocks:
        raise RuntimeError("capture benchmark leaked query-ring blocks")


def main() -> None:
    """Run the benchmark and print one machine-readable JSON document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--tokens-per-request", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--snapshot-iterations", type=int, default=10_000)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if args.requests <= 0 or args.tokens_per_request <= 0:
        raise ValueError("requests and tokens-per-request must be positive")
    if args.tokens_per_request % args.block_size != 0:
        raise ValueError("tokens-per-request must be divisible by block-size")
    if args.warmup < 0 or args.iterations <= 0 or args.snapshot_iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")

    device = torch.device(args.device)
    capture, ring, query, metadata, attn_metadata = _build_fixture(
        device,
        args.requests,
        args.tokens_per_request,
        args.block_size,
        args.hidden_dim,
    )

    for _ in range(args.warmup):
        _run_plan_once(
            capture,
            ring,
            query,
            metadata,
            attn_metadata,
            args.requests,
        )
    _synchronize(device)

    plan_samples_ns: list[int] = []
    for _ in range(args.iterations):
        started_ns = time.perf_counter_ns()
        _run_plan_once(
            capture,
            ring,
            query,
            metadata,
            attn_metadata,
            args.requests,
        )
        _synchronize(device)
        plan_samples_ns.append(time.perf_counter_ns() - started_ns)

    snapshot_samples_ns: list[int] = []
    metrics_snapshot = getattr(capture, "metrics_snapshot", None)
    if metrics_snapshot is not None:
        for _ in range(args.snapshot_iterations):
            started_ns = time.perf_counter_ns()
            metrics_snapshot()
            snapshot_samples_ns.append(time.perf_counter_ns() - started_ns)
        capture_metrics = asdict(metrics_snapshot())
    else:
        capture_metrics = None

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor()
    )
    output = {
        "schema": "lmcache.q-capture-observability-benchmark.v1",
        "environment": {
            "device": str(device),
            "device_name": device_name,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "config": {
            "requests": args.requests,
            "tokens_per_request": args.tokens_per_request,
            "block_size": args.block_size,
            "hidden_dim": args.hidden_dim,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "snapshot_iterations": args.snapshot_iterations,
        },
        "capture_plan_ns": {
            "median": int(statistics.median(plan_samples_ns)),
            "p95": _percentile(plan_samples_ns, 0.95),
            "min": min(plan_samples_ns),
            "max": max(plan_samples_ns),
        },
        "metrics_snapshot_ns": (
            {
                "median": int(statistics.median(snapshot_samples_ns)),
                "p95": _percentile(snapshot_samples_ns, 0.95),
                "min": min(snapshot_samples_ns),
                "max": max(snapshot_samples_ns),
            }
            if snapshot_samples_ns
            else None
        ),
        "capture_metrics": capture_metrics,
        "invariants": {
            "ring_fully_reclaimed": ring.num_free_blocks() == ring.num_blocks,
            "sample_count": len(plan_samples_ns) == args.iterations,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
