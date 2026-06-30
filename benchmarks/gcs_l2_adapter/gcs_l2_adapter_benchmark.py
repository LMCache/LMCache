#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark GCSL2Adapter: store, lookup, and load throughput against a real GCS bucket.

Measures end-to-end latency and throughput of the three core operations
(store / lookup / load) using the poll-driven L2AdapterInterface. Each
operation is run for --warmup iterations (discarded) then --num-iterations
measured iterations.

Usage:
    pip install google-cloud-storage
    python benchmarks/gcs_l2_adapter/gcs_l2_adapter_benchmark.py \
        --bucket my-lmcache-bench \
        --credentials-file /path/to/sa.json \
        --object-size-kb 512 \
        --batch-size 4 \
        --num-iterations 10
"""

# Future
from __future__ import annotations

# Standard
import argparse
import hashlib
import statistics
import sys
import time
import types

# ---------------------------------------------------------------------------
# Pure-Python fallback for native_storage_ops (Bitmap) so the benchmark runs
# without a compiled lmcache wheel.  Mirrors the stub in
# tests/v1/distributed/conftest.py.
# ---------------------------------------------------------------------------
if "lmcache.native_storage_ops" not in sys.modules:

    class _Bitmap:
        def __init__(self, size: int, first_n: int = 0) -> None:
            self._size = int(size)
            self._bits: set[int] = {i for i in range(min(int(first_n), self._size))}

        def __len__(self) -> int:
            return self._size

        def set(self, index: int) -> None:
            if index < 0 or index >= self._size:
                raise IndexError(index)
            self._bits.add(int(index))

        def get(self, index: int) -> bool:
            return int(index) in self._bits

        def test(self, index: int) -> bool:
            return int(index) in self._bits

        def get_indices_list(self) -> list[int]:
            return sorted(self._bits)

    _mod = types.ModuleType("lmcache.native_storage_ops")
    _mod.Bitmap = _Bitmap  # type: ignore[attr-defined]
    sys.modules["lmcache.native_storage_ops"] = _mod

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.gcs_l2_adapter import (
    GCSL2Adapter,
    GCSL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)

# Poll every 2 ms; give up after 60 s.
_POLL_INTERVAL_S = 0.002
_POLL_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keys(n: int, run_id: int = 0) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=hashlib.sha256(f"bench-{run_id}-{i}".encode()).digest(),
            model_name="bench-model",
            kv_rank=i,
        )
        for i in range(n)
    ]


def _make_objects(
    n: int, size_bytes: int, allocator: AdHocMemoryAllocator
) -> list[MemoryObj]:
    objs = []
    for _ in range(n):
        obj = allocator.allocate(
            shapes=torch.Size([size_bytes]),
            dtypes=torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
        )
        assert obj is not None
        obj.raw_data.random_()
        objs.append(obj)
    return objs


def _poll_store(adapter: GCSL2Adapter, task_id: int) -> bool:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        results = adapter.pop_completed_store_tasks()
        if task_id in results:
            return results[task_id].is_successful()
        time.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(
        f"store task {task_id} did not complete within {_POLL_TIMEOUT_S}s"
    )


def _poll_lookup(adapter: GCSL2Adapter, task_id: int):
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        result = adapter.query_lookup_and_lock_result(task_id)
        if result is not None:
            return result
        time.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(
        f"lookup task {task_id} did not complete within {_POLL_TIMEOUT_S}s"
    )


def _poll_load(adapter: GCSL2Adapter, task_id: int):
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        result = adapter.query_load_result(task_id)
        if result is not None:
            return result
        time.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(
        f"load task {task_id} did not complete within {_POLL_TIMEOUT_S}s"
    )


def _bits_set(bitmap) -> int:
    return sum(1 for i in range(len(bitmap)) if bitmap.get(i))


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------


def bench_store(
    adapter: GCSL2Adapter,
    allocator: AdHocMemoryAllocator,
    batch_size: int,
    size_bytes: int,
    iters: int,
    warmup: int,
) -> dict:
    latencies: list[float] = []
    total_bytes = batch_size * size_bytes

    for i in range(warmup + iters):
        keys = _make_keys(batch_size, run_id=i)
        objects = _make_objects(batch_size, size_bytes, allocator)

        t0 = time.monotonic()
        task_id = adapter.submit_store_task(keys, objects)
        success = _poll_store(adapter, task_id)
        elapsed = time.monotonic() - t0

        if not success:
            print(f"  [store iter {i}] FAILED")
        if i >= warmup:
            latencies.append(elapsed)

    throughputs = [total_bytes / lat / 1e6 for lat in latencies]
    return {
        "latency_ms": _stats_ms(latencies),
        "throughput_mbps": _stats(throughputs),
        "total_bytes_per_batch": total_bytes,
    }


def bench_lookup(
    adapter: GCSL2Adapter,
    keys: list[ObjectKey],
    iters: int,
    warmup: int,
) -> dict:
    latencies: list[float] = []
    hit_rates: list[float] = []
    n = len(keys)

    for i in range(warmup + iters):
        t0 = time.monotonic()
        task_id = adapter.submit_lookup_and_lock_task(
            keys,
            None,  # type: ignore[arg-type]
        )
        bitmap = _poll_lookup(adapter, task_id)
        elapsed = time.monotonic() - t0

        hits = _bits_set(bitmap)
        adapter.submit_unlock(keys)

        if i >= warmup:
            latencies.append(elapsed)
            hit_rates.append(hits / n)

    return {
        "latency_ms": _stats_ms(latencies),
        "hit_rate": {
            "mean": statistics.mean(hit_rates),
            "min": min(hit_rates),
            "max": max(hit_rates),
        },
    }


def bench_load(
    adapter: GCSL2Adapter,
    allocator: AdHocMemoryAllocator,
    keys: list[ObjectKey],
    size_bytes: int,
    iters: int,
    warmup: int,
) -> dict:
    latencies: list[float] = []
    total_bytes = len(keys) * size_bytes

    for i in range(warmup + iters):
        buffers = _make_objects(len(keys), size_bytes, allocator)

        t0 = time.monotonic()
        task_id = adapter.submit_load_task(keys, buffers)
        bitmap = _poll_load(adapter, task_id)
        elapsed = time.monotonic() - t0

        hits = _bits_set(bitmap)
        if hits < len(keys):
            print(f"  [load iter {i}] only {hits}/{len(keys)} keys loaded")
        if i >= warmup:
            latencies.append(elapsed)

    throughputs = [total_bytes / lat / 1e6 for lat in latencies]
    return {
        "latency_ms": _stats_ms(latencies),
        "throughput_mbps": _stats(throughputs),
        "total_bytes_per_batch": total_bytes,
    }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _stats_ms(latencies: list[float]) -> dict:
    ms = [x * 1000 for x in latencies]
    return {
        "mean": statistics.mean(ms),
        "median": statistics.median(ms),
        "p95": sorted(ms)[int(len(ms) * 0.95)],
        "min": min(ms),
        "max": max(ms),
    }


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _print_result(label: str, result: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")

    lat = result.get("latency_ms", {})
    print(
        f"  Latency (ms)   mean={lat.get('mean', 0):.1f}  "
        f"median={lat.get('median', 0):.1f}  "
        f"p95={lat.get('p95', 0):.1f}  "
        f"min={lat.get('min', 0):.1f}  "
        f"max={lat.get('max', 0):.1f}"
    )

    if "throughput_mbps" in result:
        tp = result["throughput_mbps"]
        print(
            f"  Throughput     mean={tp['mean']:.2f} MB/s  "
            f"median={tp['median']:.2f} MB/s"
        )
        bpb = result.get("total_bytes_per_batch", 0)
        print(f"  Bytes/batch    {bpb / 1e6:.3f} MB")

    if "hit_rate" in result:
        hr = result["hit_rate"]
        pct = {k: v * 100 for k, v in hr.items()}
        print(
            f"  Hit rate       mean={pct['mean']:.1f}%  "
            f"(min={pct['min']:.1f}%  max={pct['max']:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Token-to-bytes helper
# ---------------------------------------------------------------------------


def _chunk_size_bytes(
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    chunk_tokens: int,
    dtype_bytes: int,
) -> int:
    """Bytes for one KV-cache chunk: layers × 2(K+V) × heads × dim × tokens × dtype."""
    return num_layers * 2 * kv_heads * head_dim * chunk_tokens * dtype_bytes


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def _run_one(
    adapter: GCSL2Adapter,
    allocator: AdHocMemoryAllocator,
    batch_size: int,
    size_bytes: int,
    iters: int,
    warmup: int,
    run_id: int,
) -> dict:
    """Run store → seed → lookup → load for one (batch_size, size_bytes) combo."""
    store = bench_store(adapter, allocator, batch_size, size_bytes, iters, warmup)

    seed_keys = _make_keys(batch_size, run_id=run_id)
    seed_objs = _make_objects(batch_size, size_bytes, allocator)
    tid = adapter.submit_store_task(seed_keys, seed_objs)
    _poll_store(adapter, tid)

    lookup = bench_lookup(adapter, seed_keys, iters, warmup)
    load = bench_load(adapter, allocator, seed_keys, size_bytes, iters, warmup)
    return {"store": store, "lookup": lookup, "load": load}


def _print_matrix(rows: list[dict]) -> None:
    """Print a summary matrix table over all sweep rows."""
    w = 22
    h1 = f"{'Config':<{w}}"
    h2 = f"{'Store lat(ms)':>14} {'Store MB/s':>11}"
    h3 = f"{'Lookup lat(ms)':>15} {'Hit%':>5}"
    h4 = f"{'Load lat(ms)':>13} {'Load MB/s':>10}"
    header = h1 + h2 + h3 + h4
    print(f"\n{'=' * len(header)}")
    print("  SWEEP SUMMARY")
    print(f"{'=' * len(header)}")
    print(f"  {header}")
    print(f"  {'-' * (len(header) - 2)}")
    for row in rows:
        label = row["label"]
        s = row["store"]
        lk = row["lookup"]
        ld = row["load"]
        hit = lk["hit_rate"]["mean"] * 100
        print(
            f"  {label:<{w}}"
            f"{s['latency_ms']['mean']:>13.1f} ms"
            f"{s['throughput_mbps']['mean']:>11.2f}"
            f"{lk['latency_ms']['mean']:>14.1f} ms"
            f"{hit:>6.0f}%"
            f"{ld['latency_ms']['mean']:>12.1f} ms"
            f"{ld['throughput_mbps']['mean']:>10.2f}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark GCSL2Adapter: matrix sweep over sizes and batches"
    )
    # Connection
    p.add_argument("--bucket", required=True, help="GCS bucket name")
    p.add_argument(
        "--credentials-file", default=None, help="Path to service-account JSON key"
    )
    p.add_argument("--project", default=None, help="GCP project ID")
    p.add_argument("--num-workers", type=int, default=64, help="Thread-pool size")

    # Sweep axes
    p.add_argument(
        "--object-sizes-kb",
        default="64,256,512,1024,4096",
        help="Comma-separated object sizes in KB (default: 64,256,512,1024,4096)",
    )
    p.add_argument(
        "--batch-sizes",
        default="1,4,8,16",
        help="Comma-separated batch sizes (default: 1,4,8,16)",
    )

    # Token mode
    p.add_argument(
        "--tokens",
        type=int,
        default=None,
        help=(
            "Also benchmark N tokens worth of KV cache. "
            "Computes chunk size from model params and uses "
            "ceil(N/chunk-tokens) as batch size."
        ),
    )
    p.add_argument(
        "--model-layers",
        type=int,
        default=32,
        help="Layers for token mode (default 32)",  # noqa: E501
    )
    p.add_argument(
        "--kv-heads", type=int, default=8, help="KV heads for token mode (default 8)"
    )
    p.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dim for token mode (default 128)",  # noqa: E501
    )
    p.add_argument(
        "--chunk-tokens",
        type=int,
        default=256,
        help="Tokens per chunk for token mode (default 256)",
    )
    p.add_argument(
        "--dtype-bytes",
        type=int,
        default=2,
        help="Bytes per element: 2=bfloat16/fp16, 4=fp32 (default 2)",
    )

    # Iteration control
    p.add_argument(
        "--num-iterations", type=int, default=5, help="Measured iterations (default 5)"
    )
    p.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations, discarded (default 1)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    creds = args.credentials_file or "Application Default Credentials"

    object_sizes_kb = [int(x) for x in args.object_sizes_kb.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("\nGCS L2 Adapter Benchmark — Matrix Sweep")
    print(f"  bucket        : {args.bucket}")
    print(f"  credentials   : {creds}")
    print(f"  object sizes  : {object_sizes_kb} KB")
    print(f"  batch sizes   : {batch_sizes}")
    print(f"  iterations    : {args.num_iterations} (+ {args.warmup} warmup)")
    print(f"  num workers   : {args.num_workers}")

    if args.tokens:
        tok_chunk_bytes = _chunk_size_bytes(
            args.model_layers,
            args.kv_heads,
            args.head_dim,
            args.chunk_tokens,
            args.dtype_bytes,
        )
        # Standard
        import math

        tok_batch = math.ceil(args.tokens / args.chunk_tokens)
        print(
            f"\n  Token mode    : {args.tokens} tokens → "
            f"{tok_batch} chunks × {tok_chunk_bytes // 1024} KB each "
            f"({tok_batch * tok_chunk_bytes / 1e6:.1f} MB total)"
        )
        print(
            f"  Model config  : {args.model_layers}L × "
            f"{args.kv_heads}kv-heads × dim{args.head_dim} × "
            f"{args.chunk_tokens}tok/chunk"
        )

    config = GCSL2AdapterConfig(
        gcs_bucket=args.bucket,
        gcs_credentials_file=args.credentials_file,
        gcs_project=args.project,
        gcs_num_workers=args.num_workers,
    )

    print("\nInitializing adapter...")
    adapter = GCSL2Adapter(config)
    allocator = AdHocMemoryAllocator(device="cpu")

    sweep_rows: list[dict] = []
    run_id = 0
    total = len(object_sizes_kb) * len(batch_sizes)
    done = 0

    for size_kb in object_sizes_kb:
        for batch in batch_sizes:
            done += 1
            size_bytes = size_kb * 1024
            total_mb = batch * size_bytes / 1e6
            label = f"{size_kb}KB × batch{batch}"
            print(f"\n[{done}/{total}] {label}  ({total_mb:.2f} MB/call)")
            result = _run_one(
                adapter,
                allocator,
                batch,
                size_bytes,
                args.num_iterations,
                args.warmup,
                run_id,
            )
            run_id += 1
            sweep_rows.append({"label": label, **result})
            s = result["store"]
            lk = result["lookup"]
            ld = result["load"]
            print(
                f"  store  {s['latency_ms']['mean']:>7.1f} ms  "
                f"{s['throughput_mbps']['mean']:>6.2f} MB/s  |  "
                f"lookup {lk['latency_ms']['mean']:>6.1f} ms  "
                f"hit={lk['hit_rate']['mean'] * 100:.0f}%  |  "
                f"load  {ld['latency_ms']['mean']:>7.1f} ms  "
                f"{ld['throughput_mbps']['mean']:>6.2f} MB/s"
            )

    # ---- Token mode row ----
    if args.tokens:
        # Standard
        import math

        tok_chunk_bytes = _chunk_size_bytes(
            args.model_layers,
            args.kv_heads,
            args.head_dim,
            args.chunk_tokens,
            args.dtype_bytes,
        )
        tok_batch = math.ceil(args.tokens / args.chunk_tokens)
        total_mb = tok_batch * tok_chunk_bytes / 1e6
        label = f"{args.tokens}tok ({tok_batch}×{tok_chunk_bytes // 1024}KB)"
        print(f"\n[tokens] {label}  ({total_mb:.1f} MB/call)")
        # Use fewer iterations for very large objects
        tok_iters = max(3, args.num_iterations // 2)
        result = _run_one(
            adapter,
            allocator,
            tok_batch,
            tok_chunk_bytes,
            tok_iters,
            1,
            run_id,
        )
        sweep_rows.append({"label": label, **result})
        s = result["store"]
        lk = result["lookup"]
        ld = result["load"]
        print(
            f"  store  {s['latency_ms']['mean']:>7.1f} ms  "
            f"{s['throughput_mbps']['mean']:>6.2f} MB/s  |  "
            f"lookup {lk['latency_ms']['mean']:>6.1f} ms  "
            f"hit={lk['hit_rate']['mean'] * 100:.0f}%  |  "
            f"load  {ld['latency_ms']['mean']:>7.1f} ms  "
            f"{ld['throughput_mbps']['mean']:>6.2f} MB/s"
        )

    _print_matrix(sweep_rows)
    adapter.close()


if __name__ == "__main__":
    main()
