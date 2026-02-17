# SPDX-License-Identifier: Apache-2.0
"""Benchmark LocalDiskBackend vs RustRawBlockBackend under high write concurrency."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import argparse
import asyncio
import json
import os
import tempfile
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)

DEFAULT_SHAPE = torch.Size([2, 16, 8, 128])
DEFAULT_DTYPE = torch.bfloat16


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="bench-loop", daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop: asyncio.AbstractEventLoop, t: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


def _build_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="benchmark_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _make_memory_objs(num_ops: int) -> list:
    allocator = AdHocMemoryAllocator(device="cpu")
    objs = []
    for _ in range(num_ops):
        obj = allocator.allocate(
            [DEFAULT_SHAPE],
            [DEFAULT_DTYPE],
            fmt=MemoryFormat.KV_T2D,
        )
        assert obj is not None
        assert obj.tensor is not None
        obj.tensor.fill_(7)
        objs.append(obj)
    return objs


def _make_keys(num_ops: int) -> list[CacheEngineKey]:
    return [
        CacheEngineKey("benchmark_model", 1, 0, i, DEFAULT_DTYPE)
        for i in range(num_ops)
    ]


def _bench_local_disk(
    num_ops: int,
    concurrency: int,
    local_disk_dir: str,
    max_disk_gb: float,
    use_odirect: bool,
) -> dict:
    loop, t = _start_loop()
    metadata = _build_metadata()
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="bench_local_disk",
    )
    config.local_disk = local_disk_dir
    config.max_local_disk_size = max_disk_gb
    config.extra_config = {"use_odirect": use_odirect}

    local_cpu = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        memory_allocator=AdHocMemoryAllocator(device="cpu"),
    )
    backend = LocalDiskBackend(
        config=config,
        loop=loop,
        local_cpu_backend=local_cpu,
        dst_device="cpu",
        metadata=metadata,
    )

    keys = _make_keys(num_ops)
    objs = _make_memory_objs(num_ops)

    completed = 0
    lock = threading.Lock()
    done = threading.Event()

    def on_complete(_key: CacheEngineKey) -> None:
        nonlocal completed
        with lock:
            completed += 1
            if completed >= num_ops:
                done.set()

    def submit_slice(start: int, end: int) -> None:
        backend.batched_submit_put_task(
            keys[start:end],
            objs[start:end],
            on_complete_callback=on_complete,
        )

    slice_size = max(1, num_ops // concurrency)
    slices = []
    for i in range(0, num_ops, slice_size):
        slices.append((i, min(i + slice_size, num_ops)))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for s in slices:
            ex.submit(submit_slice, s[0], s[1])

    done.wait()
    elapsed = time.perf_counter() - start

    backend.disk_worker.close()
    _stop_loop(loop, t)

    return {
        "backend": "local_disk",
        "num_ops": num_ops,
        "concurrency": concurrency,
        "elapsed_sec": elapsed,
        "ops_per_sec": num_ops / elapsed if elapsed > 0 else 0.0,
        "use_odirect": use_odirect,
        "local_disk_dir": local_disk_dir,
    }


def _bench_rust_raw_block(
    num_ops: int,
    concurrency: int,
    raw_device: str,
    raw_device_size_gb: float,
    use_odirect: bool,
    alignment: int,
    cleanup_raw_device: bool,
) -> dict:
    loop, t = _start_loop()
    metadata = _build_metadata()
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="bench_rust_raw_block",
    )

    # Create a backing file if raw_device not provided.
    temp_dir: Optional[str] = None
    if not raw_device:
        temp_dir = tempfile.mkdtemp(prefix="raw_block_bench_")
        raw_device = os.path.join(temp_dir, "raw_block.bin")
    if raw_device:
        with open(raw_device, "wb") as f:
            f.truncate(int(raw_device_size_gb * 1024**3))

    config.extra_config = {
        "rust_raw_block.device_path": raw_device,
        "rust_raw_block.block_align": alignment,
        "rust_raw_block.header_bytes": alignment,
        "rust_raw_block.use_odirect": use_odirect,
        "rust_raw_block.manifest_write_interval": 0,
    }

    local_cpu = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        memory_allocator=AdHocMemoryAllocator(device="cpu"),
    )
    backend = RustRawBlockBackend(
        config=config,
        metadata=metadata,
        local_cpu_backend=local_cpu,
        loop=loop,
        dst_device="cpu",
    )

    keys = _make_keys(num_ops)
    objs = _make_memory_objs(num_ops)

    futures = []
    fut_lock = threading.Lock()

    def submit_slice(start: int, end: int) -> None:
        futs = backend.batched_submit_put_task(keys[start:end], objs[start:end])
        if futs:
            with fut_lock:
                futures.extend(futs)

    slice_size = max(1, num_ops // concurrency)
    slices = []
    for i in range(0, num_ops, slice_size):
        slices.append((i, min(i + slice_size, num_ops)))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for s in slices:
            ex.submit(submit_slice, s[0], s[1])

    for fut in futures:
        fut.result(timeout=120)

    elapsed = time.perf_counter() - start

    backend.close()
    _stop_loop(loop, t)

    # Best-effort cleanup for temp file or requested cleanup.
    if cleanup_raw_device or temp_dir:
        try:
            os.remove(raw_device)
        except Exception:
            pass
        if temp_dir:
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

    return {
        "backend": "rust_raw_block",
        "num_ops": num_ops,
        "concurrency": concurrency,
        "elapsed_sec": elapsed,
        "ops_per_sec": num_ops / elapsed if elapsed > 0 else 0.0,
        "use_odirect": use_odirect,
        "raw_device": raw_device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LocalDiskBackend vs RustRawBlockBackend "
            "under high write concurrency."
        )
    )
    parser.add_argument("--num-ops", type=int, default=256, help="Total put ops")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Number of submit threads"
    )
    parser.add_argument(
        "--backend",
        choices=["local_disk", "rust_raw_block", "both"],
        default="both",
    )
    parser.add_argument(
        "--local-disk-dir",
        type=str,
        default="/tmp/lmcache_local_disk_bench",
    )
    parser.add_argument("--max-local-disk-gb", type=float, default=2.0)
    parser.add_argument(
        "--local-disk-odirect",
        action="store_true",
        help="Enable O_DIRECT for local disk backend",
    )
    parser.add_argument(
        "--raw-device",
        type=str,
        default="",
        help="Raw block device path (if empty, uses a temp file)",
    )
    parser.add_argument("--raw-device-size-gb", type=float, default=1.0)
    parser.add_argument(
        "--raw-odirect",
        action="store_true",
        help="Enable O_DIRECT for raw block backend",
    )
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON file path or directory",
    )

    args = parser.parse_args()

    results = []
    if args.backend in ("local_disk", "both"):
        results.append(
            _bench_local_disk(
                num_ops=args.num_ops,
                concurrency=args.concurrency,
                local_disk_dir=args.local_disk_dir,
                max_disk_gb=args.max_local_disk_gb,
                use_odirect=args.local_disk_odirect,
            )
        )

    if args.backend in ("rust_raw_block", "both"):
        raw_device = args.raw_device
        cleanup_raw_device = False
        if not raw_device:
            # Use the same filesystem as local disk backend for apples-to-apples.
            raw_device = os.path.join(args.local_disk_dir, "raw_block.bin")
            cleanup_raw_device = True
        results.append(
            _bench_rust_raw_block(
                num_ops=args.num_ops,
                concurrency=args.concurrency,
                raw_device=raw_device,
                raw_device_size_gb=args.raw_device_size_gb,
                use_odirect=args.raw_odirect,
                alignment=args.alignment,
                cleanup_raw_device=cleanup_raw_device,
            )
        )

    for result in results:
        print(
            f"{result['backend']}: ops={result['num_ops']} "
            f"concurrency={result['concurrency']} "
            f"elapsed={result['elapsed_sec']:.3f}s "
            f"ops/sec={result['ops_per_sec']:.2f}"
        )

    if args.output_json:
        output_path = args.output_json
        if output_path.endswith(os.sep) or os.path.isdir(output_path):
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"storage_backend_io_{ts}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
