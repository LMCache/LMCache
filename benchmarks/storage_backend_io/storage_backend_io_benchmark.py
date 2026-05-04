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
import stat
import tempfile
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)

DEFAULT_SHAPE = torch.Size([2, 16, 8, 128])
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_CHUNK_SIZE = 4


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
        kv_shape=(4, 2, DEFAULT_CHUNK_SIZE, 8, 128),
        chunk_size=DEFAULT_CHUNK_SIZE,
    )


def _make_memory_objs(
    num_ops: int,
    use_aligned: bool,
    alignment: int,
    keepalive: list[torch.Tensor],
    memory_allocator: Optional[MemoryAllocatorInterface] = None,
    shapes: Optional[list[torch.Size]] = None,
    start_idx: int = 0,
) -> list:
    if memory_allocator is None:
        memory_allocator = AdHocMemoryAllocator(device="cpu")
    # Use provided shapes or default to DEFAULT_SHAPE
    if shapes is None:
        shapes = [DEFAULT_SHAPE]
    objs: list[TensorMemoryObj] = []
    for i in range(num_ops):
        if use_aligned:
            num_bytes = shapes[0].numel() * DEFAULT_DTYPE.itemsize
            base = torch.empty(
                torch.Size([num_bytes + alignment]),
                dtype=torch.uint8,
                device="cpu",
            )
            offset = (-base.data_ptr()) % alignment
            aligned = base[offset : offset + num_bytes]
            keepalive.append(base)
            obj = TensorMemoryObj(
                raw_data=aligned,
                metadata=MemoryObjMetadata(
                    shape=shapes[0],
                    dtype=DEFAULT_DTYPE,
                    address=0,
                    phy_size=0,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.KV_T2D,
                    shapes=shapes,
                    dtypes=[DEFAULT_DTYPE],
                ),
                parent_allocator=memory_allocator,
            )
        else:
            allocated_obj = memory_allocator.allocate(
                shapes,
                [DEFAULT_DTYPE],
                fmt=MemoryFormat.KV_T2D,
            )
            assert allocated_obj is not None
            obj = allocated_obj  # type: ignore[assignment]
        assert obj.tensor is not None
        obj.tensor.fill_(start_idx + i)
        objs.append(obj)
    return objs


def _release_memory_objs(objs: list) -> None:
    for obj in objs:
        try:
            obj.ref_count_down()
        except Exception:
            # Best effort for benchmark cleanup.
            pass


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
    alignment: int,
) -> dict:
    loop, t = _start_loop()
    metadata = _build_metadata()
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=DEFAULT_CHUNK_SIZE,
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
    keepalive: list[torch.Tensor] = []
    objs = _make_memory_objs(num_ops, use_odirect, alignment, keepalive)

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

    # Keep a floor for normal runs but scale for large-op runs.
    # This avoids premature timeout for long single-shot benchmarks.
    timeout_sec = max(300.0, float(num_ops) / 100.0)
    while not done.wait(timeout=1.0):
        if completed >= num_ops:
            break
        if (time.perf_counter() - start) >= timeout_sec:
            raise TimeoutError(
                "LocalDisk benchmark timed out: "
                f"completed={completed}, expected={num_ops}"
            )
    elapsed = time.perf_counter() - start

    _release_memory_objs(objs)
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
    use_uring: bool,
    alignment: int,
    cleanup_raw_device: bool,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    loop, t = _start_loop()

    # Build metadata with the provided chunk_size
    metadata = LMCacheMetadata(
        model_name="benchmark_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=(4, 2, chunk_size, 8, 128),
        chunk_size=chunk_size,
    )

    # For fixed buffer support, we need a larger CPU buffer that's aligned to chunk size
    # Calculate chunk size from metadata
    chunk_size_bytes = (
        metadata.kv_shape[0]
        * metadata.kv_shape[1]
        * chunk_size
        * metadata.kv_shape[3]
        * metadata.kv_shape[4]
        * metadata.kv_dtype.itemsize
    )
    # Calculate required buffer size (num_ops * chunk_size with some margin)
    required_buffer_gb = max(0.01, (num_ops * chunk_size_bytes) / (1024**3))

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
        max_local_cpu_size=required_buffer_gb,
        lmcache_instance_id="bench_rust_raw_block",
    )

    # Create a backing file if raw_device is not provided. For a real block
    # device path (e.g. /dev/nvme*), do not truncate.
    temp_dir: Optional[str] = None
    is_block_device = False
    if not raw_device:
        temp_dir = tempfile.mkdtemp(prefix="raw_block_bench_")
        raw_device = os.path.join(temp_dir, "raw_block.bin")
    else:
        try:
            st_mode = os.stat(raw_device).st_mode
            is_block_device = stat.S_ISBLK(st_mode)
        except FileNotFoundError:
            is_block_device = False

    if raw_device and not is_block_device:
        with open(raw_device, "wb") as f:
            f.truncate(int(raw_device_size_gb * 1024**3))

    config.extra_config = {
        "rust_raw_block.device_path": raw_device,
        "rust_raw_block.block_align": alignment,
        "rust_raw_block.header_bytes": alignment,
        "rust_raw_block.use_odirect": use_odirect,
        "rust_raw_block.use_uring": use_uring,
    }

    # Use MixedMemoryAllocator with use_paging=True for fixed buffer support
    if use_uring:
        # LocalCPUBackend will automatically create MixedMemoryAllocator with
        # use_paging=True when use_uring is enabled
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
        )
    else:
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
    # Use the memory allocator from LocalCPUBackend
    shapes = metadata.get_shapes()
    use_aligned = use_odirect and not use_uring
    objs = _make_memory_objs(
        num_ops,
        use_aligned,
        alignment,
        [],
        memory_allocator=local_cpu.memory_allocator,
        shapes=shapes,
    )

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

    _release_memory_objs(objs)
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
        "use_uring": use_uring,
        "raw_device": raw_device,
    }


def _bench_rust_raw_block_read(
    num_ops: int,
    concurrency: int,
    raw_device: str,
    raw_device_size_gb: float,
    use_odirect: bool,
    use_uring: bool,
    alignment: int,
    cleanup_raw_device: bool,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verify_integrity: bool = False,
) -> dict:
    """Benchmark RustRawBlockBackend write & read performance.

    This function first writes memory objects to the raw block device, then
    performs read benchmarks with concurrent batched blocking reads. Optionally
    verifies data integrity by comparing read data with original written data.

    Args:
        num_ops: Number of operations to perform (both write and read).
        concurrency: Number of concurrent threads for batched operations.
        raw_device: Path to raw block device (empty for temp file).
        raw_device_size_gb: Size of raw device in GB (for temp file creation).
        use_odirect: Enable O_DIRECT for I/O operations.
        use_uring: Enable io_uring for I/O operations.
        alignment: Buffer alignment in bytes.
        cleanup_raw_device: Whether to clean up the raw device after benchmark.
        chunk_size: Chunk size for io_uring case.
        verify_integrity: Whether to verify data integrity after reads.

    Returns:
        Dictionary containing benchmark results including write and read metrics.
    """
    loop, t = _start_loop()

    # Build metadata with the provided chunk_size
    metadata = LMCacheMetadata(
        model_name="benchmark_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=(4, 2, chunk_size, 8, 128),
        chunk_size=chunk_size,
    )

    # For fixed buffer support, we need a larger CPU buffer that's aligned to chunk size
    chunk_size_bytes = (
        metadata.kv_shape[0]
        * metadata.kv_shape[1]
        * chunk_size
        * metadata.kv_shape[3]
        * metadata.kv_shape[4]
        * metadata.kv_dtype.itemsize
    )
    required_buffer_gb = max(0.01, (num_ops * chunk_size_bytes) / (1024**3))

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
        max_local_cpu_size=required_buffer_gb,
        lmcache_instance_id="bench_rust_raw_block_read",
    )

    # Create a backing file if raw_device is not provided
    temp_dir: Optional[str] = None
    is_block_device = False
    if not raw_device:
        temp_dir = tempfile.mkdtemp(prefix="raw_block_read_bench_")
        raw_device = os.path.join(temp_dir, "raw_block.bin")
    else:
        try:
            st_mode = os.stat(raw_device).st_mode
            is_block_device = stat.S_ISBLK(st_mode)
        except FileNotFoundError:
            is_block_device = False

    if raw_device and not is_block_device:
        with open(raw_device, "wb") as f:
            f.truncate(int(raw_device_size_gb * 1024**3))

    config.extra_config = {
        "rust_raw_block.device_path": raw_device,
        "rust_raw_block.block_align": alignment,
        "rust_raw_block.header_bytes": alignment,
        "rust_raw_block.use_odirect": use_odirect,
        "rust_raw_block.use_uring": use_uring,
    }

    # Use MixedMemoryAllocator with use_paging=True for fixed buffer support
    if use_uring:
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
        )
    else:
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
    shapes = metadata.get_shapes()
    use_aligned = use_odirect and not use_uring
    objs = _make_memory_objs(
        num_ops,
        use_aligned,
        alignment,
        [],
        memory_allocator=local_cpu.memory_allocator,
        shapes=shapes,
    )

    # Store original data for integrity verification, keyed by CacheEngineKey
    original_data: dict[CacheEngineKey, torch.Tensor] = {}
    if verify_integrity:
        for key, obj in zip(keys, objs, strict=False):
            assert obj.tensor is not None
            original_data[key] = obj.tensor.clone()

    # Write phase
    write_futures = []
    write_fut_lock = threading.Lock()

    def submit_write_slice(start: int, end: int) -> None:
        futs = backend.batched_submit_put_task(keys[start:end], objs[start:end])
        if futs:
            with write_fut_lock:
                write_futures.extend(futs)

    slice_size = max(1, num_ops // concurrency)
    slices = []
    for i in range(0, num_ops, slice_size):
        slices.append((i, min(i + slice_size, num_ops)))

    write_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for s in slices:
            ex.submit(submit_write_slice, s[0], s[1])

    for fut in write_futures:
        fut.result(timeout=120)
    write_elapsed = time.perf_counter() - write_start

    _release_memory_objs(objs)

    # Read phase
    read_start = time.perf_counter()
    read_results: list[tuple[CacheEngineKey, Optional[MemoryObj]]] = []
    read_lock = threading.Lock()

    def submit_read_slice(start: int, end: int) -> None:
        batch_keys = keys[start:end]
        loaded = backend.batched_get_blocking(batch_keys)
        with read_lock:
            read_results.extend(zip(batch_keys, loaded, strict=False))

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for s in slices:
            ex.submit(submit_read_slice, s[0], s[1])

    read_elapsed = time.perf_counter() - read_start

    # Data integrity verification using key-based lookup
    integrity_errors = 0
    if verify_integrity:
        for key, read_obj in read_results:
            if read_obj is None:
                integrity_errors += 1
                continue
            assert read_obj.tensor is not None
            if key not in original_data:
                integrity_errors += 1
                continue
            if not torch.equal(read_obj.tensor, original_data[key]):
                integrity_errors += 1

    for _, read_obj in read_results:
        if read_obj is not None:
            try:
                read_obj.ref_count_down()
            except Exception:
                pass
    backend.close()
    _stop_loop(loop, t)

    # Best-effort cleanup for temp file or requested cleanup
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
        "backend": "rust_raw_block_read",
        "num_ops": num_ops,
        "concurrency": concurrency,
        "write_elapsed_sec": write_elapsed,
        "write_ops_per_sec": num_ops / write_elapsed if write_elapsed > 0 else 0.0,
        "read_elapsed_sec": read_elapsed,
        "read_ops_per_sec": num_ops / read_elapsed if read_elapsed > 0 else 0.0,
        "total_elapsed_sec": write_elapsed + read_elapsed,
        "use_odirect": use_odirect,
        "use_uring": use_uring,
        "raw_device": raw_device,
        "verify_integrity": verify_integrity,
        "integrity_errors": integrity_errors,
        "integrity_passed": integrity_errors == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LocalDiskBackend vs RustRawBlockBackend "
            "under high write concurrency. "
        )
    )
    parser.add_argument("--num-ops", type=int, default=256, help="Total put ops")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Number of submit threads"
    )
    parser.add_argument(
        "--backend",
        choices=["local_disk", "rust_raw_block", "both", "rust_raw_block_read"],
        default="both",
        help=(
            "Backend to benchmark. "
            "rust_raw_block_read performs write then read benchmark "
            "with optional integrity check."
        ),
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
    parser.add_argument(
        "--use-uring",
        action="store_true",
        help="Enable io_uring for raw block backend",
    )
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for io_uring case (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help="Verify data integrity after reads (only for rust_raw_block_read backend)",
    )
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
                alignment=args.alignment,
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
                use_uring=args.use_uring,
                alignment=args.alignment,
                cleanup_raw_device=cleanup_raw_device,
                chunk_size=args.chunk_size,
            )
        )

    if args.backend == "rust_raw_block_read":
        raw_device = args.raw_device
        cleanup_raw_device = False
        if not raw_device:
            # Use the same filesystem as local disk backend for apples-to-apples.
            raw_device = os.path.join(args.local_disk_dir, "raw_block_read.bin")
            cleanup_raw_device = True
        results.append(
            _bench_rust_raw_block_read(
                num_ops=args.num_ops,
                concurrency=args.concurrency,
                raw_device=raw_device,
                raw_device_size_gb=args.raw_device_size_gb,
                use_odirect=args.raw_odirect,
                use_uring=args.use_uring,
                alignment=args.alignment,
                cleanup_raw_device=cleanup_raw_device,
                chunk_size=args.chunk_size,
                verify_integrity=args.verify_integrity,
            )
        )

    for result in results:
        if result["backend"] == "rust_raw_block_read":
            print(
                f"{result['backend']}: ops={result['num_ops']} "
                f"concurrency={result['concurrency']} "
                f"write_elapsed={result['write_elapsed_sec']:.3f}s "
                f"write_ops/sec={result['write_ops_per_sec']:.2f} "
                f"read_elapsed={result['read_elapsed_sec']:.3f}s "
                f"read_ops/sec={result['read_ops_per_sec']:.2f} "
                f"total_elapsed={result['total_elapsed_sec']:.3f}s"
            )
            if result["verify_integrity"]:
                status = "PASSED" if result["integrity_passed"] else "FAILED"
                print(
                    f"  Integrity check: {status} (errors={result['integrity_errors']})"
                )
        else:
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
