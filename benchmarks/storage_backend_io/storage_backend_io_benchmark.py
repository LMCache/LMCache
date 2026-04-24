# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
#
# 2026/4/20 support benchmark write performance of hf3fs and fs backend
#   Wenwen Chen <wenwen.chen@samsung.com>
#   Ruyi Zhang <ruyi.zhang@samsung.com>

"""Benchmark storage backends under high write concurrency.
This module provides a framework for benchmarking different storage backends
(LocalDiskBackend, RustRawBlockBackend, RemoteBackend, etc.) with consistent
logic .
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional
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
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

logger = init_logger(__name__)
# Type aliases
# MemoryObj = TensorMemoryObj
OnCompleteCallback = Callable[[CacheEngineKey], None]


# ============================================================================
# Constants
# ============================================================================

DEFAULT_SHAPE = torch.Size([2, 16, 8, 128])
DEFAULT_DTYPE = torch.bfloat16  # 2Bytes

# ============================================================================
# Helper Functions
# ============================================================================


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start an async event loop in a background thread."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="bench-loop", daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop: asyncio.AbstractEventLoop, t: threading.Thread) -> None:
    """Stop the async event loop."""
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


def _build_metadata() -> LMCacheMetadata:
    """Build test metadata for benchmark."""
    return LMCacheMetadata(
        model_name="benchmark_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _make_memory_objs(
    num_ops: int,
    use_aligned: bool,
    alignment: int,
    keepalive: list[torch.Tensor],
) -> list:
    """Create memory objects for benchmark."""
    allocator = AdHocMemoryAllocator(device="cpu")
    objs = []
    for _ in range(num_ops):
        if use_aligned:
            num_bytes = DEFAULT_SHAPE.numel() * DEFAULT_DTYPE.itemsize
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
                    shape=DEFAULT_SHAPE,
                    dtype=DEFAULT_DTYPE,
                    address=0,
                    phy_size=0,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.KV_T2D,
                    shapes=[DEFAULT_SHAPE],
                    dtypes=[DEFAULT_DTYPE],
                ),
                parent_allocator=allocator,
            )
        else:
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


def _release_memory_objs(objs: list) -> None:
    """Release memory objects."""
    for obj in objs:
        try:
            obj.ref_count_down()
        except Exception:
            pass


def _make_keys(num_ops: int) -> list[CacheEngineKey]:
    """Create cache keys for benchmark."""
    return [
        CacheEngineKey("benchmark_model", 1, 0, i, DEFAULT_DTYPE)
        for i in range(num_ops)
    ]


# ============================================================================
# Abstract Base Class for Storage Backends
# ============================================================================
class StorageBackendBenchmark(ABC):
    """Abstract base class for storage backend benchmarks.

    This class provides common benchmark logic and defines abstract methods
    that each backend implementation must override.
    """

    def __init__(
        self,
        name: str,
        num_ops: int,
        concurrency: int,
        use_odirect: bool,
        alignment: int,
    ):
        self._backend_name = name
        self.num_ops = num_ops
        self.concurrency = concurrency
        self.use_odirect = use_odirect
        self.alignment = alignment

        # Runtime state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._local_cpu: LocalCPUBackend
        self._backend: StorageBackendInterface
        self._keys: list[CacheEngineKey] = []
        self._objs: list[TensorMemoryObj] = []
        self._keepalive: list[torch.Tensor] = []
        self._start_time: float
        # completed ops
        self._completed = 0
        # lock for _completed
        self._lock = threading.Lock()
        self._done = threading.Event()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    @abstractmethod
    def extra_config_keys(self) -> dict:
        """Return extra config keys specific to this backend."""
        pass

    @abstractmethod
    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> StorageBackendInterface:
        """Create and return the backend instance."""
        pass

    @abstractmethod
    def _close_backend(self) -> None:
        """Close the backend and perform cleanup."""
        pass

    def _submit_put_tasks(
        self,
        keys: list[CacheEngineKey],
        objs: list,
    ) -> list[Any]:
        """Submit put tasks using callback pattern."""
        self._completed = 0
        self._done.clear()

        def on_complete(_key: CacheEngineKey) -> None:
            with self._lock:
                self._completed += 1
                if self._completed >= self.num_ops:
                    self._done.set()

        def submit_slice(start: int, end: int) -> None:
            self._backend.batched_submit_put_task(
                keys[start:end],
                objs[start:end],
                on_complete_callback=on_complete,
            )

        slice_size = max(1, self.num_ops // self.concurrency)
        slices = [
            (i, min(i + slice_size, self.num_ops))
            for i in range(0, self.num_ops, slice_size)
        ]

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for s in slices:
                ex.submit(submit_slice, s[0], s[1])

        # Return empty list since we use callback pattern
        return []

    def _wait_for_completion(self, pending_ops: list[Any]) -> None:
        """Wait using Event pattern."""
        # Keep a floor for normal runs but scale for large-op runs.
        # This avoids premature timeout for long single-shot benchmarks.
        timeout_sec = max(300.0, float(self.num_ops) / 100.0)
        while not self._done.wait(timeout=1.0):
            if self._completed >= self.num_ops:
                break
            if float(time.perf_counter() - self._start_time) >= timeout_sec:
                raise TimeoutError(
                    f"{self.backend_name} benchmark timed out: "
                    f"completed={self._completed}, expected={self.num_ops}"
                )

    def _setup_filesystem(self) -> None:
        return None

    def _cleanup_filesystem(self) -> None:
        return None

    def _setup_config(self, base_config: LMCacheEngineConfig) -> LMCacheEngineConfig:
        """Setup backend-specific configuration. Override if needed."""
        return base_config

    def run(self) -> dict:
        """Run the benchmark and return results."""
        # Setup
        self._loop, self._loop_thread = _start_loop()
        metadata = _build_metadata()
        logger.info(f"Prepare config for {self.backend_name} ...")
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id=f"bench_{self.backend_name}",
        )

        # _setup_filesystem prepare the environment, it should be executed
        # before _setup_config. _setup_config may use some environment variables
        self._setup_filesystem()
        config.extra_config = self.extra_config_keys
        config = self._setup_config(config)

        # Create local CPU backend (common to all backends)
        self._local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=AdHocMemoryAllocator(device="cpu"),
        )
        logger.info(f"Creating {self.backend_name} ...")
        # Create the specific backend
        self._backend = self._create_backend(
            config, metadata, self._loop, self._local_cpu
        )

        # Prepare test data
        self._keys = _make_keys(self.num_ops)
        self._objs = _make_memory_objs(
            self.num_ops, self.use_odirect, self.alignment, self._keepalive
        )

        # Run benchmark
        logger.info(f"Start benchmark with {self.backend_name} ...")
        self._start_time = time.perf_counter()
        result = self._execute_benchmark()
        logger.info(f"End benchmark with {self.backend_name} ...")
        # Cleanup
        _release_memory_objs(self._objs)
        self._close_backend()
        logger.info(f"Closed {self.backend_name} ...")

        self._cleanup_filesystem()
        _stop_loop(self._loop, self._loop_thread)
        return result

    def _execute_benchmark(self) -> dict:
        """Execute the benchmark with concurrent writes."""
        # Submit tasks
        pending_ops = self._submit_put_tasks(self._keys, self._objs)

        # Wait for completion
        start = time.perf_counter()
        self._wait_for_completion(pending_ops)
        elapsed = time.perf_counter() - start

        return {
            "backend": self.backend_name,
            "num_ops": self.num_ops,
            "concurrency": self.concurrency,
            "elapsed_sec": elapsed,
            "ops_per_sec": self.num_ops / elapsed if elapsed > 0 else 0.0,
            "use_odirect": self.use_odirect,
        }


# ============================================================================
# LocalDiskBackend Implementation
# ============================================================================
class LocalDiskBackendBenchmark(StorageBackendBenchmark):
    """Benchmark for LocalDiskBackend."""

    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        local_disk_dir: str,
        max_disk_gb: float,
        use_odirect: bool,
        alignment: int,
    ):
        super().__init__("local_disk", num_ops, concurrency, use_odirect, alignment)
        self.local_disk_dir = local_disk_dir
        self.max_disk_gb = max_disk_gb

    @property
    def extra_config_keys(self) -> dict:
        return {"use_odirect": self.use_odirect}

    def _setup_config(self, config: LMCacheEngineConfig) -> LMCacheEngineConfig:
        config.local_disk = self.local_disk_dir
        config.max_local_disk_size = self.max_disk_gb
        return config

    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> LocalDiskBackend:
        return LocalDiskBackend(
            config=config,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )

    def _close_backend(self) -> None:
        if hasattr(self._backend, "disk_worker"):
            self._backend.disk_worker.close()
        else:
            self._backend.close()


# ============================================================================
# RustRawBlockBackend Implementation
# ============================================================================
class RustRawBlockBackendBenchmark(StorageBackendBenchmark):
    """Benchmark for RustRawBlockBackend."""

    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        raw_device: str,
        raw_device_size_gb: float,
        use_odirect: bool,
        alignment: int,
        cleanup_raw_device: bool,
    ):
        super().__init__("rust_raw_block", num_ops, concurrency, use_odirect, alignment)
        self.raw_device = raw_device
        self.raw_device_size_gb = raw_device_size_gb
        self.cleanup_raw_device = cleanup_raw_device
        self._temp_dir: Optional[str] = None
        self._manifest_path: Optional[str] = None

    @property
    def extra_config_keys(self) -> dict:
        return {
            "rust_raw_block.device_path": self.raw_device,
            "rust_raw_block.block_align": self.alignment,
            "rust_raw_block.header_bytes": self.alignment,
            "rust_raw_block.use_odirect": self.use_odirect,
            "rust_raw_block.manifest_path": self._manifest_path,
            "rust_raw_block.manifest_write_interval": 0,
        }

    def _setup_filesystem(self) -> None:
        """Setup raw block device or temp file."""
        is_block_device = False
        self._temp_dir = None
        if self.raw_device:
            try:
                st_mode = os.stat(self.raw_device).st_mode
                is_block_device = stat.S_ISBLK(st_mode)
            except FileNotFoundError:
                is_block_device = False

        # Create temp file if no device specified
        if not self.raw_device:
            self._temp_dir = tempfile.mkdtemp(prefix="raw_block_bench_")
            self.raw_device = os.path.join(self._temp_dir, "raw_block.bin")

        # Truncate if not a real block device
        if self.raw_device and not is_block_device:
            with open(self.raw_device, "wb") as f:
                f.truncate(int(self.raw_device_size_gb * 1024**3))

        # Create manifest path
        self._manifest_path = os.path.join(
            tempfile.gettempdir(),
            f"lmcache_rust_raw_block_bench_{os.getpid()}_{time.time_ns()}.manifest.json",
        )

    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> RustRawBlockBackend:
        return RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
            dst_device="cpu",
        )

    def _submit_put_tasks(
        self,
        keys: list[CacheEngineKey],
        objs: list,
    ) -> list[Future]:
        """Submit put tasks and return futures."""
        futures: list[Future] = []
        fut_lock = threading.Lock()

        def submit_slice(start: int, end: int) -> None:
            futs = self._backend.batched_submit_put_task(
                keys[start:end], objs[start:end]
            )
            if futs:
                with fut_lock:
                    futures.extend(futs)

        slice_size = max(1, self.num_ops // self.concurrency)
        slices = [
            (i, min(i + slice_size, self.num_ops))
            for i in range(0, self.num_ops, slice_size)
        ]

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for s in slices:
                ex.submit(submit_slice, s[0], s[1])

        return futures

    def _wait_for_completion(self, futures: list[Future]) -> None:
        """Wait for all futures to complete."""
        for fut in futures:
            fut.result(timeout=120)

    def _close_backend(self) -> None:
        if self._backend:
            self._backend.close()

    def _cleanup_filesystem(self) -> None:
        """Cleanup temp files."""
        if self.cleanup_raw_device or self._temp_dir:
            try:
                os.remove(self.raw_device)
            except Exception:
                pass
            if self._temp_dir:
                try:
                    os.rmdir(self._temp_dir)
                except Exception:
                    pass
        if self._manifest_path:
            try:
                os.remove(self._manifest_path)
            except Exception:
                pass


# ============================================================================
# RemoteBackendBenchmark Implementation
# ============================================================================
class RemoteBackendBenchmark(StorageBackendBenchmark):
    """Benchmark for RemoteBackend (e.g., S3, Redis)."""

    def __init__(
        self,
        name: str,
        num_ops: int,
        concurrency: int,
        remote_url: str,
        use_odirect: bool,
        alignment: int,
    ):
        super().__init__(name, num_ops, concurrency, use_odirect, alignment)
        self.remote_url = remote_url

    def _setup_config(self, config: LMCacheEngineConfig) -> LMCacheEngineConfig:
        config.remote_url = self.remote_url
        return config

    def _create_backend(
        self,
        config,
        metadata,
        loop,
        local_cpu_backend,
    ):
        return RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )

    def _close_backend(self):
        if self._backend:
            self._backend.close()


# ============================================================================
# Hf3fsBackendBenchmark Implementation
# ============================================================================
class Hf3fsBackendBenchmark(RemoteBackendBenchmark):
    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        remote_url: str,
        use_odirect: bool,
        alignment: int,
    ):
        super().__init__(
            "hf3fs_backend", num_ops, concurrency, remote_url, use_odirect, alignment
        )

    @property
    def extra_config_keys(self) -> dict:
        return {
            "hf3fs_mount_point": "/3fs/stage",
            "hf3fs_iov_size": 209715200,
            "hf3fs_ior_entries": 256,
            "hf3fs_io_depth": 0,
            "hf3fs_numa_id": -1,
            "hf3fs_io_thread_num": 8,
        }


# ============================================================================
# FsBackendBenchmark Implementation
# ============================================================================
class FsBackendBenchmark(RemoteBackendBenchmark):
    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        remote_url: str,
        use_odirect: bool,
        alignment: int,
    ):
        super().__init__(
            "fs_backend", num_ops, concurrency, remote_url, use_odirect, alignment
        )

    @property
    def extra_config_keys(self) -> dict:
        return {
            "save_chunk_meta": False,
            "fs_connector_read_ahead_size": 0,
            "fs_connector_use_odirect": False,
            # "fs_connector_relative_tmp_dir": "tmp",
        }


# ============================================================================
# Main Entry Point
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark storage backends under high write concurrency."
    )
    parser.add_argument("--num-ops", type=int, default=256, help="Total put ops")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Number of submit threads"
    )
    parser.add_argument(
        "--backend",
        choices=["local_disk", "rust_raw_block", "hf3fs_backend", "fs_backend", "both"],
        default="both",
    )
    parser.add_argument(
        "--local-disk-dir",
        type=str,
        default="/tmp/lmcache_local_disk_bench",
    )

    parser.add_argument(
        "--remote-url",
        type=str,
        default="hf3fs:///3fs/stage/hello,/3fs/stage/world",
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

    # Run LocalDiskBackend benchmark
    if args.backend in ("local_disk", "both"):
        localdisk_bench = LocalDiskBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            local_disk_dir=args.local_disk_dir,
            max_disk_gb=args.max_local_disk_gb,
            use_odirect=args.local_disk_odirect,
            alignment=args.alignment,
        )
        result = localdisk_bench.run()
        result["local_disk_dir"] = args.local_disk_dir
        results.append(result)

    # Run RustRawBlockBackend benchmark
    if args.backend in ("rust_raw_block", "both"):
        raw_device = args.raw_device
        cleanup_raw_device = False
        if not raw_device:
            # Use same filesystem as local disk for fair comparison
            raw_device = os.path.join(args.local_disk_dir, "raw_block.bin")
            cleanup_raw_device = True

        rustraw_bench = RustRawBlockBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            raw_device=raw_device,
            raw_device_size_gb=args.raw_device_size_gb,
            use_odirect=args.raw_odirect,
            alignment=args.alignment,
            cleanup_raw_device=cleanup_raw_device,
        )
        result = rustraw_bench.run()
        result["raw_device"] = raw_device
        results.append(result)

    # Run Hf3fsBackend benchmark
    if args.backend in ("hf3fs_backend",):
        hf3fs_bench = Hf3fsBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            remote_url=args.remote_url,
            use_odirect=False,
            alignment=args.alignment,
        )
        result = hf3fs_bench.run()
        result["hf3fs_dir"] = args.remote_url
        results.append(result)

    # Run FsBackend benchmark
    if args.backend in ("fs_backend",):
        fs_bench = FsBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            remote_url=args.remote_url,
            use_odirect=False,
            alignment=args.alignment,
        )
        result = fs_bench.run()
        result["fs_dir"] = args.remote_url
        results.append(result)

    # Print results
    for result in results:
        print(
            f"{result['backend']}: ops={result['num_ops']} "
            f"concurrency={result['concurrency']} "
            f"elapsed={result['elapsed_sec']:.3f}s "
            f"ops/sec={result['ops_per_sec']:.2f}"
        )

    # Write JSON output
    if args.output_json:
        output_path = args.output_json
        if output_path.endswith(os.sep) or os.path.isdir(output_path):
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"storage_backend_io_{ts}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
