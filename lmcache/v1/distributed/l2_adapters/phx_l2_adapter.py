# SPDX-License-Identifier: Apache-2.0
"""Phoenix L2 adapter for LMCache MP mode.

Asymmetric PHX: POSIX store (CPU MemoryObj → disk) + phxfs_read DMA load
(disk → device MemoryObj).  Falls back to POSIX read for CPU MemoryObj.

Configuration (JSON)::

    {"type": "phx", "base_path": "/path/to/kv_cache", "device_ids": [4]}
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional
import os
import threading
import time

# First Party
from lmcache import torch_device_type
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l1_manager import get_current_l1_manager
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


# ── Instrumentation: perf log (hit rate + per-phase timing) ──
# Conditionally enabled via PhxL2AdapterConfig.perf_log_dir. When None (the
# default), all instrumentation is no-ops with near-zero overhead.
# When set to a directory path, writes structured per-task lines to
# perf_log_dir/phx_perf.log (file-only, no logger output).


_lmc_store_keys: set[str] = set()
_lmc_store_lock = threading.Lock()
_lmc_perf_log_file = None
_lmc_perf_log_lock = threading.Lock()


def _configure_perf_log(perf_log_dir: str | None) -> None:
    """Enable/disable perf logging at runtime (called from __init__).

    When *perf_log_dir* is a non-empty string, opens
    ``perf_log_dir/phx_perf.log`` for append writing (block-buffered, 64 KiB)
    to amortize small file IOs on the hot path.
    When None, perf logging is disabled and all instrumentation is no-op.
    """
    global _lmc_perf_log_file
    if perf_log_dir:
        os.makedirs(perf_log_dir, exist_ok=True)
        log_path = os.path.join(perf_log_dir, "phx_perf.log")
        _lmc_perf_log_file = open(log_path, "a", buffering=1 << 16)
        logger.info("PhxL2Adapter: perf log enabled (%s)", log_path)
    else:
        _lmc_perf_log_file = None


def _perf_log_enabled() -> bool:
    return _lmc_perf_log_file is not None


def _perf_write(line: str) -> None:
    """Write a line to the perf log file (thread-safe)."""
    f = _lmc_perf_log_file
    if f is None:
        return
    with _lmc_perf_log_lock:
        f.write(line + "\n")


def _key_str(key: "ObjectKey") -> str:
    """Compact key representation: chunk_hash:kv_rank:ogid"""
    return f"{key.chunk_hash.hex()[:16]}:{key.kv_rank:08x}:{key.object_group_id}"


def _record_store_keys(keys: list) -> None:
    """Record all keys submitted for store (before execution)."""
    if not _perf_log_enabled():
        return
    with _lmc_store_lock:
        for k in keys:
            _lmc_store_keys.add(_key_str(k))


class _StorePerf:
    """Timing + log helper for _process_store.

    Usage::

        perf = _StorePerf(enabled)
        # ... do store work, accumulate success_keys/sizes ...
        perf.finish(task_id, len(keys), len(success_keys), total_bytes, mode)
    """

    __slots__ = ("enabled", "t0")

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.t0 = time.perf_counter() if enabled else 0.0

    def finish(
        self, task_id: int, n_keys: int, n_ok: int, total_bytes: int, mode: str
    ) -> None:
        if not self.enabled:
            return
        dt = time.perf_counter() - self.t0
        bw = total_bytes / dt / 1024 / 1024 if dt > 0 and total_bytes > 0 else 0.0
        _perf_write(
            f"STORE task={task_id} keys={n_keys} ok={n_ok} "
            f"{total_bytes / 1024 / 1024:.1f}MB in {dt * 1000:.1f}ms "
            f"({bw:.0f}MB/s) mode={mode}"
        )


class _LoadPerf:
    """Timing + log helper for _process_load.

    Accumulates per-phase timing (path resolve, alloc, fd open,
    batch read, result processing, fallback) and emits a single
    summary log on :meth:`finish`.

    Usage::

        perf = _LoadPerf(enabled)
        t0 = perf.mark("path")
        # ... resolve paths ...
        perf.measure("path", t0)
        # ... batch read (use perf.measure for sub-phases) ...
        perf.finish(task_id, ...)
    """

    __slots__ = (
        "enabled",
        "t_start",
        "_timings",
        "_fd_hits",
        "_fd_misses",
        "_batch_bytes",
        "_batch_ok",
        "_n_batch_reqs",
        "_path_misses",
        "_n_bp_timeouts",
    )

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.t_start = time.perf_counter() if enabled else 0.0
        self._timings: dict[str, float] = {}
        self._fd_hits = 0
        self._fd_misses = 0
        self._batch_bytes = 0
        self._batch_ok = 0
        self._n_batch_reqs = 0
        self._path_misses = 0
        self._n_bp_timeouts = 0

    def mark(self) -> float:
        """Return a timestamp for later :meth:`measure`."""
        return time.perf_counter() if self.enabled else 0.0

    def measure(self, name: str, t0: float) -> None:
        """Accumulate elapsed time into *name*."""
        if not self.enabled:
            return
        self._timings[name] = self._timings.get(name, 0.0) + (time.perf_counter() - t0)

    def set_path_misses(self, n: int) -> None:
        self._path_misses = n

    def set_batch_stats(self, n_reqs: int, bytes_read: int, ok_count: int) -> None:
        self._n_batch_reqs = n_reqs
        self._batch_bytes = bytes_read
        self._batch_ok = ok_count

    def add_fd_hit(self) -> None:
        if self.enabled:
            self._fd_hits += 1

    def add_fd_miss(self) -> None:
        if self.enabled:
            self._fd_misses += 1

    def add_fd_stats(self, hits: int, misses: int) -> None:
        """Batch add fd hit/miss counts (for parallel open path)."""
        if self.enabled:
            self._fd_hits += hits
            self._fd_misses += misses

    def add_bp_timeout(self) -> None:
        """Record a backpressure timeout."""
        if self.enabled:
            self._n_bp_timeouts += 1

    def finish_early(self, task_id: int, n_keys: int) -> None:
        """Log for the early-return (no files found) path."""
        if not self.enabled:
            return
        dt = time.perf_counter() - self.t_start
        _perf_write(
            f"LOAD task={task_id} keys={n_keys} hit=0 miss={n_keys} "
            f"path_miss={self._path_misses} hit_rate=0.0% "
            f"total={dt * 1000:.1f}ms (no files found, early return)"
        )

    def finish(
        self,
        task_id: int,
        n_keys: int,
        odirect: str,
        n_fallback: int,
        fb_bytes: int,
        n_device_objs: int,
        n_hit: int,
        n_miss: int,
    ) -> None:
        """Emit the full per-phase breakdown log."""
        if not self.enabled:
            return
        t_total = time.perf_counter() - self.t_start
        t_path = self._timings.get("path", 0.0)
        t_alloc = self._timings.get("alloc", 0.0)
        t_bp = self._timings.get("backpressure", 0.0)
        t_fd = self._timings.get("fd", 0.0)
        t_read = self._timings.get("read_batch", 0.0)
        t_result = self._timings.get("result", 0.0)
        t_fb = self._timings.get("fallback", 0.0)
        t_prep = t_bp + t_alloc + t_fd
        read_bw = (
            self._batch_bytes / t_read / 1024 / 1024
            if t_read > 0 and self._batch_bytes > 0
            else 0.0
        )
        hit_rate = (n_hit / n_keys * 100) if n_keys > 0 else 0.0

        _perf_write(
            f"LOAD task={task_id} keys={n_keys} hit={n_hit} miss={n_miss} "
            f"hit_rate={hit_rate:.1f}% dev_objs={n_device_objs} "
            f"total={t_total * 1000:.1f}ms | "
            f"path={t_path * 1000:.1f}ms(miss={self._path_misses}) | "
            f"prep={t_prep * 1000:.1f}ms[bp={t_bp * 1000:.1f}ms"
            f"(to={self._n_bp_timeouts}) "
            f"alloc={t_alloc * 1000:.1f}ms "
            f"fd={t_fd * 1000:.1f}ms(hits={self._fd_hits} miss={self._fd_misses})] | "
            f"read={t_read * 1000:.1f}ms[{self._n_batch_reqs}reqs "
            f"{self._batch_bytes / 1024 / 1024:.1f}MB {read_bw:.0f}MB/s] | "
            f"result={t_result * 1000:.1f}ms(ok={self._batch_ok}) | "
            f"fallback={t_fb * 1000:.1f}ms({n_fallback}keys "
            f"{fb_bytes / 1024 / 1024:.1f}MB)"
        )


# ── helpers ──


def _object_key_to_filename(key: "ObjectKey") -> str:
    """Convert ObjectKey to a flat filename (same scheme as S3/HFBucket).

    The model_name may contain path separators (e.g. '/mnt/nvme4/.../model');
    replace them with '_' to produce a safe flat filename.
    """
    safe_model = key.model_name.replace("/", "_")
    base = (
        f"{safe_model}@{key.kv_rank:08x}@{key.object_group_id:x}@{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _key_to_path(base_path: str, key: "ObjectKey") -> str:
    """Map an ObjectKey to a two-level directory path under *base_path*."""
    fname = _object_key_to_filename(key)
    if len(fname) >= 4:
        l1, l2, rest = fname[:2], fname[2:4], fname[4:]
        return os.path.join(base_path, l1, l2, rest)
    return os.path.join(base_path, fname)


# ── config ──


@dataclass
class PhxL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for :class:`PhxL2Adapter`."""

    base_path: str
    device_ids: list[int] | None = None
    buffer_size_mb: int = 2048
    use_direct_io: bool = True
    max_capacity_bytes: int = 0
    perf_log_dir: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhxL2AdapterConfig":
        device_ids_raw = d.get("device_ids")
        device_ids = (
            [int(x) for x in device_ids_raw] if device_ids_raw is not None else None
        )
        return cls(
            base_path=d["base_path"],
            device_ids=device_ids,
            buffer_size_mb=int(d.get("buffer_size_mb", 2048)),
            use_direct_io=bool(d.get("use_direct_io", True)),
            max_capacity_bytes=int(d.get("max_capacity_bytes", 0)),
            perf_log_dir=d.get("perf_log_dir"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": get_type_name_for_config(self),
            "base_path": self.base_path,
            "buffer_size_mb": self.buffer_size_mb,
            "use_direct_io": self.use_direct_io,
            "max_capacity_bytes": self.max_capacity_bytes,
        }
        if self.device_ids is not None:
            result["device_ids"] = self.device_ids
        if self.perf_log_dir is not None:
            result["perf_log_dir"] = self.perf_log_dir
        return result

    def help_params(self) -> list[tuple[str, str, str]]:
        return [
            ("base_path", "str", "Root directory for KV cache files"),
            (
                "device_ids",
                "list[int]",
                "Device IDs for phxfs DMA (one buffer per device)",
            ),
            (
                "buffer_size_mb",
                "int",
                "Device buffer size in MiB per device (default: 2048)",
            ),
            ("use_direct_io", "bool", "Use O_DIRECT (default: true)"),
            ("max_capacity_bytes", "int", "Max bytes, 0=unlimited (default: 0)"),
            (
                "perf_log_dir",
                "str",
                "Enable perf log (hit rate + per-phase timing) "
                "to this dir (default: None/off)",
            ),
        ]

    @classmethod
    def help(cls) -> str:
        return (
            "Phoenix L2 adapter config fields:\n"
            "- base_path (str): root directory for KV cache "
            "files (required)\n"
            "- device_ids (list[int]): Device IDs for phxfs DMA — "
            "one device buffer per ID. Single-device: [4]; "
            "multi-device: [4, 5, 6, 7].\n"
            "- buffer_size_mb (int): Device buffer size in MiB "
            "per device (optional, default 2048)\n"
            "- use_direct_io (bool): use O_DIRECT for I/O "
            "(optional, default true)\n"
            "- max_capacity_bytes (int): max bytes to store, "
            "0=unlimited (optional, default 0)\n"
            "- perf_log_dir (str): when set, writes perf log "
            "(hit rate + per-phase timing for store/load/lookup) "
            "to perf_log_dir/phx_perf.log. When None (default), "
            "no perf logging occurs."
        )


# ── adapter ──


class PhxL2Adapter(L2AdapterInterface):
    """Phoenix KV cache L2 adapter (asymmetric PHX).

    Store:  CPU MemoryObj → POSIX write → Phoenix storage
    Load:   Phoenix storage → phxfs_read DMA → device MemoryObj (or POSIX fallback)
    Lookup: hot_cache + os.path.exists
    """

    def __init__(self, config: PhxL2AdapterConfig) -> None:
        super().__init__(max_capacity_bytes=config.max_capacity_bytes)

        self._config = config
        self._base_path = config.base_path
        os.makedirs(self._base_path, exist_ok=True)

        # ── Logging flags ──
        self._perf_breakdown = config.perf_log_dir is not None

        # ── Instrumentation (optional) ──
        _configure_perf_log(config.perf_log_dir)

        # ── Phoenix device DMA setup ──
        # One PhxCache + Allocator per configured device.
        # device_ids=[4] → single device; device_ids=[4,5,6,7] → multi-device.
        self._phx_caches: dict[int, Any] = {}
        self._phx_allocators: dict[int, Any] = {}
        self._phx_base_pointers: dict[int, int] = {}

        if config.device_ids:
            self._init_devices(config)
        else:
            logger.warning(
                "PhxL2Adapter: no device_ids configured, falling back to POSIX read"
            )

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # L1Manager reference for self-admission: when PHX DMA produces
        # device-resident objs, query_load_result replaces the pre-allocated
        # CPU objs in L1 with them (mark_temporary=True) so retrieve serves
        # via D2D and finish_read auto-recycles the DMA buffer.
        self._l1_manager = get_current_l1_manager()
        if self._l1_manager is None and self._phx_caches:
            raise RuntimeError(
                "PhxL2Adapter: L1Manager not yet constructed; "
                "adapter must be created after StorageManager's L1Manager"
            )

        # ── task queues ──
        self._store_queue: list[tuple[L2TaskId, list, list]] = []
        self._store_lock = threading.Lock()
        self._store_completed: dict[L2TaskId, L2StoreResult] = {}
        self._store_completed_lock = threading.Lock()
        self._next_store_id = 0

        self._lookup_queue: list[tuple[L2TaskId, list]] = []
        self._lookup_lock = threading.Lock()
        self._lookup_results: dict[L2TaskId, Bitmap] = {}
        self._lookup_results_lock = threading.Lock()
        self._next_lookup_id = 0

        self._load_queue: list[tuple[L2TaskId, list, list]] = []
        self._load_lock = threading.Lock()
        self._load_results: dict[L2TaskId, Bitmap] = {}
        self._load_results_lock = threading.Lock()
        self._next_load_id = 0
        # Device-resident MemoryObjs produced by PHX DMA load (task_id ->
        # {key: obj}). Populated by _process_load when is_phx_available();
        # consumed by query_load_result() which self-admits them into L1
        # via replace_memory_obj(mark_temporary=True) so retrieve serves
        # via D2D and finish_read auto-recycles the DMA buffer.
        self._load_device_objs: dict[L2TaskId, dict[ObjectKey, MemoryObj]] = {}

        # ── hot cache ──
        self._hot_cache: dict[str, str] = {}
        self._hot_lock = threading.Lock()

        # ── fd cache (read-only, LRU) ──
        # Avoids repeated os.open/os.close for the same cache files across
        # batch read requests.  Only used by _process_load (read path);
        # store uses .tmp files which are temporary and not worth caching.
        self._fd_cache: OrderedDict[str, int] = OrderedDict()
        self._fd_cache_lock = threading.Lock()
        # One fd per cache file (one chunk = 256 tokens).  Sized to cover
        # the full working set observed in production traces (~256K unique
        # paths).  Each fd uses ~4 KB of kernel memory (~1 GB total) which
        # is negligible on servers with 1M fd limit (ulimit -n).
        self._fd_cache_max = 262144

        # ── background worker ──
        self._stop_flag = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="phx-l2-worker"
        )
        self._worker_thread.start()
        logger.info("PhxL2Adapter started (base_path=%s)", self._base_path)

    def _init_devices(self, config: PhxL2AdapterConfig) -> None:
        """Initialize one PhxCache + Allocator per configured device."""
        assert config.device_ids is not None
        try:
            # Third Party
            import phxcache  # type: ignore[import-untyped]

            # First Party
            from lmcache.v1.memory_allocators.phx_device_memory_allocator import (
                PhxDeviceMemoryAllocator,
            )

            buffer_bytes = config.buffer_size_mb * 1024 * 1024
            for dev_id in config.device_ids:
                cache = phxcache.PhxCache(device_id=dev_id)
                allocator = PhxDeviceMemoryAllocator(
                    buffer_bytes,
                    device=f"{torch_device_type}:{dev_id}",
                    phx_cache=cache,
                )
                self._phx_caches[dev_id] = cache
                self._phx_allocators[dev_id] = allocator
                self._phx_base_pointers[dev_id] = allocator.base_pointer
                logger.info(
                    "PhxL2Adapter: device %d buffer ready (size=%d MiB, base=0x%x)",
                    dev_id,
                    config.buffer_size_mb,
                    allocator.base_pointer,
                )
            logger.info(
                "PhxL2Adapter: initialized (%d devices, %d MiB each)",
                len(self._phx_caches),
                config.buffer_size_mb,
            )
        except ImportError:
            logger.warning(
                "PhxL2Adapter: phxcache not available, falling back to POSIX read"
            )
            self._phx_caches.clear()
            self._phx_allocators.clear()
            self._phx_base_pointers.clear()
        except Exception as e:
            logger.warning(
                "PhxL2Adapter: device init failed (%s), falling back to POSIX read",
                e,
            )
            self._phx_caches.clear()
            self._phx_allocators.clear()
            self._phx_base_pointers.clear()

    # ── event fd interface ──

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ── store ──

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._store_lock:
            task_id = self._next_store_id
            self._next_store_id += 1
            self._store_queue.append((task_id, keys, objects))
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._store_completed_lock:
            result = dict(self._store_completed)
            self._store_completed.clear()
            return result

    # ── lookup ──

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> L2TaskId:
        with self._lookup_lock:
            task_id = self._next_lookup_id
            self._next_lookup_id += 1
            self._lookup_queue.append((task_id, keys))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lookup_results_lock:
            return self._lookup_results.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        pass  # no-op: file-based, no explicit lock

    # ── load ──

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._load_lock:
            task_id = self._next_load_id
            self._next_load_id += 1
            self._load_queue.append((task_id, keys, objects))
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        """Pop the load bitmap for *task_id* and self-admit device objs.

        When PHX DMA was used, device-resident MemoryObjs were allocated
        from the phx pool during ``_process_load``.  Here we replace the
        pre-allocated CPU placeholder objs in L1 with these device objs,
        marking them ``is_temporary=True`` so that ``finish_read``
        automatically deletes the L1 entry and frees the device obj back
        to the phx pool (via ``PhxL1MemoryManager.free`` dispatch) when
        read locks reach zero — no manual ``release_device_objs`` needed.

        Thread-safety: ``_load_results`` and ``_load_device_objs`` are
        written under ``_load_results_lock`` in ``_process_load`` and
        atomically popped here under the same lock, so the bitmap and
        device objs are always consistent.

        ``device_objs`` only contains keys whose DMA succeeded
        (``_load_batch`` sets ``device_objs[key]`` and ``bitmap.set(idx)``
        together; failed DMA objs are freed immediately and never enter
        ``device_objs``).  POSIX-fallback keys fill the CPU obj directly
        and are not in ``device_objs``.
        """
        with self._load_results_lock:
            bitmap = self._load_results.pop(task_id, None)
            if bitmap is None:
                return None
            device_objs = self._load_device_objs.pop(task_id, {})

        # Self-admission: swap CPU placeholders for device-resident objs.
        if device_objs and self._l1_manager is not None:
            for key, obj in device_objs.items():
                err = self._l1_manager.replace_memory_obj(key, obj, mark_temporary=True)
                if err != L1Error.SUCCESS:
                    # Entry was deleted (e.g. request aborted between
                    # reserve_write and query_load_result).  Free the
                    # device obj directly to avoid leaking pool memory.
                    parent = obj.parent()
                    if parent is not None:
                        parent.free(obj)

        return bitmap

    # ── background worker ──

    def _worker_loop(self) -> None:
        while not self._stop_flag.is_set():
            busy = False

            with self._store_lock:
                store_tasks: list[tuple[L2TaskId, list[ObjectKey], list[MemoryObj]]] = (
                    list(self._store_queue)
                )
                self._store_queue.clear()
            for store_task in store_tasks:
                self._process_store(store_task)
                busy = True

            with self._lookup_lock:
                lookup_tasks: list[tuple[L2TaskId, list[ObjectKey]]] = list(
                    self._lookup_queue
                )
                self._lookup_queue.clear()
            for lookup_task in lookup_tasks:
                self._process_lookup(lookup_task)
                busy = True

            with self._load_lock:
                load_tasks: list[tuple[L2TaskId, list[ObjectKey], list[MemoryObj]]] = (
                    list(self._load_queue)
                )
                self._load_queue.clear()
            for load_task in load_tasks:
                self._process_load(load_task)
                busy = True

            if not busy:
                self._stop_flag.wait(timeout=0.01)

    def _process_store(
        self, task: tuple[L2TaskId, list[ObjectKey], list[MemoryObj]]
    ) -> None:
        """Store: CPU MemoryObj → batch write (phxfs_write_batch) or POSIX."""
        perf = _StorePerf(self._perf_breakdown)
        task_id, keys, objects = task
        # Third Party
        import torch

        success_keys: list[ObjectKey] = []
        success_sizes: list[int] = []

        # ── Instrumentation: record store keys for lookup cross-check ──
        _record_store_keys(keys)

        # ── Batch write path (phxfs_write_batch with CPU buffers) ──
        # Store uses CPU MemoryObj → phxfs_write_batch (NVMe write, no device
        # buffer needed). Use the first available PhxCache for write_batch
        # (it operates on NVMe, not device memory).
        write_cache = next(iter(self._phx_caches.values()), None)

        if write_cache is not None:
            batch_reqs: list[tuple[int, int, int, int, int]] = []
            batch_keys: list[ObjectKey] = []
            batch_tmps: list[str] = []
            batch_finals: list[str] = []
            batch_sizes: list[int] = []
            batch_tensors: list[Any] = []  # keep refs alive during write
            batch_fds: list[int] = []

            for key, obj in zip(keys, objects, strict=False):
                try:
                    path = _key_to_path(self._base_path, key)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp_path = path + ".tmp"

                    tensor = obj.raw_tensor
                    if tensor is None:
                        continue
                    buf = tensor.contiguous()
                    size = buf.nbytes

                    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                    buf_ptr = buf.data_ptr()

                    batch_reqs.append((fd, buf_ptr, 0, size, 0))
                    batch_keys.append(key)
                    batch_tmps.append(tmp_path)
                    batch_finals.append(path)
                    batch_sizes.append(size)
                    batch_tensors.append(buf)
                    batch_fds.append(fd)
                except Exception as e:
                    logger.error("PhxL2Adapter store prep failed for %s: %s", key, e)

            if batch_reqs:
                try:
                    results = write_cache.write_batch(batch_reqs)
                except Exception as e:
                    logger.error("PhxL2Adapter batch write failed: %s", e)
                    results = [-1] * len(batch_reqs)
                finally:
                    for fd in batch_fds:
                        os.close(fd)

                for i, (key, result, tmp_path, final_path, size) in enumerate(
                    zip(
                        batch_keys,
                        results,
                        batch_tmps,
                        batch_finals,
                        batch_sizes,
                        strict=False,
                    )
                ):
                    if result == size:
                        os.rename(tmp_path, final_path)
                        with self._hot_lock:
                            self._hot_cache[_object_key_to_filename(key)] = final_path
                        success_keys.append(key)
                        success_sizes.append(size)
                    else:
                        logger.error(
                            "PhxL2Adapter batch write failed for %s: result=%d",
                            key,
                            result,
                        )
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
        else:
            # ── Fallback: POSIX per-key write ──
            for key, obj in zip(keys, objects, strict=False):
                try:
                    path = _key_to_path(self._base_path, key)
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                    tensor = obj.raw_tensor
                    if tensor is None:
                        continue
                    size = tensor.nbytes

                    tmp_path = path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        if tensor.dtype == torch.bfloat16:
                            f.write(tensor.contiguous().view(torch.uint8).numpy())  # type: ignore[arg-type]
                        else:
                            f.write(tensor.contiguous().numpy())  # type: ignore[arg-type]
                    os.rename(tmp_path, path)

                    with self._hot_lock:
                        self._hot_cache[_object_key_to_filename(key)] = path

                    success_keys.append(key)
                    success_sizes.append(size)
                except Exception as e:
                    logger.error("PhxL2Adapter store failed for %s: %s", key, e)

        self._notify_keys_stored(success_keys, success_sizes)
        total_bytes = sum(success_sizes)
        with self._store_completed_lock:
            self._store_completed[task_id] = L2StoreResult(
                success=len(success_keys) == len(keys), bytes_transferred=total_bytes
            )
        self._store_efd.notify()
        perf.finish(
            task_id,
            len(keys),
            len(success_keys),
            total_bytes,
            "batch" if self.is_phx_available() else "posix",
        )

    def _process_lookup(self, task: tuple[L2TaskId, list[ObjectKey]]) -> None:
        """Lookup: check hot_cache + file existence."""
        task_id, keys = task
        bitmap = Bitmap(len(keys))
        hit_keys: list[ObjectKey] = []
        miss_keys: list[ObjectKey] = []
        with self._hot_lock:
            for i, key in enumerate(keys):
                fname = _object_key_to_filename(key)
                if fname in self._hot_cache:
                    bitmap.set(i)
                    hit_keys.append(key)
                    continue
                path = _key_to_path(self._base_path, key)
                if os.path.exists(path):
                    self._hot_cache[fname] = path
                    bitmap.set(i)
                    hit_keys.append(key)
                else:
                    miss_keys.append(key)
        with self._lookup_results_lock:
            self._lookup_results[task_id] = bitmap
        self._lookup_efd.notify()
        # ── Perf log: LOOKUP line with hit/miss + stored_but_miss ──
        if _perf_log_enabled():
            stored_but_miss = 0
            with _lmc_store_lock:
                for k in miss_keys:
                    if _key_str(k) in _lmc_store_keys:
                        stored_but_miss += 1
            hit_rate = (len(hit_keys) / len(keys) * 100) if keys else 0.0
            _perf_write(
                f"LOOKUP task={task_id} keys={len(keys)} "
                f"hit={len(hit_keys)} miss={len(miss_keys)} "
                f"stored_but_miss={stored_but_miss} "
                f"hit_rate={hit_rate:.1f}% "
                f"store_set_size={len(_lmc_store_keys)}"
            )

    def _process_load(
        self, task: tuple[L2TaskId, list[ObjectKey], list[MemoryObj]]
    ) -> None:
        """Load: batch PHX DMA (preferred) or per-key POSIX read.

        When PHX is available, allocate device MemoryObjs for all keys,
        open all files, and submit a single phxfs_read_batch for concurrent
        I/O.  Keys that can't be batched (file missing, pool exhausted, or
        batch read failure) fall back to per-key POSIX/DMA read.

        Instrumented with per-phase timing for performance breakdown.
        """
        perf = _LoadPerf(self._perf_breakdown)
        task_id, keys, objects = task
        bitmap = Bitmap(len(keys))
        device_objs: dict[ObjectKey, MemoryObj] = {}

        # ── Phase 1: Resolve file paths (hot_cache + os.path.exists) ──
        t_path_start = perf.mark()
        key_paths: dict[int, str] = {}  # index -> path
        path_misses = 0  # keys not found on disk
        for i, (key, obj) in enumerate(zip(keys, objects, strict=False)):
            with self._hot_lock:
                fname = _object_key_to_filename(key)
                path = self._hot_cache.get(fname)
            if path is None:
                path = _key_to_path(self._base_path, key)
                if not os.path.exists(path):
                    path_misses += 1
                    continue
            key_paths[i] = path
        perf.measure("path", t_path_start)
        perf.set_path_misses(path_misses)

        if not key_paths:
            with self._load_results_lock:
                self._load_results[task_id] = bitmap
            self._load_efd.notify()
            perf.finish_early(task_id, len(keys))
            return

        batch_bytes = 0  # total bytes in batch read
        batch_ok_count = 0  # keys successfully read in batch
        n_batch_reqs = 0  # number of batch read requests

        # ── Phase 2: Batch read path (phxfs_read_batch) ──
        if self.is_phx_available():
            batch_ok_count, batch_bytes, n_batch_reqs = self._load_batch(
                keys,
                objects,
                key_paths,
                device_objs,
                bitmap,
                perf,
            )

        # ── Phase 3: Fallback per-key POSIX/DMA read for remaining keys ──
        t_fb_start = perf.mark()
        fb_bytes = 0
        for i, path in key_paths.items():
            try:
                obj = objects[i]
                cpu_tensor = obj.raw_tensor
                if cpu_tensor is None:
                    continue
                size = cpu_tensor.nbytes

                if cpu_tensor.device.type != "cpu" and self.is_phx_available():
                    self._load_dma(path, cpu_tensor, size)
                else:
                    self._load_posix(path, cpu_tensor, size)

                bitmap.set(i)
                self._notify_keys_accessed([keys[i]])
                fb_bytes += size
            except Exception as e:
                logger.error("PhxL2Adapter load failed for %s: %s", keys[i], e)
        perf.measure("fallback", t_fb_start)

        with self._load_results_lock:
            self._load_results[task_id] = bitmap
            if device_objs:
                self._load_device_objs[task_id] = device_objs
        self._load_efd.notify()

        # ── Timing summary ──
        perf.set_batch_stats(n_batch_reqs, batch_bytes, batch_ok_count)
        odirect = "O_DIRECT" if self._config.use_direct_io else "buffered"
        n_hit = bitmap.popcount()
        n_miss = len(keys) - n_hit
        perf.finish(
            task_id,
            len(keys),
            odirect,
            len(key_paths),
            fb_bytes,
            len(device_objs),
            n_hit,
            n_miss,
        )

    def _get_read_fd(self, path: str, flags: int) -> int:
        """Get a read fd from cache, or open a new one and cache it.

        Uses an LRU cache so that frequently-read cache files keep their fds
        open across batch read requests, avoiding repeated os.open/os.close.
        """
        with self._fd_cache_lock:
            if path in self._fd_cache:
                self._fd_cache.move_to_end(path)
                return self._fd_cache[path]
        fd = os.open(path, flags)
        with self._fd_cache_lock:
            if path in self._fd_cache:
                # Another thread opened the same path; close the duplicate
                os.close(fd)
                self._fd_cache.move_to_end(path)
                return self._fd_cache[path]
            if len(self._fd_cache) >= self._fd_cache_max:
                _, old_fd = self._fd_cache.popitem(last=False)
                os.close(old_fd)
            self._fd_cache[path] = fd
        return fd

    def _invalidate_read_fd(self, path: str) -> None:
        """Close and remove a cached fd (e.g. after a read failure)."""
        with self._fd_cache_lock:
            fd = self._fd_cache.pop(path, None)
        if fd is not None:
            os.close(fd)

    def _batch_open_fds(self, paths: list[str], flags: int) -> dict[str, int]:
        """Open fds for multiple paths in parallel.

        Returns a dict mapping path -> fd (or -1 if open failed).
        Uses a thread pool to parallelize os.open syscalls, which are
        I/O-bound (especially with O_DIRECT) and release the GIL.
        """
        if not paths:
            return {}
        results: dict[str, int] = {}

        def _open_one(p: str) -> tuple[str, int]:
            try:
                return p, os.open(p, flags)
            except OSError:
                return p, -1

        # Small batches: serial is faster (no thread pool overhead)
        if len(paths) <= 4:
            for p in paths:
                try:
                    results[p] = os.open(p, flags)
                except OSError:
                    results[p] = -1
            return results

        n_workers = min(32, len(paths))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for p, fd in pool.map(_open_one, paths):
                results[p] = fd
        return results

    def _batch_get_read_fds(
        self, paths: list[str], flags: int, perf: _LoadPerf
    ) -> dict[str, int]:
        """Batch get fds: parallel-open cache misses, reuse cache hits.

        1. Check which paths are already in the fd cache (hits).
        2. Parallel os.open all unique misses.
        3. Insert opened fds into the cache.
        4. Return path -> fd mapping for all requested paths.

        Records hit/miss counts into *perf*.
        """
        # Phase 1: classify hits vs misses
        hit_paths: list[str] = []
        miss_paths: list[str] = []
        miss_set: set[str] = set()
        for p in paths:
            with self._fd_cache_lock:
                cached = p in self._fd_cache
            if cached:
                hit_paths.append(p)
            elif p not in miss_set:
                miss_set.add(p)
                miss_paths.append(p)

        perf.add_fd_stats(len(hit_paths), len(miss_paths))

        # Phase 2: parallel open misses
        if miss_paths:
            opened = self._batch_open_fds(miss_paths, flags)
            with self._fd_cache_lock:
                for p, fd in opened.items():
                    if fd < 0:
                        continue
                    if p in self._fd_cache:
                        # Another thread cached it; close duplicate
                        os.close(fd)
                        continue
                    if len(self._fd_cache) >= self._fd_cache_max:
                        _, old_fd = self._fd_cache.popitem(last=False)
                        os.close(old_fd)
                    self._fd_cache[p] = fd

        # Phase 3: collect fds (all should be cache hits now)
        result: dict[str, int] = {}
        for p in paths:
            try:
                fd = self._get_read_fd(p, flags)  # moves to end (LRU update)
                result[p] = fd
            except OSError:
                result[p] = -1
        return result

    def _load_dma(self, path: str, tensor: Any, size: int) -> None:
        """Load via phxfs_read DMA to device buffer.

        Looks up the PhxCache and base pointer for the tensor's device.
        Falls back to POSIX read if no PhxCache is configured for that device.
        """
        # Third Party
        import phxcache

        dev_idx = tensor.device.index
        if dev_idx is None:
            self._load_posix(path, tensor, size)
            return
        cache = self._phx_caches.get(dev_idx)
        base_ptr = self._phx_base_pointers.get(dev_idx)
        if cache is None or base_ptr is None:
            self._load_posix(path, tensor, size)
            return

        dev_ptr = tensor.data_ptr()
        buf_offset = dev_ptr - base_ptr

        flags = os.O_RDONLY
        if self._config.use_direct_io:
            flags |= os.O_DIRECT

        with phxcache.PhxFile(cache, path, flags) as f:
            ret = f.read(
                buf=base_ptr,
                buf_offset=buf_offset,
                nbyte=size,
                f_offset=0,
            )
        if ret != size:
            raise IOError(f"phxfs_read short read: {ret}/{size} bytes from {path}")

    def _load_batch(
        self,
        keys: list,
        objects: list,
        key_paths: dict[int, str],
        device_objs: dict,
        bitmap,
        perf: _LoadPerf,
    ) -> tuple:
        """Batch read: group keys by device, batch per device.

        When only one device is configured, all keys are assigned to that
        device regardless of kv_rank (preserving the old single-device
        behavior where one GPU serves all TP ranks).

        Returns:
            (batch_ok_count, batch_bytes, n_reqs)
        """
        batch_ok_count = 0
        batch_bytes = 0
        n_reqs = 0

        flags = os.O_RDONLY
        if self._config.use_direct_io:
            flags |= os.O_DIRECT

        # Group keys by device
        device_groups: dict[int, list[tuple[int, str]]] = {}
        if len(self._phx_caches) == 1:
            # Single device: all keys go to the one available device
            sole_dev_id = next(iter(self._phx_caches))
            device_groups[sole_dev_id] = list(key_paths.items())
        else:
            # Multi-device: route by kv_rank
            for i, path in key_paths.items():
                dev_id = self._kv_rank_to_device(keys[i].kv_rank)
                device_groups.setdefault(dev_id, []).append((i, path))

        for dev_id, group in device_groups.items():
            allocator = self._phx_allocators.get(dev_id)
            base_ptr = self._phx_base_pointers.get(dev_id)
            cache = self._phx_caches.get(dev_id)
            if allocator is None or cache is None or base_ptr is None:
                # Device not initialized, skip (will go to fallback)
                continue

            dev_batch_indices: list[int] = []
            dev_batch_reqs: list[tuple[int, int, int, int]] = []
            dev_batch_dev_objs: list[MemoryObj] = []
            dev_batch_paths: list[str] = []

            # ── Phase 1: Backpressure + Alloc (serial) ──
            t_alloc_start = perf.mark()
            alloc_results: list[tuple[int, str, int, MemoryObj]] = []
            for i, path in group:
                obj = objects[i]
                cpu_tensor = obj.raw_tensor
                if cpu_tensor is None:
                    continue
                size = cpu_tensor.nbytes

                # Backpressure: wait for pool space before allocating.
                # This avoids fallback to slow POSIX reads when the DMA
                # buffer is temporarily full (device objs pending release
                # by the retrieve path).
                if hasattr(allocator, "wait_for_available"):
                    t_bp = perf.mark()
                    got = allocator.wait_for_available(size, timeout=1.0)
                    perf.measure("backpressure", t_bp)
                    if not got:
                        perf.add_bp_timeout()
                        logger.warning(
                            "PhxL2: backpressure timeout on dev %d "
                            "(need %d bytes, free %d, key %s)",
                            dev_id,
                            size,
                            allocator.get_free_bytes()
                            if hasattr(allocator, "get_free_bytes")
                            else -1,
                            keys[i],
                        )
                        continue  # timeout – fall back to POSIX

                device_obj = allocator.allocate(
                    shapes=obj.metadata.shape,
                    dtypes=obj.metadata.dtype,
                    fmt=obj.metadata.fmt,
                )
                if device_obj is None:
                    logger.warning(
                        "PhxL2: pool exhausted on dev %d (key %s)", dev_id, keys[i]
                    )
                    continue  # pool exhausted, skip for fallback

                alloc_results.append((i, path, size, device_obj))
            perf.measure("alloc", t_alloc_start)

            # ── Phase 2: Batch open fds (parallel for cache misses) ──
            t_fd_start = perf.mark()
            if alloc_results:
                unique_paths = []
                seen_paths: set[str] = set()
                for _, path, _, _ in alloc_results:
                    if path not in seen_paths:
                        seen_paths.add(path)
                        unique_paths.append(path)
                fd_map = self._batch_get_read_fds(unique_paths, flags, perf)
            else:
                fd_map = {}
            perf.measure("fd", t_fd_start)

            # ── Phase 3: Assemble batch requests ──
            for i, path, size, device_obj in alloc_results:
                fd = fd_map.get(path, -1)
                if fd < 0:
                    allocator.free(device_obj)
                    logger.error("PhxL2Adapter open failed for %s", keys[i])
                    continue

                dev_tensor = device_obj.raw_tensor
                if dev_tensor is None:
                    allocator.free(device_obj)
                    logger.error(
                        "PhxL2Adapter device obj has no tensor for %s", keys[i]
                    )
                    continue
                buf_offset = dev_tensor.data_ptr() - base_ptr
                dev_batch_indices.append(i)
                dev_batch_reqs.append((fd, buf_offset, size, 0))
                dev_batch_dev_objs.append(device_obj)
                dev_batch_paths.append(path)
                batch_bytes += size

            # ── Phase 4: Batch read on this device ──
            if dev_batch_reqs:
                t_read_start = perf.mark()
                try:
                    results = cache.read_batch(base_ptr, dev_batch_reqs)
                except Exception as e:
                    logger.error(
                        "PhxL2Adapter batch read failed on dev %d: %s", dev_id, e
                    )
                    results = [-1] * len(dev_batch_reqs)
                perf.measure("read_batch", t_read_start)

                # Process results
                t_result_start = perf.mark()
                for j, (idx, dev_obj, result, path) in enumerate(
                    zip(
                        dev_batch_indices,
                        dev_batch_dev_objs,
                        results,
                        dev_batch_paths,
                        strict=False,
                    )
                ):
                    key = keys[idx]
                    expected = objects[idx].raw_tensor.nbytes
                    if result == expected:
                        device_objs[key] = dev_obj
                        bitmap.set(idx)
                        self._notify_keys_accessed([key])
                        batch_ok_count += 1
                        key_paths.pop(idx, None)
                    else:
                        allocator.free(dev_obj)
                        self._invalidate_read_fd(path)
                        logger.error(
                            "PhxL2Adapter batch read failed for %s: "
                            "result=%d/%d (dev %d)",
                            key,
                            result,
                            expected,
                            dev_id,
                        )
                perf.measure("result", t_result_start)
                n_reqs += len(dev_batch_reqs)

        return (batch_ok_count, batch_bytes, n_reqs)

    def _load_posix(self, path: str, tensor: Any, size: int) -> None:
        """Fallback: POSIX read into tensor."""
        # Third Party
        import torch

        buf = tensor.contiguous()
        if buf.dtype == torch.bfloat16:
            buf = buf.view(torch.uint8)

        with open(path, "rb") as f:
            f.readinto(buf.numpy())

    def is_phx_available(self) -> bool:
        """Whether PHX DMA is available (at least one device initialized)."""
        return len(self._phx_caches) > 0 and len(self._phx_allocators) > 0

    def _kv_rank_to_device(self, kv_rank: int) -> int:
        """Map kv_rank to CUDA device id.

        ComputeKVRank packs (world_size, global_rank, local_world_size,
        local_rank) into 8-bit fields:
            kv_rank = (world_size << 24) | (global_rank << 16)
                      | (local_world_size << 8) | local_rank
        global_rank is the TP worker index (0..7), which maps 1:1 to
        CUDA_VISIBLE_DEVICES index.
        """
        return (kv_rank >> 16) & 0xFF

    # ── lifecycle ──

    def close(self) -> None:
        self._stop_flag.set()
        self._worker_thread.join(timeout=5)

        # Close all cached read fds
        with self._fd_cache_lock:
            for fd in self._fd_cache.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fd_cache.clear()

        # Release any device objs never popped by the controller (e.g. aborted
        # prefetch requests) to avoid leaking phx pool memory.
        with self._load_results_lock:
            leaked_device_maps = list(self._load_device_objs.values())
            self._load_device_objs.clear()

        # Free leaked device objs from whichever device they were on
        for device_map in leaked_device_maps:
            for dev_obj in device_map.values():
                for allocator in self._phx_allocators.values():
                    try:
                        allocator.free(dev_obj)
                        break
                    except Exception:
                        continue
        for dev_id, allocator in self._phx_allocators.items():
            del allocator
        self._phx_allocators.clear()
        for dev_id, cache in self._phx_caches.items():
            try:
                cache.close()
            except Exception:
                pass
        self._phx_caches.clear()
        self._phx_base_pointers.clear()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("PhxL2Adapter closed")

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "type": "phx",
            "base_path": self._base_path,
            "dma_enabled": self.is_phx_available(),
            "hot_cache_size": len(self._hot_cache),
        }


# Self-register config type and adapter factory
register_l2_adapter_type("phx", PhxL2AdapterConfig)


def _create_phx_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional["L1MemoryDesc"] = None,
) -> "L2AdapterInterface":
    """Create a PhxL2Adapter from config."""
    return PhxL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("phx", _create_phx_adapter)
