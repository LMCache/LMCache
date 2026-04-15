# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, cast
import asyncio
import ctypes
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops
else:
    # First Party
    import lmcache.non_cuda_equivalents as lmc_ops

logger = init_logger(__name__)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


@dataclass
class _Entry:
    """In-memory index entry for a stored chunk."""

    offset: int
    meta: DiskCacheMetadata
    slot_id: int
    generation: int


@dataclass
class _Inflight:
    """In-progress put operation tracking."""

    offset: int
    meta: DiskCacheMetadata
    slot_id: int
    generation: int
    canceled: bool = False


@dataclass
class _SlotState:
    """Slot state for a stored DAX chunk."""

    generation: int
    committed: bool = False
    borrow_count: int = 0
    pending_free: bool = False


@dataclass
class _RestoreItem:
    """Reserved DAX read metadata for one item in a batched restore."""

    result_index: int
    key: CacheEngineKey
    offset: int
    size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    cached_positions: Optional[torch.Tensor]
    slot_id: int
    generation: int
    memory_obj: Optional[MemoryObj] = None
    slab_offset: int = 0


@dataclass
class _RestoreSpan:
    """One contiguous source span copied from DAX into the staging slab."""

    src_offset: int
    slab_offset: int
    size: int


@dataclass
class _RestoreRegion:
    """One restore region executed by a persistent worker."""

    region_index: int
    slab_offset: int
    total_bytes: int
    items: list[_RestoreItem]
    spans: list[_RestoreSpan]


@dataclass
class _RestoreWave:
    """One wave of region work against the fixed-size retrieve slab."""

    regions: list[_RestoreRegion]


class DaxBackend(StoragePluginInterface):
    """Storage plugin backend for /dev/dax mmap-backed KV cache."""

    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
        local_cpu_backend: Optional[LocalCPUBackend] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cpu",
    ) -> None:
        """Initialize a DAX-backed storage backend."""
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        if self.config is None:
            raise ValueError("DaxBackend requires config")
        if self.metadata is None:
            raise ValueError("DaxBackend requires metadata")

        if self.metadata.world_size != 1:
            raise ValueError(
                "DaxBackend currently only supports TP=1 "
                f"(world_size={self.metadata.world_size})"
            )
        if self.metadata.get_num_groups() != 1:
            raise ValueError(
                "DaxBackend currently supports only single-group KV layout"
            )

        extra = self.config.extra_config or {}
        self.device_path = str(extra.get("dax.device_path", "")).strip()
        if not self.device_path:
            raise ValueError("extra_config['dax.device_path'] is required")

        self.async_put = _to_bool(extra.get("dax.async_put", False))
        if self.async_put and self.loop is None:
            raise ValueError("DaxBackend async_put=true requires an asyncio event loop")

        self.max_dax_size = float(extra.get("dax.max_dax_size", 0))
        if self.max_dax_size <= 0:
            raise ValueError("extra_config['dax.max_dax_size'] must be > 0")

        if self.local_cpu_backend is None:
            raise ValueError("DaxBackend requires local_cpu_backend")

        # Total size in bytes of the mapped DAX arena.
        self._arena_bytes = int(self.max_dax_size * 1024**3)
        if self._arena_bytes <= 0:
            raise ValueError("dax.max_dax_size results in zero-sized arena")

        self._fd: Optional[int] = None
        self._mmap_obj: Optional[mmap.mmap] = None
        self._base_ptr: int = 0
        # Python memoryview exposing the mapped arena for byte-level access.
        self._arena_view: Optional[memoryview] = None
        self._restore_executor: Optional[ThreadPoolExecutor] = None
        self._restore_dispatch_executor: Optional[ThreadPoolExecutor] = None
        self._retrieve_staging_slab_ptr: int = 0
        self._retrieve_staging_slab_bytes: int = 0
        self._restore_region_bytes: int = 0
        self._restore_workers: int = 0
        self._restore_max_regions: int = 0
        self._open_arena()
        try:
            assert self.local_cpu_backend is not None
            full_chunk_size = int(self.local_cpu_backend.get_full_chunk_size_bytes())
            self.slot_bytes = max(1, int(full_chunk_size))
            self._max_slots = self._arena_bytes // self.slot_bytes
            if self._max_slots <= 0:
                raise RuntimeError(
                    "dax.max_dax_size is too small for the configured chunk size"
                )

            default_restore_workers = min(8, max(1, os.cpu_count() or 1))
            self._restore_workers = self._get_positive_int_extra(
                extra,
                "dax.restore_workers",
                default_restore_workers,
            )
            self._restore_max_regions = self._get_positive_int_extra(
                extra,
                "dax.restore_max_regions",
                self._restore_workers,
            )
            default_staging_slab_bytes = max(
                256 * 1024 * 1024,
                self._restore_max_regions * self.slot_bytes,
            )
            self._retrieve_staging_slab_bytes = self._get_positive_int_extra(
                extra,
                "dax.retrieve_staging_slab_bytes",
                default_staging_slab_bytes,
            )
            min_required_slab = self._restore_max_regions * self.slot_bytes
            if self._retrieve_staging_slab_bytes < min_required_slab:
                raise ValueError(
                    "extra_config['dax.retrieve_staging_slab_bytes'] must be at "
                    f"least {min_required_slab} bytes"
                )
            self._restore_region_bytes = (
                self._retrieve_staging_slab_bytes // self._restore_max_regions
            )
            if self._restore_region_bytes < self.slot_bytes:
                raise ValueError(
                    "dax.retrieve_staging_slab_bytes does not leave enough space "
                    "per restore region for one full chunk"
                )
            self._retrieve_staging_slab_ptr = int(
                lmc_ops.alloc_pinned_ptr(self._retrieve_staging_slab_bytes, 0)
            )
            self._restore_executor = ThreadPoolExecutor(
                max_workers=self._restore_workers,
                thread_name_prefix="dax-restore",
            )
            self._restore_dispatch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="dax-restore-dispatch",
            )

            self._state_lock = threading.RLock()
            self._state_condition = threading.Condition(self._state_lock)

            self._index: dict[CacheEngineKey, _Entry] = {}
            self._pin_counts: dict[CacheEngineKey, int] = {}
            self._inflight: dict[CacheEngineKey, _Inflight] = {}
            self._lru: "OrderedDict[CacheEngineKey, None]" = OrderedDict()
            self._slot_states: dict[int, _SlotState] = {}

            self._next_slot = 0
            self._free_slots: set[int] = set()
            self._active_ops = 0
            self._active_puts = 0
            self._closing = False
            self._closed = False

            logger.info(
                "DaxBackend init: device=%s dax_size=%d slot=%d max_slots=%d "
                "restore_workers=%d restore_regions=%d restore_slab=%d",
                self.device_path,
                self._arena_bytes,
                self.slot_bytes,
                self._max_slots,
                self._restore_workers,
                self._restore_max_regions,
                self._retrieve_staging_slab_bytes,
            )
        except Exception:
            fd, mmap_obj, arena_view = self._fd, self._mmap_obj, self._arena_view
            self._fd = None
            self._mmap_obj = None
            self._base_ptr = 0
            self._arena_view = None
            self._release_restore_resources()
            self._release_arena_resources(fd, mmap_obj, arena_view)
            raise

    def __str__(self) -> str:
        return "DaxBackend"

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether ``key`` exists in the backend.

        Args:
            key: The cache key to look up.
            pin: If ``True`` and the key is present, atomically
                increment its pin count.

        Returns:
            ``True`` if the key is present, ``False`` otherwise.
        """
        with self._state_lock:
            ok = key in self._index
            if ok and pin:
                self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
            return ok

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether ``key`` is tracked as an in-flight put task.

        Args:
            key: The cache key to check.

        Returns:
            ``True`` if ``key`` is in the in-flight put task set.
        """
        with self._state_lock:
            return key in self._inflight

    def pin(self, key: CacheEngineKey) -> bool:
        """Increment the pin count for ``key`` if it exists.

        Args:
            key: The cache key to pin.

        Returns:
            ``True`` if the key was found and pinned, ``False`` otherwise.
        """
        with self._state_lock:
            if key in self._index:
                self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
                return True
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        """Decrement the pin count for ``key``.

        Args:
            key: The cache key to unpin.

        Returns:
            ``True`` if ``key`` is present in the backend after the
            operation, ``False`` otherwise.
        """
        with self._state_lock:
            count = self._pin_counts.get(key, 0)
            if count > 0:
                if count == 1:
                    del self._pin_counts[key]
                else:
                    self._pin_counts[key] = count - 1
                return True
            return key in self._index

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove ``key`` from the backend if present.

        If the key is in-flight, it is marked canceled. If it is
        committed, its slot is scheduled for reclamation according to
        the current slot state.

        Args:
            key: The cache key to remove.
            force: Unused; accepted for interface compatibility.

        Returns:
            ``True`` if the key was present (committed or in-flight).
        """
        del force
        with self._state_lock:
            existed = key in self._index or key in self._inflight
            entry = self._index.pop(key, None)
            inflight = self._inflight.get(key)
            self._pin_counts.pop(key, None)
            self._lru.pop(key, None)
            if entry is not None:
                self._schedule_slot_reclaim_locked(entry.slot_id, entry.generation)
            if inflight is not None:
                inflight.canceled = True
            return existed

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        """Store a batch of memory objects in the DAX arena."""
        del transfer_spec
        if len(keys) != len(objs):
            raise ValueError(
                "keys and objs must have the same length, "
                f"got {len(keys)} and {len(objs)}"
            )
        futures: List[Future] = []

        for key, obj in zip(keys, objs, strict=True):
            should_finish_put = False
            try:
                # Multi-tensor objects are not yet supported.
                num_shapes = len(obj.get_shapes())
                if num_shapes > 1:
                    logger.error(
                        "DaxBackend does not support multi-tensor allocations: "
                        "key=%s has %d tensors. "
                        "Use single-tensor format or extend metadata.",
                        key,
                        num_shapes,
                    )
                    continue
                size = int(obj.get_size())
                obj_metadata = obj.metadata
                shape = obj_metadata.shape
                dtype = obj_metadata.dtype
                cached_positions = obj_metadata.cached_positions
                fmt = obj_metadata.fmt

                with self._state_lock:
                    if self._closing:
                        raise RuntimeError("DaxBackend is closing")
                    if key in self._index or key in self._inflight:
                        continue

                    if size > self.slot_bytes:
                        raise ValueError(
                            f"DaxBackend: object size {size} for key {key} "
                            f"exceeds slot size {self.slot_bytes}"
                        )
                    while True:
                        try:
                            slot_id = self._allocate_slot_locked()
                            break
                        except RuntimeError:
                            if not self._evict_one_locked():
                                raise
                    offset = slot_id * self.slot_bytes
                    generation = self._reserve_slot_state_locked(slot_id)

                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=size,
                        shape=shape,
                        dtype=dtype,
                        cached_positions=cached_positions,
                        fmt=fmt,
                        pin_count=0,
                    )

                    self._inflight[key] = _Inflight(
                        offset=offset,
                        meta=meta,
                        slot_id=slot_id,
                        generation=generation,
                        canceled=False,
                    )
                    self._active_puts += 1
                    should_finish_put = True

                if self.async_put and self.loop is not None and self.loop.is_running():
                    obj.ref_count_up()
                    try:
                        fut = asyncio.run_coroutine_threadsafe(
                            self._submit_write(
                                key=key,
                                offset=offset,
                                size=size,
                                memory_obj=obj,
                                on_complete_callback=on_complete_callback,
                            ),
                            self.loop,
                        )
                    except Exception:
                        with self._state_lock:
                            self._finalize_inflight_locked(key, write_failed=True)
                        obj.ref_count_down()
                        raise
                    futures.append(fut)
                    should_finish_put = False
                    continue

                try:
                    self._do_write(offset, obj, size)
                except Exception as e:
                    with self._state_lock:
                        self._finalize_inflight_locked(key, write_failed=True)
                    raise RuntimeError(
                        f"DaxBackend write failed for key {key}: {e}"
                    ) from e

                with self._state_lock:
                    should_invoke_callback = self._finalize_inflight_locked(
                        key,
                        write_failed=False,
                    )

                if should_invoke_callback:
                    self._invoke_on_complete_callback(key, on_complete_callback)
            finally:
                if should_finish_put:
                    with self._state_lock:
                        if self._active_puts > 0:
                            self._active_puts -= 1
                        else:
                            logger.warning(
                                "DaxBackend active put count underflow for key %s", key
                            )
                        self._state_condition.notify_all()

        return futures or None

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Return the memory object for a key, or ``None`` if unavailable."""
        with self._state_lock:
            if self._closing:
                return None
            entry = self._index.get(key)
            if entry is None:
                return None
            meta = entry.meta
            shape = meta.shape
            dtype = meta.dtype
            fmt = meta.fmt
            if shape is None or dtype is None or fmt is None:
                return None
            state = self._slot_states.get(entry.slot_id)
            if (
                state is None
                or state.generation != entry.generation
                or not state.committed
            ):
                return None
            state.borrow_count += 1
            self._active_ops += 1
            offset, size = entry.offset, int(meta.size)
            cached_positions = meta.cached_positions
            slot_id, generation = entry.slot_id, entry.generation

        assert self.local_cpu_backend is not None
        memory_obj: Optional[MemoryObj] = None
        read_ok = False
        try:
            memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
            if memory_obj is None:
                return None
            self._do_read(offset, memory_obj, size)
            memory_obj.metadata.cached_positions = cached_positions
            read_ok = True
            return memory_obj
        except Exception:
            if memory_obj is not None:
                memory_obj.ref_count_down()
            raise
        finally:
            with self._state_lock:
                if self._active_ops > 0:
                    self._active_ops -= 1
                state = self._slot_states.get(slot_id)
                if state is not None and state.generation == generation:
                    if state.borrow_count > 0:
                        state.borrow_count -= 1
                    if read_ok:
                        current = self._index.get(key)
                        if (
                            current is not None
                            and current.slot_id == slot_id
                            and current.generation == generation
                        ):
                            self._touch_locked(key)
                    if state.pending_free and state.borrow_count == 0:
                        state.pending_free = False
                        self._free_slot_locked(slot_id)
                self._state_condition.notify_all()

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Return the number of consecutive keys present in the index.

        Iterates ``keys`` in order and stops at the first miss.

        Args:
            lookup_id: Caller-supplied identifier (not used by this backend).
            keys: Ordered list of cache keys to check.
            pin: If ``True``, pin each found key.

        Returns:
            The count of consecutive hits from the start of ``keys``.
        """
        del lookup_id
        hit = 0
        with self._state_lock:
            for key in keys:
                if key not in self._index:
                    break
                if pin:
                    self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
                hit += 1
        return hit

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Retrieve memory objects for consecutive keys asynchronously.

        Schedules one batched restore job on the persistent dispatch
        executor and returns only the consecutive hit prefix. Stops at the
        first key that is not found or is no longer readable.

        Args:
            lookup_id: Caller-supplied identifier (not used by this backend).
            keys: Ordered list of cache keys to retrieve.
            transfer_spec: Transfer hint (not used by this backend).

        Returns:
            A list of ``MemoryObj`` instances for the consecutive hits.
        """
        del lookup_id, transfer_spec
        if not keys:
            return []

        dispatch_executor = self._restore_dispatch_executor
        if dispatch_executor is None:
            raise RuntimeError("DaxBackend restore dispatch executor is not available")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            dispatch_executor,
            self._restore_batch,
            list(keys),
            True,
        )
        return cast(list[MemoryObj], result)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Restore a batch of DAX-backed cache entries synchronously.

        The returned list preserves the input order. Entries that are missing
        or no longer readable remain ``None`` so callers keep positional
        alignment with ``keys``.

        Args:
            keys: Ordered cache keys to restore from the DAX arena.

        Returns:
            A list aligned with ``keys`` containing restored ``MemoryObj``
            instances or ``None`` for entries that could not be read.
        """
        if not keys:
            return []

        dispatch_executor = self._restore_dispatch_executor
        if dispatch_executor is None:
            raise RuntimeError("DaxBackend restore dispatch executor is not available")

        batch_keys = list(keys)
        if threading.current_thread().name.startswith("dax-restore-dispatch"):
            return cast(
                List[Optional[MemoryObj]],
                self._restore_batch(batch_keys, False),
            )

        future = dispatch_executor.submit(
            self._restore_batch,
            batch_keys,
            False,
        )
        return cast(List[Optional[MemoryObj]], future.result())

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Return the number of consecutive keys present in the index.

        Synchronous variant of :meth:`batched_async_contains`.

        Args:
            keys: Ordered list of cache keys to check.
            pin: If ``True``, pin each found key.

        Returns:
            The count of consecutive hits from the start of ``keys``.
        """
        hit = 0
        with self._state_lock:
            for key in keys:
                if key not in self._index:
                    break
                if pin:
                    self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
                hit += 1
        return hit

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """Remove multiple keys from the backend.

        Args:
            keys: The cache keys to remove.
            force: Passed through to :meth:`remove`.

        Returns:
            The number of keys that were actually present and removed.
        """
        removed = 0
        for key in keys:
            removed += int(self.remove(key, force=force))
        return removed

    def get_allocator_backend(self) -> LocalCPUBackend:
        """Return the CPU allocator backend used for read buffers.

        Raises:
            RuntimeError: If no ``local_cpu_backend`` is available.
        """
        if self.local_cpu_backend is None:
            raise RuntimeError("DaxBackend has no allocator backend available")
        return self.local_cpu_backend

    def close(self) -> None:
        """Quiesce outstanding operations and release the mapped DAX arena."""
        restore_executor = None
        restore_dispatch_executor = None
        staging_slab_ptr = 0
        with self._state_lock:
            if self._closed:
                return
            self._closing = True
            while self._active_puts > 0 or self._active_ops > 0:
                if not self._state_condition.wait(timeout=30.0):
                    logger.warning(
                        "DaxBackend close: still waiting for %d puts, %d ops",
                        self._active_puts,
                        self._active_ops,
                    )
            if self._closed:
                return
            self._closed = True
            restore_executor = self._restore_executor
            restore_dispatch_executor = self._restore_dispatch_executor
            staging_slab_ptr = self._retrieve_staging_slab_ptr
            self._restore_executor = None
            self._restore_dispatch_executor = None
            self._retrieve_staging_slab_ptr = 0
            self._retrieve_staging_slab_bytes = 0
            self._restore_region_bytes = 0
            self._index.clear()
            self._inflight.clear()
            self._lru.clear()
            self._pin_counts.clear()
            self._slot_states.clear()
            self._free_slots.clear()
            fd = self._fd
            mmap_obj = self._mmap_obj
            arena_view = self._arena_view
            self._fd = None
            self._mmap_obj = None
            self._base_ptr = 0
            self._arena_view = None

        if restore_dispatch_executor is not None:
            restore_dispatch_executor.shutdown(wait=True)
        if restore_executor is not None:
            restore_executor.shutdown(wait=True)
        self._release_restore_resources(
            restore_slab_ptr=staging_slab_ptr,
        )
        self._release_arena_resources(fd, mmap_obj, arena_view)

    # ------------------------------------------------------------------
    # Private / helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_positive_int_extra(
        extra_config: dict[str, Any],
        key: str,
        default: int,
    ) -> int:
        value = extra_config.get(key, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"extra_config['{key}'] must be a positive integer") from e
        if parsed <= 0:
            raise ValueError(f"extra_config['{key}'] must be a positive integer")
        return parsed

    def _release_restore_resources(
        self,
        restore_slab_ptr: Optional[int] = None,
    ) -> None:
        """Shut down restore workers and free the pinned retrieve slab.

        Args:
            restore_slab_ptr: Optional explicit slab pointer to free. When not
                provided, the backend releases its current staging slab and
                clears the associated bookkeeping fields.
        """
        dispatch_executor = self._restore_dispatch_executor
        if dispatch_executor is not None:
            dispatch_executor.shutdown(wait=True)
            self._restore_dispatch_executor = None

        restore_executor = self._restore_executor
        if restore_executor is not None:
            restore_executor.shutdown(wait=True)
            self._restore_executor = None

        ptr = (
            self._retrieve_staging_slab_ptr
            if restore_slab_ptr is None
            else restore_slab_ptr
        )
        if ptr:
            try:
                lmc_ops.free_pinned_ptr(ptr)
            except Exception as e:
                logger.warning("Failed to free DAX retrieve slab: %s", e)

        if restore_slab_ptr is None:
            self._retrieve_staging_slab_ptr = 0
            self._retrieve_staging_slab_bytes = 0
            self._restore_region_bytes = 0

    @staticmethod
    def _release_arena_resources(
        fd: Optional[int],
        mmap_obj: Optional[mmap.mmap],
        arena_view: Optional[memoryview],
    ) -> None:
        """Release the mapped DAX arena resources in close order.

        Args:
            fd: File descriptor for the DAX device.
            mmap_obj: Mmap object backing the arena mapping.
            arena_view: Memoryview exported from the mmap.
        """
        if arena_view is not None:
            try:
                arena_view.release()
            except Exception as e:
                logger.warning("Failed to release DAX memoryview: %s", e)

        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception as e:
                logger.warning("Failed to close DAX mmap: %s", e)

        if fd is not None:
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Failed to close DAX fd: %s", e)

    def _open_arena(self) -> None:
        fd: Optional[int] = None
        mmap_obj: Optional[mmap.mmap] = None
        arena_view: Optional[memoryview] = None
        try:
            fd = os.open(self.device_path, os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                f"Failed to open dax device {self.device_path}: {e}"
            ) from e
        try:
            try:
                capacity_bytes = os.fstat(fd).st_size
                if capacity_bytes > 0 and self._arena_bytes > capacity_bytes:
                    raise RuntimeError(
                        f"dax.max_dax_size ({self._arena_bytes} bytes) exceeds "
                        f"device capacity ({capacity_bytes} bytes)"
                    )
            except OSError:
                # Some dax devices may not report size via fstat.
                logger.warning(
                    "Could not determine DAX device capacity via fstat; "
                    "skipping dax.max_dax_size validation"
                )

            mmap_obj = mmap.mmap(
                fd,
                self._arena_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mmap_obj))
            arena_view = memoryview(mmap_obj)
            self._fd = fd
            self._mmap_obj = mmap_obj
            self._base_ptr = base_ptr
            self._arena_view = arena_view
        except Exception as e:
            DaxBackend._release_arena_resources(fd, mmap_obj, arena_view)
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(
                f"Failed to mmap dax arena ({self._arena_bytes} bytes) from "
                f"{self.device_path}: {e}"
            ) from e

    def _reserve_slot_state_locked(self, slot_id: int) -> int:
        existing = self._slot_states.get(slot_id)
        new_gen = (existing.generation if existing is not None else 0) + 1
        self._slot_states[slot_id] = _SlotState(generation=new_gen)
        return new_gen

    def _mark_slot_committed_locked(self, slot_id: int, generation: int) -> None:
        state = self._slot_states.get(slot_id)
        if state is None or state.generation != generation:
            return
        state.committed = True
        state.pending_free = False

    def _schedule_slot_reclaim_locked(self, slot_id: int, generation: int) -> None:
        """Mark a slot uncommitted and free it immediately or defer if borrowed."""
        state = self._slot_states.get(slot_id)
        if state is None or state.generation != generation:
            return
        state.committed = False
        if state.borrow_count == 0:
            state.pending_free = False
            self._free_slot_locked(slot_id)
        else:
            state.pending_free = True

    def _finalize_inflight_locked(
        self,
        key: CacheEngineKey,
        write_failed: bool,
    ) -> bool:
        """Resolve an in-flight put: commit on success, reclaim on failure."""
        inflight = self._inflight.pop(key, None)
        if inflight is None:
            return False
        if inflight.canceled or write_failed:
            self._schedule_slot_reclaim_locked(inflight.slot_id, inflight.generation)
            return False
        self._mark_slot_committed_locked(inflight.slot_id, inflight.generation)
        self._index[key] = _Entry(
            offset=inflight.offset,
            meta=inflight.meta,
            slot_id=inflight.slot_id,
            generation=inflight.generation,
        )
        self._touch_locked(key)
        return True

    def _invoke_on_complete_callback(
        self,
        key: CacheEngineKey,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        if on_complete_callback is None:
            return
        try:
            on_complete_callback(key)
        except Exception as e:
            logger.warning("on_complete_callback failed for key %s: %s", key, e)

    def _allocate_slot_locked(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            return slot
        raise RuntimeError("No free slots available; eviction required")

    def _free_slot_locked(self, slot_id: int) -> None:
        if slot_id < 0:
            return
        self._free_slots.add(slot_id)

    def _touch_locked(self, key: CacheEngineKey) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None

    def _evict_one_locked(self) -> bool:
        for victim in list(self._lru.keys()):
            if self._pin_counts.get(victim, 0) > 0 or victim in self._inflight:
                continue
            entry = self._index.get(victim)
            if entry is None:
                continue
            state = self._slot_states.get(entry.slot_id)
            if (
                state is None
                or state.generation != entry.generation
                or state.borrow_count > 0
            ):
                continue
            self._index.pop(victim, None)
            self._lru.pop(victim, None)
            self._pin_counts.pop(victim, None)
            self._schedule_slot_reclaim_locked(entry.slot_id, entry.generation)
            return True
        return False

    def _reserve_restore_items(
        self,
        keys: Sequence[CacheEngineKey],
        *,
        prefix_only: bool,
    ) -> tuple[list[_RestoreItem], list[Optional[MemoryObj]]]:
        """Reserve readable entries and build the aligned result list.

        Args:
            keys: Ordered keys requested by the caller.
            prefix_only: When ``True``, stop at the first miss or unreadable
                entry so async-prefetch keeps prefix-hit semantics.

        Returns:
            A tuple containing the reserved restore items and a result list
            aligned with ``keys`` that is prefilled with ``None`` placeholders.
        """
        results: list[Optional[MemoryObj]] = [None] * len(keys)
        reserved: list[_RestoreItem] = []

        with self._state_lock:
            if self._closing:
                return reserved, results

            for result_index, key in enumerate(keys):
                entry = self._index.get(key)
                if entry is None:
                    if prefix_only:
                        break
                    continue

                meta = entry.meta
                shape = meta.shape
                dtype = meta.dtype
                fmt = meta.fmt
                if shape is None or dtype is None or fmt is None:
                    if prefix_only:
                        break
                    continue

                state = self._slot_states.get(entry.slot_id)
                if (
                    state is None
                    or state.generation != entry.generation
                    or not state.committed
                ):
                    if prefix_only:
                        break
                    continue

                state.borrow_count += 1
                reserved.append(
                    _RestoreItem(
                        result_index=result_index,
                        key=key,
                        offset=entry.offset,
                        size=int(meta.size),
                        shape=shape,
                        dtype=dtype,
                        fmt=fmt,
                        cached_positions=meta.cached_positions,
                        slot_id=entry.slot_id,
                        generation=entry.generation,
                    )
                )

            if reserved:
                self._active_ops += 1

        return reserved, results

    def _allocate_restore_outputs(self, reserved: Sequence[_RestoreItem]) -> None:
        """Allocate CPU restore buffers for the reserved DAX items.

        Args:
            reserved: Restore items whose output ``MemoryObj`` fields will be
                populated in-place.

        Raises:
            RuntimeError: If the local CPU allocator cannot provide enough
                output buffers for the batch.
        """
        assert self.local_cpu_backend is not None

        grouped_items: OrderedDict[
            tuple[tuple[int, ...], torch.dtype, MemoryFormat], list[_RestoreItem]
        ] = OrderedDict()
        for item in reserved:
            grouped_items.setdefault(
                (tuple(item.shape), item.dtype, item.fmt),
                [],
            ).append(item)

        for group_items in grouped_items.values():
            first = group_items[0]
            outputs: Optional[list[MemoryObj]] = None
            if len(group_items) > 1:
                outputs = self.local_cpu_backend.batched_allocate(
                    first.shape,
                    first.dtype,
                    len(group_items),
                    first.fmt,
                )

            if outputs is None:
                outputs = []
                for _ in group_items:
                    memory_obj = self.local_cpu_backend.allocate(
                        first.shape,
                        first.dtype,
                        first.fmt,
                    )
                    if memory_obj is None:
                        for allocated in outputs:
                            allocated.ref_count_down()
                        raise RuntimeError(
                            "DaxBackend batched restore allocation failed"
                        )
                    outputs.append(memory_obj)

            for item, memory_obj in zip(group_items, outputs, strict=True):
                item.memory_obj = memory_obj

    def _build_restore_waves(
        self,
        reserved: Sequence[_RestoreItem],
    ) -> list[_RestoreWave]:
        """Plan slab-backed restore work as waves of parallel regions.

        Args:
            reserved: Restore items that already have output buffers assigned.

        Returns:
            A list of restore waves, where each wave contains region copies that
            can run in parallel without overlapping slab space.
        """
        if not reserved:
            return []

        sorted_items = sorted(reserved, key=lambda item: item.offset)
        waves: list[_RestoreWave] = []
        next_item_idx = 0

        while next_item_idx < len(sorted_items):
            regions: list[_RestoreRegion] = []
            for region_index in range(self._restore_max_regions):
                if next_item_idx >= len(sorted_items):
                    break

                region_items: list[_RestoreItem] = []
                region_spans: list[_RestoreSpan] = []
                used_bytes = 0

                while next_item_idx < len(sorted_items):
                    item = sorted_items[next_item_idx]
                    if item.size > self._restore_region_bytes:
                        raise RuntimeError(
                            f"DaxBackend restore item size {item.size} exceeds "
                            "region capacity "
                            f"{self._restore_region_bytes}"
                        )
                    if (
                        used_bytes > 0
                        and used_bytes + item.size > self._restore_region_bytes
                    ):
                        break

                    item.slab_offset = used_bytes
                    region_items.append(item)
                    if (
                        region_spans
                        and region_spans[-1].src_offset + region_spans[-1].size
                        == item.offset
                        and region_spans[-1].slab_offset + region_spans[-1].size
                        == item.slab_offset
                    ):
                        region_spans[-1].size += item.size
                    else:
                        region_spans.append(
                            _RestoreSpan(
                                src_offset=item.offset,
                                slab_offset=item.slab_offset,
                                size=item.size,
                            )
                        )

                    used_bytes += item.size
                    next_item_idx += 1

                regions.append(
                    _RestoreRegion(
                        region_index=region_index,
                        slab_offset=region_index * self._restore_region_bytes,
                        total_bytes=used_bytes,
                        items=region_items,
                        spans=region_spans,
                    )
                )

            waves.append(_RestoreWave(regions=regions))

        return waves

    def _batched_memcpy(
        self,
        src_ptrs: Sequence[int],
        dst_ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        """Copy a batch of byte ranges, preferring the native helper.

        Args:
            src_ptrs: Source addresses for each copy.
            dst_ptrs: Destination addresses for each copy.
            sizes: Byte counts for each copy.
        """
        if not src_ptrs:
            return
        if hasattr(lmc_ops, "batched_memcpy"):
            lmc_ops.batched_memcpy(list(src_ptrs), list(dst_ptrs), list(sizes))
            return

        for src_ptr, dst_ptr, size in zip(src_ptrs, dst_ptrs, sizes, strict=True):
            ctypes.memmove(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                size,
            )

    def _restore_region(self, region: _RestoreRegion) -> None:
        """Restore one region from DAX into the assigned output buffers.

        Args:
            region: Copy plan describing the DAX spans to stage and the output
                buffers to populate from the shared slab.

        Raises:
            RuntimeError: If the shared retrieve slab is unavailable.
        """
        if region.total_bytes <= 0 or not region.items:
            return
        if self._retrieve_staging_slab_ptr == 0:
            raise RuntimeError("DaxBackend retrieve slab is not allocated")

        slab_base_ptr = self._retrieve_staging_slab_ptr + region.slab_offset
        dax_src_ptrs = [self._base_ptr + span.src_offset for span in region.spans]
        slab_dst_ptrs = [slab_base_ptr + span.slab_offset for span in region.spans]
        dax_copy_sizes = [span.size for span in region.spans]
        self._batched_memcpy(dax_src_ptrs, slab_dst_ptrs, dax_copy_sizes)

        slab_src_ptrs = [slab_base_ptr + item.slab_offset for item in region.items]
        dst_ptrs = [cast(MemoryObj, item.memory_obj).data_ptr for item in region.items]
        out_sizes = [item.size for item in region.items]
        self._batched_memcpy(slab_src_ptrs, dst_ptrs, out_sizes)

    def _run_restore_waves(self, waves: Sequence[_RestoreWave]) -> None:
        """Execute restore waves and wait for all region copies to finish.

        Args:
            waves: Ordered restore waves produced by
                :meth:`_build_restore_waves`.

        Raises:
            RuntimeError: If the restore worker pool is unavailable.
        """
        restore_executor = self._restore_executor
        if restore_executor is None:
            raise RuntimeError("DaxBackend restore executor is not available")

        for wave in waves:
            futures = [
                restore_executor.submit(self._restore_region, region)
                for region in wave.regions
                if region.items
            ]
            for future in futures:
                future.result()

    def _cleanup_restore_outputs(self, reserved: Sequence[_RestoreItem]) -> None:
        """Release any output buffers allocated for a failed restore batch.

        Args:
            reserved: Restore items whose temporary output buffers should be
                decremented and cleared.
        """
        for item in reserved:
            if item.memory_obj is not None:
                item.memory_obj.ref_count_down()
                item.memory_obj = None

    def _finalize_reserved_items(
        self,
        reserved: Sequence[_RestoreItem],
        *,
        touched_keys: Optional[set[CacheEngineKey]] = None,
    ) -> None:
        """Release restore borrows and update post-restore slot state.

        Args:
            reserved: Restore items previously reserved by
                :meth:`_reserve_restore_items`.
            touched_keys: Keys that completed successfully and should refresh
                their LRU state before borrow counts are dropped.
        """
        if not reserved:
            return
        touched_keys = touched_keys or set()
        with self._state_lock:
            if self._active_ops > 0:
                self._active_ops -= 1
            else:
                logger.warning("DaxBackend active op count underflow during restore")

            for item in reserved:
                state = self._slot_states.get(item.slot_id)
                if state is None or state.generation != item.generation:
                    continue
                if state.borrow_count > 0:
                    state.borrow_count -= 1

                if item.key in touched_keys:
                    current = self._index.get(item.key)
                    if (
                        current is not None
                        and current.slot_id == item.slot_id
                        and current.generation == item.generation
                    ):
                        self._touch_locked(item.key)

                if state.pending_free and state.borrow_count == 0:
                    state.pending_free = False
                    self._free_slot_locked(item.slot_id)

            self._state_condition.notify_all()

    def _restore_batch(
        self,
        keys: list[CacheEngineKey],
        prefix_only: bool,
    ) -> list[Optional[MemoryObj]]:
        """Restore one batch of keys through the staged DAX retrieve pipeline.

        Args:
            keys: Ordered keys to restore.
            prefix_only: When ``True``, return only the consecutive readable
                prefix used by async-prefetch retrieval. When ``False``, return
                an input-aligned list and preserve ``None`` holes for misses.

        Returns:
            Restored outputs for the batch. The returned list is input-aligned
            for blocking retrieval and prefix-compacted for async-prefetch.
        """
        reserved, results = self._reserve_restore_items(keys, prefix_only=prefix_only)
        if not reserved:
            return [] if prefix_only else results

        touched_keys: set[CacheEngineKey] = set()
        try:
            self._allocate_restore_outputs(reserved)
            waves = self._build_restore_waves(reserved)
            self._run_restore_waves(waves)
            for item in reserved:
                memory_obj = cast(MemoryObj, item.memory_obj)
                memory_obj.metadata.cached_positions = item.cached_positions
                results[item.result_index] = memory_obj
                touched_keys.add(item.key)
        except Exception:
            self._cleanup_restore_outputs(reserved)
            self._finalize_reserved_items(reserved)
            raise

        self._finalize_reserved_items(reserved, touched_keys=touched_keys)
        if prefix_only:
            return cast(
                list[Optional[MemoryObj]],
                [cast(MemoryObj, results[item.result_index]) for item in reserved],
            )
        return results

    def _do_write(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(self._base_ptr + offset),
            ctypes.c_void_p(memory_obj.data_ptr),
            size,
        )

    def _do_read(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(memory_obj.data_ptr),
            ctypes.c_void_p(self._base_ptr + offset),
            size,
        )

    async def _submit_write(
        self,
        key: CacheEngineKey,
        offset: int,
        size: int,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        write_error: Optional[Exception] = None
        should_invoke_callback = False
        try:
            try:
                await asyncio.to_thread(self._do_write, offset, memory_obj, size)
            except Exception as e:
                write_error = e
                logger.warning("Async DAX write failed for key %s: %s", key, e)
            finally:
                with self._state_lock:
                    should_invoke_callback = self._finalize_inflight_locked(
                        key,
                        write_failed=write_error is not None,
                    )

            if write_error is not None:
                raise RuntimeError(
                    f"DaxBackend write failed for key {key}: {write_error}"
                ) from write_error

            if should_invoke_callback:
                self._invoke_on_complete_callback(key, on_complete_callback)
        finally:
            memory_obj.ref_count_down()
            with self._state_lock:
                if self._active_puts > 0:
                    self._active_puts -= 1
                else:
                    logger.warning(
                        "DaxBackend active put count underflow for key %s", key
                    )
                self._state_condition.notify_all()
