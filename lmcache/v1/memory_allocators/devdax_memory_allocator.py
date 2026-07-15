# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union
import ctypes
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.buffer_allocator import BufferAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
    logger,
)
import lmcache.v1.memory_management as memory_management


class DevDaxArenaState(Enum):
    """Arena lifecycle. ``REMOVED`` is terminal and report-only: the arena has
    already been unmapped and dropped from the pool."""

    ACTIVE = "active"
    DRAINING = "draining"
    REMOVED = "removed"


class DevDaxRemoveMode(Enum):
    """Arena removal strategy. Only ``DRAIN`` (unmap after the last live
    allocation is freed) is supported; migrate/evict are deferred."""

    DRAIN = "drain"


@dataclass(frozen=True)
class DevDaxArenaStatus:
    """Immutable snapshot of one arena's runtime state."""

    device_path: str
    size_in_bytes: int
    used_bytes: int
    free_bytes: int
    active_allocations: int
    state: DevDaxArenaState
    is_primary: bool


@dataclass
class _DevDaxArena:
    """One mmap-backed arena: fd, mmap, flat ``torch.uint8`` view, and its own
    :class:`TensorMemoryAllocator`. Frees are routed to the owning arena by
    pointer range."""

    device_path: str
    size: int
    fd: int
    mmap_obj: mmap.mmap
    mmap_buffer: Any
    buffer: torch.Tensor
    allocator: TensorMemoryAllocator
    is_primary: bool
    state: DevDaxArenaState = DevDaxArenaState.ACTIVE
    pinned_ptr: int | None = None
    # Captured at map time so ``contains`` stays correct after a deferred
    # close has dropped the buffer references.
    base_ptr: int = 0

    def __post_init__(self) -> None:
        self.base_ptr = self.buffer.data_ptr()

    def contains(self, ptr: int) -> bool:
        """Return whether ``ptr`` falls within this arena's mapped range."""
        return self.base_ptr <= ptr < self.base_ptr + self.size

    def status(self) -> DevDaxArenaStatus:
        """Return an immutable snapshot of this arena's runtime state."""
        address_manager = self.allocator.address_manager
        total = address_manager.get_heap_size()
        free = address_manager.get_free_size()
        return DevDaxArenaStatus(
            device_path=self.device_path,
            size_in_bytes=self.size,
            used_bytes=total - free,
            free_bytes=free,
            active_allocations=self.allocator.num_active_allocations,
            state=self.state,
            is_primary=self.is_primary,
        )


class DevDaxMemoryAllocator(MemoryAllocatorInterface):
    """Allocates L1 objects from DRAM and/or a pool of Device-DAX arenas.

    DRAM (if configured) is tried first; active arenas serve overflow in pool
    order. The pool grows/shrinks at runtime via :meth:`add_device` and
    :meth:`remove_device` (drain-based: unmap once the last live allocation is
    freed). In pure Device-DAX mode the initial arena is primary and
    non-removable; in hybrid mode every arena is removable overflow.
    ``host_mem_lock`` serializes all pool and per-arena mutations; the DRAM
    local allocator synchronizes itself.
    """

    def __init__(
        self,
        size: int,
        device_path: str,
        *,
        local_allocator: MixedMemoryAllocator | None = None,
        local_size: int = 0,
        shm_name: str | None = None,
        align_bytes: int = AddressManager.ALIGN_BYTES,
    ) -> None:
        if not device_path:
            raise ValueError("device_path must be a non-empty string")
        if size <= 0:
            raise ValueError("size must be > 0")
        if local_size < 0:
            raise ValueError("local_size must be >= 0")
        if local_size and local_allocator is not None:
            raise ValueError("local_size cannot be used with local_allocator")
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        # ``size``/``device_path`` describe the initial arena (kept for
        # backwards compatibility); the live pool is ``self._arenas``.
        self.size = size
        self.device_path = device_path
        self.align_bytes = align_bytes
        self.local_allocator = local_allocator
        if local_size:
            self.local_allocator = MixedMemoryAllocator(
                local_size,
                align_bytes=self.align_bytes,
                shm_name=shm_name,
            )
        self.host_mem_lock = threading.Lock()
        self.buffer_allocator = BufferAllocator("cpu")
        self._arenas: list[_DevDaxArena] = []
        self._unregistered = False

        # Primary (non-removable) only in pure Device-DAX mode. The constructor
        # runs single-threaded, so no lock is needed here.
        self._map_and_append_arena(
            device_path, size, is_primary=self.local_allocator is None
        )

    @property
    def buffer(self) -> torch.Tensor:
        """Return the primary L1 buffer (DRAM if hybrid, else the primary arena)."""
        if self.local_allocator is not None:
            return self.local_allocator.buffer
        if self._arenas:
            return self._arenas[0].buffer
        return torch.empty(0, dtype=torch.uint8)

    @property
    def devdax_buffer(self) -> torch.Tensor:
        """Return the initial arena buffer (backwards-compat shim)."""
        if self._arenas:
            return self._arenas[0].buffer
        return torch.empty(0, dtype=torch.uint8)

    @property
    def devdax_allocator(self) -> TensorMemoryAllocator | None:
        """Return the initial Device-DAX arena allocator, or ``None`` if closed."""
        if self._arenas:
            return self._arenas[0].allocator
        return None

    @property
    def address_manager(self) -> AddressManager | None:
        """Return the initial Device-DAX arena address manager, or ``None``."""
        if self._arenas:
            return self._arenas[0].allocator.address_manager
        return None

    def _open_devdax_mapping(
        self,
        device_path: str,
        size: int,
    ) -> tuple[int, mmap.mmap, Any, torch.Tensor]:
        fd: int | None = None
        mmap_obj: mmap.mmap | None = None
        try:
            fd = os.open(device_path, os.O_RDWR)
            capacity = os.fstat(fd).st_size
            if capacity > 0 and size > capacity:
                raise RuntimeError(
                    f"l1 devdax size ({size} bytes) exceeds "
                    f"{device_path} capacity ({capacity} bytes)"
                )

            mmap_obj = mmap.mmap(
                fd,
                size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            array_type = ctypes.c_uint8 * size
            mmap_buffer = array_type.from_buffer(mmap_obj)
            buffer = torch.frombuffer(mmap_buffer, dtype=torch.uint8)
            return fd, mmap_obj, mmap_buffer, buffer
        except Exception:
            if mmap_obj is not None:
                mmap_obj.close()
            if fd is not None:
                os.close(fd)
            raise

    def _map_and_append_arena(
        self,
        device_path: str,
        size: int,
        *,
        is_primary: bool,
    ) -> _DevDaxArena:
        """Map a device and append it to the pool as ACTIVE.

        Caller holds ``host_mem_lock`` (except in the single-threaded
        ``__init__``).
        """
        fd, mmap_obj, mmap_buffer, buffer = self._open_devdax_mapping(device_path, size)
        arena: _DevDaxArena | None = None
        try:
            allocator = TensorMemoryAllocator(buffer, align_bytes=self.align_bytes)
            arena = _DevDaxArena(
                device_path=device_path,
                size=size,
                fd=fd,
                mmap_obj=mmap_obj,
                mmap_buffer=mmap_buffer,
                buffer=buffer,
                allocator=allocator,
                is_primary=is_primary,
            )
            self._register_arena_pin(arena)
        except Exception:
            # Release the mapping on any setup failure; all references into
            # the mmap must be dropped before it can close.
            buffer = torch.empty(0, dtype=torch.uint8)
            mmap_buffer = None
            if arena is not None:
                self._close_arena_locked(arena)
            else:
                mmap_obj.close()
                os.close(fd)
            raise

        self._arenas.append(arena)
        return arena

    def _close_arena_locked(self, arena: _DevDaxArena) -> None:
        """Unpin and unmap one arena, dropping every reference into the mmap
        first (CPython refuses to close a buffer with exported pointers).
        Caller holds ``host_mem_lock``."""
        self._unregister_arena_pin(arena)
        arena.allocator.buffer = torch.empty(0, dtype=torch.uint8)
        arena.buffer = torch.empty(0, dtype=torch.uint8)
        arena.mmap_buffer = None
        arena.mmap_obj.close()
        os.close(arena.fd)

    def _register_arena_pin(self, arena: _DevDaxArena) -> None:
        if not memory_management.current_device_spec.is_pin_supported:
            return
        if not memory_management.current_device_spec.pin_memory(
            arena.buffer.data_ptr(), arena.size
        ):
            logger.warning(
                "pin_memory failed for Device-DAX L1 mapping %s; "
                "falling back to pageable host copies",
                arena.device_path,
            )
            return
        arena.pinned_ptr = arena.buffer.data_ptr()

    def _unregister_arena_pin(self, arena: _DevDaxArena) -> None:
        if arena.pinned_ptr is None:
            return
        memory_management.current_device_spec.unpin_memory(arena.pinned_ptr)
        arena.pinned_ptr = None

    def _active_arenas(self) -> list[_DevDaxArena]:
        """Return arenas accepting new allocations. Caller holds ``host_mem_lock``."""
        return [
            arena for arena in self._arenas if arena.state == DevDaxArenaState.ACTIVE
        ]

    def _find_arena_locked(self, device_path: str) -> _DevDaxArena:
        """Return the arena mapped at ``device_path``. Caller holds the lock."""
        for arena in self._arenas:
            if arena.device_path == device_path:
                return arena
        raise ValueError(f"no Device-DAX arena mapped at {device_path}")

    def _arena_for_obj_locked(self, memory_obj: MemoryObj) -> _DevDaxArena:
        """Return the arena owning ``memory_obj`` by pointer range; must run
        before the free invalidates the object. Caller holds the lock."""
        ptr = memory_obj.data_ptr
        for arena in self._arenas:
            if arena.contains(ptr):
                return arena
        raise ValueError("Memory object does not belong to any Device-DAX arena")

    def _maybe_reap_arena_locked(self, arena: _DevDaxArena) -> None:
        """Unmap a DRAINING arena once it has no live allocations. If lingering
        external views block the unmap, it stays DRAINING and later frees
        retry. Caller holds ``host_mem_lock``."""
        if arena.state != DevDaxArenaState.DRAINING:
            return
        if arena.allocator.num_active_allocations > 0:
            return
        try:
            self._close_arena_locked(arena)
        except BufferError:
            logger.warning(
                "Device-DAX arena %s has no live allocations but its mapping "
                "is still referenced by exported views; deferring unmap",
                arena.device_path,
            )
            return
        self._arenas.remove(arena)
        arena.state = DevDaxArenaState.REMOVED

    def _reap_draining_arenas_locked(self) -> None:
        """Retry reaping every draining arena. Caller holds ``host_mem_lock``."""
        for arena in list(self._arenas):
            self._maybe_reap_arena_locked(arena)

    def _is_local_obj(self, memory_obj: MemoryObj) -> bool:
        return (
            self.local_allocator is not None
            and memory_obj.parent() is self.local_allocator
        )

    def _is_devdax_obj(self, memory_obj: MemoryObj) -> bool:
        return memory_obj.parent() is self

    def _available_count(
        self,
        address_manager: AddressManager,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> int:
        """Return how many objects of the given shape fit in ``address_manager``."""
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        unit_raw_size = get_size_bytes(shapes, dtypes)
        unit_aligned_size = address_manager.compute_aligned_size(unit_raw_size)
        if unit_aligned_size <= 0:
            return 0
        return address_manager.get_free_size() // unit_aligned_size

    def _local_available_count(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> int:
        if self.local_allocator is None:
            return 0
        local_pin_allocator = self.local_allocator.pin_allocator
        assert isinstance(local_pin_allocator, TensorMemoryAllocator)
        return self._available_count(
            local_pin_allocator.address_manager, shapes, dtypes
        )

    def _local_memory_usage(self) -> tuple[int, int]:
        if self.local_allocator is None:
            return 0, 0
        local_pin_allocator = self.local_allocator.pin_allocator
        assert isinstance(local_pin_allocator, TensorMemoryAllocator)
        local_total = local_pin_allocator.address_manager.get_heap_size()
        local_used = local_total - local_pin_allocator.address_manager.get_free_size()
        return local_used, local_total

    def _allocate_from_arenas(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """Allocate one object from the first active arena that can serve it."""
        with self.host_mem_lock:
            for arena in self._active_arenas():
                # Skip full arenas: a failed probe logs a warning per miss,
                # flooding the log once earlier arenas fill up.
                if (
                    self._available_count(
                        arena.allocator.address_manager, shapes, dtypes
                    )
                    <= 0
                ):
                    continue
                obj = arena.allocator.allocate(shapes, dtypes, fmt, str(self))
                if obj is not None:
                    if isinstance(obj, TensorMemoryObj):
                        obj.parent_allocator = self
                    return obj
            return None

    def _batched_allocate_from_arenas(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        count: int,
        fmt: MemoryFormat,
    ) -> Optional[List[MemoryObj]]:
        """Fill ``count`` objects greedily across active arenas, all-or-nothing:
        a partial allocation is rolled back and ``None`` returned."""
        with self.host_mem_lock:
            remaining = count
            allocated: list[tuple[_DevDaxArena, list[MemoryObj]]] = []
            for arena in self._active_arenas():
                if remaining == 0:
                    break
                available = self._available_count(
                    arena.allocator.address_manager, shapes, dtypes
                )
                take = min(remaining, available)
                if take <= 0:
                    continue
                objs = arena.allocator.batched_allocate(
                    shapes, dtypes, take, fmt, str(self)
                )
                if not objs:
                    continue
                for obj in objs:
                    if isinstance(obj, TensorMemoryObj):
                        obj.parent_allocator = self
                allocated.append((arena, objs))
                remaining -= len(objs)

            if remaining > 0:
                # Roll back directly on each arena (the lock is already held).
                for arena, objs in allocated:
                    arena.allocator.batched_free(objs, update_stats=False)
                return None

            result: list[MemoryObj] = []
            for _, objs in allocated:
                result.extend(objs)
            return result

    def _free_devdax_obj(self, memory_obj: MemoryObj) -> None:
        with self.host_mem_lock:
            arena = self._arena_for_obj_locked(memory_obj)
            arena.allocator.free(memory_obj)
            if isinstance(memory_obj, TensorMemoryObj):
                memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)
            self._reap_draining_arenas_locked()

    def _batched_free_devdax_objs(
        self,
        memory_objs: List[MemoryObj],
        update_stats: bool,
    ) -> None:
        with self.host_mem_lock:
            # Resolve each object's owning arena before freeing invalidates it.
            groups: dict[int, tuple[_DevDaxArena, list[MemoryObj]]] = {}
            for memory_obj in memory_objs:
                arena = self._arena_for_obj_locked(memory_obj)
                groups.setdefault(id(arena), (arena, []))[1].append(memory_obj)
            for arena, objs in groups.values():
                arena.allocator.batched_free(objs, update_stats=update_stats)
                for memory_obj in objs:
                    if isinstance(memory_obj, TensorMemoryObj):
                        memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)
            self._reap_draining_arenas_locked()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shapes, dtypes, fmt)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            if self.local_allocator is not None:
                obj = self.local_allocator.allocate(shapes, dtypes, fmt, str(self))
                if obj is not None:
                    return obj
            return self._allocate_from_arenas(shapes, dtypes, fmt)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            local_objs: list[MemoryObj] = []
            if self.local_allocator is not None:
                local_count = min(
                    batch_size, self._local_available_count(shapes, dtypes)
                )
                if local_count:
                    local_objs = (
                        self.local_allocator.batched_allocate(
                            shapes, dtypes, local_count, fmt, str(self)
                        )
                        or []
                    )

            remaining = batch_size - len(local_objs)
            if remaining == 0:
                return local_objs

            dax_objs = self._batched_allocate_from_arenas(
                shapes, dtypes, remaining, fmt
            )
            if dax_objs is None:
                if local_objs and self.local_allocator is not None:
                    self.local_allocator.batched_free(local_objs, update_stats=False)
                return None
            return local_objs + dax_objs
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        fmt = memory_obj.meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.free(memory_obj)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            if self._is_local_obj(memory_obj):
                assert self.local_allocator is not None
                self.local_allocator.free(memory_obj)
                return
            if self._is_devdax_obj(memory_obj):
                self._free_devdax_obj(memory_obj)
                return
            raise ValueError("Memory object does not belong to DevDaxMemoryAllocator")
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        if not memory_objs:
            return

        fmt = memory_objs[0].meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.batched_free(memory_objs)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            local_objs = [
                memory_obj
                for memory_obj in memory_objs
                if self._is_local_obj(memory_obj)
            ]
            devdax_objs = [
                memory_obj
                for memory_obj in memory_objs
                if self._is_devdax_obj(memory_obj)
            ]
            if len(local_objs) + len(devdax_objs) != len(memory_objs):
                raise ValueError(
                    "One or more memory objects do not belong to DevDaxMemoryAllocator"
                )
            if local_objs:
                assert self.local_allocator is not None
                self.local_allocator.batched_free(local_objs, update_stats=update_stats)
            if devdax_objs:
                self._batched_free_devdax_objs(devdax_objs, update_stats=update_stats)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def add_device(self, device_path: str, size_in_bytes: int) -> DevDaxArenaStatus:
        """Map an additional Device-DAX device into the pool; it serves
        overflow immediately. Rejects empty/duplicate paths, non-positive
        sizes, and a closed allocator."""
        if not device_path:
            raise ValueError("device_path must be a non-empty string")
        if size_in_bytes <= 0:
            raise ValueError("size_in_bytes must be > 0")
        with self.host_mem_lock:
            if self._unregistered:
                raise RuntimeError(
                    "cannot add a device to a closed DevDaxMemoryAllocator"
                )
            for arena in self._arenas:
                if arena.device_path == device_path:
                    raise ValueError(
                        f"Device-DAX arena {device_path} is already mapped"
                    )
            arena = self._map_and_append_arena(
                device_path, size_in_bytes, is_primary=False
            )
            return arena.status()

    def remove_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
    ) -> DevDaxArenaStatus:
        """Drain-remove an arena: it stops accepting allocations and is
        unmapped once idle. Returns ``REMOVED`` if unmapped immediately, else
        ``DRAINING``. Rejects unknown paths and the primary arena."""
        if mode != DevDaxRemoveMode.DRAIN:
            raise ValueError(f"unsupported Device-DAX L1 remove mode: {mode}")
        with self.host_mem_lock:
            arena = self._find_arena_locked(device_path)
            if arena.is_primary:
                raise ValueError(
                    f"cannot remove primary Device-DAX arena {device_path}"
                )
            arena.state = DevDaxArenaState.DRAINING
            self._maybe_reap_arena_locked(arena)
            return arena.status()

    def arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return a status snapshot of every arena currently in the pool."""
        with self.host_mem_lock:
            return [arena.status() for arena in self._arenas]

    def memcheck(self) -> bool:
        local_ok = True
        if self.local_allocator is not None:
            local_ok = self.local_allocator.memcheck()
        with self.host_mem_lock:
            dax_ok = all(arena.allocator.memcheck() for arena in self._arenas)
        return local_ok and dax_ok

    def get_memory_usage(self) -> tuple[int, int]:
        local_used, local_total = self._local_memory_usage()
        dax_used = 0
        dax_total = 0
        with self.host_mem_lock:
            for arena in self._arenas:
                address_manager = arena.allocator.address_manager
                arena_total = address_manager.get_heap_size()
                dax_total += arena_total
                dax_used += arena_total - address_manager.get_free_size()
        return local_used + dax_used, local_total + dax_total

    def close(self) -> None:
        if self._unregistered:
            return
        if memory_management.torch_dev.is_available():
            memory_management.torch_dev.synchronize()

        with self.host_mem_lock:
            if any(
                arena.allocator.num_active_allocations > 0 for arena in self._arenas
            ):
                raise BufferError(
                    "cannot close DevDaxMemoryAllocator with active allocations"
                )
            for arena in self._arenas:
                self._close_arena_locked(arena)
            self._arenas = []

        if self.local_allocator is not None:
            self.local_allocator.close()
            self.local_allocator = None
        self._unregistered = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __str__(self) -> str:
        return "DevDaxMemoryAllocator"
