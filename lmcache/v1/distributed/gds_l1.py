# SPDX-License-Identifier: Apache-2.0
"""Slab-file GDS L1 backend for MP mode.

This module turns NVMe disk into the durable L1 medium in MP mode by
pre-allocating one large *slab file* on disk and treating it the same
way ``L1MemoryManager`` treats its pinned-DRAM slab today: a single
allocation, a single cuFile handle registration, and per-chunk
sub-allocations as ``(offset, size)`` pairs inside the slab.

The motivation for the single-slab design is throughput. cuFile's per-
file ``cuFileHandleRegister`` is the dominant cost when chunks are
small; the prior file-per-chunk layout paid it on every read and
write. With one slab handle held for the lifetime of the backend,
per-chunk I/O cost collapses to a single ``cuFileWriteAsync`` /
``cuFileReadAsync`` submit plus the stream-side DMA.

See ``docs/design/v1/distributed/gds_l1_backend.md`` for the full
architecture and the slab-design decision log.

Surface this module exposes:

- :class:`SlabAddressManager` — first-fit free-list allocator over
  byte offsets within the slab file. Mirrors the role of
  ``TensorMemoryAllocator`` for the slab file.
- :class:`GdsL1Backend` — owns the slab file, the cuFile handle, the
  address manager, and the persisted index of resident keys.
- :class:`GdsScratchAllocator` — tag class used for ``isinstance``
  dispatch in ``gpu_ops.py``; also the home for
  :meth:`cufile_read_into` and :meth:`cufile_write_from`.
- :class:`GdsMemoryObj` — slab-anchored ``MemoryObj``. Carries
  ``slab_offset + size`` instead of a per-chunk path. ``.tensor`` is
  always ``None``; ``.byte_array`` / ``.data_ptr`` raise.

cuFile DMA fast path is engaged by:

1. Opening the slab file once with ``O_DIRECT``
   (``GdsL1Config.use_direct_io``, default ``True``).
2. Registering it with ``cuFileHandleRegister`` once and keeping the
   handle alive for the backend's lifetime.
3. Pre-allocating the slab with ``posix_fallocate`` so cuFile writes
   never have to grow the file underneath us.

The slab medium itself is the durable cache. Eviction frees a slab
region back to the address manager. The on-disk slab survives
restart; an index file (JSON) at ``<gds_path>/lmcache_gds_index.json``
lets a fresh backend instance recover the
``ObjectKey → (offset, size)`` map.
"""

# Standard
from collections import OrderedDict
from typing import Optional, Union
import asyncio
import bisect
import ctypes
import json
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed import _cufile_async as ca
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
)
from lmcache.v1.storage_backend.path_sharder import PathSharder

logger = init_logger(__name__)


# --- Constants ------------------------------------------------------

_SLAB_FILENAME = "lmcache_gds_slab.bin"
_INDEX_FILENAME = "lmcache_gds_index.json"
_INDEX_VERSION = 1
_CUFILE_ALIGNMENT = 4096
_TORCH_DTYPE_TO_STR = {
    torch.half: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.uint8: "U8",
    torch.uint16: "U16",
    torch.uint32: "U32",
    torch.uint64: "U64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float8_e4m3fn: "F8E4M3FN",
    torch.float8_e5m2: "F8E5M2",
}
_STR_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_STR.items()}


# --- Helpers --------------------------------------------------------


def _round_up(n: int, align: int) -> int:
    """Round ``n`` up to the nearest multiple of ``align``."""
    return (n + align - 1) // align * align


def _object_key_to_string(key: ObjectKey) -> str:
    """Stable string form of an ObjectKey for index serialisation."""
    return f"{key.model_name}@{key.kv_rank}@{key.chunk_hash.hex()}@{key.cache_salt}"


def _string_to_object_key(s: str) -> Optional[ObjectKey]:
    """Inverse of :func:`_object_key_to_string`. ``None`` on parse error."""
    parts = s.split("@")
    if len(parts) != 4:
        return None
    model_name, kv_rank_str, hash_hex, cache_salt = parts
    try:
        kv_rank = int(kv_rank_str)
        chunk_hash = bytes.fromhex(hash_hex)
    except ValueError:
        return None
    try:
        return ObjectKey(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            cache_salt=cache_salt,
        )
    except (ValueError, TypeError):
        return None


def get_fstype(path: str) -> str:
    """Detect the filesystem type backing ``path`` via ``/proc/mounts``.

    Args:
        path: Filesystem path to probe.

    Returns:
        Filesystem type string (e.g. ``"ext4"``, ``"wekafs"``, ``"tmpfs"``).

    Raises:
        RuntimeError: If no mount point covers ``path``.
    """
    with open("/proc/mounts", "r") as f:
        lines = f.readlines()
    best_match = ""
    best_fstype = ""
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            mount_point, fstype = parts[1], parts[2]
            if path.startswith(mount_point) and len(mount_point) > len(best_match):
                best_match = mount_point
                best_fstype = fstype
    if not best_fstype:
        raise RuntimeError(f"GDS L1: unable to detect fstype for {path}")
    return best_fstype


def _compute_layout_bytes(layout_desc: MemoryLayoutDesc) -> int:
    """Total byte size of a ``MemoryLayoutDesc`` (sum across groups)."""
    total = 0
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        total += shape.numel() * dtype.itemsize
    return total


# --- Slab address manager -------------------------------------------


class SlabAddressManager:
    """First-fit free-list allocator over a fixed-size byte offset space.

    Used by :class:`GdsL1Backend` to track which byte ranges of the
    slab file are in use. Same data structure pattern as the
    ``TensorMemoryAllocator`` that backs CPU L1's pinned slab, but
    operating on file offsets instead of pointers.

    Args:
        total_size: Total addressable space in bytes.
        align: Allocation alignment; every ``allocate`` and ``free``
            rounds up to this multiple. Default 4 KiB to match
            cuFile's alignment requirement on ext4.

    Raises:
        ValueError: If ``total_size`` <= 0 or ``align`` is not a
            positive power of two.

    Thread-safe: all public methods take an internal lock.
    """

    def __init__(self, total_size: int, align: int = _CUFILE_ALIGNMENT) -> None:
        if total_size <= 0:
            raise ValueError(
                f"SlabAddressManager: total_size must be > 0, got {total_size}"
            )
        if align <= 0 or (align & (align - 1)) != 0:
            raise ValueError(
                "SlabAddressManager: align must be a positive power of two, "
                f"got {align}"
            )
        self._total = total_size
        self._align = align
        self._lock = threading.Lock()
        # Sorted list of (offset, size) free regions. Starts as one
        # region covering the whole slab.
        self._free: list[tuple[int, int]] = [(0, total_size)]
        self._used = 0

    @property
    def total_bytes(self) -> int:
        return self._total

    def used_bytes(self) -> int:
        """Total bytes currently reserved by callers."""
        with self._lock:
            return self._used

    def free_bytes(self) -> int:
        """Bytes still available for ``allocate``."""
        with self._lock:
            return self._total - self._used

    def allocate(self, size: int) -> Optional[int]:
        """Reserve ``size`` bytes and return the offset, or ``None`` on OOM.

        First-fit: scans the free list for the first region big enough.

        Args:
            size: Number of bytes to reserve. Rounded up to ``align``.

        Returns:
            Byte offset of the allocation, or ``None`` if no region
            is big enough.

        Raises:
            ValueError: If ``size`` <= 0.
        """
        if size <= 0:
            raise ValueError(
                f"SlabAddressManager.allocate: size must be > 0, got {size}"
            )
        size = _round_up(size, self._align)
        with self._lock:
            for i, (off, free_size) in enumerate(self._free):
                if free_size >= size:
                    if free_size == size:
                        del self._free[i]
                    else:
                        self._free[i] = (off + size, free_size - size)
                    self._used += size
                    return off
            return None

    def free(self, offset: int, size: int) -> None:
        """Return ``[offset, offset + size)`` to the free list.

        Adjacent regions are coalesced so the free list stays compact.

        Args:
            offset: Start of the region returned by :meth:`allocate`.
            size: Size of the region (will be rounded up to ``align``).
        """
        if size <= 0:
            return
        size = _round_up(size, self._align)
        with self._lock:
            self._used = max(0, self._used - size)
            offsets = [o for o, _ in self._free]
            idx = bisect.bisect_left(offsets, offset)
            self._free.insert(idx, (offset, size))
            if idx + 1 < len(self._free):
                nxt_off, nxt_size = self._free[idx + 1]
                if offset + size == nxt_off:
                    self._free[idx] = (offset, size + nxt_size)
                    del self._free[idx + 1]
            if idx > 0:
                prev_off, prev_size = self._free[idx - 1]
                cur_off, cur_size = self._free[idx]
                if prev_off + prev_size == cur_off:
                    self._free[idx - 1] = (prev_off, prev_size + cur_size)
                    del self._free[idx]

    def mark_used(self, offset: int, size: int) -> None:
        """Mark ``[offset, offset + size)`` as already in use.

        Called during startup index replay so the free list reflects
        regions occupied by entries we recovered from the index file.

        Args:
            offset: Start of the resident region.
            size: Size of the resident region.

        Raises:
            RuntimeError: If the range overlaps an already-used region
                or extends beyond the slab.
        """
        if size <= 0:
            return
        size = _round_up(size, self._align)
        with self._lock:
            for i, (off, free_size) in enumerate(self._free):
                if off <= offset and offset + size <= off + free_size:
                    head = (off, offset - off)
                    tail = (offset + size, off + free_size - (offset + size))
                    del self._free[i]
                    insertions = []
                    if head[1] > 0:
                        insertions.append(head)
                    if tail[1] > 0:
                        insertions.append(tail)
                    for j, r in enumerate(insertions):
                        self._free.insert(i + j, r)
                    self._used += size
                    return
            raise RuntimeError(
                f"SlabAddressManager.mark_used: range [{offset}, {offset + size}) "
                f"is not free (slab is corrupted or index is stale)"
            )


# --- GdsMemoryObj ---------------------------------------------------


class GdsMemoryObj(MemoryObj):
    """Slab-anchored ``MemoryObj`` for the GDS L1 backend.

    Carries ``(slab_offset, size)`` within the backend's single slab
    file plus the standard ``MemoryObjMetadata``. ``.tensor`` is
    always ``None``; ``.byte_array`` and ``.data_ptr`` raise. Under
    the exclusive-L2 mode that GDS L1 enforces, neither field is read
    on the GDS path — the data path is the ``gpu_ops`` dispatch + the
    ``gpu_buffer`` parameter, not field access on the MemoryObj.

    Args:
        key: The ObjectKey this MemoryObj represents.
        slab_offset: Byte offset into the slab file where this chunk's
            payload starts.
        size: Payload size in bytes.
        metadata: Standard :class:`MemoryObjMetadata`.
        parent_allocator: The :class:`GdsScratchAllocator` returned by
            :meth:`parent` so ``gpu_ops``'s ``isinstance`` dispatch
            picks the cuFile path.
    """

    def __init__(
        self,
        key: ObjectKey,
        slab_offset: int,
        size: int,
        metadata: MemoryObjMetadata,
        parent_allocator: "GdsScratchAllocator",
    ) -> None:
        super().__init__(metadata)
        self.key = key
        self.slab_offset = slab_offset
        self.size = size
        self._parent_allocator = parent_allocator
        self._lock = threading.Lock()
        self._valid = True

    def invalidate(self) -> None:
        self._valid = False

    def is_valid(self) -> bool:
        return self._valid

    def get_size(self) -> int:
        return self.size

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> Optional[torch.dtype]:
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        if self.meta.shapes is not None:
            return self.meta.shapes
        return [self.meta.shape]

    def get_dtypes(self) -> list[torch.dtype]:
        if self.meta.dtypes is not None:
            return self.meta.dtypes
        if self.meta.dtype is None:
            raise RuntimeError("GdsMemoryObj.meta.dtype is None")
        return [self.meta.dtype]

    def get_memory_format(self) -> MemoryFormat:
        with self._lock:
            return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def get_num_tokens(self) -> int:
        with self._lock:
            token_dim = self.meta.fmt.token_dim()
            if token_dim < 0 or token_dim >= len(self.meta.shape):
                return 0
            return self.meta.shape[token_dim]

    def pin(self) -> bool:
        with self._lock:
            self.meta.pin_count += 1
            return True

    def unpin(self) -> bool:
        with self._lock:
            if self.meta.pin_count > 0:
                self.meta.pin_count -= 1
            return True

    def ref_count_up(self) -> None:
        with self._lock:
            self.meta.ref_count += 1

    def ref_count_down(self) -> None:
        with self._lock:
            self.meta.ref_count -= 1
            if self.meta.ref_count < 0:
                logger.warning(
                    "GdsMemoryObj for key %s: ref_count went negative (%d), clamping",
                    self.key,
                    self.meta.ref_count,
                )
                self.meta.ref_count = 0

    def get_ref_count(self) -> int:
        with self._lock:
            return self.meta.ref_count

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def is_pinned(self) -> bool:
        with self._lock:
            return self.meta.pin_count > 0

    @property
    def can_evict(self) -> bool:
        with self._lock:
            return self.meta.pin_count == 0 and self.meta.ref_count == 0

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return None

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    @property
    def byte_array(self) -> bytes:
        raise NotImplementedError(
            f"GdsMemoryObj(slab_offset={self.slab_offset}).byte_array is not "
            "supported; bytes live in the GDS slab file and the staging "
            "buffer is registered VRAM (no buffer protocol)."
        )

    @property
    def data_ptr(self) -> int:
        raise NotImplementedError(
            f"GdsMemoryObj(slab_offset={self.slab_offset}).data_ptr is not "
            "supported; GDS reads/writes use gpu_buffer.data_ptr() via the "
            "gpu_ops dispatch, never the MemoryObj's data_ptr."
        )

    def parent(self) -> Optional[MemoryAllocatorInterface]:
        return self._parent_allocator


# --- GdsScratchAllocator -------------------------------------------


class GdsScratchAllocator(MemoryAllocatorInterface):
    """Tag class for ``gpu_ops`` dispatch + home of cuFile I/O helpers.

    Holds one or more cuFile-registered ``tmp_gpu_buffer`` regions and
    routes every I/O through the backend's single slab handle. The
    actual ``cuFileReadAsync`` / ``cuFileWriteAsync`` call uses the
    chunk's ``slab_offset`` as ``file_offset``, the slot's offset
    inside its containing registered region as ``dev_offset``, and
    that region's base pointer as ``buf_base``.

    Multi-registration striping (path #2 from the design notes):
    cuFile on ext4 caps a single ``cuFileBufRegister`` at the host's
    nvidia-fs slab size (16 MiB on the reference host). To stage more
    concurrent in-flight I/Os without P2P mode, ``register_gpu_buffer``
    is called multiple times, each with a ≤16 MiB sub-view of the
    overall ``tmp_gpu_buffer_``. Each registration claims its own
    nvidia-fs slab, allowing N concurrent ``cuFileReadAsync`` /
    ``cuFileWriteAsync`` submissions before the next stream sync.

    Args:
        backend: The owning :class:`GdsL1Backend`. The allocator
            reaches back into it for the slab handle and the address
            manager.
    """

    def __init__(self, backend: "GdsL1Backend") -> None:
        self._backend = backend
        # Parallel lists; index i describes one registered region.
        # Sorted by ``_base_ptrs`` ascending so ``_resolve_buffer``
        # can binary-search.
        self._buffers: list[torch.Tensor] = []
        self._base_ptrs: list[int] = []
        self._nbytes: list[int] = []
        # Cached scalar for callers that want a single representative
        # pointer (e.g. logs). Not used in the hot path.
        self._first_base_ptr: int = 0

    @property
    def registered_base_ptr(self) -> int:
        """Base pointer of the first registered slot (legacy accessor).

        Multi-registration consumers should use the ``buf_base``
        carried into :meth:`cufile_read_into` / :meth:`cufile_write_from`
        via the gpu_buffer's residency, not this scalar.
        """
        return self._first_base_ptr

    @property
    def registered_nbytes(self) -> int:
        """Total bytes across all registered regions."""
        return sum(self._nbytes)

    @property
    def num_registered_buffers(self) -> int:
        """How many distinct cuFile registrations the allocator holds."""
        return len(self._buffers)

    @property
    def has_registered_buffer(self) -> bool:
        """``True`` once at least one ``register_gpu_buffer`` succeeded."""
        return len(self._buffers) > 0

    # --- Buffer registration ----------------------------------------

    def register_gpu_buffer(self, buffer: torch.Tensor) -> None:
        """Register ``buffer`` with cuFile and the backend's CUDA stream.

        Can be called multiple times; each call adds one more
        registered region. nvidia-fs caps each call at 16 MiB, but
        multiple registrations stripe across the slab pool's 16 MiB
        tier and unlock N-way cuFile concurrency.

        Args:
            buffer: Must be contiguous, on CUDA, 4 KiB-aligned in size,
                and no larger than 16 MiB (single nvidia-fs slab).

        Raises:
            ValueError: If ``buffer`` violates the contiguity, device,
                alignment, or size constraints.
            RuntimeError: If ``buffer``'s pointer range overlaps an
                already-registered region. Overlapping registrations
                would make ``_resolve_buffer`` ambiguous.
        """
        if not buffer.is_cuda:
            raise ValueError("register_gpu_buffer: buffer must be on CUDA")
        if not buffer.is_contiguous():
            raise ValueError("register_gpu_buffer: buffer must be contiguous")
        nbytes = buffer.numel() * buffer.element_size()
        if nbytes % _CUFILE_ALIGNMENT != 0:
            raise ValueError(
                f"register_gpu_buffer: buffer size {nbytes} is not a multiple of "
                f"{_CUFILE_ALIGNMENT} (cuFile requires 4 KiB alignment)."
            )
        if nbytes > 16 * 1024 * 1024:
            raise ValueError(
                f"register_gpu_buffer: a single cuFileBufRegister is capped at "
                f"16 MiB on hosts with the standard nvidia-fs slab config; got "
                f"{nbytes} bytes. Reduce lmcache_chunk_size or split into "
                "multiple smaller registrations."
            )
        base = buffer.data_ptr()
        for existing_base, existing_n in zip(
            self._base_ptrs, self._nbytes, strict=True
        ):
            if base < existing_base + existing_n and existing_base < base + nbytes:
                raise RuntimeError(
                    f"register_gpu_buffer: new region [0x{base:x}, "
                    f"0x{base + nbytes:x}) overlaps existing region "
                    f"[0x{existing_base:x}, 0x{existing_base + existing_n:x})"
                )
        if self._backend.use_gds:
            ca.register_buffer(buffer)
            # Stream registration is backend-owned (see backend.close
            # for why) — the first registered buffer triggers it.
            self._backend.ensure_stream_registered()
            log_label = "cuFile"
        else:
            log_label = "POSIX fallback (no cuFile registration)"
        # Insert keeping ``_base_ptrs`` sorted so ``_resolve_buffer``
        # can use ``bisect``.
        idx = bisect.bisect_left(self._base_ptrs, base)
        self._buffers.insert(idx, buffer)
        self._base_ptrs.insert(idx, base)
        self._nbytes.insert(idx, nbytes)
        if not self._first_base_ptr:
            self._first_base_ptr = base
        logger.info(
            "GdsScratchAllocator: registered %d bytes at 0x%x via %s "
            "(total registrations: %d)",
            nbytes,
            base,
            log_label,
            len(self._buffers),
        )

    def deregister_gpu_buffer(self) -> None:
        """Deregister every previously-registered region + stream.

        Idempotent: safe to call from ``GdsL1Backend.close`` even if
        registration never happened.
        """
        if not self._buffers:
            return
        if self._backend.use_gds:
            torch_dev.synchronize(device=self._buffers[0].device)
            for buf in self._buffers:
                try:
                    ca.deregister_buffer(buf)
                except Exception as e:
                    logger.warning("deregister_gpu_buffer: %s", e)
        # Stream stays registered for the backend's lifetime; the
        # backend's close() deregisters it once on shutdown.
        self._buffers.clear()
        self._base_ptrs.clear()
        self._nbytes.clear()
        self._first_base_ptr = 0

    # --- cuFile I/O (dispatch target for gpu_ops) -------------------

    def cufile_read_into(
        self,
        memory_obj: "GdsMemoryObj",
        gpu_buffer: torch.Tensor,
    ) -> None:
        """Submit a ``cuFileReadAsync`` from the slab → registered slot.

        No per-call sync: the read is enqueued on the current torch
        CUDA stream and the subsequent kernel on the same stream
        (``multi_layer_block_kv_transfer``) auto-waits via stream
        ordering.

        Args:
            memory_obj: The :class:`GdsMemoryObj` to read.
            gpu_buffer: A view into one of the registered staging
                regions; the matching registration is resolved by
                pointer comparison.

        Raises:
            RuntimeError: If no buffer has been registered.
            ValueError: If ``gpu_buffer`` lies outside every registered
                region or is smaller than the chunk.
        """
        base_ptr, dev_offset = self._resolve_buffer(gpu_buffer)
        nbytes = memory_obj.size
        if nbytes > gpu_buffer.numel() * gpu_buffer.element_size():
            raise ValueError(
                f"cufile_read_into: chunk size {nbytes} exceeds gpu_buffer "
                f"capacity {gpu_buffer.numel() * gpu_buffer.element_size()}"
            )
        self._backend.slab_read(memory_obj.slab_offset, nbytes, dev_offset, base_ptr)

    def cufile_write_from(
        self,
        memory_obj: "GdsMemoryObj",
        gpu_buffer: torch.Tensor,
    ) -> None:
        """Submit a ``cuFileWriteAsync`` from registered slot → slab.

        Same stream-ordered semantics as :meth:`cufile_read_into`.
        Registers the chunk in the backend's in-memory index after
        the submit; persistence to the index file happens at backend
        close (or on next periodic flush).

        Args:
            memory_obj: The :class:`GdsMemoryObj` to write.
            gpu_buffer: A view into one of the registered staging
                regions.
        """
        base_ptr, dev_offset = self._resolve_buffer(gpu_buffer)
        nbytes = memory_obj.size
        if nbytes > gpu_buffer.numel() * gpu_buffer.element_size():
            raise ValueError(
                f"cufile_write_from: chunk size {nbytes} exceeds gpu_buffer "
                f"capacity {gpu_buffer.numel() * gpu_buffer.element_size()}"
            )
        self._backend.slab_write(memory_obj.slab_offset, nbytes, dev_offset, base_ptr)
        self._backend.record_entry(memory_obj)

    # --- MemoryAllocatorInterface (mostly no-ops) -------------------

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        raise NotImplementedError(
            "GdsScratchAllocator.allocate: use "
            "GdsL1Backend.create_memory_obj(key, layout_desc) instead"
        )

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[list[MemoryObj]]:
        raise NotImplementedError(
            "GdsScratchAllocator.batched_allocate: use "
            "GdsL1Backend.create_memory_obj(key, layout_desc) instead"
        )

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        if isinstance(memory_obj, GdsMemoryObj):
            self._backend.free_entry_from_index(memory_obj)

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        for mo in memory_objs:
            self.free(mo)

    def _resolve_buffer(self, gpu_buffer: torch.Tensor) -> tuple[int, int]:
        """Find which registered region ``gpu_buffer`` belongs to.

        Returns:
            ``(base_ptr, dev_offset)`` where ``base_ptr`` is the
            matching registration's base pointer (used as the
            ``buf_base`` argument to ``cuFileReadAsync`` /
            ``cuFileWriteAsync``) and ``dev_offset`` is
            ``gpu_buffer.data_ptr() - base_ptr``.

        Raises:
            RuntimeError: If no buffer has been registered.
            ValueError: If ``gpu_buffer`` does not lie entirely inside
                any single registered region.
        """
        if not self._base_ptrs:
            raise RuntimeError(
                "GdsScratchAllocator: no GPU buffer has been registered yet"
            )
        ptr = gpu_buffer.data_ptr()
        # ``_base_ptrs`` is sorted ascending. The candidate region is
        # the rightmost one whose base ≤ ptr.
        idx = bisect.bisect_right(self._base_ptrs, ptr) - 1
        if idx < 0:
            raise ValueError(
                f"GdsScratchAllocator: gpu_buffer pointer 0x{ptr:x} is below "
                f"the lowest registered base 0x{self._base_ptrs[0]:x}"
            )
        base = self._base_ptrs[idx]
        nbytes = self._nbytes[idx]
        offset = ptr - base
        if offset < 0 or offset >= nbytes:
            raise ValueError(
                f"GdsScratchAllocator: gpu_buffer pointer 0x{ptr:x} is "
                f"outside every registered region (closest candidate: "
                f"[0x{base:x}, 0x{base + nbytes:x}))"
            )
        return base, offset


# --- GdsL1Backend ---------------------------------------------------


class _IndexEntry:
    """One record in the on-disk index.

    Small enough to keep as a tiny class rather than a full dataclass
    so it serialises cheaply to JSON.
    """

    __slots__ = ("slab_offset", "size", "shape", "dtype", "fmt")

    def __init__(
        self,
        slab_offset: int,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> None:
        self.slab_offset = slab_offset
        self.size = size
        self.shape = shape
        self.dtype = dtype
        self.fmt = fmt


class GdsL1Backend:
    """Slab-file GDS L1 backend.

    Owns the slab file, the cuFile handle for it, the address
    manager, and the in-memory ``ObjectKey → _IndexEntry`` map. All
    per-chunk I/O routes through this single handle — no per-chunk
    open/close, no per-chunk ``cuFileHandleRegister``.

    Lifecycle:

    1. ``__init__``:
       - Resolves ``gds_path`` via :class:`PathSharder`.
       - Opens or creates ``<gds_path>/lmcache_gds_slab.bin``,
         ``posix_fallocate``s it to ``slab_size_gb`` GiB.
       - Registers the slab fd with cuFile (single
         ``cuFileHandleRegister``).
       - Loads ``<gds_path>/lmcache_gds_index.json`` if present and
         replays the recorded entries into the address manager.
    2. Steady state:
       - ``lookup`` checks the index.
       - ``create_memory_obj`` ``allocate``s a slab region and
         returns a :class:`GdsMemoryObj`.
       - ``slab_read`` / ``slab_write`` submit one ``cuFileReadAsync``
         / ``WriteAsync`` per call, no sync.
       - ``record_entry`` / ``free_entry_from_index`` keep the index in sync.
    3. ``close``: stream-syncs, writes the index file, deregisters
       the cuFile handle, closes the slab fd.

    Args:
        config: :class:`GdsL1Config` with ``gds_path``,
            ``slab_size_gb``, ``use_direct_io``, etc.
        loop: An asyncio event loop. Currently unused — preserved for
            API stability with previous versions.
        dst_device: Target GPU device string used by
            :class:`PathSharder`.
    """

    def __init__(
        self,
        config: GdsL1Config,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
    ) -> None:
        if not config.gds_path:
            raise ValueError("GdsL1Backend requires gds_path to be set")
        if not dst_device.startswith("cuda"):
            raise ValueError(f"GdsL1Backend requires cuda dst_device, got {dst_device}")
        self.config = config
        self._loop = loop
        self.dst_device = dst_device

        sharder = PathSharder(
            raw_csv=config.gds_path,
            strategy=config.gds_path_sharding,
            dst_device=dst_device,
            create_dirs=True,
        )
        self.gds_paths: list[str] = sharder.all_paths
        self.gds_path: str = sharder.selected
        self.fstype: str = get_fstype(self.gds_path)
        logger.info(
            "GdsL1Backend: fstype=%r path=%r (%d configured)",
            self.fstype,
            self.gds_path,
            len(self.gds_paths),
        )

        self.use_gds: bool = config.use_gds
        if self.fstype in ("tmpfs", "overlayfs") and config.use_gds:
            logger.info("GdsL1Backend: auto-disabling cuFile on fstype=%r", self.fstype)
            self.use_gds = False

        self._slab_path = os.path.join(self.gds_path, _SLAB_FILENAME)
        self._index_path = os.path.join(self.gds_path, _INDEX_FILENAME)
        self._slab_size = _round_up(
            int(config.slab_size_gb * (1 << 30)), _CUFILE_ALIGNMENT
        )

        self._slab_handle: Optional[ca.AsyncHandle] = None
        self._posix_fd: int = -1
        self._cudart: Optional[ctypes.CDLL] = None
        # cuFile requires the stream that ``cuFileReadAsync`` /
        # ``cuFileWriteAsync`` will run on to be registered exactly
        # once via ``cuFileStreamRegister``. We register the caller's
        # current torch CUDA stream on first use and keep it for the
        # backend's lifetime. cuFile + the scatter kernel share this
        # one stream — same pattern CPU L1 uses with
        # ``cudaMemcpyAsync`` — so stream-ordering naturally
        # serialises the per-batch ``cuFile reads → kernel`` chain
        # without needing CUDA-event barriers.
        self._registered_stream: Optional[int] = None
        self._stream_lock = threading.Lock()
        # cuFileReadAsync/WriteAsync writes through host-side
        # ctypes storage held by each Submission. The store has to
        # outlive the async op — cuFile dereferences these pointers
        # when the CUDA stream eventually executes the op, not at
        # submit time. We keep every Submission alive for the
        # backend's lifetime; each one is ~40 bytes so even a
        # million-chunk run is sub-megabyte.
        self._pending_submissions: list[ca.Submission] = []
        self._submissions_lock = threading.Lock()
        if self.use_gds:
            self._open_and_register_slab(config.use_direct_io)
        else:
            self._open_slab_posix()

        self._address_manager = SlabAddressManager(
            total_size=self._slab_size, align=_CUFILE_ALIGNMENT
        )
        self._index_lock = threading.Lock()
        self._index: OrderedDict[ObjectKey, _IndexEntry] = OrderedDict()
        self._load_index()

        self.scratch_allocator = GdsScratchAllocator(self)
        logger.info(
            "GdsL1Backend: slab=%s size=%.1f GiB, %d resident entries",
            self._slab_path,
            config.slab_size_gb,
            len(self._index),
        )

    # --- Public API -------------------------------------------------

    def lookup(self, keys: list[ObjectKey]) -> list[bool]:
        """Return whether each key is in the slab's index."""
        with self._index_lock:
            return [k in self._index for k in keys]

    def create_memory_obj(
        self,
        key: ObjectKey,
        layout_desc: MemoryLayoutDesc,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[GdsMemoryObj]:
        """Reserve a slab region for ``key`` and return a fresh MemoryObj.

        Args:
            key: The ObjectKey being inserted.
            layout_desc: Layout descriptor for the chunk's payload.
            fmt: The chunk's memory format.

        Returns:
            A :class:`GdsMemoryObj` pointing at a freshly-reserved
            slab region, or ``None`` if the slab is full (caller
            should treat this as an L1 OOM and trigger eviction).
        """
        shape = layout_desc.shapes[0]
        dtype = layout_desc.dtypes[0]
        nbytes = _compute_layout_bytes(layout_desc)
        slab_offset = self._address_manager.allocate(nbytes)
        if slab_offset is None:
            return None
        meta = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=slab_offset,
            phy_size=_round_up(nbytes, _CUFILE_ALIGNMENT),
            ref_count=0,
            pin_count=0,
            fmt=fmt,
            shapes=list(layout_desc.shapes),
            dtypes=list(layout_desc.dtypes),
        )
        return GdsMemoryObj(
            key=key,
            slab_offset=slab_offset,
            size=nbytes,
            metadata=meta,
            parent_allocator=self.scratch_allocator,
        )

    def create_memory_obj_from_index(self, key: ObjectKey) -> Optional[GdsMemoryObj]:
        """Synthesise a :class:`GdsMemoryObj` for an already-resident key.

        Used by ``L1Manager.reserve_read`` on the fill-on-miss path —
        the key is in the durable index, so we return a fresh
        ``GdsMemoryObj`` pointing at the recorded slab region.

        Args:
            key: The ObjectKey to look up.

        Returns:
            A fresh :class:`GdsMemoryObj` if ``key`` is resident; ``None``
            otherwise.
        """
        with self._index_lock:
            entry = self._index.get(key)
        if entry is None:
            return None
        meta = MemoryObjMetadata(
            shape=entry.shape,
            dtype=entry.dtype,
            address=entry.slab_offset,
            phy_size=_round_up(entry.size, _CUFILE_ALIGNMENT),
            ref_count=0,
            pin_count=0,
            fmt=entry.fmt,
        )
        return GdsMemoryObj(
            key=key,
            slab_offset=entry.slab_offset,
            size=entry.size,
            metadata=meta,
            parent_allocator=self.scratch_allocator,
        )

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` of the slab.

        Feeds the existing L1 eviction-controller signal.
        """
        return self._address_manager.used_bytes(), self._slab_size

    def record_entry(self, memory_obj: GdsMemoryObj) -> None:
        """Insert ``memory_obj`` into the in-memory index.

        Called by ``GdsScratchAllocator.cufile_write_from`` after the
        async write submit. Persistence to the on-disk index file is
        deferred to :meth:`close`.

        Args:
            memory_obj: The freshly-written :class:`GdsMemoryObj`.
        """
        if memory_obj.meta.dtype is None:
            raise RuntimeError(
                f"GdsL1Backend.record_entry: memory_obj.meta.dtype is None for "
                f"key {memory_obj.key}"
            )
        entry = _IndexEntry(
            slab_offset=memory_obj.slab_offset,
            size=memory_obj.size,
            shape=memory_obj.meta.shape,
            dtype=memory_obj.meta.dtype,
            fmt=memory_obj.meta.fmt,
        )
        with self._index_lock:
            existing = self._index.get(memory_obj.key)
            if existing is not None:
                self._address_manager.free(existing.slab_offset, existing.size)
            self._index[memory_obj.key] = entry

    def free_entry_from_index(self, memory_obj: GdsMemoryObj) -> None:
        """Free the slab region and drop the index entry."""
        with self._index_lock:
            entry = self._index.pop(memory_obj.key, None)
        if entry is not None:
            self._address_manager.free(entry.slab_offset, entry.size)

    def free_entry(self, memory_obj: GdsMemoryObj) -> None:
        """Return a slab region without an idex."""
        self._address_manager.free(memory_obj.slab_offset, memory_obj.size)

    def slab_read(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """Submit one ``cuFileReadAsync`` against the slab handle.

        Stream-ordered: no per-call sync. The bytes land in the
        registered GPU buffer at ``buf_base + dev_offset`` before any
        subsequent kernel on the same stream runs.

        Args:
            slab_offset: Byte offset within the slab file.
            size: Bytes to read.
            dev_offset: Byte offset within the registered region whose
                base is ``buf_base``.
            buf_base: Base pointer of the cuFile-registered region the
                read should land in. Picked by
                ``GdsScratchAllocator._resolve_buffer`` so multi-
                registration striping ends up at the correct region.
        """
        if not self.use_gds:
            self._posix_slab_read(slab_offset, size, dev_offset, buf_base)
            return
        if self._slab_handle is None:
            raise RuntimeError("GdsL1Backend.slab_read: slab handle not open")
        if self._registered_stream is None:
            raise RuntimeError(
                "GdsL1Backend.slab_read: cuFile stream not registered; "
                "call register_gpu_buffer first"
            )
        sub = self._slab_handle.read_async(
            buf_base,
            size,
            slab_offset,
            dev_offset,
            self._registered_stream,
        )
        with self._submissions_lock:
            self._pending_submissions.append(sub)

    def slab_write(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """Submit one ``cuFileWriteAsync`` against the slab handle.

        Args:
            slab_offset: Byte offset within the slab file.
            size: Bytes to write.
            dev_offset: Byte offset within the registered region whose
                base is ``buf_base``.
            buf_base: Base pointer of the cuFile-registered region the
                write should read from.
        """
        if not self.use_gds:
            self._posix_slab_write(slab_offset, size, dev_offset, buf_base)
            return
        if self._slab_handle is None:
            raise RuntimeError("GdsL1Backend.slab_write: slab handle not open")
        if self._registered_stream is None:
            raise RuntimeError(
                "GdsL1Backend.slab_write: cuFile stream not registered; "
                "call register_gpu_buffer first"
            )
        sub = self._slab_handle.write_async(
            buf_base,
            size,
            slab_offset,
            dev_offset,
            self._registered_stream,
        )
        with self._submissions_lock:
            self._pending_submissions.append(sub)

    def close(self) -> None:
        """Sync stream, persist index, deregister cuFile state, close slab."""
        if self.scratch_allocator._buffers:  # noqa: SLF001
            torch_dev.synchronize(
                device=self.scratch_allocator._buffers[0].device  # noqa: SLF001
            )
        self._persist_index()
        self.scratch_allocator.deregister_gpu_buffer()
        with self._stream_lock:
            if self._registered_stream is not None and self.use_gds:
                try:
                    ca.deregister_stream(self._registered_stream)
                except Exception as e:
                    logger.warning("GdsL1Backend.close: deregister_stream: %s", e)
                self._registered_stream = None
        if self._slab_handle is not None:
            try:
                self._slab_handle.close()
            except Exception as e:
                logger.warning("GdsL1Backend.close: slab handle close failed: %s", e)
            self._slab_handle = None
        if self._posix_fd != -1:
            try:
                os.close(self._posix_fd)
            except OSError:
                pass
            self._posix_fd = -1
        logger.info("GdsL1Backend: closed")

    def ensure_stream_registered(self) -> None:
        """Register the caller's current torch CUDA stream with cuFile.

        Idempotent: the first call records the stream that
        ``cuFileReadAsync`` / ``cuFileWriteAsync`` will run on; later
        calls are no-ops. cuFile mis-handles repeated register /
        deregister cycles on the same stream (assertion
        ``tmp == sinfo`` in ``putStreamOp``), so we register exactly
        once and keep it for the backend's lifetime.

        Sharing the kernel's stream (rather than using a dedicated
        cuFile stream) is intentional: cuFile + scatter kernel share
        the GPU's SMs anyway because ``cuFileCopyGpu`` (cuFile's
        internal bounce-buffer copy kernel) saturates the SMs while
        running. Putting them on different streams couldn't unlock
        SM parallelism that doesn't exist, so we save the
        cross-stream event handshake cost by keeping one stream.
        """
        if not self.use_gds:
            return
        with self._stream_lock:
            if self._registered_stream is not None:
                return
            raw_stream = torch_dev.current_stream().cuda_stream
            ca.register_stream(raw_stream)
            self._registered_stream = raw_stream

    def get_hot_cache_size(self) -> int:
        """Number of resident keys. Used by tests + ``gds-check``."""
        with self._index_lock:
            return len(self._index)

    def wait_for_scan(self, timeout: float = 0.0) -> None:
        """Compat stub: the slab design has no async scan to wait for."""
        return

    # --- Internal ---------------------------------------------------

    def _open_and_register_slab(self, use_direct_io: bool) -> None:
        """Create / open the slab file and register it with cuFile.

        Pre-allocates to ``self._slab_size`` so subsequent
        ``cuFileWriteAsync``s don't have to grow the file.

        Args:
            use_direct_io: If ``True``, opens with ``O_DIRECT`` (required
                for the cuFile GDS DMA fast path on ext4).
        """
        # Create the file first with a regular fd so posix_fallocate
        # works even when we'd otherwise open with O_DIRECT (some
        # kernels disallow fallocate via O_DIRECT fds).
        creator_fd = os.open(self._slab_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            current_size = os.fstat(creator_fd).st_size
            if current_size < self._slab_size:
                os.posix_fallocate(creator_fd, 0, self._slab_size)
        finally:
            os.close(creator_fd)
        flags = os.O_RDWR
        if use_direct_io:
            flags |= os.O_DIRECT
        fd = os.open(self._slab_path, flags)
        try:
            # Third Party
            from cufile.bindings import cuFileHandleRegister

            handle = cuFileHandleRegister(fd)
        except Exception:
            os.close(fd)
            raise
        # Build an AsyncHandle by hand so we share the same close/
        # read_async/write_async surface as elsewhere in this module.
        self._slab_handle = ca.AsyncHandle.__new__(ca.AsyncHandle)
        self._slab_handle._fd = fd  # noqa: SLF001
        self._slab_handle._handle = handle  # noqa: SLF001
        self._slab_handle.path = self._slab_path
        self._slab_handle.writable = True
        logger.info(
            "GdsL1Backend: slab opened at %s (O_DIRECT=%s), cuFile handle registered",
            self._slab_path,
            use_direct_io,
        )

    def _open_slab_posix(self) -> None:
        """POSIX fallback: open the slab without cuFile registration."""
        creator_fd = os.open(self._slab_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            current_size = os.fstat(creator_fd).st_size
            if current_size < self._slab_size:
                os.posix_fallocate(creator_fd, 0, self._slab_size)
        finally:
            os.close(creator_fd)
        self._posix_fd = os.open(self._slab_path, os.O_RDWR)
        logger.info("GdsL1Backend: slab opened at %s (POSIX fallback)", self._slab_path)

    def _posix_slab_read(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """POSIX fallback read: pread + cudaMemcpy H2D into ``buf_base``."""
        if self._cudart is None:
            self._cudart = ctypes.CDLL("libcudart.so")
        host_bytes = os.pread(self._posix_fd, size, slab_offset)
        if len(host_bytes) != size:
            raise RuntimeError(
                f"POSIX slab_read: short read at offset {slab_offset}, "
                f"got {len(host_bytes)} expected {size}"
            )
        host_buf = ctypes.create_string_buffer(host_bytes, size)
        res = self._cudart.cudaMemcpy(
            ctypes.c_void_p(buf_base + dev_offset),
            ctypes.cast(host_buf, ctypes.c_void_p),
            ctypes.c_size_t(size),
            ctypes.c_int(1),  # cudaMemcpyHostToDevice
        )
        if res != 0:
            raise RuntimeError(f"POSIX slab_read cudaMemcpy failed: {res}")

    def _posix_slab_write(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """POSIX fallback write: cudaMemcpy D2H from ``buf_base`` + pwrite."""
        if self._cudart is None:
            self._cudart = ctypes.CDLL("libcudart.so")
        host_buf = ctypes.create_string_buffer(size)
        res = self._cudart.cudaMemcpy(
            ctypes.cast(host_buf, ctypes.c_void_p),
            ctypes.c_void_p(buf_base + dev_offset),
            ctypes.c_size_t(size),
            ctypes.c_int(2),  # cudaMemcpyDeviceToHost
        )
        if res != 0:
            raise RuntimeError(f"POSIX slab_write cudaMemcpy failed: {res}")
        written = os.pwrite(self._posix_fd, host_buf.raw[:size], slab_offset)
        if written != size:
            raise RuntimeError(f"POSIX slab_write: short write at offset {slab_offset}")

    def _load_index(self) -> None:
        """Read the on-disk index and replay it into the address manager.

        Missing or corrupted index → start empty (the slab data is
        effectively orphaned). This is best-effort: production
        deployments should manage the slab + index as a unit.
        """
        if not os.path.exists(self._index_path):
            return
        start = time.perf_counter()
        try:
            with open(self._index_path, "rb") as f:
                doc = json.loads(f.read())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "GdsL1Backend: failed to read index %s: %s — starting empty",
                self._index_path,
                e,
            )
            return
        if doc.get("version") != _INDEX_VERSION:
            logger.warning(
                "GdsL1Backend: index version mismatch (got %r, want %d) — ignoring",
                doc.get("version"),
                _INDEX_VERSION,
            )
            return
        if doc.get("slab_size") != self._slab_size:
            logger.warning(
                "GdsL1Backend: index slab_size mismatch (got %d, want %d) — ignoring",
                doc.get("slab_size"),
                self._slab_size,
            )
            return
        loaded = 0
        for key_str, raw in doc.get("entries", {}).items():
            key = _string_to_object_key(key_str)
            if key is None:
                continue
            try:
                shape = torch.Size(raw["shape"])
                dtype = _STR_TO_TORCH_DTYPE[raw["dtype"]]
                fmt = MemoryFormat(raw["fmt"])
                slab_offset = int(raw["offset"])
                size = int(raw["size"])
            except (KeyError, ValueError):
                continue
            entry = _IndexEntry(
                slab_offset=slab_offset,
                size=size,
                shape=shape,
                dtype=dtype,
                fmt=fmt,
            )
            try:
                self._address_manager.mark_used(slab_offset, size)
            except RuntimeError as e:
                logger.warning("GdsL1Backend._load_index: %s", e)
                continue
            self._index[key] = entry
            loaded += 1
        elapsed = time.perf_counter() - start
        logger.info("GdsL1Backend: loaded %d index entries in %.2fs", loaded, elapsed)

    def _persist_index(self) -> None:
        """Write the current in-memory index to ``self._index_path``."""
        with self._index_lock:
            entries = {
                _object_key_to_string(k): {
                    "offset": e.slab_offset,
                    "size": e.size,
                    "shape": list(e.shape),
                    "dtype": _TORCH_DTYPE_TO_STR.get(e.dtype, "F16"),
                    "fmt": e.fmt.value,
                }
                for k, e in self._index.items()
            }
        doc = {
            "version": _INDEX_VERSION,
            "slab_size": self._slab_size,
            "entries": entries,
        }
        tmp = self._index_path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(json.dumps(doc).encode("utf-8"))
            os.rename(tmp, self._index_path)
        except OSError as e:
            logger.warning(
                "GdsL1Backend._persist_index: failed to write %s: %s",
                self._index_path,
                e,
            )
