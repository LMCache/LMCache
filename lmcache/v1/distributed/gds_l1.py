# SPDX-License-Identifier: Apache-2.0
"""Slab-file GDS L1 backend for MP mode.

This module turns NVMe disk into the durable L1 medium in MP mode by
pre-allocating one large *slab file* on disk and treating it the same
way ``L1MemoryManager`` treats its pinned-DRAM slab: a single
allocation, a single cuFile handle registration, and per-chunk
sub-allocations as ``(offset, size)`` pairs inside the slab.

The single-slab design is about throughput. cuFile's per-file
``cuFileHandleRegister`` dominates the cost when chunks are small, and
the prior file-per-chunk layout paid it on every read and write. Here
the slab is opened once with ``O_DIRECT``, registered once with
``cuFileHandleRegister``, and pre-allocated with ``posix_fallocate`` so
writes never grow the file; the handle is held for the backend's
lifetime. Per-chunk I/O then collapses to a single
``cuFileWriteAsync`` / ``cuFileReadAsync`` submit plus the DMA.

The slab medium itself is the durable cache: eviction frees a region
back to the address manager, and the on-disk slab survives restart. A
JSON index at ``<gds_path>/lmcache_gds_index.json`` lets a fresh
backend instance recover the ``ObjectKey → (offset, size)`` map.

Classes:

- :class:`SlabAddressManager` — first-fit free-list allocator over byte
  offsets within the slab file (the slab's ``TensorMemoryAllocator``).
- :class:`GdsCuFileIO` — the cuFile data path: the slab's cuFile handle,
  the registered GPU staging-buffer table, and the stream-ordered
  ``cuFileReadAsync`` / ``cuFileWriteAsync`` submissions.
- :class:`GdsSlabAllocator` — owns the address manager, the persisted
  index of resident keys, and a ``GdsCuFileIO``. The ``isinstance``
  dispatch target in ``gpu_ops.py`` and home of ``create_memory_obj`` /
  :meth:`cufile_read_into` / :meth:`cufile_write_from`.
- :class:`GdsL1Backend` — thin L1-tier facade (``allocate`` / ``free``)
  over ``GdsSlabAllocator``, mirroring ``L1MemoryManager``.
- :class:`GdsMemoryObj` — slab-anchored ``MemoryObj`` whose ``(offset,
  size)`` come from its metadata. ``.tensor`` is always ``None``;
  ``.byte_array`` / ``.data_ptr`` raise.
"""

# Standard
from collections import OrderedDict
from typing import Optional, Union
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
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.v1.distributed import _cufile_async as ca
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
)
from lmcache.v1.storage_backend.gds_backend import get_fstype
from lmcache.v1.storage_backend.path_sharder import PathSharder
from lmcache.v1.storage_backend.raw_block.core import round_up

logger = init_logger(__name__)


# --- Constants ------------------------------------------------------

_SLAB_FILENAME = "lmcache_gds_slab.bin"
_INDEX_FILENAME = "lmcache_gds_index.json"
_INDEX_VERSION = 1
_CUFILE_ALIGNMENT = 4096
# cuFile submissions to accumulate before recording a completion event and
# draining finished ones (see ``GdsCuFileIO._record_submission``).
_SUBMISSION_CHECKPOINT_EVERY = 64


# --- Helpers --------------------------------------------------------


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


# --- Classes --------------------------------------------------------


class SlabAddressManager:
    """First-fit free-list allocator over a fixed-size byte offset space.

    Used by :class:`GdsSlabAllocator` to track which byte ranges of the
    slab file are in use. Same data structure pattern as the
    ``TensorMemoryAllocator`` that backs CPU L1's pinned slab, but
    operating on file offsets instead of pointers.

    Sizes and offsets are tracked verbatim; callers are responsible for
    passing 4 KiB-aligned sizes (the cuFile/O_DIRECT requirement is
    enforced upstream at the object-creation and index-load boundaries).

    Args:
        total_size: Total addressable space in bytes.

    Raises:
        ValueError: If ``total_size`` <= 0.

    Thread-safe: all public methods take an internal lock.
    """

    def __init__(self, total_size: int) -> None:
        if total_size <= 0:
            raise ValueError(
                f"SlabAddressManager: total_size must be > 0, got {total_size}"
            )
        self._total = total_size
        self._lock = threading.Lock()
        # Sorted (offset, size) free regions; starts covering the whole slab.
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

    def memcheck(self) -> bool:
        """Verify the free list and the used counter still add up to the slab.

        Returns ``True`` when the sum of the free regions plus the tracked
        used bytes equals the total size; ``False`` (with a log) on a
        bookkeeping inconsistency (a lost, leaked, or double-counted region).
        """
        with self._lock:
            free_in_list = sum(size for _, size in self._free)
            consistent = free_in_list + self._used == self._total
            if not consistent:
                logger.error(
                    "SlabAddressManager inconsistent: free_list=%d + used=%d "
                    "!= total=%d",
                    free_in_list,
                    self._used,
                    self._total,
                )
            return consistent

    def allocate(self, size: int) -> Optional[int]:
        """Reserve ``size`` bytes and return the offset, or ``None`` on OOM.

        First-fit: scans the free list for the first region big enough.

        Args:
            size: Number of bytes to reserve (caller must pass a
                4 KiB-aligned size).

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
            size: Size of the region (must match the allocated size).
        """
        if size <= 0:
            return
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


class GdsMemoryObj(MemoryObj):
    """
    Slab-anchored ``MemoryObj`` for the GDS L1 backend.
    """

    def __init__(
        self,
        key: ObjectKey,
        metadata: MemoryObjMetadata,
        parent_allocator: "GdsSlabAllocator",
    ) -> None:
        super().__init__(metadata)
        self.key = key
        self._parent_allocator = parent_allocator
        self.valid = True

    @property
    def slab_offset(self) -> int:
        """Byte offset of this chunk within the slab file (== meta.address)."""
        return self.meta.address

    def invalidate(self) -> None:
        self.valid = False

    def is_valid(self) -> bool:
        return self.valid

    def get_size(self) -> int:
        return self.meta.phy_size

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> Optional[torch.dtype]:
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        raise NotImplementedError(
            "GdsMemoryObj.get_shapes: per-group shapes are not tracked on the "
            "MP path (only the singular meta.shape is); use get_shape()"
        )

    def get_dtypes(self) -> list[torch.dtype]:
        raise NotImplementedError(
            "GdsMemoryObj.get_dtypes: per-group dtypes are not tracked on the "
            "MP path (only the singular meta.dtype is); use get_dtype()"
        )

    def get_memory_format(self) -> MemoryFormat:
        return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self) -> None:
        raise NotImplementedError("GdsMemoryObj.ref_count_up: not used on the MP path")

    def ref_count_down(self) -> None:
        raise NotImplementedError(
            "GdsMemoryObj.ref_count_down: not used on the MP path"
        )

    def get_ref_count(self) -> int:
        raise NotImplementedError("GdsMemoryObj.get_ref_count: not used on the MP path")

    def get_num_tokens(self) -> int:
        raise NotImplementedError(
            "GdsMemoryObj.get_num_tokens: not used on the MP path"
        )

    def pin(self) -> bool:
        raise NotImplementedError("GdsMemoryObj.pin: not used on the MP path")

    def unpin(self) -> bool:
        raise NotImplementedError("GdsMemoryObj.unpin: not used on the MP path")

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
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

    @property
    def is_pinned(self) -> bool:
        raise NotImplementedError("GdsMemoryObj.is_pinned: not used on the MP path")

    @property
    def can_evict(self) -> bool:
        raise NotImplementedError("GdsMemoryObj.can_evict: not used on the MP path")

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional[MemoryAllocatorInterface]:
        return self._parent_allocator


class _IndexEntry:
    """One record in the on-disk index.

    Small enough to keep as a tiny class rather than a full dataclass
    so it serialises cheaply to JSON.
    """

    __slots__ = ("slab_offset", "size")

    def __init__(
        self,
        slab_offset: int,
        size: int,
    ) -> None:
        self.slab_offset = slab_offset
        self.size = size


class GdsCuFileIO:
    """cuFile data path for the GDS slab.

    Owns the single ``cuFileHandleRegister`` handle, the registered GPU
    staging-buffer table, and the stream-ordered ``cuFileReadAsync`` /
    ``cuFileWriteAsync`` submissions. Owned by :class:`GdsSlabAllocator`:
    ``register_gpu_buffer`` (one call per staging slot at ``GPUCacheContext``
    init) and ``read`` / ``write`` (per-chunk DMA) are the surface.
    """

    def __init__(
        self,
        slab_path: str,
        slab_size: int,
        use_gds: bool,
        use_direct_io: bool,
    ) -> None:
        self._slab_path = slab_path
        self._slab_size = slab_size
        self.use_gds = use_gds
        self._slab_handle: Optional[ca.AsyncHandle] = None
        self._posix_fd: int = -1
        self._cudart: Optional[ctypes.CDLL] = None
        # The CUDA stream the async ops run on, registered once and held for
        # the engine's lifetime: the torch ``Stream`` (for events) and the raw
        # ``CUstream`` int (for the cuFile C API).
        self._registered_stream: Optional[torch.Stream] = None
        self._registered_stream_handle: Optional[int] = None
        self._stream_lock = threading.Lock()
        # In-flight submissions, released once a CUDA event recorded after them
        # on the cuFile stream completes (see ``_checkpoint_submissions_locked``).
        self._uncommitted_submissions: list[ca.Submission] = []
        self._inflight_submissions: list[tuple[torch.Event, list[ca.Submission]]] = []
        self._ops_since_checkpoint = 0
        self._submissions_lock = threading.Lock()
        # One entry per cuFile-registered region, parallel lists kept sorted by
        # ``_base_ptrs`` ascending for ``_resolve_buffer``'s bisect.
        self._buffers: list[torch.Tensor] = []
        self._base_ptrs: list[int] = []
        self._nbytes: list[int] = []
        if use_gds:
            self._open_and_register_slab(use_direct_io)
        else:
            self._open_slab_posix()

    # --- Public API ---------------------------------------------------

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
        if self.use_gds:
            ca.register_buffer(buffer)
            # Register the backend's CUDA stream on first buffer.
            self._ensure_stream_registered()
            log_label = "cuFile"
        else:
            log_label = "POSIX fallback (no cuFile registration)"
        # Insert keeping ``_base_ptrs`` sorted for ``_resolve_buffer``.
        idx = bisect.bisect_left(self._base_ptrs, base)
        self._buffers.insert(idx, buffer)
        self._base_ptrs.insert(idx, base)
        self._nbytes.insert(idx, nbytes)
        logger.info(
            "GdsCuFileIO: registered %d bytes at 0x%x via %s (total registrations: %d)",
            nbytes,
            base,
            log_label,
            len(self._buffers),
        )

    def deregister_gpu_buffer(self) -> None:
        """Deregister every previously-registered region + stream.

        Idempotent: safe to call from ``GdsCuFileIO.close`` even if
        registration never happened.
        """
        if not self._buffers:
            return
        if self.use_gds:
            torch_dev.synchronize(device=self._buffers[0].device)
            for buf in self._buffers:
                try:
                    ca.deregister_buffer(buf)
                except Exception as e:
                    logger.warning("deregister_gpu_buffer: %s", e)
        # The stream is deregistered in backend.close(), not here.
        self._buffers.clear()
        self._base_ptrs.clear()
        self._nbytes.clear()

    def read(self, memory_obj: "GdsMemoryObj", gpu_buffer: torch.Tensor) -> None:
        """DMA ``memory_obj``'s chunk from the slab into ``gpu_buffer``.

        Stream-ordered (no per-call sync); the matching registered region is
        resolved by pointer comparison.

        Raises:
            RuntimeError: If no buffer has been registered.
            ValueError: If ``gpu_buffer`` is outside every registered region
                or smaller than the chunk.
        """
        base_ptr, dev_offset = self._resolve_buffer(gpu_buffer)
        nbytes = memory_obj.get_size()
        if nbytes > gpu_buffer.numel() * gpu_buffer.element_size():
            raise ValueError(
                f"GdsCuFileIO.read: chunk size {nbytes} exceeds gpu_buffer "
                f"capacity {gpu_buffer.numel() * gpu_buffer.element_size()}"
            )
        self._slab_read(memory_obj.slab_offset, nbytes, dev_offset, base_ptr)

    def write(self, memory_obj: "GdsMemoryObj", gpu_buffer: torch.Tensor) -> None:
        """DMA ``gpu_buffer`` into ``memory_obj``'s slab region (no per-call sync)."""
        base_ptr, dev_offset = self._resolve_buffer(gpu_buffer)
        nbytes = memory_obj.get_size()
        if nbytes > gpu_buffer.numel() * gpu_buffer.element_size():
            raise ValueError(
                f"GdsCuFileIO.write: chunk size {nbytes} exceeds gpu_buffer "
                f"capacity {gpu_buffer.numel() * gpu_buffer.element_size()}"
            )
        self._slab_write(memory_obj.slab_offset, nbytes, dev_offset, base_ptr)

    def close(self) -> None:
        """Sync the stream, deregister cuFile state, close the slab handle."""
        if self._buffers:
            torch_dev.synchronize(device=self._buffers[0].device)
        with self._submissions_lock:
            self._uncommitted_submissions = []
            self._inflight_submissions = []
            self._ops_since_checkpoint = 0
        self.deregister_gpu_buffer()
        with self._stream_lock:
            if self._registered_stream_handle is not None and self.use_gds:
                try:
                    ca.deregister_stream(self._registered_stream_handle)
                except Exception as e:
                    logger.warning("GdsCuFileIO.close: deregister_stream: %s", e)
                self._registered_stream_handle = None
                self._registered_stream = None
        if self._slab_handle is not None:
            try:
                self._slab_handle.close()
            except Exception as e:
                logger.warning("GdsCuFileIO.close: slab handle close failed: %s", e)
            self._slab_handle = None
        if self._posix_fd != -1:
            try:
                os.close(self._posix_fd)
            except OSError:
                pass
            self._posix_fd = -1

    # --- Internal -----------------------------------------------------

    def _open_and_register_slab(self, use_direct_io: bool) -> None:
        """Create / open the slab file and register it with cuFile.

        Pre-allocates to ``self._slab_size`` so subsequent
        ``cuFileWriteAsync``s don't have to grow the file.

        Args:
            use_direct_io: If ``True``, opens with ``O_DIRECT`` (required
                for the cuFile GDS DMA fast path on ext4).
        """
        # Create and fallocate via a regular (non-O_DIRECT) fd.
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
        # Build an AsyncHandle exposing close/read_async/write_async.
        self._slab_handle = ca.AsyncHandle.__new__(ca.AsyncHandle)
        self._slab_handle._fd = fd  # noqa: SLF001
        self._slab_handle._handle = handle  # noqa: SLF001
        self._slab_handle.path = self._slab_path
        self._slab_handle.writable = True
        logger.info(
            "GdsCuFileIO: slab opened at %s (O_DIRECT=%s), cuFile handle registered",
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
        logger.info("GdsCuFileIO: slab opened at %s (POSIX fallback)", self._slab_path)

    def _ensure_stream_registered(self) -> None:
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
            if self._registered_stream_handle is not None:
                return
            current_stream = torch_dev.current_stream()
            raw_stream = current_stream.cuda_stream
            ca.register_stream(raw_stream)
            self._registered_stream_handle = raw_stream
            self._registered_stream = current_stream

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
            raise RuntimeError("GdsCuFileIO: no GPU buffer has been registered yet")
        ptr = gpu_buffer.data_ptr()
        # Candidate region is the rightmost one whose base ≤ ptr.
        idx = bisect.bisect_right(self._base_ptrs, ptr) - 1
        if idx < 0:
            raise ValueError(
                f"GdsCuFileIO: gpu_buffer pointer 0x{ptr:x} is below "
                f"the lowest registered base 0x{self._base_ptrs[0]:x}"
            )
        base = self._base_ptrs[idx]
        nbytes = self._nbytes[idx]
        offset = ptr - base
        if offset < 0 or offset >= nbytes:
            raise ValueError(
                f"GdsCuFileIO: gpu_buffer pointer 0x{ptr:x} is "
                f"outside every registered region (closest candidate: "
                f"[0x{base:x}, 0x{base + nbytes:x}))"
            )
        return base, offset

    def _slab_read(
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
                ``GdsCuFileIO._resolve_buffer`` so multi-
                registration striping ends up at the correct region.
        """
        if not self.use_gds:
            self._posix_slab_read(slab_offset, size, dev_offset, buf_base)
            return
        if self._slab_handle is None:
            raise RuntimeError("GdsCuFileIO._slab_read: slab handle not open")
        if self._registered_stream_handle is None:
            raise RuntimeError(
                "GdsCuFileIO._slab_read: cuFile stream not registered; "
                "call register_gpu_buffer first"
            )
        sub = self._slab_handle.read_async(
            buf_base,
            size,
            slab_offset,
            dev_offset,
            self._registered_stream_handle,
        )
        self._record_submission(sub)

    def _slab_write(
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
            raise RuntimeError("GdsCuFileIO._slab_write: slab handle not open")
        if self._registered_stream_handle is None:
            raise RuntimeError(
                "GdsCuFileIO._slab_write: cuFile stream not registered; "
                "call register_gpu_buffer first"
            )
        sub = self._slab_handle.write_async(
            buf_base,
            size,
            slab_offset,
            dev_offset,
            self._registered_stream_handle,
        )
        self._record_submission(sub)

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

    def _record_submission(self, sub: "ca.Submission") -> None:
        """Track one in-flight cuFile submission, draining completed ones.

        The ``Submission``'s ctypes storage must outlive the stream op,
        so submissions are accumulated and only released once a CUDA
        event recorded after them on the cuFile stream reports complete.
        A checkpoint (event record + non-blocking drain) is taken every
        ``_SUBMISSION_CHECKPOINT_EVERY`` ops, keeping the live set
        bounded to a few in-flight batches instead of growing for the
        backend's lifetime (which would inflate Python's gen-2 GC pause).

        Args:
            sub: The submission returned by ``read_async`` /
                ``write_async``.
        """
        with self._submissions_lock:
            self._uncommitted_submissions.append(sub)
            self._ops_since_checkpoint += 1
            if self._ops_since_checkpoint >= _SUBMISSION_CHECKPOINT_EVERY:
                self._checkpoint_submissions_locked()

    def _checkpoint_submissions_locked(self) -> None:
        """Close the current submission batch and release completed ones.

        Records a CUDA event on the registered cuFile stream that marks
        the point after every currently-uncommitted submission, then
        drops any earlier batch whose event has already completed. Uses
        a non-blocking ``query()`` so the hot path never synchronizes.

        Must be called while holding ``self._submissions_lock``.
        """
        if self._uncommitted_submissions:
            event = torch_dev.Event()
            # Record on the stream the ops were enqueued on.
            if self._registered_stream is not None:
                event.record(self._registered_stream)
            else:
                event.record()
            self._inflight_submissions.append((event, self._uncommitted_submissions))
            self._uncommitted_submissions = []
        self._ops_since_checkpoint = 0
        # Drop batches whose event has completed (stream passed it).
        self._inflight_submissions = [
            (event, subs)
            for (event, subs) in self._inflight_submissions
            if not event.query()
        ]


class GdsSlabAllocator(MemoryAllocatorInterface):
    """Slab-file GDS L1 allocator.

    Owns the slab address space (:class:`SlabAddressManager`), the durable
    ``ObjectKey -> _IndexEntry`` map, and a :class:`GdsCuFileIO` for the
    GPU <-> slab byte movement. ``create_memory_obj`` reserves a slab region
    and returns a :class:`GdsMemoryObj`; the cuFile transfer and the GPU
    staging-buffer registration live on the owned ``GdsCuFileIO``.
    """

    def __init__(
        self,
        config: GdsL1Config,
        dst_device: str = "cuda",
    ) -> None:
        if not config.gds_path:
            raise ValueError("GdsSlabAllocator requires gds_path to be set")
        if not dst_device.startswith("cuda"):
            raise ValueError(
                f"GdsSlabAllocator requires cuda dst_device, got {dst_device}"
            )
        self.config = config
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
            "GdsSlabAllocator: fstype=%r path=%r (%d configured)",
            self.fstype,
            self.gds_path,
            len(self.gds_paths),
        )

        self.use_gds: bool = config.use_gds
        if self.fstype in ("tmpfs", "overlayfs") and config.use_gds:
            logger.info(
                "GdsSlabAllocator: auto-disabling cuFile on fstype=%r", self.fstype
            )
            self.use_gds = False

        self._slab_path = os.path.join(self.gds_path, _SLAB_FILENAME)
        self._index_path = os.path.join(self.gds_path, _INDEX_FILENAME)
        self._slab_size = round_up(
            int(config.slab_size_gb * (1 << 30)), _CUFILE_ALIGNMENT
        )
        self._io = GdsCuFileIO(
            slab_path=self._slab_path,
            slab_size=self._slab_size,
            use_gds=self.use_gds,
            use_direct_io=config.use_direct_io,
        )

        self._address_manager = SlabAddressManager(total_size=self._slab_size)
        self._index_lock = threading.Lock()
        self._index: OrderedDict[ObjectKey, _IndexEntry] = OrderedDict()
        self._load_index()
        logger.info(
            "GdsSlabAllocator: slab=%s size=%.1f GiB, %d resident entries",
            self._slab_path,
            config.slab_size_gb,
            len(self._index),
        )

    def create_memory_obj(
        self,
        key: ObjectKey,
        layout_desc: MemoryLayoutDesc,
    ) -> Optional[GdsMemoryObj]:
        """Reserve a slab region for ``key`` and return a fresh MemoryObj.

        Args:
            key: The ObjectKey being inserted.
            layout_desc: Layout descriptor for the chunk's payload.

        Returns:
            A :class:`GdsMemoryObj` pointing at a freshly-reserved
            slab region, or ``None`` if the slab is full (caller
            should treat this as an L1 OOM and trigger eviction).
        """
        shape = layout_desc.shapes[0]
        dtype = layout_desc.dtypes[0]
        nbytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        if nbytes % _CUFILE_ALIGNMENT != 0:
            raise ValueError(
                f"GdsSlabAllocator.create_memory_obj: payload size {nbytes} is not a "
                f"multiple of {_CUFILE_ALIGNMENT}, which cuFile/O_DIRECT requires. "
                f"This should be unreachable since register_gpu_buffer already "
                f"enforces 4 KiB alignment on the larger staging slot."
            )
        slab_offset = self._address_manager.allocate(nbytes)
        if slab_offset is None:
            return None
        meta = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=slab_offset,
            phy_size=nbytes,
            ref_count=0,
        )
        return GdsMemoryObj(
            key=key,
            metadata=meta,
            parent_allocator=self,
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
        # Also unreachable in practice: entries only enter ``self._index``
        # via ``_record_entry`` (sizes already accepted by
        # ``create_memory_obj`` above) or via ``_load_index``, which skips
        # any persisted entry whose size is not 4 KiB-aligned. This guard
        # is the in-memory analogue of that index-load check, so a
        # corrupt-but-aligned-looking entry can't reach the cuFile
        # transfer length with a misaligned size.
        if entry.size % _CUFILE_ALIGNMENT != 0:
            raise ValueError(
                f"GdsSlabAllocator.create_memory_obj_from_index: index entry size "
                f"{entry.size} is not a multiple of {_CUFILE_ALIGNMENT}. This "
                f"should be unreachable -- _load_index rejects misaligned persisted "
                f"sizes and create_memory_obj rejects misaligned new ones."
            )
        # The index persists only (offset, size); per-group shape/dtype are not
        # stored (unused on the GDS path). Describe the chunk as raw bytes so
        # the metadata is self-consistent (meta.get_size() == size).
        meta = MemoryObjMetadata(
            shape=torch.Size([entry.size]),
            dtype=torch.uint8,
            address=entry.slab_offset,
            phy_size=entry.size,
            ref_count=0,
        )
        return GdsMemoryObj(
            key=key,
            metadata=meta,
            parent_allocator=self,
        )

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` of the slab.

        Feeds the existing L1 eviction-controller signal.
        """
        return self._address_manager.used_bytes(), self._slab_size

    def memcheck(self) -> bool:
        """Verify the slab address manager's free-list bookkeeping is consistent."""
        return self._address_manager.memcheck()

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Free the slab region and drop the index entry.

        Removal semantics: returns the object's slab region and drops its
        index entry.
        """
        if not isinstance(memory_obj, GdsMemoryObj):
            raise TypeError(
                f"GdsSlabAllocator.free expects GdsMemoryObj, got "
                f"{type(memory_obj).__name__}"
            )
        with self._index_lock:
            entry = self._index.pop(memory_obj.key, None)
        if entry is not None:
            self._address_manager.free(entry.slab_offset, entry.size)

    def free_not_from_index(self, memory_obj: GdsMemoryObj) -> None:
        """Return a slab region without touching the index.

        Rollback semantics: for a reserved-but-unrecorded object (e.g. an
        allocation aborted before it was written), frees the region directly.
        """
        self._address_manager.free(memory_obj.slab_offset, memory_obj.get_size())

    def close(self) -> None:
        """Persist the index and tear down the cuFile data path."""
        self._persist_index()
        self._io.close()

    def _record_entry(self, memory_obj: GdsMemoryObj) -> None:
        """Insert ``memory_obj`` into the in-memory index.

        Called by ``GdsSlabAllocator.cufile_write_from`` after the
        async write submit. Persistence to the on-disk index file is
        deferred to :meth:`close`.

        Args:
            memory_obj: The freshly-written :class:`GdsMemoryObj`.
        """
        entry = _IndexEntry(
            slab_offset=memory_obj.slab_offset,
            size=memory_obj.get_size(),
        )
        with self._index_lock:
            existing = self._index.get(memory_obj.key)
            if existing is not None:
                self._address_manager.free(existing.slab_offset, existing.size)
            self._index[memory_obj.key] = entry

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
                "GdsSlabAllocator: failed to read index %s: %s — starting empty",
                self._index_path,
                e,
            )
            return
        if doc.get("version") != _INDEX_VERSION:
            logger.warning(
                "GdsSlabAllocator: index version mismatch (got %r, want %d) — ignoring",
                doc.get("version"),
                _INDEX_VERSION,
            )
            return
        if doc.get("slab_size") != self._slab_size:
            logger.warning(
                "GdsSlabAllocator: index slab_size mismatch (got %d, want %d) "
                "— ignoring",
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
                slab_offset = int(raw["offset"])
                size = int(raw["size"])
            except (KeyError, ValueError):
                continue
            if size % _CUFILE_ALIGNMENT != 0:
                logger.warning(
                    "GdsSlabAllocator._load_index: entry size %d is not a multiple "
                    "of %d — skipping",
                    size,
                    _CUFILE_ALIGNMENT,
                )
                continue
            entry = _IndexEntry(slab_offset=slab_offset, size=size)
            try:
                self._address_manager.mark_used(slab_offset, size)
            except RuntimeError as e:
                logger.warning("GdsSlabAllocator._load_index: %s", e)
                continue
            self._index[key] = entry
            loaded += 1
        elapsed = time.perf_counter() - start
        logger.info(
            "GdsSlabAllocator: loaded %d index entries in %.2fs", loaded, elapsed
        )

    def _persist_index(self) -> None:
        """Write the current in-memory index to ``self._index_path``."""
        with self._index_lock:
            entries = {
                _object_key_to_string(k): {
                    "offset": e.slab_offset,
                    "size": e.size,
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
                "GdsSlabAllocator._persist_index: failed to write %s: %s",
                self._index_path,
                e,
            )

    def cufile_read_into(
        self, memory_obj: "GdsMemoryObj", gpu_buffer: torch.Tensor
    ) -> None:
        """Read ``memory_obj``'s chunk from the slab into ``gpu_buffer``.

        Dispatch target for ``gpu_ops``; delegates the DMA to the owned
        :class:`GdsCuFileIO`.
        """
        self._io.read(memory_obj, gpu_buffer)

    def cufile_write_from(
        self, memory_obj: "GdsMemoryObj", gpu_buffer: torch.Tensor
    ) -> None:
        """Write ``gpu_buffer`` to ``memory_obj``'s slab region, then index it."""
        self._io.write(memory_obj, gpu_buffer)
        self._record_entry(memory_obj)

    @property
    def cufile_io(self) -> "GdsCuFileIO":
        """The owned cuFile data-path engine (GPU registration + DMA)."""
        return self._io

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        raise NotImplementedError(
            "GdsSlabAllocator.allocate is unsupported; use create_memory_obj(key, "
            "layout_desc). GDS allocation needs an ObjectKey for its index."
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
            "GdsSlabAllocator.batched_allocate is unsupported; use "
            "create_memory_obj(key, layout_desc). GDS allocation needs an "
            "ObjectKey for its index."
        )

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        for memory_obj in memory_objs:
            self.free(memory_obj)


class GdsL1Backend:
    """
    Mirrors :class:`L1MemoryManager` on the CPU tier: owns the allocator
    and exposes the ``allocate`` / ``free`` interface ``L1Manager`` drives.
    """

    def __init__(self, config: GdsL1Config, dst_device: str = "cuda") -> None:
        self._allocator = GdsSlabAllocator(config, dst_device)

    def allocate(
        self, layout_desc: MemoryLayoutDesc, keys: list[ObjectKey]
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Reserve one slab region per key.

        All-or-nothing: on the first slab OOM, frees what was reserved and
        returns ``(L1Error.OUT_OF_MEMORY, [])``.
        """
        allocated: list[MemoryObj] = []
        for key in keys:
            obj = self._allocator.create_memory_obj(key, layout_desc)
            if obj is None:
                self.free(allocated)
                return L1Error.OUT_OF_MEMORY, []
            allocated.append(obj)
        return L1Error.SUCCESS, allocated

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Return the slab regions of reserved-but-unrecorded ``mem_objs``.

        For the reserve-time rollback path: the objects are not yet in the
        index, so only the slab region is returned (no index touch).
        """
        for mo in mem_objs:
            if not isinstance(mo, GdsMemoryObj):
                raise TypeError(
                    f"GdsL1Backend.free expects GdsMemoryObj, got {type(mo).__name__}"
                )
            self._allocator.free_not_from_index(mo)
        return L1Error.SUCCESS

    def free_from_index(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Remove resident (recorded) ``mem_objs``: drop index entry + region.

        For the removal paths (delete / clear / eviction): each object is in
        the durable index, so both the index entry and its slab region go.
        The allocator's ``free`` raises on a non-GDS object.
        """
        for mo in mem_objs:
            self._allocator.free(mo)
        return L1Error.SUCCESS

    @property
    def cufile_io(self) -> "GdsCuFileIO":
        """The cuFile data-path engine (GPU buffer registration + DMA)."""
        return self._allocator.cufile_io

    def create_memory_obj_from_index(self, key: ObjectKey) -> Optional[MemoryObj]:
        return self._allocator.create_memory_obj_from_index(key)

    def get_memory_usage(self) -> tuple[int, int]:
        return self._allocator.get_memory_usage()

    def memcheck(self) -> bool:
        return self._allocator.memcheck()

    def close(self) -> None:
        self._allocator.close()
