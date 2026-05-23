# SPDX-License-Identifier: Apache-2.0
"""GDS L1 backend for MP mode.

This module makes NVMe disk the durable L1 medium, with cuFile DMA
between disk files and the GPU staging buffer that ``GPUCacheContext``
already owns. There is no in-VRAM warm cache: every L1 access goes
through ``kvikio.CuFile.pread`` / ``pwrite`` directly into the existing
``tmp_gpu_buffer_``.

We use kvikio (NVIDIA RAPIDS) as the cuFile wrapper rather than the
bare ``cufile`` Python bindings because kvikio exposes the async
APIs that the bare bindings omit (``IOFuture`` + future-based wait,
plus a ``raw_read_async`` variant that's stream-ordered). Today we
use ``pread``/``pwrite`` (thread-pool async); ``raw_read_async`` is a
follow-up when the caller can submit a batch of reads before waiting.

See ``docs/design/v1/distributed/gds_l1_backend.md`` for the full
architecture and ``gds_l1_backend_plan.md`` for the decision log.

Surface this module exposes:

- :class:`GdsL1Backend` — owns the hot index, the cuFile handle cache,
  the metadata scan, and the :class:`GdsScratchAllocator`.
  ``L1Manager`` consumes this via an optional hook.
- :class:`GdsScratchAllocator` — tag class used for ``isinstance``
  dispatch in ``gpu_ops.py``; also the home for
  :meth:`cufile_read_into` and :meth:`cufile_write_from`.
- :class:`GdsMemoryObj` — disk-anchored ``MemoryObj``. ``.tensor`` is
  ``None`` always; ``.byte_array`` and ``.data_ptr`` raise. Under the
  exclusive-L2 mode that GDS L1 enforces, neither field is read on
  the GDS path — the data path is gpu_ops dispatch + the gpu_buffer
  parameter, not field access on the MemoryObj.
"""

# Standard
from collections import OrderedDict
from typing import Optional, Union
import asyncio
import ctypes
import mmap
import os
import struct
import threading
import time
import urllib.parse
import uuid

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import DiskCacheMetadata
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


# --- Constants (kept compatible with storage_backend/gds_backend.py
#     so the disk layout is shared across MP and non-MP modes) -----

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_WEKA_DATA_FILE_SUFFIX = ".weka1"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # 4 KiB reserved header inside the data file

_CUFILE_ALIGNMENT = 4096
_DEFAULT_HANDLE_CACHE_SIZE = 1024

_TORCH_DTYPES = {
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
_TORCH_DTYPES_INVERSE = {v: k for k, v in _TORCH_DTYPES.items()}


class UnsupportedMetadataVersion(Exception):
    """Raised when a metadata file carries an unrecognised version."""


# --- Module-level helpers --------------------------------------------


def pack_metadata(
    nbytes: int,
    shape: torch.Size,
    dtype: torch.dtype,
    fmt: MemoryFormat,
    **extra_metadata: object,
) -> bytes:
    """Encode a 4 KiB metadata header for a kvcache file.

    Args:
        nbytes: Size of the data payload in bytes.
        shape: Logical tensor shape.
        dtype: Logical tensor dtype.
        fmt: Memory format of the chunk.
        **extra_metadata: Free-form metadata stored under ``__metadata__``.

    Returns:
        Exactly ``_METADATA_MAX_SIZE`` bytes: an 8-byte length prefix
        plus a space-padded JSON blob.

    Raises:
        RuntimeError: If ``dtype`` is not one of the supported dtypes.
    """
    # Standard
    import json

    if dtype not in _TORCH_DTYPES:
        raise RuntimeError(f"Unsupported dtype for GDS L1: {dtype}")
    tensor_meta = {
        "dtype": _TORCH_DTYPES[dtype],
        "shape": list(shape),
        "data_offsets": [0, nbytes],
        "fmt": fmt.value,
        "__metadata__": extra_metadata,
    }
    str_meta = json.dumps({"kvcache": tensor_meta}).encode("utf-8")
    if len(str_meta) > _METADATA_MAX_SIZE - 8:
        raise RuntimeError(
            f"GDS L1 metadata header overflowed budget: "
            f"{len(str_meta)} > {_METADATA_MAX_SIZE - 8}"
        )
    str_meta += b" " * (_METADATA_MAX_SIZE - 8 - len(str_meta))
    return struct.pack("<Q", len(str_meta)) + str_meta


def unpack_metadata(
    buffer: bytes,
) -> tuple[torch.Size, torch.dtype, int, MemoryFormat, dict]:
    """Decode a metadata header produced by :func:`pack_metadata`.

    Args:
        buffer: At least ``_METADATA_MAX_SIZE`` bytes from the file head.

    Returns:
        ``(shape, dtype, nbytes, fmt, extra_metadata)``.

    Raises:
        UnsupportedMetadataVersion: If ``__metadata__.lmcache_version``
            does not match :data:`_METADATA_VERSION`.
    """
    # Standard
    import json

    meta_len = struct.unpack("<Q", buffer[:8])[0]
    json_meta = buffer[8 : 8 + meta_len].rstrip(b" ")
    meta = json.loads(json_meta.decode("utf-8"))
    tensor_meta = meta["kvcache"]
    shape = torch.Size(tensor_meta["shape"])
    dtype = _TORCH_DTYPES_INVERSE[tensor_meta["dtype"]]
    data_offsets = tensor_meta["data_offsets"]
    nbytes = data_offsets[1] - data_offsets[0]
    fmt = MemoryFormat(tensor_meta["fmt"])
    extra = tensor_meta.get("__metadata__", {})
    if str(extra.get("lmcache_version")) != str(_METADATA_VERSION):
        raise UnsupportedMetadataVersion(
            f"GDS L1 metadata version mismatch: got {extra.get('lmcache_version')!r}"
        )
    return shape, dtype, nbytes, fmt, extra


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


def key_to_disk_path(
    key: ObjectKey,
    base_path: str,
    data_suffix: str = _DATA_FILE_SUFFIX,
) -> tuple[str, str, str, str]:
    """Map an ``ObjectKey`` to its on-disk file path.

    Layout mirrors ``storage_backend/gds_backend.py`` so the same
    directory tree can be shared across MP and non-MP modes:
    ``<base>/<hash[:2]>/<hash[2:4]>/<urlquote(key)>.kvcache.safetensors``.

    Args:
        key: The object key to map.
        base_path: Root directory for the cache (post-sharding).
        data_suffix: File suffix; ``_WEKA_DATA_FILE_SUFFIX`` for wekafs,
            else ``_DATA_FILE_SUFFIX``.

    Returns:
        ``(full_path, subdir_key, l1_dir, l2_dir)``. ``subdir_key`` is
        the concatenation ``l1_dir + l2_dir`` used as the dedup key
        for directory creation.
    """
    hash_str = _hash_str(key)
    l1_dir = hash_str[:2]
    l2_dir = hash_str[2:4]
    key_str = _object_key_to_string(key)
    full_path = os.path.join(
        base_path,
        l1_dir,
        l2_dir,
        urllib.parse.quote(key_str, safe="") + data_suffix,
    )
    return full_path, l1_dir + l2_dir, l1_dir, l2_dir


def _hash_str(key: ObjectKey) -> str:
    """Convert an ObjectKey's chunk_hash bytes to a hex string."""
    return key.chunk_hash.hex()


def _object_key_to_string(key: ObjectKey) -> str:
    """Stable string form of an ObjectKey for filename embedding.

    Format mirrors the L2 fs adapter so the on-disk layout is
    cross-mode portable.
    """
    return f"{key.model_name}@{key.kv_rank}@{_hash_str(key)}@{key.cache_salt}"


def _rand_suffix(n: int = 8) -> str:
    """Generate ``n`` hex chars of randomness for tmp-file suffixes."""
    return uuid.uuid4().hex[:n]


# --- cuFile handle cache --------------------------------------------


class CuFileHandleCache:
    """Thread-safe LRU cache of opened cuFile handles.

    Each entry maps ``(disk_path, mode)`` to a context-managed
    ``cufile.CuFile`` object. Entries are reference-counted: an entry
    in active use will not be evicted out from under a caller. The LRU
    bound caps fd usage; idle entries are evicted at insert time when
    the cache is full.

    This is a v1 implementation. It does not currently honor the
    handle's idle time — once the cache is at capacity, the
    least-recently-used idle handle is closed.

    Args:
        max_handles: LRU capacity. Default 1024.
        gds_module: The imported ``kvikio`` module; passed in so the
            cache doesn't import it at module load time. ``None``
            disables the cache (acquire raises).
    """

    def __init__(
        self,
        max_handles: int = _DEFAULT_HANDLE_CACHE_SIZE,
        gds_module: Optional[object] = None,
    ) -> None:
        if max_handles <= 0:
            raise ValueError(f"max_handles must be positive, got {max_handles}")
        self._max = max_handles
        self._gds_module = gds_module
        self._lock = threading.Lock()
        # entries are (file_obj, in_use_count); LRU order is dict order
        self._entries: OrderedDict[tuple[str, str], list] = OrderedDict()

    def acquire(self, disk_path: str, mode: str) -> object:
        """Borrow a cuFile handle for ``(disk_path, mode)``.

        The handle is reference-counted while in use so a concurrent
        :meth:`acquire` for the same key shares it without re-opening,
        and a concurrent capacity-driven eviction will not close it.

        Args:
            disk_path: File to open.
            mode: ``"r"``, ``"r+"``, etc. — passed to ``CuFile``.

        Returns:
            The open ``cufile.CuFile`` instance. **Caller must call
            :meth:`release` when done.**

        Raises:
            RuntimeError: If cuFile is not configured on this backend.
        """
        if self._gds_module is None:
            raise RuntimeError(
                "CuFileHandleCache: gds_module is None; cuFile not configured"
            )
        cache_key = (disk_path, mode)
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                self._entries.move_to_end(cache_key)
                entry[1] += 1
                return entry[0]
            # Cache miss: open under the lock to keep races simple.
            # ``kvikio.CuFile(path, flags)`` opens the file and
            # registers the handle with cuFile in the constructor, so
            # there is no separate ``.open()`` step.
            file_obj = self._gds_module.CuFile(disk_path, mode)
            self._entries[cache_key] = [file_obj, 1]
            self._evict_idle_if_full_locked()
            return file_obj

    def release(self, disk_path: str, mode: str) -> None:
        """Return a previously-acquired handle. Decrements ref count.

        Args:
            disk_path: File path passed to :meth:`acquire`.
            mode: Mode passed to :meth:`acquire`.
        """
        cache_key = (disk_path, mode)
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                logger.warning("CuFileHandleCache.release on unknown key %r", cache_key)
                return
            entry[1] -= 1
            if entry[1] < 0:
                logger.warning(
                    "CuFileHandleCache: release underflow for %r, clamping to 0",
                    cache_key,
                )
                entry[1] = 0

    def invalidate(self, disk_path: str, mode: Optional[str] = None) -> None:
        """Forcibly drop the cached handle(s) for ``disk_path``.

        Used when a file is deleted or rewritten so subsequent calls
        re-open. Idle handles are closed; in-use handles are left to
        be closed by :meth:`release` semantics on completion (though
        we drop the cache entry so no new acquirer reuses it).

        Args:
            disk_path: File whose cached handle(s) should be dropped.
            mode: If given, only invalidate this specific mode.
        """
        with self._lock:
            keys_to_drop = [
                k
                for k in self._entries
                if k[0] == disk_path and (mode is None or k[1] == mode)
            ]
            for k in keys_to_drop:
                entry = self._entries.pop(k)
                if entry[1] == 0:
                    self._safe_close(entry[0], k)

    def close(self) -> None:
        """Close every cached handle. Called at backend shutdown."""
        with self._lock:
            for cache_key, entry in self._entries.items():
                self._safe_close(entry[0], cache_key)
            self._entries.clear()

    def _evict_idle_if_full_locked(self) -> None:
        """Drop the LRU idle entry if the cache is over capacity.

        Must be called with ``self._lock`` held.
        """
        if len(self._entries) <= self._max:
            return
        # Walk from oldest to newest; close the first idle one.
        for cache_key in list(self._entries.keys()):
            entry = self._entries[cache_key]
            if entry[1] == 0:
                self._entries.pop(cache_key)
                self._safe_close(entry[0], cache_key)
                return
        # No idle entries: cache is fully busy. Log and keep going
        # (we will exceed capacity briefly rather than block).
        logger.debug(
            "CuFileHandleCache: capacity %d exceeded but all entries busy",
            self._max,
        )

    @staticmethod
    def _safe_close(file_obj: object, cache_key: tuple) -> None:
        """Close ``file_obj`` swallowing errors so shutdown is robust."""
        try:
            file_obj.close()
        except Exception as e:
            logger.warning("CuFileHandleCache: error closing %r: %s", cache_key, e)


# --- GdsMemoryObj -----------------------------------------------------


class GdsMemoryObj(MemoryObj):
    """Disk-anchored ``MemoryObj`` for the GDS L1 backend.

    A ``GdsMemoryObj`` carries the disk location and the logical
    metadata of a cached chunk. It owns no live in-memory body:
    ``.tensor`` is always ``None``, ``.byte_array`` always raises,
    and ``.data_ptr`` always raises. Bytes move between disk and the
    GPU staging buffer via the dispatch in ``gpu_ops.py``, which
    receives the staging buffer as a separate ``gpu_buffer`` argument
    and does not read these fields.

    Args:
        key: The ObjectKey this MemoryObj represents. Carried alongside
            the disk path so the backend's hot index can be updated on
            successful writes without re-parsing the path.
        disk_path: Absolute path to the kvcache data file.
        file_offset: Byte offset within the file where the payload
            starts; always :data:`_METADATA_MAX_SIZE` for the v1
            layout.
        metadata: Standard :class:`MemoryObjMetadata` for the chunk.
        parent_allocator: The :class:`GdsScratchAllocator` that
            handles cuFile I/O on this object's behalf. Returned by
            :meth:`parent` and used by ``gpu_ops`` ``isinstance``
            dispatch.
    """

    def __init__(
        self,
        key: ObjectKey,
        disk_path: str,
        file_offset: int,
        metadata: MemoryObjMetadata,
        parent_allocator: "GdsScratchAllocator",
    ) -> None:
        super().__init__(metadata)
        self.key = key
        self.disk_path = disk_path
        self.file_offset = file_offset
        self._parent_allocator = parent_allocator
        self._lock = threading.Lock()
        self._valid = True

    # --- Standard MemoryObj surface ------------------------------------

    def invalidate(self) -> None:
        self._valid = False

    def is_valid(self) -> bool:
        return self._valid

    def get_size(self) -> int:
        return self.meta.get_size()

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
                    "GdsMemoryObj for %s: ref_count went negative (%d), clamping",
                    self.disk_path,
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
            f"GdsMemoryObj({self.disk_path}).byte_array is not supported; "
            "bytes live on disk and the GPU staging is registered VRAM."
        )

    @property
    def data_ptr(self) -> int:
        raise NotImplementedError(
            f"GdsMemoryObj({self.disk_path}).data_ptr is not supported; "
            "GDS reads/writes use gpu_buffer.data_ptr() directly via the "
            "gpu_ops dispatch, never the MemoryObj's data_ptr."
        )

    def parent(self) -> Optional[MemoryAllocatorInterface]:
        return self._parent_allocator


# --- GdsScratchAllocator ---------------------------------------------


class GdsScratchAllocator(MemoryAllocatorInterface):
    """Tag class for ``gpu_ops`` ``isinstance`` dispatch, and the home
    for ``cufile_read_into`` / ``cufile_write_from``.

    This allocator does **not** participate in :meth:`allocate` /
    :meth:`free` the way ``MixedMemoryAllocator`` does. Allocation
    happens at the :class:`GdsL1Backend` level (which knows the
    ``ObjectKey`` and so can derive the disk path);
    ``GdsScratchAllocator``'s public role is:

    1. Hold a reference to the cuFile-registered GPU staging buffer
       (``GPUCacheContext.tmp_gpu_buffer_``) and its base pointer, so
       ``cufile.read`` / ``cufile.write`` can pass the registered
       address + a computed ``dev_offset``.
    2. Be the return value of ``GdsMemoryObj.parent()`` so the
       ``isinstance(..., GdsScratchAllocator)`` branch in
       ``gpu_ops.py`` selects the cuFile path instead of memcpy.

    Args:
        backend: The owning :class:`GdsL1Backend`. The allocator calls
            back into it for the cuFile handle cache and config.
    """

    def __init__(self, backend: "GdsL1Backend") -> None:
        self._backend = backend
        self._registered_base_ptr: Optional[int] = None
        self._registered_buffer: Optional[torch.Tensor] = None
        self._registered_nbytes: int = 0
        # ctypes-async path state. Populated only when
        # ``backend.config.use_async_ctypes`` is True.
        self._use_async_ctypes: bool = bool(
            getattr(backend.config, "use_async_ctypes", False)
        )
        # Submission objects keep host-side ctypes storage alive until
        # the CUDA stream actually executes the cuFile op. We hold them
        # here for the lifetime of the allocator; freed at
        # ``deregister_gpu_buffer``.
        self._inflight_subs: list[object] = []
        # Per-disk_path cache of opened ``AsyncHandle``s so we don't pay
        # the cuFileHandleRegister cost on every chunk.
        self._async_handles: dict[str, object] = {}
        self._registered_stream: Optional[int] = None

    # --- Buffer registration -----------------------------------------

    def register_gpu_buffer(self, buffer: torch.Tensor) -> None:
        """Register ``buffer`` with cuFile for zero-copy DMA.

        Called once per ``GPUCacheContext`` when GDS L1 is enabled.
        ``buffer`` must be a contiguous CUDA tensor whose byte size is
        a multiple of :data:`_CUFILE_ALIGNMENT` (4 KiB) — the L1
        backend hard-errors at this call site if not, since silently
        misaligned slot offsets would corrupt cuFile reads.

        Args:
            buffer: GPU tensor to register. Must be the underlying
                contiguous buffer (not a slice).

        Raises:
            ValueError: If the buffer is not on CUDA, not contiguous,
                or its size is not a 4 KiB multiple.
            RuntimeError: If cuFile is not configured on this backend.
        """
        if not buffer.is_cuda:
            raise ValueError(
                "GdsScratchAllocator.register_gpu_buffer: buffer must be on CUDA"
            )
        if not buffer.is_contiguous():
            raise ValueError(
                "GdsScratchAllocator.register_gpu_buffer: buffer must be contiguous"
            )
        nbytes = buffer.numel() * buffer.element_size()
        if nbytes % _CUFILE_ALIGNMENT != 0:
            raise ValueError(
                f"GdsScratchAllocator.register_gpu_buffer: buffer size "
                f"{nbytes} is not a multiple of {_CUFILE_ALIGNMENT} (cuFile "
                "requires 4 KiB alignment). Adjust lmcache_chunk_size or "
                "max_batch_size, or disable GDS L1."
            )
        if not self._backend.use_gds:
            # POSIX fallback path doesn't need cuFile registration.
            self._registered_buffer = buffer
            self._registered_base_ptr = buffer.data_ptr()
            self._registered_nbytes = nbytes
            return
        gds_module = self._backend.gds_module
        if gds_module is None:
            raise RuntimeError(
                "GdsScratchAllocator.register_gpu_buffer: gds_module is None"
            )
        if self._use_async_ctypes:
            # Direct-ctypes path: cuFileBufRegister via libcufile.
            # See class docstring for the trade-offs.
            # First Party
            from lmcache.v1.distributed import _cufile_async as ca

            # cuFile on this host caps a single registration at the
            # largest nvidia-fs slab (16 MiB). Bigger buffers would
            # need to be split across multiple registrations + a
            # per-slot base-pointer table; that's deferred.
            if nbytes > 16 * 1024 * 1024:
                raise ValueError(
                    f"GdsScratchAllocator: use_async_ctypes requires the "
                    f"staging buffer to fit in a single cuFile registration "
                    f"(<= 16 MiB on this host); got {nbytes} bytes. "
                    f"Reduce lmcache_chunk_size or max_batch_size, or set "
                    f"use_async_ctypes=False."
                )
            ca.register_buffer(buffer)
            raw_stream = torch.cuda.current_stream().cuda_stream
            ca.register_stream(raw_stream)
            self._registered_stream = raw_stream
            log_label = "cuFile (ctypes async)"
        else:
            # kvikio path: memory_register takes the buffer-like
            # directly and auto-discovers the underlying CUDA
            # allocation.
            gds_module.memory_register(buffer)
            log_label = "kvikio"
        self._registered_buffer = buffer
        self._registered_base_ptr = buffer.data_ptr()
        self._registered_nbytes = nbytes
        logger.info(
            "GdsScratchAllocator: registered %d bytes at 0x%x with %s",
            nbytes,
            buffer.data_ptr(),
            log_label,
        )

    def deregister_gpu_buffer(self) -> None:
        """Deregister the previously-registered buffer. Safe to call
        when no buffer was ever registered."""
        if self._registered_base_ptr is None:
            return
        # Sync the CUDA stream first so any in-flight cuFile async ops
        # complete and won't write into ctypes storage we're about to
        # release.
        if self._use_async_ctypes and self._registered_buffer is not None:
            torch.cuda.synchronize(device=self._registered_buffer.device)
        # Close any cached AsyncHandles (ctypes path only).
        for h in list(self._async_handles.values()):
            try:
                h.close()
            except Exception as e:
                logger.warning("GdsScratchAllocator: AsyncHandle close failed: %s", e)
        self._async_handles.clear()
        self._inflight_subs.clear()
        if (
            self._backend.use_gds
            and self._backend.gds_module is not None
            and self._registered_buffer is not None
        ):
            try:
                if self._use_async_ctypes:
                    # First Party
                    from lmcache.v1.distributed import _cufile_async as ca

                    ca.deregister_buffer(self._registered_buffer)
                    if self._registered_stream is not None:
                        ca.deregister_stream(self._registered_stream)
                        self._registered_stream = None
                else:
                    self._backend.gds_module.memory_deregister(self._registered_buffer)
            except Exception as e:
                logger.warning(
                    "GdsScratchAllocator: deregister failed: %s",
                    e,
                )
        self._registered_buffer = None
        self._registered_base_ptr = None
        self._registered_nbytes = 0

    # --- cuFile I/O for gpu_ops dispatch ----------------------------

    def cufile_read_into(
        self,
        memory_obj: GdsMemoryObj,
        gpu_buffer: torch.Tensor,
    ) -> None:
        """Read ``memory_obj``'s disk payload into ``gpu_buffer``.

        ``gpu_buffer`` is a slice of (or alias for) the registered
        staging buffer; this method computes the device offset from
        ``gpu_buffer.data_ptr() - registered_base_ptr`` and submits
        the read via the cuFile handle cache.

        Runs on the backend's thread pool when called from
        ``gpu_ops``; in PR 1 we expose it as a synchronous helper for
        unit tests and the future dispatch call site.

        Args:
            memory_obj: The disk-anchored MemoryObj.
            gpu_buffer: Destination GPU tensor (slice of registered
                staging buffer).

        Raises:
            ValueError: If ``gpu_buffer`` is not within the registered
                region, or sizes mismatch.
            RuntimeError: If the read returns short.
        """
        dev_offset = self._dev_offset(gpu_buffer)
        nbytes = memory_obj.get_size()
        if nbytes > gpu_buffer.numel() * gpu_buffer.element_size():
            raise ValueError(
                f"cufile_read_into: memory_obj size {nbytes} exceeds "
                f"gpu_buffer capacity "
                f"{gpu_buffer.numel() * gpu_buffer.element_size()}"
            )
        if self._registered_buffer is None:
            raise RuntimeError("cufile_read_into: no GPU buffer has been registered")
        if self._use_async_ctypes:
            self._ctypes_read(memory_obj, dev_offset, nbytes)
            return
        ret = self._backend.do_load(
            memory_obj.disk_path,
            file_offset=memory_obj.file_offset,
            buf=self._registered_buffer,
            dev_offset=dev_offset,
            size_in_bytes=nbytes,
        )
        if ret != nbytes:
            raise RuntimeError(
                f"cufile_read_into: short read for {memory_obj.disk_path} — "
                f"got {ret} bytes, expected {nbytes}"
            )

    def cufile_write_from(
        self,
        memory_obj: GdsMemoryObj,
        gpu_buffer: torch.Tensor,
    ) -> None:
        """Write ``gpu_buffer`` to ``memory_obj``'s disk file.

        Creates the file (atomically via tmp + rename), writes the
        metadata header and payload, and registers the entry in the
        backend's hot index.

        Args:
            memory_obj: The disk-anchored MemoryObj to populate.
            gpu_buffer: Source GPU tensor (slice of registered
                staging buffer).

        Raises:
            ValueError: If sizes mismatch.
            RuntimeError: If the write fails.
        """
        dev_offset = self._dev_offset(gpu_buffer)
        nbytes = memory_obj.get_size()
        if nbytes > gpu_buffer.numel() * gpu_buffer.element_size():
            raise ValueError(
                f"cufile_write_from: memory_obj size {nbytes} exceeds "
                f"gpu_buffer capacity "
                f"{gpu_buffer.numel() * gpu_buffer.element_size()}"
            )
        if self._registered_buffer is None:
            raise RuntimeError("cufile_write_from: no GPU buffer has been registered")
        if self._use_async_ctypes:
            self._ctypes_write(memory_obj, dev_offset, nbytes)
            return
        self._backend.do_save(
            memory_obj=memory_obj,
            buf=self._registered_buffer,
            dev_offset=dev_offset,
            nbytes=nbytes,
        )

    # --- direct-ctypes cuFile async path ---------------------------

    def _ctypes_read(
        self, memory_obj: GdsMemoryObj, dev_offset: int, nbytes: int
    ) -> None:
        """Submit a cuFileReadAsync on the current torch stream.

        No per-call sync — the submission is kept alive in
        ``_inflight_subs`` and the bytes land in the registered VRAM
        before any subsequent kernel on the same stream runs (CUDA
        stream ordering). Callers wanting to see ``Submission.
        bytes_done`` must first synchronise the stream.
        """
        # First Party

        handle = self._get_or_open_async_handle(memory_obj.disk_path, writable=False)
        raw_stream = torch.cuda.current_stream().cuda_stream
        sub = handle.read_async(
            self._registered_base_ptr,
            nbytes,
            memory_obj.file_offset,
            dev_offset,
            raw_stream,
        )
        self._inflight_subs.append(sub)

    def _ctypes_write(
        self, memory_obj: GdsMemoryObj, dev_offset: int, nbytes: int
    ) -> None:
        """Submit a cuFileWriteAsync on the current torch stream.

        The file is pre-allocated to the chunk's payload size + the
        4 KiB metadata header before submitting, then the metadata
        sidecar + hot-index entry are written synchronously. Same
        no-per-call-sync pattern as :meth:`_ctypes_read`.
        """
        # First Party
        from lmcache.v1.distributed import _cufile_async as ca

        disk_path = memory_obj.disk_path
        os.makedirs(os.path.dirname(disk_path), exist_ok=True)
        tmp_path = disk_path + ".tmp" + _rand_suffix(8)

        metadata_bytes = pack_metadata(
            nbytes=nbytes,
            shape=memory_obj.meta.shape,
            dtype=memory_obj.meta.dtype,
            fmt=memory_obj.meta.fmt,
            lmcache_version=str(_METADATA_VERSION),
        )
        # Lay down the metadata header on a regular fd (O_DIRECT
        # writes need aligned sizes/offsets and a 4 KiB header isn't
        # the world; let the kernel write it normally), then re-open
        # via the ctypes AsyncHandle to write the GDS payload.
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata_bytes)
            payload_size = nbytes
            # Pre-allocate the full file (header + payload) so the
            # cuFile write doesn't have to grow the file.
            fd = os.open(tmp_path, os.O_RDWR)
            try:
                os.posix_fallocate(fd, 0, _METADATA_MAX_SIZE + payload_size)
            finally:
                os.close(fd)
            with ca.AsyncHandle(tmp_path, writable=True) as handle:
                raw_stream = torch.cuda.current_stream().cuda_stream
                sub = handle.write_async(
                    self._registered_base_ptr,
                    payload_size,
                    _METADATA_MAX_SIZE,
                    dev_offset,
                    raw_stream,
                )
                # The AsyncHandle context manager calls close() on exit;
                # the cuFile handle deregister implicitly waits for
                # in-flight ops on it (kernel-side ref counting). Force
                # a stream sync here too so the bytes_done we read is
                # actually settled.
                torch.cuda.current_stream().synchronize()
                actual = sub.bytes_done
            if actual != payload_size:
                raise RuntimeError(
                    f"_ctypes_write: short write for {disk_path} — "
                    f"got {actual} bytes, expected {payload_size}"
                )
            os.rename(tmp_path, disk_path)
            for mode in ("r", "r+"):
                self.handle_cache_invalidate(disk_path, mode)
            self._backend._record_save(memory_obj, disk_path, nbytes)  # noqa: SLF001
            self._backend._write_metadata_sidecar(disk_path, metadata_bytes)  # noqa: SLF001
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            logger.exception("_ctypes_write failed for %s", disk_path)
            raise

    def _get_or_open_async_handle(self, disk_path: str, writable: bool) -> object:
        """Cache opened ``AsyncHandle``s per disk_path.

        cuFileHandleRegister is non-trivial; opening once per file and
        keeping the handle alive across reads/writes is a meaningful
        speedup for the chunk-heavy access pattern.
        """
        # First Party
        from lmcache.v1.distributed import _cufile_async as ca

        cache_key = (disk_path, writable)
        h = self._async_handles.get(cache_key)
        if h is None:
            h = ca.AsyncHandle(disk_path, writable=writable)
            self._async_handles[cache_key] = h
        return h

    def handle_cache_invalidate(self, disk_path: str, mode: str) -> None:
        """Forward to the kvikio handle cache.

        Exposed so ``_ctypes_write`` (and any external caller that
        replaces a file under us) can drop stale fds — see the
        analogous call in ``GdsL1Backend.do_save``.
        """
        # First Party
        # The async path doesn't share the kvikio handle cache, but a
        # mode-flagged ``AsyncHandle`` for the same path would be
        # stale after a rename; drop it from our own cache too.
        for cache_key in [(disk_path, True), (disk_path, False)]:
            h = self._async_handles.pop(cache_key, None)
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
        if self._backend.handle_cache is not None:
            self._backend.handle_cache.invalidate(disk_path, mode)

    # --- MemoryAllocatorInterface (mostly no-ops) -------------------

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Not supported — GDS L1 allocates via ``GdsL1Backend``.

        Callers should use :meth:`GdsL1Backend.create_memory_obj`
        instead, which takes the ``ObjectKey`` needed to derive the
        disk path.

        Raises:
            NotImplementedError: Always.
        """
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
        """Not supported — see :meth:`allocate`."""
        raise NotImplementedError(
            "GdsScratchAllocator.batched_allocate: use "
            "GdsL1Backend.create_memory_objs(keys, layout_desc) instead"
        )

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """No-op. ``GdsMemoryObj`` carries no in-memory allocation."""
        return

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """No-op. See :meth:`free`."""
        return

    # --- Private -----------------------------------------------------

    def _dev_offset(self, gpu_buffer: torch.Tensor) -> int:
        """Compute the offset of ``gpu_buffer`` within the registered base.

        Raises:
            ValueError: If no buffer has been registered, or
                ``gpu_buffer`` does not lie within the registered region.
        """
        if self._registered_base_ptr is None:
            raise ValueError(
                "GdsScratchAllocator: no GPU buffer has been registered yet"
            )
        offset = gpu_buffer.data_ptr() - self._registered_base_ptr
        if offset < 0 or offset >= self._registered_nbytes:
            raise ValueError(
                f"GdsScratchAllocator: gpu_buffer pointer 0x{gpu_buffer.data_ptr():x} "
                f"is outside the registered region "
                f"[0x{self._registered_base_ptr:x}, "
                f"0x{self._registered_base_ptr + self._registered_nbytes:x})"
            )
        return offset


# --- GdsL1Backend ---------------------------------------------------


class GdsL1Backend:
    """The GDS L1 backend: disk-resident L1 with cuFile DMA to GPU.

    Owns the in-memory hot index of disk-resident keys, the cuFile
    handle cache, an I/O thread pool, and the
    :class:`GdsScratchAllocator` that tags
    :class:`GdsMemoryObj` instances.

    Construction kicks off an async filesystem scan that populates
    the hot index; ``lookup`` calls during the scan window may miss
    keys that have not yet been indexed. This matches the non-MP
    ``GdsBackend`` behaviour.

    Args:
        config: :class:`GdsL1Config` with the GDS-specific knobs.
            ``gds_path`` is required; everything else has defaults.
        loop: An asyncio event loop on which the metadata scan runs.
        dst_device: Target GPU device string (``"cuda:N"``); used by
            :class:`PathSharder` for multi-path sharding.
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
            raise ValueError(
                f"GdsL1Backend requires a cuda dst_device, got {dst_device}"
            )

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

        # Resolve cuFile vs POSIX fallback. Auto-disable cuFile on
        # filesystems that don't support GDS (e.g. tmpfs, overlayfs)
        # unless the user explicitly asked for cuFile.
        self.use_gds: bool = config.use_gds
        if self.fstype in ("tmpfs", "overlayfs") and config.use_gds:
            logger.info("GdsL1Backend: auto-disabling cuFile on fstype=%r", self.fstype)
            self.use_gds = False

        self.data_suffix = _DATA_FILE_SUFFIX
        if self.fstype == "wekafs":
            self.data_suffix = _WEKA_DATA_FILE_SUFFIX

        self.gds_module: Optional[object] = None
        self.cudart: Optional[object] = None
        if self.use_gds:
            # Third Party
            import kvikio  # noqa: WPS433 — lazy import to avoid load-time failures
            import kvikio._lib.defaults as _kvikio_defaults
            import kvikio.defaults

            self.gds_module = kvikio
            # cuFile's GDS fast path on ext4 (and most local FSes
            # other than wekafs) requires the file to be opened with
            # ``O_DIRECT``. kvikio's ``CuFile`` constructor doesn't
            # accept that flag directly, but it does expose
            # ``auto_direct_io_read`` / ``auto_direct_io_write``
            # process-wide defaults that add ``O_DIRECT`` to its
            # internal open(). Flip both when the operator passes
            # ``use_direct_io=True`` so GDS DMA can actually engage.
            if config.use_direct_io:
                _kvikio_defaults.set_auto_direct_io_read(True)
                _kvikio_defaults.set_auto_direct_io_write(True)
            logger.info(
                "GdsL1Backend: kvikio loaded "
                "(compat_mode_preferred=%s, "
                "auto_direct_io_read=%s, auto_direct_io_write=%s)",
                kvikio.defaults.is_compat_mode_preferred(),
                _kvikio_defaults.auto_direct_io_read(),
                _kvikio_defaults.auto_direct_io_write(),
            )
        else:
            self.cudart = ctypes.CDLL("libcudart.so")
            logger.info("GdsL1Backend: cuFile disabled, using POSIX fallback")

        self.handle_cache = CuFileHandleCache(
            max_handles=config.handle_cache_size,
            gds_module=self.gds_module,
        )

        self._hot_lock = threading.Lock()
        self._hot_cache: OrderedDict[ObjectKey, DiskCacheMetadata] = OrderedDict()
        self._metadata_dirs: set[str] = set()

        self._max_bytes: int = config.disk_max_bytes
        self._disk_bytes: int = 0

        # Public allocator: used by L1Manager via the gds_backend hook
        # and by gpu_ops via ``isinstance(memory_obj.parent(), ...)``.
        self.scratch_allocator = GdsScratchAllocator(self)

        # Kick off the async metadata scan. Misses during the scan
        # window are acceptable; see class docstring.
        self._scan_future = asyncio.run_coroutine_threadsafe(
            self._scan_metadata(), self._loop
        )

    # --- Public API ----------------------------------------------------

    def lookup(self, keys: list[ObjectKey]) -> list[bool]:
        """Return whether each key is present in the hot index.

        Args:
            keys: Object keys to test.

        Returns:
            ``[True if key resident else False, ...]`` in the same
            order as ``keys``.
        """
        with self._hot_lock:
            return [key in self._hot_cache for key in keys]

    def create_memory_obj(
        self,
        key: ObjectKey,
        layout_desc: MemoryLayoutDesc,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> GdsMemoryObj:
        """Construct a ``GdsMemoryObj`` for a *new* write.

        Used by ``L1Manager.reserve_write`` to mint the MemoryObj
        before the data is written. The disk file does not exist yet
        at this point; it is created via
        :meth:`GdsScratchAllocator.cufile_write_from` when the MP
        server completes the store.

        Args:
            key: The ObjectKey to be stored.
            layout_desc: Logical layout for the chunk.
            fmt: Memory format for the chunk.

        Returns:
            A new disk-anchored ``GdsMemoryObj``. Not yet registered
            in the hot index — that happens on successful write.
        """
        disk_path, _, _, _ = key_to_disk_path(
            key, base_path=self.gds_path, data_suffix=self.data_suffix
        )
        shape = layout_desc.shapes[0]
        dtype = layout_desc.dtypes[0]
        nbytes = _compute_layout_bytes(layout_desc)
        meta = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=0,
            phy_size=nbytes,
            ref_count=0,
            pin_count=0,
            fmt=fmt,
            shapes=list(layout_desc.shapes),
            dtypes=list(layout_desc.dtypes),
        )
        return GdsMemoryObj(
            key=key,
            disk_path=disk_path,
            file_offset=_METADATA_MAX_SIZE,
            metadata=meta,
            parent_allocator=self.scratch_allocator,
        )

    def create_memory_obj_from_index(self, key: ObjectKey) -> Optional[GdsMemoryObj]:
        """Construct a ``GdsMemoryObj`` for a *resident* (on-disk) key.

        Used by ``L1Manager.reserve_read`` on the fill-on-miss path:
        the key is in the hot index, so we synthesise a MemoryObj
        from the recorded :class:`DiskCacheMetadata`.

        Args:
            key: The key to resolve.

        Returns:
            ``None`` if the key is not in the hot index, otherwise a
            new ``GdsMemoryObj`` referencing the existing disk file.
        """
        with self._hot_lock:
            entry = self._hot_cache.get(key)
        if entry is None:
            return None
        if entry.shape is None or entry.dtype is None or entry.fmt is None:
            logger.warning(
                "GdsL1Backend: hot-index entry for %s missing shape/dtype/fmt",
                entry.path,
            )
            return None
        meta = MemoryObjMetadata(
            shape=entry.shape,
            dtype=entry.dtype,
            address=0,
            phy_size=entry.size,
            ref_count=0,
            pin_count=0,
            fmt=entry.fmt,
        )
        return GdsMemoryObj(
            key=key,
            disk_path=entry.path,
            file_offset=_METADATA_MAX_SIZE,
            metadata=meta,
            parent_allocator=self.scratch_allocator,
        )

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(disk_bytes_used, disk_bytes_total)`` for the L1
        eviction signal.

        ``disk_bytes_total`` is ``gds_disk_max_bytes`` from extra
        config; ``0`` means no capacity advertised, which the
        eviction controller treats as "no global eviction signal."
        """
        with self._hot_lock:
            return self._disk_bytes, self._max_bytes

    def get_hot_cache_size(self) -> int:
        """Return the number of resident keys. Useful for tests."""
        with self._hot_lock:
            return len(self._hot_cache)

    def wait_for_scan(self, timeout: float = 30.0) -> None:
        """Block until the startup metadata scan completes.

        Optional helper for tests and callers that want deterministic
        startup; production code should rely on the async scan.

        Args:
            timeout: Seconds to wait. ``0`` waits indefinitely.

        Raises:
            TimeoutError: If the scan does not complete in time.
        """
        try:
            self._scan_future.result(timeout=timeout if timeout > 0 else None)
        except Exception as e:
            logger.warning("GdsL1Backend: scan failed or timed out: %s", e)
            raise

    def close(self) -> None:
        """Flush and release all resources."""
        # Best-effort: wait for an in-flight scan to settle before
        # tearing down the handle cache and registered buffer.
        try:
            self._scan_future.result(timeout=30)
        except Exception as e:
            logger.warning("GdsL1Backend.close: scan wait failed: %s", e)
        self.handle_cache.close()
        self.scratch_allocator.deregister_gpu_buffer()
        logger.info("GdsL1Backend: closed")

    # --- I/O methods used by GdsScratchAllocator ----------------------

    def do_load(
        self,
        disk_path: str,
        file_offset: int,
        buf: torch.Tensor,
        dev_offset: int,
        size_in_bytes: int,
    ) -> int:
        """Read ``size_in_bytes`` from ``disk_path`` into ``buf`` at
        ``dev_offset`` bytes from the start of the registered region.

        Returns:
            Number of bytes read, or a negative value on failure
            (mirrors :class:`GdsBackend` semantics so callers can
            log and drop the hot-index entry).
        """
        if self.gds_module is not None:
            return self._gds_read(
                disk_path, file_offset, buf, dev_offset, size_in_bytes
            )
        return self._posix_read(
            disk_path, file_offset, buf.data_ptr(), dev_offset, size_in_bytes
        )

    def do_save(
        self,
        memory_obj: GdsMemoryObj,
        buf: torch.Tensor,
        dev_offset: int,
        nbytes: int,
    ) -> None:
        """Write ``nbytes`` from ``buf`` at ``dev_offset`` to
        ``memory_obj.disk_path``.

        On success, registers the key in the hot index. On failure,
        logs and re-raises so the caller can handle.
        """
        disk_path = memory_obj.disk_path
        # Late create the parent dirs so we tolerate first writes per
        # subdir cleanly without preallocating the full tree.
        os.makedirs(os.path.dirname(disk_path), exist_ok=True)

        tmp_suffix = ".tmp" + _rand_suffix(8)
        tmp_path = disk_path + tmp_suffix

        metadata_bytes = pack_metadata(
            nbytes=nbytes,
            shape=memory_obj.meta.shape,
            dtype=memory_obj.meta.dtype,
            fmt=memory_obj.meta.fmt,
            lmcache_version=str(_METADATA_VERSION),
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata_bytes)
            if self.gds_module is not None:
                self._gds_write(
                    tmp_path,
                    file_offset=_METADATA_MAX_SIZE,
                    buf=buf,
                    dev_offset=dev_offset,
                    nbytes=nbytes,
                )
            else:
                self._posix_write(
                    tmp_path,
                    file_offset=_METADATA_MAX_SIZE,
                    base_pointer=buf.data_ptr(),
                    dev_offset=dev_offset,
                    nbytes=nbytes,
                )
            os.rename(tmp_path, disk_path)
            # Invalidate any cached "r"/"r+" cuFile handles for the
            # old inode at this path. ``os.rename`` replaces the
            # directory entry but doesn't disturb the previously
            # opened fd, so a stale handle would happily keep reading
            # the unlinked inode (whose size is the old write's
            # nbytes, not the new one). This shows up as short reads
            # immediately after an overwrite.
            for mode in ("r", "r+"):
                self.handle_cache.invalidate(disk_path, mode)
            self._record_save(memory_obj, disk_path, nbytes)
            # Write the sibling metadata file atomically.
            self._write_metadata_sidecar(disk_path, metadata_bytes)
        except Exception:
            # Clean up the half-written tmp file if it still exists.
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            logger.exception("GdsL1Backend.do_save failed for %s", disk_path)
            raise

    # --- Private internals --------------------------------------------

    async def _scan_metadata(self) -> None:
        """Async startup scan over all gds_paths.

        Populates :attr:`_hot_cache` and :attr:`_disk_bytes`. Logs
        elapsed time and entry count on completion.
        """
        start = time.perf_counter()
        tasks = []
        for p in self.gds_paths:
            try:
                entries = os.scandir(p)
            except FileNotFoundError:
                continue
            with entries as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    l1_dir = os.path.basename(entry.name)
                    if len(l1_dir) != 2:
                        continue
                    tasks.append(
                        asyncio.to_thread(
                            self._scan_metadata_subdir,
                            os.path.join(p, l1_dir),
                            l1_dir,
                        )
                    )
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - start
        logger.info(
            "GdsL1Backend: scan complete (%d resident, %.2fs)",
            self.get_hot_cache_size(),
            elapsed,
        )

    def _scan_metadata_subdir(self, path: str, l1_dir: str) -> None:
        """Scan a single first-level directory.

        Picks up every ``*<data_suffix>.metadata`` file under
        ``path/<l2_dir>/`` and feeds them through
        :meth:`_read_metadata`.
        """
        target_suffix = self.data_suffix + _METADATA_FILE_SUFFIX
        try:
            l2_entries = os.scandir(path)
        except FileNotFoundError:
            return
        with l2_entries as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l2_dir = os.path.basename(entry.name)
                if len(l2_dir) != 2:
                    continue
                try:
                    file_entries = os.scandir(os.path.join(path, l2_dir))
                except FileNotFoundError:
                    continue
                with file_entries as it2:
                    for fentry in it2:
                        if not fentry.is_file() or not fentry.name.endswith(
                            target_suffix
                        ):
                            continue
                        filename = os.path.basename(fentry.name)
                        key_str = urllib.parse.unquote(filename[: -len(target_suffix)])
                        key = _parse_object_key(key_str)
                        if key is None:
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for %s, ignoring",
                                fentry.path,
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to read metadata for %s: %s",
                                fentry.path,
                                e,
                            )

    def _read_metadata(
        self, key: ObjectKey, metadata_path: str, subdir_key: str
    ) -> DiskCacheMetadata:
        """Read a single metadata sidecar file and register the entry."""
        with open(metadata_path, "rb") as f:
            buf = f.read(_METADATA_MAX_SIZE)
        shape, dtype, nbytes, fmt, _ = unpack_metadata(buf)
        data_path = metadata_path.removesuffix(_METADATA_FILE_SUFFIX)
        meta = DiskCacheMetadata(data_path, nbytes, shape, dtype, None, fmt)
        with self._hot_lock:
            self._metadata_dirs.add(subdir_key)
            self._hot_cache[key] = meta
            self._disk_bytes += nbytes
        return meta

    def _write_metadata_sidecar(self, data_path: str, metadata_bytes: bytes) -> None:
        """Atomically write the ``.metadata`` sidecar next to ``data_path``."""
        sidecar = data_path + _METADATA_FILE_SUFFIX
        tmp = sidecar + ".tmp" + _rand_suffix(8)
        with open(tmp, "wb") as f:
            f.write(metadata_bytes)
        os.rename(tmp, sidecar)

    def _record_save(
        self, memory_obj: GdsMemoryObj, disk_path: str, nbytes: int
    ) -> None:
        """Insert a fresh write into the hot index.

        Called from the I/O worker thread on the store path; holds
        ``_hot_lock`` only briefly.
        """
        entry = DiskCacheMetadata(
            disk_path,
            nbytes,
            memory_obj.meta.shape,
            memory_obj.meta.dtype,
            None,
            memory_obj.meta.fmt,
        )
        with self._hot_lock:
            existing = self._hot_cache.get(memory_obj.key)
            if existing is not None:
                # Overwrite: account for the size delta only.
                self._disk_bytes += nbytes - existing.size
            else:
                self._disk_bytes += nbytes
            self._hot_cache[memory_obj.key] = entry

    def _gds_read(
        self,
        disk_path: str,
        file_offset: int,
        buf: torch.Tensor,
        dev_offset: int,
        size_in_bytes: int,
    ) -> int:
        """CUDA-stream-aware cuFile read into the registered region.

        Uses ``kvikio.CuFile.raw_read_async`` so the read is enqueued
        on the current torch CUDA stream — the same dispatch shape as
        ``multi_layer_block_kv_transfer`` and the existing
        ``cudaMemcpyAsync`` in ``lmcache_memcpy_async_h2d``. When
        ``nvidia-fs`` is loaded the read is truly stream-ordered and
        the next kernel on the same stream waits for it implicitly;
        in compat mode kvikio fakes the stream semantics with an
        internal thread + completion callback, so
        ``check_bytes_done()`` forces the wait to keep the API
        behaving consistently across environments.

        Trade-off: ``raw_read_async`` is the peer of
        ``multi_layer_block_kv_transfer`` in the stream-ordering
        model and is the right primitive once we can defer the sync.
        Today we still call ``check_bytes_done()`` here because
        callers invoke ``_gds_read`` one chunk at a time — batching
        the deferred sync requires a downstream API change.
        """
        try:
            handle = self.handle_cache.acquire(disk_path, "r")
            try:
                stream = torch.cuda.current_stream().cuda_stream
                future = handle.raw_read_async(
                    buf, stream, size_in_bytes, file_offset, dev_offset
                )
                return future.check_bytes_done()
            finally:
                self.handle_cache.release(disk_path, "r")
        except Exception as e:
            logger.error("GDS read failed for %s: %s", disk_path, e, exc_info=True)
            return -1

    def _gds_write(
        self,
        disk_path: str,
        file_offset: int,
        buf: torch.Tensor,
        dev_offset: int,
        nbytes: int,
    ) -> None:
        """CUDA-stream-aware cuFile write from the registered region.

        Uses ``raw_write_async`` for the same reasons as
        :meth:`_gds_read` — see that method's docstring.
        """
        handle = self.handle_cache.acquire(disk_path, "r+")
        try:
            stream = torch.cuda.current_stream().cuda_stream
            future = handle.raw_write_async(
                buf, stream, nbytes, file_offset, dev_offset
            )
            future.check_bytes_done()
        finally:
            self.handle_cache.release(disk_path, "r+")

    def _posix_read(
        self,
        disk_path: str,
        file_offset: int,
        base_pointer: int,
        dev_offset: int,
        size_in_bytes: int,
    ) -> int:
        """Fallback read path: mmap + cudaMemcpy. Mirrors GdsBackend."""
        if self.cudart is None:
            raise RuntimeError("GdsL1Backend._posix_read: cudart not loaded")
        try:
            fd = os.open(disk_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            if file_size < file_offset + size_in_bytes:
                os.close(fd)
                logger.error(
                    "POSIX read: %s too small (size=%d, need=%d)",
                    disk_path,
                    file_size,
                    file_offset + size_in_bytes,
                )
                return -1
            mm = mmap.mmap(
                fd,
                file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)
            arr = np.frombuffer(mm, dtype=np.uint8)
            src_addr = arr.__array_interface__["data"][0]
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(base_pointer + dev_offset),
                ctypes.c_void_p(src_addr + file_offset),
                ctypes.c_size_t(size_in_bytes),
                ctypes.c_int(1),  # cudaMemcpyHostToDevice
            )
            del arr
            mm.close()
            if res != 0:
                logger.error("cudaMemcpy failed for %s with code %d", disk_path, res)
                return -1
            return size_in_bytes
        except Exception as e:
            logger.error("POSIX read failed for %s: %s", disk_path, e, exc_info=True)
            return -1

    def _posix_write(
        self,
        disk_path: str,
        file_offset: int,
        base_pointer: int,
        dev_offset: int,
        nbytes: int,
    ) -> None:
        """Fallback write path: ftruncate + mmap + cudaMemcpy."""
        if self.cudart is None:
            raise RuntimeError("GdsL1Backend._posix_write: cudart not loaded")
        fd = os.open(disk_path, os.O_RDWR)
        try:
            os.ftruncate(fd, nbytes + file_offset)
            mm = mmap.mmap(
                fd,
                nbytes + file_offset,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED,
            )
        finally:
            os.close(fd)
        try:
            arr = np.frombuffer(mm, dtype=np.uint8)
            dst_addr = arr.__array_interface__["data"][0]
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(dst_addr + file_offset),
                ctypes.c_void_p(base_pointer + dev_offset),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(2),  # cudaMemcpyDeviceToHost
            )
            if res != 0:
                raise RuntimeError(f"cudaMemcpy D2H failed with code {res}")
            del arr
        finally:
            mm.close()


# --- helpers --------------------------------------------------------


def _parse_object_key(key_str: str) -> Optional[ObjectKey]:
    """Parse the filename-embedded ObjectKey string back into an ``ObjectKey``.

    Args:
        key_str: String produced by :func:`_object_key_to_string`.

    Returns:
        ``ObjectKey`` on success, ``None`` on parse failure (with a log).
    """
    parts = key_str.split("@")
    if len(parts) != 4:
        logger.warning(
            "GdsL1Backend: cannot parse ObjectKey from %r (got %d parts)",
            key_str,
            len(parts),
        )
        return None
    model_name, kv_rank_str, hash_hex, cache_salt = parts
    try:
        kv_rank = int(kv_rank_str)
        chunk_hash = bytes.fromhex(hash_hex)
    except ValueError as e:
        logger.warning("GdsL1Backend: invalid kv_rank/chunk_hash in %r: %s", key_str, e)
        return None
    try:
        return ObjectKey(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            cache_salt=cache_salt,
        )
    except ValueError as e:
        logger.warning("GdsL1Backend: invalid ObjectKey fields in %r: %s", key_str, e)
        return None


def _compute_layout_bytes(layout_desc: MemoryLayoutDesc) -> int:
    """Total byte size of a MemoryLayoutDesc (sum across groups)."""
    total = 0
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        total += shape.numel() * dtype.itemsize
    return total
