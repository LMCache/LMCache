# SPDX-License-Identifier: Apache-2.0
"""CPU-only KV-cache IPC wrapper backed by POSIX shared memory.

Mirrors the GPU-mode CUDA-IPC zero-copy semantics for hosts without an
accelerator: client and LMCache mp server map the **same** physical
pages so transfers are pointer-shuffles rather than memcpys.

Bound to ``device_type="cpu"`` via
:attr:`~lmcache.v1.platform.cpu.CpuDeviceSpec.ipc_wrapper_cls`, so the
multiprocess adapter can dispatch by ``tensor.device.type`` without
any if/elif chain.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import ctypes
import itertools
import os
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_map_readwrite,
    shm_munmap,
    shm_unlink,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

logger = init_logger(__name__)

# Re-export POSIX-SHM primitives so existing callers keep working.
# The canonical home is :mod:`lmcache.v1.multiprocess.posix_shm`; new
# code (e.g. the MP non-GPU SHM transport) should import from there.
__all__ = [
    "CpuShmTensorWrapper",
    "inject_stale_cache_entry_for_test",
    "migrate_to_shm_and_wrap",
    "shm_create_readwrite",
    "shm_map_readwrite",
    "shm_munmap",
    "shm_unlink",
]

# ---------------------------------------------------------------------------
# Wrapper class                                                             #
# ---------------------------------------------------------------------------


class CpuShmTensorWrapper(DeviceIPCWrapper):
    """IPC wrapper for CPU tensors backed by POSIX shared memory.

    Used by the ``lmcache bench kvcache --mode cpu`` path and engine
    CPU integrations so that the client and the LMCache mp server
    map the **same** physical pages for the KV cache, mirroring the
    GPU-mode CUDA-IPC zero-copy semantics.

    Subclassing :class:`DeviceIPCWrapper` is load-bearing for the same
    reason :class:`RawCudaIPCWrapper` does it: msgspec does not
    support unions of custom ext-encoded types, so all wire-level
    KV-cache wrappers must share the single ext code (1) registered
    for ``DeviceIPCWrapper``. Pickle preserves the subclass identity
    so ``to_tensor`` dispatches correctly on both sides.
    """

    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding
    #: without instantiating the wrapper.
    device_type: ClassVar[str] = "cpu"

    # POSIX shared-memory name (``/lmcache_...``) -- leading ``/`` is
    # required by ``shm_open(3)`` on both Linux and macOS.
    SHM_NAME_PREFIX = "/lmcache_kv_"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "CpuShmTensorWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Delegates to :func:`migrate_to_shm_and_wrap`, which migrates the
        tensor's storage to a POSIX SHM segment so the LMCache mp server
        can map the same physical pages.

        Args:
            tensor: A contiguous CPU tensor to migrate and wrap.

        Returns:
            A new :class:`CpuShmTensorWrapper` referencing the SHM
            segment that now backs ``tensor``.
        """
        return migrate_to_shm_and_wrap(tensor)

    def __init__(
        self,
        tensor: torch.Tensor,
        shm_name: str,
        segment_nbytes: int | None = None,
    ) -> None:
        """Describe ``tensor``'s view of the SHM segment ``shm_name``.

        Args:
            tensor: The (already SHM-backed) CPU tensor to describe.
            shm_name: POSIX SHM name of the backing segment, or ``""``
                for empty tensors that carry no segment.
            segment_nbytes: Byte length of the SHM segment. Defaults to
                the tensor's own span (``numel * element_size``); pass
                the storage's byte size when views share one segment.

        Raises:
            ValueError: If the tensor is not a contiguous CPU tensor,
                or its view does not fit inside the segment.
        """
        if tensor.device.type != "cpu":
            raise ValueError(
                "CpuShmTensorWrapper requires a CPU tensor, got %s" % tensor.device
            )
        if not tensor.is_contiguous():
            raise ValueError("CpuShmTensorWrapper requires a contiguous tensor")

        self.shm_name = shm_name
        # ``nbytes`` is the mmap length on the receiving side.
        view_nbytes = tensor.numel() * tensor.element_size()
        self.nbytes = view_nbytes if segment_nbytes is None else segment_nbytes

        # DeviceIPCWrapper interface fields. ``handle`` / ``device_uuid``
        # are unused on the CPU path but kept to satisfy the base
        # contract used by equality checks.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())
        self.device_uuid = "cpu"

        # The view must land inside the segment, or ``to_tensor`` on
        # the receiving side would read past the mapping.
        view_end = (self.storage_offset * tensor.element_size()) + view_nbytes
        if self.shm_name and view_end > self.nbytes:
            raise ValueError(
                "CpuShmTensorWrapper: view ends at byte %d but the SHM "
                "segment is only %d bytes; a nonzero storage_offset "
                "requires a segment covering the whole backing storage"
                % (view_end, self.nbytes)
            )

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor by mapping the same SHM segment.

        The returned tensor owns the mmap: a ``weakref.finalize`` hook
        runs ``munmap`` once the tensor (and any views derived from it)
        is garbage-collected, so the per-process virtual address space
        does not leak across repeated ``to_tensor`` calls. Wrappers
        sharing one segment map it independently (pages are shared).

        We rebuild the view through ``as_strided`` so the original
        memory layout (stride / storage_offset / memory_format) is
        replayed faithfully on the receiving side; reshape would
        silently re-coalesce strides and lose, e.g., channels_last.
        """
        # Empty tensors carry no SHM segment (mmap with length 0 is
        # undefined / EINVAL on POSIX); rebuild the empty view in-process.
        if self.nbytes == 0:
            return torch.empty(self.shape, dtype=self.dtype)
        addr = shm_map_readwrite(self.shm_name, self.nbytes)
        # ``torch.frombuffer`` requires a writable buffer; build one
        # via ctypes so the resulting torch tensor shares storage
        # with the SHM mapping (zero copy across processes).
        buf_type = ctypes.c_uint8 * self.nbytes
        buf = buf_type.from_address(addr)
        flat = torch.frombuffer(buf, dtype=torch.uint8)
        # Truncate to a multiple of the element size before the typed
        # view: a whole-storage segment is not guaranteed to divide
        # evenly by this view's dtype width.
        itemsize = self.dtype.itemsize
        typed = flat[: self.nbytes - (self.nbytes % itemsize)].view(self.dtype)
        out = torch.as_strided(typed, self.shape, self.stride, self.storage_offset)
        # Pin the mmap to the *storage*, not the outer tensor: views
        # (reshape / slicing) create new tensor objects that share the
        # storage but do not inherit Python attributes, so a finalizer
        # attached to ``out`` would munmap as soon as ``out`` is GC'd
        # even when a view is still reading the SHM segment.
        # ``UntypedStorage`` is shared across views, so finalizing on it
        # only fires once every view is also dropped.
        storage = out.untyped_storage()
        _CPU_SHM_KEEP_ALIVE[id(storage)] = flat
        weakref.finalize(storage, _release_shm_segment, id(storage), addr, self.nbytes)
        return out


# ---------------------------------------------------------------------------
# Migrate-and-wrap factory (used by the multiprocess adapter)              #
# ---------------------------------------------------------------------------


# Registry record for one migrated backing storage:
# ``(origin_ref, shm_storage_ref, shm_name, segment_nbytes)``.
# ``origin_ref`` weak-references the storage the entry was keyed for (a
# CPython id-recycling collision reads as a miss); ``shm_storage_ref``
# weak-references the SHM-backed storage so a hit can re-point sibling
# views without the registry keeping the segment alive.
_ShmSegmentRecord = tuple[
    "weakref.ReferenceType[object]",
    "weakref.ReferenceType[object]",
    str,
    int,
]


# Per-process registry of SHM segments we have created, so each backing
# storage is migrated only once no matter how many of its views are
# wrapped. Keyed by ``id(untyped_storage)``; PyTorch preserves the
# ``UntypedStorage`` PyObject for the lifetime of its C++ impl, so the
# id is stable while any view is alive. Each record is inserted under
# two keys -- the original storage and the SHM-backed one -- so both
# "wrap a sibling view" and "re-wrap a migrated tensor" hit.
_CPU_SHM_SEGMENTS: dict[int, _ShmSegmentRecord] = {}
_CPU_SHM_LOCK = threading.Lock()
_CPU_SHM_COUNTER = itertools.count()


# Process-level registry that pins the base ``flat`` buffer of every live
# ``to_tensor()`` mmap until its storage is finalized. Keyed by ``id(storage)``,
# which is stable across views because PyTorch caches the storage Python
# wrapper (so reshape / slicing returns the same ``UntypedStorage`` object).
_CPU_SHM_KEEP_ALIVE: dict[int, torch.Tensor] = {}


def _release_shm_segment(storage_id: int, addr: int, nbytes: int) -> None:
    """Drop the pinned base buffer and ``munmap`` the mapping.

    Invoked by ``weakref.finalize`` on the tensor's ``UntypedStorage`` once
    every view of the mapping is gone, so views (e.g. ``reshape`` returning
    a new tensor without ``_lmcache_shm_buf``) cannot trigger a premature
    unmap that would turn into a use-after-free in the next read.
    """
    _CPU_SHM_KEEP_ALIVE.pop(storage_id, None)
    shm_munmap(addr, nbytes)


def _cleanup_shm_segment(
    origin_sid: int,
    shm_sid: int,
    shm_name: str,
    addr: int,
    nbytes: int,
) -> None:
    """Release the mmap, unlink, and forget both registry keys.

    Registered via ``weakref.finalize`` on the SHM-backed storage, so it
    fires only once the *last* view sharing the segment is gone.
    """
    with _CPU_SHM_LOCK:
        for sid in (origin_sid, shm_sid):
            # Only drop an entry still pointing at *this* segment; a
            # future storage reusing the id may already have replaced it.
            cached = _CPU_SHM_SEGMENTS.get(sid)
            if cached is not None and cached[2] == shm_name:
                _CPU_SHM_SEGMENTS.pop(sid, None)
    shm_munmap(addr, nbytes)
    shm_unlink(shm_name)


def migrate_to_shm_and_wrap(tensor: torch.Tensor) -> CpuShmTensorWrapper:
    """Re-point ``tensor``'s storage at a POSIX SHM segment, then wrap.

    Used as the registered ``"cpu"`` KV-wrapper factory: the LMCache mp
    server can mmap the same physical pages on the receiving side.

    The unit of migration is the tensor's *backing storage*: the first
    view of a storage to arrive copies the whole storage into a fresh
    SHM segment; every later view is re-pointed at that segment with
    its own ``storage_offset`` / stride preserved, so all views keep
    aliasing the same bytes. Views must be migrated *before* the buffer
    is written -- only the tensors handed in are re-pointed.

    The segment is released (``munmap`` + ``shm_unlink``) once the last
    migrated view of it is garbage-collected. Concurrent first-time
    migration of views sharing one storage is not supported; the
    register-time ``wrap_kv_caches`` loop wraps sequentially.
    """
    # First Party
    from lmcache.v1.gpu_connector.kv_format.contiguity import (
        attempt_permute_to_contiguous_view,
    )

    # Validate and normalise the tensor *before* touching the registry
    # or mutating storage, so a bad input never leaves things half-done.
    normalized = attempt_permute_to_contiguous_view(tensor)
    if not isinstance(normalized, torch.Tensor):
        raise TypeError(
            "attempt_permute_to_contiguous_view returned %s, expected a tensor"
            % type(normalized)
        )
    if tensor.device.type != "cpu":
        raise ValueError(
            "migrate_to_shm_and_wrap requires a CPU tensor, got %s" % tensor.device
        )
    if not normalized.is_contiguous():
        raise ValueError("migrate_to_shm_and_wrap requires a contiguous tensor")

    if tensor.numel() == 0:
        # No SHM segment for empty tensors: ``mmap`` with length 0
        # is undefined / EINVAL on POSIX. ``to_tensor`` rebuilds an
        # empty view directly when ``shm_name`` is empty.
        return CpuShmTensorWrapper(tensor, "")

    storage = tensor.untyped_storage()
    segment_nbytes = storage.nbytes()

    # Fast path: the storage was already migrated (either this very
    # tensor, or a sibling view of the same buffer). Validate the hit
    # under the lock, re-point outside it.
    shm_storage: torch.UntypedStorage | None = None
    hit_name = ""
    hit_nbytes = 0
    with _CPU_SHM_LOCK:
        cached = _CPU_SHM_SEGMENTS.get(id(storage))
        if cached is not None:
            origin_ref, shm_storage_ref, cached_name, cached_nbytes = cached
            if origin_ref() is storage:
                resolved = shm_storage_ref()
                if resolved is not None:
                    shm_storage = resolved
                    hit_name = cached_name
                    hit_nbytes = cached_nbytes
            if shm_storage is None:
                # Stale entry: a recycled id, or a segment whose last
                # view already died. Fall through to a fresh migration.
                _CPU_SHM_SEGMENTS.pop(id(storage), None)

    if shm_storage is not None:
        if storage is not shm_storage:
            # Sibling view: re-point it, keeping its offset / layout.
            tensor.set_(
                shm_storage,
                normalized.storage_offset(),
                normalized.shape,
                normalized.stride(),
            )
            logger.debug(
                "Re-pointed sibling KV view (offset=%d) onto SHM %s",
                int(tensor.storage_offset()),
                hit_name,
            )
        return CpuShmTensorWrapper(tensor, hit_name, segment_nbytes=hit_nbytes)

    shm_name = "%s%d_%d" % (
        CpuShmTensorWrapper.SHM_NAME_PREFIX,
        os.getpid(),
        next(_CPU_SHM_COUNTER),
    )
    # Perform the heavy work (syscall + copy + tensor mutation) outside
    # the lock to keep the critical section small.
    addr = shm_create_readwrite(shm_name, segment_nbytes)
    try:
        buf_type = ctypes.c_uint8 * segment_nbytes
        buf = buf_type.from_address(addr)
        shm_flat = torch.frombuffer(buf, dtype=torch.uint8)
        # Copy the whole origin storage so sibling views keep their
        # bytes and their offsets stay valid.
        src_flat = torch.empty(0, dtype=torch.uint8)
        src_flat.set_(storage)
        shm_flat.copy_(src_flat)
        shm_storage = shm_flat.untyped_storage()
        tensor.set_(
            shm_storage,
            normalized.storage_offset(),
            normalized.shape,
            normalized.stride(),
        )
    except Exception:
        # Make sure the SHM resources don't leak if migration fails
        # part-way (e.g. ``set_`` rejects an unusual stride).
        shm_munmap(addr, segment_nbytes)
        shm_unlink(shm_name)
        raise

    shm_storage_ref = weakref.ref(shm_storage)
    with _CPU_SHM_LOCK:
        _CPU_SHM_SEGMENTS[id(storage)] = (
            weakref.ref(storage),
            shm_storage_ref,
            shm_name,
            segment_nbytes,
        )
        _CPU_SHM_SEGMENTS[id(shm_storage)] = (
            shm_storage_ref,
            shm_storage_ref,
            shm_name,
            segment_nbytes,
        )
    # Unlink fires when the SHM-backed storage dies, i.e. once the last
    # view sharing the segment is garbage-collected.
    weakref.finalize(
        shm_storage,
        _cleanup_shm_segment,
        id(storage),
        id(shm_storage),
        shm_name,
        addr,
        segment_nbytes,
    )
    logger.info(
        "Migrated CPU KV backing storage (segment_nbytes=%d, view_offset=%d) to SHM %s",
        segment_nbytes,
        int(tensor.storage_offset()),
        shm_name,
    )
    return CpuShmTensorWrapper(tensor, shm_name, segment_nbytes=segment_nbytes)


def inject_stale_cache_entry_for_test(
    tensor: torch.Tensor,
    dead_ref: "weakref.ReferenceType[object]",
    stale_shm_name: str,
) -> None:
    """Test-only hook: pre-seed the registry with a stale entry.

    Lets unit tests reproduce the CPython id-reuse race -- where a
    fresh storage lands on the same id as a previously migrated and
    garbage-collected one -- without the per-test global-state
    surgery that would otherwise have to reach into the module's
    private dict / lock.
    """
    with _CPU_SHM_LOCK:
        _CPU_SHM_SEGMENTS[id(tensor.untyped_storage())] = (
            dead_ref,
            dead_ref,
            stale_shm_name,
            0,
        )
