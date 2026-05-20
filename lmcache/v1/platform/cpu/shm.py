# SPDX-License-Identifier: Apache-2.0
"""CPU-only KV-cache IPC wrapper backed by POSIX shared memory.

Mirrors the GPU-mode CUDA-IPC zero-copy semantics for hosts without an
accelerator: client and LMCache mp server map the **same** physical
pages so transfers are pointer-shuffles rather than memcpys.

Self-registers a ``"cpu"`` factory with
:mod:`lmcache.v1.platform._registry` at import time, so the
multiprocess adapter can dispatch by ``tensor.device.type`` without
any if/elif chain.
"""

# Future
from __future__ import annotations

# Standard
import ctypes
import ctypes.util
import itertools
import os
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Hand-rolled POSIX shared-memory helpers (shm_open + mmap via ctypes).
#
# Kept here (rather than in ``multiprocess/custom_types``) so the
# CPU-specific dependency on libc/librt lives next to the only place
# that needs it. TODO(maobaolong): replace with ``posix_ipc`` once we
# are willing to take that runtime dependency.
#
# We deliberately do not use stdlib ``mmap`` here: ``mmap.mmap`` would
# work for the in-process side, but we still need ``shm_open`` /
# ``shm_unlink`` (not exposed by stdlib), and we hand the raw address
# to ``ctypes.from_address`` + ``torch.frombuffer`` to share storage
# with the migrated tensor. So we keep the raw mmap pointers and
# pair every successful mmap with a matching ``munmap`` -- on the
# error paths via ``try/finally``, on the happy path via
# ``weakref.finalize`` hooks attached to the migrated tensor and to
# every ``CpuShmTensorWrapper.to_tensor()`` view.
# ---------------------------------------------------------------------------

_O_RDWR = os.O_RDWR
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_SHARED = 0x01

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
# macOS exposes shm_open in libSystem (== libc), Linux needs librt.
_librt = _libc
if not hasattr(_libc, "shm_open"):
    _librt = ctypes.CDLL(ctypes.util.find_library("rt"), use_errno=True)

_librt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
_librt.shm_open.restype = ctypes.c_int
_librt.shm_unlink.argtypes = [ctypes.c_char_p]
_librt.shm_unlink.restype = ctypes.c_int

_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_int64]
_libc.ftruncate.restype = ctypes.c_int
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int64,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int
_libc.close.argtypes = [ctypes.c_int]
_libc.close.restype = ctypes.c_int

_MAP_FAILED = ctypes.c_void_p(-1).value


def shm_create_readwrite(name: str, nbytes: int) -> int:
    """Create + size a POSIX SHM segment, return mapped address.

    Every failure path tears down whatever has already been allocated
    (fd, named segment) so the caller never has to compensate.
    """
    name_b = name.encode("ascii")
    fd = _librt.shm_open(name_b, _O_RDWR | _O_CREAT | _O_EXCL, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "shm_open(create) failed for %s" % name)
    addr = 0
    try:
        if _libc.ftruncate(fd, nbytes) != 0:
            raise OSError(ctypes.get_errno(), "ftruncate failed for %s" % name)
        addr = _libc.mmap(None, nbytes, _PROT_READ | _PROT_WRITE, _MAP_SHARED, fd, 0)
        if addr in (0, _MAP_FAILED):
            addr = 0
            raise OSError(ctypes.get_errno(), "mmap failed for %s" % name)
    except BaseException:
        # Roll back whatever we have so far: the named segment is
        # always created at this point; mmap may or may not have
        # succeeded.
        if addr:
            _libc.munmap(ctypes.c_void_p(addr), nbytes)
        _librt.shm_unlink(name_b)
        raise
    finally:
        _libc.close(fd)
    return addr


def shm_map_readwrite(name: str, nbytes: int) -> int:
    """Open an existing POSIX SHM segment, return mapped address.

    The fd is always closed before returning (success or failure) so
    we never leak a file descriptor even when ``mmap`` fails.
    """
    fd = _librt.shm_open(name.encode("ascii"), _O_RDWR, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "shm_open(open) failed for %s" % name)
    try:
        addr = _libc.mmap(None, nbytes, _PROT_READ | _PROT_WRITE, _MAP_SHARED, fd, 0)
        if addr in (0, _MAP_FAILED):
            raise OSError(ctypes.get_errno(), "mmap failed for %s" % name)
    finally:
        _libc.close(fd)
    return addr


def shm_munmap(addr: int, nbytes: int) -> None:
    """Best-effort ``munmap`` of a previously ``mmap``-ed SHM segment."""
    if not addr or addr == _MAP_FAILED:
        return
    _libc.munmap(ctypes.c_void_p(addr), nbytes)


def shm_unlink(name: str) -> None:
    """Best-effort SHM segment removal."""
    _librt.shm_unlink(name.encode("ascii"))


# ---------------------------------------------------------------------------
# Wrapper class                                                             #
# ---------------------------------------------------------------------------


class CpuShmTensorWrapper(CudaIPCWrapper):
    """IPC wrapper for CPU tensors backed by POSIX shared memory.

    Used by the ``lmcache bench kvcache --mode cpu`` path and the
    vLLM CPU integration so that the client and the LMCache mp server
    map the **same** physical pages for the KV cache, mirroring the
    GPU-mode CUDA-IPC zero-copy semantics.

    Subclassing :class:`CudaIPCWrapper` is load-bearing for the same
    reason :class:`RawCudaIPCWrapper` does it: msgspec does not
    support unions of custom ext-encoded types, so all wire-level
    KV-cache wrappers must share the single ext code (1) registered
    for ``CudaIPCWrapper``. Pickle preserves the subclass identity
    so ``to_tensor`` dispatches correctly on both sides.
    """

    # POSIX shared-memory name (``/lmcache_...``) -- leading ``/`` is
    # required by ``shm_open(3)`` on both Linux and macOS.
    SHM_NAME_PREFIX = "/lmcache_kv_"

    def __init__(self, tensor: torch.Tensor, shm_name: str) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            attempt_permute_to_contiguous_view,
        )

        tensor = attempt_permute_to_contiguous_view(tensor)
        if tensor.device.type != "cpu":
            raise ValueError(
                "CpuShmTensorWrapper requires a CPU tensor, got %s" % tensor.device
            )
        if not tensor.is_contiguous():
            raise ValueError("CpuShmTensorWrapper requires a contiguous tensor")

        self.shm_name = shm_name
        # ``numel * element_size`` is the correct logical byte size; the
        # underlying storage may be larger when the tensor is a view.
        self.nbytes = tensor.numel() * tensor.element_size()

        # CudaIPCWrapper interface fields. ``handle`` / ``device_uuid``
        # are unused on the CPU path but kept to satisfy the parent
        # contract used by equality checks.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())
        self.device_uuid = "cpu"

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor by mapping the same SHM segment.

        The returned tensor owns the mmap: a ``weakref.finalize`` hook
        runs ``munmap`` once the tensor (and any views derived from it)
        is garbage-collected, so the per-process virtual address space
        does not leak across repeated ``to_tensor`` calls.

        We rebuild the view through ``as_strided`` so the original
        memory layout (stride / storage_offset / memory_format) is
        replayed faithfully on the receiving side; reshape would
        silently re-coalesce strides and lose, e.g., channels_last.
        """
        addr = shm_map_readwrite(self.shm_name, self.nbytes)
        # ``torch.frombuffer`` requires a writable buffer; build one
        # via ctypes so the resulting torch tensor shares storage
        # with the SHM mapping (zero copy across processes).
        buf_type = ctypes.c_uint8 * self.nbytes
        buf = buf_type.from_address(addr)
        flat = torch.frombuffer(buf, dtype=torch.uint8)
        typed = flat.view(self.dtype)
        out = torch.as_strided(typed, self.shape, self.stride, self.storage_offset)
        # Keep ``flat`` alive for the lifetime of ``out`` so its mmap
        # is not released while still in use, then munmap on cleanup.
        out._lmcache_shm_buf = flat  # type: ignore[attr-defined]
        weakref.finalize(out, shm_munmap, addr, self.nbytes)
        return out


# ---------------------------------------------------------------------------
# Migrate-and-wrap factory (used by the multiprocess adapter)              #
# ---------------------------------------------------------------------------

# Per-process registry of SHM segments we have created, so the same
# tensor object is only migrated to SHM once even if the factory is
# called multiple times.
#
# Keyed by ``id(tensor)`` for cheap O(1) lookup, but each entry also
# holds a ``weakref.ref`` to the original tensor and we *verify the
# referent is still that exact object* before reusing the cached SHM
# name. CPython recycles object IDs, so a fresh tensor allocated at
# the same address as a previously migrated (now garbage-collected)
# one would otherwise inherit a stale name -- and because
# :func:`shm_create_readwrite` uses ``O_EXCL``, the next migration
# would crash with ``EEXIST`` ("File exists"). The weakref-validated
# lookup below makes that race impossible: a stale entry can only
# point at a dead referent, which we treat as a miss.
_CPU_SHM_NAMES: dict[int, tuple["weakref.ReferenceType[torch.Tensor]", str]] = {}
_CPU_SHM_LOCK = threading.Lock()
_CPU_SHM_COUNTER = itertools.count()


def _cleanup_shm_segment(tid: int, shm_name: str, addr: int, nbytes: int) -> None:
    """Release the mmap, unlink, and forget the cached SHM name."""
    with _CPU_SHM_LOCK:
        # Only drop the entry if it still points at *this* segment;
        # a future tensor reusing ``tid`` may already have replaced it.
        cached = _CPU_SHM_NAMES.get(tid)
        if cached is not None and cached[1] == shm_name:
            _CPU_SHM_NAMES.pop(tid, None)
    shm_munmap(addr, nbytes)
    shm_unlink(shm_name)


def migrate_to_shm_and_wrap(tensor: torch.Tensor) -> CpuShmTensorWrapper:
    """Re-point ``tensor``'s storage at a POSIX SHM segment, then wrap.

    Used as the registered ``"cpu"`` KV-wrapper factory: the LMCache mp
    server can mmap the same physical pages on the receiving side.
    Idempotent per tensor identity (validated via a stored weakref so
    Python's id-recycling cannot produce a stale-name hit). The SHM
    segment is released (``munmap`` + ``shm_unlink``) automatically
    when the migrated tensor is garbage-collected.
    """
    tid = id(tensor)
    with _CPU_SHM_LOCK:
        cached = _CPU_SHM_NAMES.get(tid)
        if cached is not None:
            ref, cached_name = cached
            if ref() is tensor:
                return CpuShmTensorWrapper(tensor, cached_name)
            # Stale entry from a GC'd tensor whose id has been
            # reused; drop it and fall through to allocate fresh.
            _CPU_SHM_NAMES.pop(tid, None)

        nbytes = tensor.numel() * tensor.element_size()
        shm_name = "%s%d_%d" % (
            CpuShmTensorWrapper.SHM_NAME_PREFIX,
            os.getpid(),
            next(_CPU_SHM_COUNTER),
        )
        addr = shm_create_readwrite(shm_name, nbytes)
        try:
            buf_type = ctypes.c_uint8 * nbytes
            buf = buf_type.from_address(addr)
            shm_storage = torch.frombuffer(buf, dtype=torch.uint8).untyped_storage()
            tensor.set_(
                shm_storage,
                tensor.storage_offset(),
                tensor.shape,
                tensor.stride(),
            )
        except Exception:
            # Make sure the SHM resources don't leak if migration fails
            # part-way (e.g. ``set_`` rejects an unusual stride).
            shm_munmap(addr, nbytes)
            shm_unlink(shm_name)
            raise

        _CPU_SHM_NAMES[tid] = (weakref.ref(tensor), shm_name)
        weakref.finalize(tensor, _cleanup_shm_segment, tid, shm_name, addr, nbytes)
        logger.info(
            "Migrated CPU KV cache tensor (nbytes=%d) to SHM %s",
            nbytes,
            shm_name,
        )

    return CpuShmTensorWrapper(tensor, shm_name)


def inject_stale_cache_entry_for_test(
    tensor: torch.Tensor,
    dead_ref: "weakref.ReferenceType[torch.Tensor]",
    stale_shm_name: str,
) -> None:
    """Test-only hook: pre-seed the registry with a stale entry.

    Lets unit tests reproduce the CPython id-reuse race -- where a
    fresh tensor lands on the same id as a previously migrated and
    garbage-collected one -- without the per-test global-state
    surgery that would otherwise have to reach into the module's
    private dict / lock.
    """
    with _CPU_SHM_LOCK:
        _CPU_SHM_NAMES[id(tensor)] = (dead_ref, stale_shm_name)
