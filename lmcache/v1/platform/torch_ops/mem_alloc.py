# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing import shared_memory
from typing import Optional, Tuple
import ctypes
import os
import warnings

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform._device_detect import (
    current_device_spec,
    get_torch_device,
)

# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}
_shm_registry: dict[int, shared_memory.SharedMemory] = {}
_buf_registry: dict[int, ctypes.Array] = {}
_pinned_ptr_registry: dict[int, int] = {}  # ptr -> size, for cudaHostUnregister

# Cuda path goes through func cudaHostAlloc, which is
# already page aligned by CUDA spec. This fallback shim mirrors that
# guarantee so consumers that require page-aligned host buffers, in
# particular the Rust raw-block backend when O_DIRECT is enabled, which
# requires page-aligned buffer pointer
try:
    _PAGE_SIZE = os.sysconf("SC_PAGESIZE")
except (AttributeError, ValueError, OSError):
    _PAGE_SIZE = 4096

logger = init_logger(__name__)

# Cached one-shot decision: pin host buffers only when an accelerator is
# present. Probed lazily on first allocation; if a pinned allocation ever
# fails at runtime we flip this to False permanently and fall back to
# pageable memory for all subsequent allocations.
_use_pinned: Optional[bool] = None


def _alloc_page_aligned_pinned_view(size: int) -> Tuple[torch.Tensor, int]:
    """
    Allocate a pinned CPU buffer whose first usable byte is page-aligned,
    and return a torch view of ``size`` bytes plus its base pointer.

    Internally over-allocates one extra page on a backing tensor, then
    slices the aligned region out. The slice shares storage with the
    backing tensor, so keeping the slice alive keeps the underlying
    memory alive (no need to track the backing tensor separately).
    """
    # Pin the host buffer when an accelerator is present (probed once).
    # StubCPUDevice.is_available returns False on CPU-only hosts.
    torch_dev, torch_device_type = get_torch_device()

    global _use_pinned
    if _use_pinned is None:
        _use_pinned = torch_dev.is_available()
    try:
        backing = torch.empty(
            size + _PAGE_SIZE, dtype=torch.uint8, pin_memory=_use_pinned
        )
    except RuntimeError:
        if not _use_pinned:
            # Pure host allocation failed (e.g. OOM); nothing to fall back to.
            raise
        logger.warning(
            "Pinned host allocation failed on device '%s'; falling back to "
            "unpinned allocation from now on.",
            torch_device_type,
        )
        _use_pinned = False
        backing = torch.empty(size + _PAGE_SIZE, dtype=torch.uint8, pin_memory=False)
    # First-touch initialization on the entire backing region
    backing.fill_(0)
    base = backing.data_ptr()
    # Distance from `base` to the next page boundary (0..PAGE_SIZE-1).
    offset = (-base) % _PAGE_SIZE
    aligned_view = backing[offset : offset + size]
    return aligned_view, aligned_view.data_ptr()


def alloc_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory with NUMA awareness.
    On XPU, uses pin_memory=True (SYCL USM host allocation) for fast transfers.
    Note: NUMA node selection is not supported on non-CUDA."""

    view, aligned_ptr = _alloc_page_aligned_pinned_view(size)
    # view shares storage with its over-allocated backing tensor;
    # holding the view in the registry transitively keeps the underlying
    # memory alive.
    _tensor_registry[aligned_ptr] = view
    return aligned_ptr


def free_pinned_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory and returning pointer
    to it. On XPU, uses pin_memory=True (SYCL USM host allocation) for
    fast DMA transfers. On other non-CUDA platforms, pinning is not supported."""

    view, aligned_ptr = _alloc_page_aligned_pinned_view(size)
    _tensor_registry[aligned_ptr] = view
    return aligned_ptr


def free_pinned_ptr(ptr: int) -> None:
    """Non-CUDA equivalent of freeing a previously allocated pinned pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def batched_memcpy(src_ptrs: list[int], dst_ptrs: list[int], sizes: list[int]) -> None:
    """Non-CUDA equivalent of the native batched memcpy helper."""

    if len(src_ptrs) != len(dst_ptrs) or len(src_ptrs) != len(sizes):
        raise ValueError(
            "batched_memcpy expects equally sized src_ptrs, dst_ptrs, and sizes"
        )

    for src_ptr, dst_ptr, size in zip(src_ptrs, dst_ptrs, sizes, strict=True):
        if size <= 0:
            continue
        ctypes.memmove(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(src_ptr),
            size,
        )


def alloc_shm_pinned_ptr(size: int, shm_name: str = "") -> int:
    """Non-CUDA equivalent of allocating shared memory pinned pointer.
    Uses multiprocessing.shared_memory for cross-platform POSIX shm.
    Attempts to pin the buffer via cudaHostRegister for async D2H;
    if pinning fails, continues without pinning."""

    # Strip leading '/' for SharedMemory name
    name = shm_name.lstrip("/") if shm_name else None

    # Clean up stale shm segment if it exists
    if name:
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass

    shm = shared_memory.SharedMemory(name=name, create=True, size=size)

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_buffer(shm.buf)
    ptr = ctypes.addressof(buf)

    # Store references to keep them alive
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    _tensor_registry[ptr] = tensor
    _buf_registry[ptr] = buf
    _shm_registry[ptr] = shm

    # Try to pin the SHM buffer for async D2H copies
    if current_device_spec().pin_memory(ptr, size):
        _pinned_ptr_registry[ptr] = size

    return ptr


def free_shm_pinned_ptr(ptr: int, size: int = 0, shm_name: str = "") -> None:
    """Non-CUDA equivalent of freeing a shared memory
    pinned pointer. Unregisters pinned memory if it was pinned."""

    # Unpin if previously registered
    if ptr in _pinned_ptr_registry:
        current_device_spec().unpin_memory(ptr)
        _pinned_ptr_registry.pop(ptr, None)

    # Release in order: tensor -> ctypes buf -> shm
    _tensor_registry.pop(ptr, None)
    _buf_registry.pop(ptr, None)
    shm = _shm_registry.pop(ptr, None)
    if shm is not None:
        shm.close()
        shm.unlink()


# Hugepage variants: non-CUDA platforms do not support hugepages, so these
# fall back to the same regular pinned allocation.


def alloc_hugepage_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA fallback for alloc_hugepage_pinned_ptr (no hugepage support)."""
    warnings.warn(
        "Hugepages requested but not available on non-CUDA platforms; "
        "falling back to regular allocation.",
        RuntimeWarning,
        stacklevel=2,
    )
    return alloc_pinned_ptr(size, device_id)


def free_hugepage_pinned_ptr(ptr: int, size: int = 0) -> None:
    """Non-CUDA fallback for free_hugepage_pinned_ptr (no hugepage support)."""
    free_pinned_ptr(ptr)


def alloc_hugepage_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA fallback for alloc_hugepage_pinned_numa_ptr (no hugepage support)."""
    warnings.warn(
        "Hugepages requested but not available on non-CUDA platforms; "
        "falling back to regular allocation.",
        RuntimeWarning,
        stacklevel=2,
    )
    return alloc_pinned_numa_ptr(size, numa_id)


def free_hugepage_pinned_numa_ptr(ptr: int, size: int = 0) -> None:
    """Non-CUDA fallback for free_hugepage_pinned_numa_ptr (no hugepage support)."""
    free_pinned_numa_ptr(ptr, size)


def alloc_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating numa memory and returning pointer
    to it. Note: Numa memory is not supported on non-CUDA."""
    return alloc_pinned_numa_ptr(size, numa_id)


def free_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""
    return free_pinned_numa_ptr(ptr, size)
