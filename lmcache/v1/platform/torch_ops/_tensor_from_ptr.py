# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import cast
import ctypes
import ctypes.util

# Third Party
import torch

# First Party
from lmcache.v1.platform import resolve_device_ops

__all__ = [
    "_get_copy_lib",
    "_tensor_from_ptr",
    "_copy_bytes_with_tensor",
]

# Cached copy library for lmcache_memcpy_async (lazy-initialized)
_COPY_LIB_NOT_LOADED = object()
_copy_lib: ctypes.CDLL | object | None = _COPY_LIB_NOT_LOADED


def _get_copy_lib() -> ctypes.CDLL | None:
    """Lazily load and cache the CUDA/ROCm runtime library, or None for CPU fallback."""
    global _copy_lib
    if _copy_lib is _COPY_LIB_NOT_LOADED:
        # Try to load GPU runtime libraries in priority order: CUDA first, then ROCm
        # TODO: ROCm path to be validated on real device
        for name, fallback in [
            ("cudart", "libcudart.so"),  # NVIDIA CUDA Runtime
            ("amdhip64", "libamdhip64.so"),  # AMD ROCm HIP Runtime
        ]:
            try:
                path = ctypes.util.find_library(name)
                if path:
                    _copy_lib = ctypes.CDLL(path)
                else:
                    _copy_lib = ctypes.CDLL(fallback)
                break  # Successfully loaded, stop trying
            except OSError:
                continue  # Current library not available, try next
        else:
            # All GPU libraries failed to load, fall back to CPU
            _copy_lib = None
    return cast("ctypes.CDLL | None", _copy_lib)


def _tensor_from_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Create a tensor view over a raw pointer (zero-copy where possible).

    Device-specific construction is delegated to the resolved
    :class:`~lmcache.v1.platform.base.device_ops.DeviceOps` implementation.

    Args:
        ptr:    Raw memory pointer as int (must be non-zero).
        shape:  Desired tensor shape.
        dtype:  Desired tensor dtype, must match the memory layout.
        device: Device that owns the pointer. ``None`` selects CPU.

    Returns:
        A tensor created by the resolved device backend.

    Raises:
        ValueError: If ``ptr`` is zero or the resolved backend does not support
            pointer-backed tensor construction.
        RuntimeError: If the device backend cannot be resolved or cannot create
            the tensor.

    Warning:
        The caller is responsible for keeping the underlying memory alive
        for the entire lifetime of the returned tensor.
    """
    if ptr == 0:
        raise ValueError("Pointer must be non-zero")

    resolved_device = torch.device("cpu" if device is None else device)
    return resolve_device_ops(resolved_device.type).tensor_from_ptr(
        ptr,
        shape,
        dtype,
        resolved_device,
    )


def _copy_bytes_with_tensor(dst: int, src: int, num_bytes: int) -> None:
    """Copy raw bytes between pointers using torch tensor semantics.

    Note: This function only works for CPU-accessible memory. For device
    memory (CUDA/XPU), use lmcache_memcpy_async with the appropriate runtime
    library or PyTorch's tensor copy operations.
    """
    if num_bytes <= 0:
        return

    buffer_type = ctypes.c_uint8 * num_bytes
    dst_tensor = torch.frombuffer(buffer_type.from_address(dst), dtype=torch.uint8)
    src_tensor = torch.frombuffer(buffer_type.from_address(src), dtype=torch.uint8)
    dst_tensor.copy_(src_tensor)
