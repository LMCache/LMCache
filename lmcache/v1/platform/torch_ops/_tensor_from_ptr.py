# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional
import ctypes

# Third Party
# Third party
import torch

__all__ = [
    "_get_copy_lib",
    "_tensor_from_ptr",
    "_copy_bytes_with_tensor",
]

# Cached copy library for lmcache_memcpy_async (lazy-initialized)
_copy_lib_NOT_LOADED = object()
_copy_lib: Optional[ctypes.CDLL] = _copy_lib_NOT_LOADED  # type: ignore


# HIP ships the same runtime entry points as CUDA under ``hip*`` names, so a
# library loaded from the ROCm branch below answers to none of the ``cuda*``
# names this module resolves on it. Map each spelling this module uses onto its
# HIP equivalent. The memcpy-kind values passed alongside these calls (3 for
# device-to-device, 4 for default) are the same in both runtimes, so only the
# name differs.
_HIP_SYMBOL_ALIASES = {"cudaMemcpy": "hipMemcpy"}


def _alias_hip_symbols(lib: ctypes.CDLL) -> None:
    """Expose the CUDA spelling of every runtime call this module resolves.

    Args:
        lib: A loaded HIP runtime library.
    """
    for cuda_name, hip_name in _HIP_SYMBOL_ALIASES.items():
        if hasattr(lib, cuda_name):
            continue
        hip_symbol = getattr(lib, hip_name, None)
        if hip_symbol is not None:
            setattr(lib, cuda_name, hip_symbol)


def _get_copy_lib() -> Optional[ctypes.CDLL]:
    """Lazily load and cache the CUDA/ROCm runtime library, or None for CPU fallback."""
    global _copy_lib
    if _copy_lib is _copy_lib_NOT_LOADED:
        # Try to load GPU runtime libraries in priority order: CUDA first, then ROCm
        for name, fallback in [
            ("cudart", "libcudart.so"),  # NVIDIA CUDA Runtime
            ("amdhip64", "libamdhip64.so"),  # AMD ROCm HIP Runtime
        ]:
            try:
                path = ctypes.util.find_library(name)
                lib = ctypes.CDLL(path) if path else ctypes.CDLL(fallback)
            except OSError:
                continue  # Current library not available, try next
            if name == "amdhip64":
                _alias_hip_symbols(lib)
            _copy_lib = lib
            break  # Successfully loaded, stop trying
        else:
            # All GPU libraries failed to load, fall back to CPU
            _copy_lib = None
    return _copy_lib


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

    # First Party
    from lmcache.v1.platform import resolve_device_ops

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
