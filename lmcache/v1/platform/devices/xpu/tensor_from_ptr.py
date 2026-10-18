# SPDX-License-Identifier: Apache-2.0
"""Build non-owning XPU tensor views from process-local device pointers.

The multiprocess transfer path hands the device backend raw pointers.  This
module is the XPU-only boundary that restores Tensor metadata before the
transfer implementation runs.

The storage is always built by the device-agnostic
:func:`torch._C._construct_storage_from_data_pointer` (available since 2.1).
Metadata is then applied with
``torch._C._construct_XPU_Tensor_From_Storage_And_Metadata(metadata, storage)``,
the XPU counterpart of the constructor the MUSA adapter calls, when this torch
build provides it.  torch-xpu 2.14 does not, so the fallback is
``torch.empty(())`` followed by :meth:`torch.Tensor.set_` -- exactly how this
package's XPU IPC wrapper applies metadata to an imported storage.  Both paths
return a non-owning view over the same storage.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# Resolved once: ``torch._C`` does not change after import.
_CONSTRUCT_XPU_TENSOR = getattr(
    torch._C, "_construct_XPU_Tensor_From_Storage_And_Metadata", None
)


def contiguous_row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return element strides for a contiguous row-major tensor.

    Args:
        shape: Tensor dimensions.

    Returns:
        Element strides matching a dense contiguous layout.
    """
    if not shape:
        return ()
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * int(shape[index + 1])
    return tuple(strides)


def _storage_nbytes(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
    storage_offset: int,
) -> int:
    """Compute bytes needed to address a strided tensor view."""
    if any(int(size) == 0 for size in shape):
        return 0
    last_element = int(storage_offset)
    for size, step in zip(shape, stride, strict=True):
        last_element += (int(size) - 1) * int(step)
    return max(0, last_element + 1) * dtype.itemsize


def _normalize_device(device: torch.device | str | int) -> torch.device:
    """Normalize a device-like value to an indexed XPU device.

    The storage and the tensor that aliases it must agree on the device index,
    so an unindexed device is resolved against the current XPU device rather
    than left as ``xpu:None``.

    Args:
        device: XPU device owning the pointer.

    Returns:
        A :class:`torch.device` of type ``xpu`` with an explicit index.

    Raises:
        ValueError: If ``device`` does not denote an XPU device.
        RuntimeError: If no XPU device index can be resolved.
    """
    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, int):
        resolved = torch.device("xpu", device)
    else:
        resolved = torch.device(device)
    if resolved.type != "xpu":
        raise ValueError(
            f"construct_xpu_tensor_from_data_pointer requires an XPU device; "
            f"got {resolved!r}"
        )
    if resolved.index is not None:
        return resolved
    if not torch.xpu.is_available():
        raise RuntimeError(
            "Cannot resolve the current XPU device index: torch.xpu reports "
            "no available device."
        )
    return torch.device("xpu", torch.xpu.current_device())


def construct_xpu_tensor_from_data_pointer(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str | int,
    *,
    stride: tuple[int, ...] | None = None,
    storage_offset: int = 0,
    nbytes: int | None = None,
) -> torch.Tensor:
    """Create a non-owning XPU tensor view over an existing allocation.

    Args:
        ptr: Non-zero device data pointer valid in the current process.
        shape: Logical tensor dimensions.
        dtype: Tensor element dtype.
        device: XPU device owning ``ptr``.
        stride: Optional element strides; contiguous strides are used by
            default.
        storage_offset: Offset from ``ptr`` in elements.
        nbytes: Optional storage size. Otherwise it is derived from metadata.

    Returns:
        A Tensor aliasing ``ptr``. The caller owns the allocation lifetime.

    Raises:
        ValueError: If pointer, shape, stride, device, or size is invalid.
        TypeError: If ``dtype`` is not a :class:`torch.dtype`.
        RuntimeError: If Torch's storage-construction helper is unavailable.

    Warning:
        The returned tensor does not keep ``ptr`` alive.  The caller must
        ensure the allocation outlives every view derived from it.
    """
    pointer = int(ptr)
    if pointer <= 0:
        raise ValueError("ptr must be a non-zero positive integer")
    if not isinstance(shape, tuple) or any(int(size) < 0 for size in shape):
        raise ValueError("shape must be a tuple of non-negative integers")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be torch.dtype, got {type(dtype).__name__}")
    resolved_stride = (
        contiguous_row_major_strides(shape) if stride is None else tuple(stride)
    )
    if len(resolved_stride) != len(shape) or any(
        int(step) < 0 for step in resolved_stride
    ):
        raise ValueError("stride must match shape and contain non-negative integers")
    if int(storage_offset) < 0:
        raise ValueError("storage_offset must be non-negative")
    storage_bytes = (
        _storage_nbytes(shape, dtype, resolved_stride, int(storage_offset))
        if nbytes is None
        else int(nbytes)
    )
    if storage_bytes < 0:
        raise ValueError("nbytes must be non-negative")

    resolved_device = _normalize_device(device)
    construct_storage = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if not callable(construct_storage):
        raise RuntimeError(
            "torch._C._construct_storage_from_data_pointer is unavailable"
        )

    # Level Zero places USM device allocations high in the address space
    # (0xffff_...), so ``Tensor.data_ptr()`` on XPU routinely exceeds int64
    # max.  The pybind signature takes an int64_t and ``reinterpret_cast``s it
    # straight back to ``void*``, so hand it the two's-complement form of the
    # same bits; passing the unsigned value raises TypeError.
    signed_pointer = pointer - (1 << 64) if pointer >= (1 << 63) else pointer
    storage = construct_storage(signed_pointer, resolved_device, storage_bytes)
    if _CONSTRUCT_XPU_TENSOR is not None:
        metadata = {
            "size": tuple(int(size) for size in shape),
            "stride": tuple(int(step) for step in resolved_stride),
            "dtype": dtype,
            "storage_offset": int(storage_offset),
        }
        return _CONSTRUCT_XPU_TENSOR(metadata, storage)
    tensor = torch.empty((), dtype=dtype, device=resolved_device)
    tensor.set_(
        storage,
        int(storage_offset),
        tuple(int(size) for size in shape),
        tuple(int(step) for step in resolved_stride),
    )
    return tensor
