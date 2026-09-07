# SPDX-License-Identifier: Apache-2.0
"""Build non-owning MUSA tensor views from process-local device pointers.

The generic multiprocess transfer path deliberately keeps the existing pointer
contract.  This module is the MUSA-only boundary that restores Tensor metadata
before invoking the MUSA transfer implementation.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch


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


def _normalize_device(device: torch.device | str | int | object) -> torch.device:
    """Normalize a device-like value and require the MUSA backend."""
    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, int):
        resolved = torch.device("musa", device)
    elif isinstance(device, str):
        resolved = torch.device(device)
    else:
        device_type = getattr(device, "type", None)
        if device_type != "musa":
            raise ValueError(
                "construct_musa_tensor_from_data_pointer requires a MUSA "
                f"device; got {device!r}"
            )
        index = getattr(device, "index", None)
        resolved = torch.device("musa", index)
    if resolved.type != "musa":
        raise ValueError(
            "construct_musa_tensor_from_data_pointer requires a MUSA "
            f"device; got {resolved!r}"
        )
    return resolved


def _musac_module() -> Any:
    """Return Torch-MUSA's tensor-construction extension module."""
    try:
        # Third Party
        import torch_musa
    except Exception as exc:
        raise RuntimeError("TorchMUSA is required for pointer reconstruction") from exc
    musac = getattr(torch_musa, "_MUSAC", None)
    if musac is None:
        raise RuntimeError("torch_musa._MUSAC is unavailable")
    return musac


def construct_musa_tensor_from_data_pointer(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str | int | object,
    *,
    stride: tuple[int, ...] | None = None,
    storage_offset: int = 0,
    nbytes: int | None = None,
) -> torch.Tensor:
    """Create a non-owning MUSA tensor view over an existing allocation.

    Args:
        ptr: Non-zero device data pointer in the current process.
        shape: Logical tensor dimensions.
        dtype: Tensor element dtype.
        device: MUSA device owning ``ptr``.
        stride: Optional element strides; contiguous strides are used by default.
        storage_offset: Offset from ``ptr`` in elements.
        nbytes: Optional storage size. Otherwise it is derived from metadata.

    Returns:
        A Tensor aliasing ``ptr``. The caller owns the allocation lifetime.

    Raises:
        ValueError: If pointer, shape, stride, device, or size is invalid.
        RuntimeError: If Torch-MUSA's construction helpers are unavailable.
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
    construct_tensor = getattr(
        _musac_module(), "_construct_MUSA_Tensor_From_Storage_And_Metadata", None
    )
    if not callable(construct_tensor):
        raise RuntimeError(
            "torch_musa._MUSAC._construct_MUSA_Tensor_From_Storage_And_Metadata "
            "is unavailable"
        )

    storage = construct_storage(pointer, resolved_device, storage_bytes)
    metadata = {
        "size": tuple(int(size) for size in shape),
        "stride": tuple(int(step) for step in resolved_stride),
        "dtype": dtype,
        "device": resolved_device,
        "storage_offset": int(storage_offset),
    }
    return construct_tensor(metadata, storage)
