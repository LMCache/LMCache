# SPDX-License-Identifier: Apache-2.0
"""Shared CUDA helpers for the cuda platform package.

Holds the lazy cuda-python bindings accessor and small conversion/
validation helpers used by the IPC modules. ``cuda-python`` is not a
declared lmcache dependency and the platform package is imported eagerly
by device discovery, so the bindings must not be imported at module load;
the first attribute access imports them.
"""

# Future
from __future__ import annotations

# Standard
from functools import lru_cache
from types import ModuleType
from typing import TypeAlias
import ctypes

# Third Party
import torch

#: Raw ``cudaStream_t`` value (the driver-level stream id), as carried in
#: Python ints -- e.g. ``torch.cuda.Stream.cuda_stream``.
cudaStream_t: TypeAlias = int


@lru_cache(maxsize=1)
def _import_cuda_bindings() -> tuple[ModuleType, ModuleType]:
    """Import and cache the ``(driver, runtime)`` CUDA bindings."""
    try:
        # Third Party
        from cuda.bindings import driver, runtime
    except ImportError:
        # Third Party
        from cuda import cuda as driver
        from cuda import cudart as runtime
    return driver, runtime


class _CudaBindings:
    """Lazy accessor; see the module docstring."""

    @property
    def driver(self) -> ModuleType:
        """The CUDA driver-API bindings module."""
        return _import_cuda_bindings()[0]

    @property
    def runtime(self) -> ModuleType:
        """The CUDA runtime-API bindings module."""
        return _import_cuda_bindings()[1]


#: Shared lazy accessor for modules of the cuda platform package.
_cuda = _CudaBindings()


def _cuda_ipc_handle_size() -> int:
    """Return the byte size of a CUDA IPC memory handle."""
    return int(getattr(_cuda.runtime, "CUDA_IPC_HANDLE_SIZE", 64))


def cuda_ipc_handle_to_bytes(handle: object) -> bytes:
    """Copy an opaque CUDA IPC memory handle into a byte string.

    cuda-bindings 13.4 no longer exposes the underlying ``reserved`` field on
    ``cudaIpcMemHandle_t``. ``getPtr`` is available on both the old structured
    representation and the new opaque representation.

    Args:
        handle: A cuda-python ``cudaIpcMemHandle_t`` instance.

    Returns:
        The fixed-size serialized CUDA IPC memory handle.

    Raises:
        TypeError: If ``handle`` does not expose cuda-python's ``getPtr`` API.
    """
    get_ptr = getattr(handle, "getPtr", None)
    if not callable(get_ptr):
        raise TypeError("CUDA IPC memory handle does not expose getPtr()")
    return ctypes.string_at(int(get_ptr()), _cuda_ipc_handle_size())


def cuda_ipc_handle_from_bytes(handle_bytes: bytes) -> object:
    """Reconstruct an opaque CUDA IPC memory handle from bytes.

    The bytes are copied through cuda-python's stable ``getPtr`` interface so
    this works with both structured and opaque ``cudaIpcMemHandle_t`` objects.

    Args:
        handle_bytes: Bytes previously returned by
            :func:`cuda_ipc_handle_to_bytes`.

    Returns:
        A cuda-python ``cudaIpcMemHandle_t`` instance owning a copy of the
        supplied bytes.

    Raises:
        ValueError: If ``handle_bytes`` has the wrong size.
        TypeError: If the cuda-python handle does not expose ``getPtr``.
    """
    handle_size = _cuda_ipc_handle_size()
    if len(handle_bytes) != handle_size:
        raise ValueError(
            "Invalid CUDA IPC memory handle size: "
            f"expected {handle_size} bytes, got {len(handle_bytes)}"
        )

    handle = _cuda.runtime.cudaIpcMemHandle_t()
    get_ptr = getattr(handle, "getPtr", None)
    if not callable(get_ptr):
        raise TypeError("CUDA IPC memory handle does not expose getPtr()")
    ctypes.memmove(int(get_ptr()), handle_bytes, handle_size)
    return handle


def _CHECK_CUDA(result: tuple[object, ...], what: str) -> None:
    """Validate the ``(err, *payload)`` tuple of a cuda-python call.

    Args:
        result: The tuple returned by the call.
        what: Operation name for the error message.

    Raises:
        RuntimeError: If the call did not return success (0).
    """
    if result[0] != 0:
        raise RuntimeError(f"{what} failed: {result[0]}")


def _resolve_device_index(device: object) -> int:
    """Resolve a CUDA device ordinal from a device-like object.

    Args:
        device: An integer ordinal, a device string (``"cuda:0"``), or a
            ``torch.device``-like object with an ``index`` attribute.

    Returns:
        The CUDA device ordinal; the current device when the object
        carries no explicit index (e.g. ``torch.device("cuda")``).
    """
    if isinstance(device, int):
        return device
    index: int | None
    if isinstance(device, str):
        index = torch.device(device).index
    else:
        attr: object = getattr(device, "index", None)
        index = attr if isinstance(attr, int) else None
    if index is None:
        return torch.cuda.current_device()
    return int(index)


def _raw_stream_handle(stream: object, device_index: int) -> cudaStream_t:
    """Resolve the raw ``cudaStream_t`` value from a stream-like object.

    Args:
        stream: A ``torch.cuda.Stream``-like object (has ``cuda_stream``),
            a raw integer handle, a cuda-bindings ``cudaStream_t``, or
            ``None`` for the current stream of ``device_index``.
        device_index: Device whose current stream is used for ``None``.

    Returns:
        The raw stream handle as an integer.

    Raises:
        RuntimeError: If the object cannot be resolved to a stream handle.
    """
    if stream is None:
        return torch.cuda.current_stream(device_index).cuda_stream
    raw: object = getattr(stream, "cuda_stream", None)
    if isinstance(raw, int):
        return raw
    try:
        return int(stream)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Cannot resolve a CUDA stream handle from {type(stream)!r}; "
            "expected a torch.cuda.Stream-like object, an integer handle, "
            "or None."
        ) from None
