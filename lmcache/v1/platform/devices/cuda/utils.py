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
