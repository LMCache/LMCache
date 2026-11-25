# SPDX-License-Identifier: Apache-2.0
"""Fallback completion recorder.

The compiled recorder schedules a host callback on the CUDA stream, so the
handler runs only after the work queued before it -- in particular the staging
copy that fills an object -- has completed. Without that primitive a completion
published here would be visible immediately, and a store could be marked written
while its bytes are still in flight: a later read or an L2 write-back would then
observe the staging buffer's old contents.

The fallback therefore waits for the stream before publishing. That is the same
trade the MUSA backend makes, and it keeps the ownership contract that the
asynchronous copy relies on.
"""

# Standard
from typing import Any
import ctypes
import threading

# Third Party
import torch

# First Party
from lmcache.v1.platform.torch_ops._tensor_from_ptr import _get_copy_lib

_completion_lock = threading.Lock()
_completion_buffer: list[tuple[str, bytes]] = []


def _wait_for_stream(stream_ptr: int) -> None:
    """Wait until the work queued on ``stream_ptr`` before this call finishes.

    Args:
        stream_ptr: Raw CUDA/HIP stream handle, or ``0`` for the legacy default
            stream when the caller did not resolve one.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    if stream_ptr:
        libcudart = _get_copy_lib()
        if libcudart is not None:
            synchronize: Any = getattr(libcudart, "cudaStreamSynchronize", None)
            if synchronize is None:
                synchronize = getattr(libcudart, "hipStreamSynchronize", None)
            if (
                synchronize is not None
                and synchronize(ctypes.c_void_p(stream_ptr)) == 0
            ):
                return
    # No stream handle, or no runtime entry point to wait on it: fall back to
    # the device so the completion can never be published early.
    torch.cuda.synchronize()


def record_completion_on_stream(
    cuda_stream_ptr: int, kind: str, payload: bytes
) -> None:
    """Publish a completion once the work queued before it has finished.

    Args:
        cuda_stream_ptr: Stream the completion belongs to. ``0`` means the
            caller did not resolve one.
        kind: Dispatch key identifying the handler (e.g. "finish_write").
        payload: Opaque msgpack-encoded bytes forwarded to the handler.
    """
    _wait_for_stream(cuda_stream_ptr)
    with _completion_lock:
        _completion_buffer.append((kind, payload))


def drain_recorded_completions() -> list[tuple[str, bytes]]:
    """Fallback: atomically drain and return all pending completions.

    Returns:
        List of (kind, payload) pairs recorded since the last drain.
    """
    with _completion_lock:
        items = list(_completion_buffer)
        _completion_buffer.clear()
    return items
