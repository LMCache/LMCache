# SPDX-License-Identifier: Apache-2.0
"""Cross-platform device-context and interprocess Event abstractions.

The LMCache multiprocess server historically wrapped GPU work in::

    with torch.cuda.device(dev), torch.cuda.stream(stream):
        event = torch.cuda.Event(interprocess=True)
        ...

Both ``torch.cuda.device`` and ``torch.cuda.Event(interprocess=True)``
require a real CUDA build at runtime; on CPU-only hosts they raise
``ValueError``.  This module hides that platform split behind two tiny
factories so callers stay device-agnostic:

* :func:`make_device_context` returns a context manager that activates
  the right device + stream on CUDA hosts, and is a no-op on CPU.
* :func:`make_interprocess_event` returns an object exposing the small
  subset of ``torch.cuda.Event`` we use (``record``, ``wait``,
  ``ipc_handle``).  On CPU-only hosts it is a pure-Python stub.
* :func:`event_from_ipc_handle` rebuilds a peer-process event handle.

Backend implementations live under ``platform/<device>/device_ctx.py``
so each accelerator can evolve independently — same convention as
``platform/stream.py``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, ContextManager, Protocol

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class InterprocessEventLike(Protocol):
    """Structural type for the subset of ``torch.cuda.Event`` we use."""

    def record(self, stream: Any = ...) -> None: ...

    def wait(self, stream: Any = ...) -> None: ...

    def ipc_handle(self) -> bytes: ...


def make_device_context(
    device: torch.device,
    stream: Any | None,
) -> ContextManager[None]:
    """Return a context manager that activates ``device`` and ``stream``.

    Dispatch order:

    1. On CUDA-capable hosts, return the real
       ``torch.cuda.device`` + ``torch.cuda.stream`` combo.
    2. Otherwise fall back to the CPU no-op context manager.
    """
    if torch.cuda.is_available() and device.type == "cuda":
        # First Party
        from lmcache.v1.platform.cuda.device_ctx import (
            make_cuda_device_context,
        )

        return make_cuda_device_context(device, stream)

    # First Party
    from lmcache.v1.platform.cpu.device_ctx import NoopDeviceContext

    return NoopDeviceContext()


def make_interprocess_event(device: torch.device) -> InterprocessEventLike:
    """Build an interprocess-capable Event for the given device."""
    if torch.cuda.is_available() and device.type == "cuda":
        # Third Party
        return torch.cuda.Event(interprocess=True)

    # First Party
    from lmcache.v1.platform.cpu.device_ctx import MockInterprocessEvent

    return MockInterprocessEvent()


def event_from_ipc_handle(device: torch.device, handle: bytes) -> InterprocessEventLike:
    """Rebuild an Event from a peer process IPC handle."""
    if torch.cuda.is_available() and device.type == "cuda":
        return torch.cuda.Event.from_ipc_handle(device, handle)

    # First Party
    from lmcache.v1.platform.cpu.device_ctx import MockInterprocessEvent

    return MockInterprocessEvent()
