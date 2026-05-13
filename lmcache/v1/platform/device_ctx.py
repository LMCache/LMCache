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

Routing strategy:

* The active backend is selected via ``lmcache.torch_device_type`` and
  the table maintained in :mod:`lmcache.v1.platform._registry`.
* When no concrete factory is registered for the active accelerator
  (e.g. an ``xpu``/``hpu`` host without its own platform sub-package
  yet) and ``torch_dev`` exposes the matching primitive, we use
  ``torch_dev`` directly.  This honours :doc:`docs/design/
  ARCHITECTURE_MULTI_HARDWARE` "use ``torch_dev`` as the unified
  middle-layer entry, fall back to per-device backends below" rule.
* Anything still unresolved falls through to the CPU stub.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, ContextManager, Protocol

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform._registry import (
    get_device_ctx_factory,
    get_event_factory,
    get_ipc_event_factory,
)

logger = init_logger(__name__)


def _torch_dev_type() -> str:
    """Lazy import to avoid the ``lmcache`` -> ``platform`` cycle."""
    # First Party
    from lmcache import torch_device_type

    return torch_device_type


def _torch_dev() -> Any:
    """Lazy import to avoid the ``lmcache`` -> ``platform`` cycle."""
    # First Party
    from lmcache import torch_dev

    return torch_dev


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

    Routing order:

    1. Backend factory registered for the running ``torch_device_type``.
    2. CPU fallback (always-registered ``NoopDeviceContext``).
    """
    factory = get_device_ctx_factory(_torch_dev_type())
    if factory is None:
        # First Party
        from lmcache.v1.platform.cpu.device_ctx import NoopDeviceContext

        return NoopDeviceContext()
    return factory(device, stream)


def make_interprocess_event(device: torch.device) -> InterprocessEventLike:
    """Build an interprocess-capable Event for the given device."""
    device_type = _torch_dev_type()
    factory = get_event_factory(device_type)
    if factory is not None:
        return factory(device)

    # Generic ``torch_dev`` path: lets xpu/hpu hosts reuse the abstraction
    # before they ship their own platform sub-package.
    torch_dev = _torch_dev()
    if torch_dev is not None and hasattr(torch_dev, "Event"):
        try:
            return torch_dev.Event(interprocess=True)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.debug("torch_dev.Event(interprocess=True) failed: %s", exc)

    # First Party
    from lmcache.v1.platform.cpu.device_ctx import MockInterprocessEvent

    return MockInterprocessEvent()


def event_from_ipc_handle(device: torch.device, handle: bytes) -> InterprocessEventLike:
    """Rebuild an Event from a peer process IPC handle."""
    device_type = _torch_dev_type()
    factory = get_ipc_event_factory(device_type)
    if factory is not None:
        return factory(device, handle)

    torch_dev = _torch_dev()
    if (
        torch_dev is not None
        and hasattr(torch_dev, "Event")
        and hasattr(torch_dev.Event, "from_ipc_handle")
    ):
        try:
            return torch_dev.Event.from_ipc_handle(device, handle)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.debug("torch_dev.Event.from_ipc_handle failed: %s", exc)

    # First Party
    from lmcache.v1.platform.cpu.device_ctx import MockInterprocessEvent

    return MockInterprocessEvent()
