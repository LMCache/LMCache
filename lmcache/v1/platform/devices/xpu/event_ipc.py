# SPDX-License-Identifier: Apache-2.0
"""XPU device-event IPC backend.

The ``lmcache_driven`` MP handle path orders cross-process KV transfers
with device events. ``torch.xpu.Event`` is expected to grow the same
interprocess surface ``torch.cuda.Event`` already has, at
:class:`~lmcache.v1.platform.base.event_ipc.EventIPCBackend`

"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import (
    DefaultEventIPCBackend,
    EventIPCBackend,
    _accepts_interprocess,
)

#: Instance/class methods ``DefaultEventIPCBackend`` calls on an event.
_REQUIRED_TORCH_XPU_EVENT_SYMBOLS = (
    "from_ipc_handle",
    "ipc_handle",
    "record",
    "wait",
    "query",
    "synchronize",
)

_TORCH_XPU_EVENT_REQUIREMENT = "torch.xpu.Event does not support interprocess handles."


def check_torch_xpu_event_support() -> bool:
    """Return whether ``torch.xpu.Event`` exposes the event IPC API.

    Covers everything ``DefaultEventIPCBackend.check_event_support``
    checks, reusing its
    :func:`~lmcache.v1.platform.base.event_ipc._accepts_interprocess`.
    ``interprocess`` must appear on the *Python* ``torch.xpu.Event``
    class: ``torch/xpu/streams.py`` defines its own ``__new__``, which
    shadows the C++ binding's signature.

    Returns:
        ``True`` when the event class accepts ``interprocess=True`` and
        provides every method the default backend calls.
    """
    xpu_module = getattr(torch, "xpu", None)
    event_cls = getattr(xpu_module, "Event", None)
    return all(
        callable(getattr(event_cls, symbol, None))
        for symbol in _REQUIRED_TORCH_XPU_EVENT_SYMBOLS
    ) and _accepts_interprocess(event_cls)


class XpuEventIPCBackend(EventIPCBackend):
    """Event IPC backend backed by torch XPU interprocess events."""

    device_type: str = "xpu"

    def __init__(self, event_module: Any | None = None) -> None:
        """Create an XPU event IPC backend.

        Args:
            event_module: Optional module exposing the ``Event`` class.
                Defaults to ``torch.xpu``; injectable for tests.
        """
        self._event_module = event_module
        self._default_backend_cache: DefaultEventIPCBackend | None = None

    def _module(self) -> Any:
        """Return the torch XPU module, resolving it lazily."""
        if self._event_module is None:
            xpu_module = getattr(torch, "xpu", None)
            if xpu_module is None:
                raise RuntimeError("XPU event IPC requires torch.xpu")
            self._event_module = xpu_module
        return self._event_module

    def _default_backend(self) -> DefaultEventIPCBackend:
        """Return the default event adapter bound to ``torch.xpu``.

        Returns:
            A CUDA-style event adapter over the torch XPU event API.

        Raises:
            RuntimeError: If ``torch.xpu`` cannot be resolved.
        """
        backend = self._default_backend_cache
        if backend is not None:
            return backend

        backend = DefaultEventIPCBackend(
            event_module=self._module(),
            device_type=self.device_type,
        )
        self._default_backend_cache = backend
        return backend

    def check_event_support(self, device: object) -> None:
        """Fail closed unless torch XPU event IPC is available.

        Args:
            device: Target XPU device (unused; part of the protocol
                signature).

        Raises:
            RuntimeError: If the torch XPU event IPC API is unavailable.
        """
        if not check_torch_xpu_event_support():
            raise RuntimeError(
                "Device backend 'xpu' does not support interprocess "
                f"events: {_TORCH_XPU_EVENT_REQUIREMENT}"
            )

    def create_event(self, device: object) -> object:
        """Create a torch XPU interprocess event."""
        return self._default_backend().create_event(device)

    def export_event(self, event: object, device: object) -> bytes:
        """Serialize a torch XPU event for another process.

        Note:
            The returned bytes stay valid only while the exporting
            process keeps ``event`` alive.
        """
        return self._default_backend().export_event(event, device)

    def import_event(self, handle: bytes, device: object) -> object:
        """Import a torch XPU event handle on ``device``."""
        return self._default_backend().import_event(handle, device)

    def record_event(self, event: object, stream: object) -> None:
        """Record an event on a torch XPU stream."""
        self._default_backend().record_event(event, stream)

    def wait_event(self, event: object, stream: object) -> None:
        """Make a torch XPU stream wait for an event."""
        self._default_backend().wait_event(event, stream)

    def query_event(self, event: object) -> bool:
        """Return whether a torch XPU event has completed."""
        return self._default_backend().query_event(event)

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until a torch XPU event completes."""
        self._default_backend().synchronize_event(event, device)
