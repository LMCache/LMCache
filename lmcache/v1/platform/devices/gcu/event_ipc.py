# SPDX-License-Identifier: Apache-2.0
"""GCU device-event IPC backend.

GCU's ``torch.gcu.Event`` exposes the CUDA-style interprocess API
(``Event(interprocess=True)``, ``ipc_handle()``, ``from_ipc_handle()``). Its C
extension kwlist is not reflected by :func:`inspect.signature`, so the generic
capability probe in :class:`DefaultEventIPCBackend` reports a false negative.
This backend overrides the capability check to probe by direct construction
while reusing the generic event operations.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend


class GcuEventIPCBackend(DefaultEventIPCBackend):
    """Event IPC backend backed by GCU interprocess events."""

    def __init__(self) -> None:
        """Bind the backend to the ``torch.gcu`` event module."""
        super().__init__(event_module=torch.gcu, device_type="gcu")

    def check_event_support(self, device: object) -> None:
        """Validate GCU interprocess event support by direct probe.

        The generic probe relies on :func:`inspect.signature`, which cannot
        see the GCU C extension's ``interprocess`` kwarg even though the
        ``Event`` docstring documents it. Construct an interprocess event
        directly to fail closed on the real capability.

        Args:
            device: Target GCU device (used only for error reporting).

        Raises:
            RuntimeError: If ``torch.gcu.Event`` is missing, does not accept
                ``interprocess=True``, or lacks ``from_ipc_handle``.
        """
        event_cls = getattr(torch.gcu, "Event", None)
        if event_cls is None:
            raise RuntimeError(
                "Device backend 'gcu' does not support interprocess events: "
                "no Event class is available."
            )
        try:
            event_cls(interprocess=True)
        except TypeError as exc:
            raise RuntimeError(
                "Device backend 'gcu' does not support interprocess events: "
                "Event does not accept interprocess=True."
            ) from exc
        if not hasattr(event_cls, "from_ipc_handle"):
            raise RuntimeError(
                "Device backend 'gcu' does not support interprocess event "
                "handles: Event.from_ipc_handle is missing."
            )
