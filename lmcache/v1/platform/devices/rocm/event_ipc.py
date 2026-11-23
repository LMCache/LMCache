# SPDX-License-Identifier: Apache-2.0
"""ROCm event IPC backend.

``torch.cuda.Event.from_ipc_handle`` and ``Event.query`` keep the GIL during
``hipIpcOpenEventHandle`` / ``hipEventQuery``, and the open can block for
seconds. ``rocm.bindings.hip`` releases the GIL around both calls, so
:meth:`RocmEventIPCBackend.import_event_for_query` opens through it the events
that are only polled on the host (``--rocm-host-event-wait``).

Events imported with :meth:`RocmEventIPCBackend.import_event` stay torch
events: a stream wait needs one, and each handle is opened once.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend

logger = init_logger(__name__)

_UNSET = object()
# sizeof(hipIpcEventHandle_t); the bindings read that many bytes of the handle.
_HIP_IPC_HANDLE_SIZE = 64


class HipIpcEvent:
    """An IPC event opened through ``rocm.bindings.hip``; host queries only."""

    __slots__ = ("_hip", "event", "__weakref__")

    def __init__(self, hip: Any, event: object) -> None:
        self._hip = hip
        self.event: object | None = event

    def __del__(self) -> None:
        event, self.event = self.event, None
        if event is None:
            return
        try:
            _hip_ok(self._hip, self._hip.hipEventDestroy(event))
        except Exception:
            # Interpreter shutdown: the bindings may already be gone.
            pass


def _hip_ok(hip: Any, result: tuple) -> bool:
    """Return whether a HIP call succeeded; clear its sticky error otherwise.

    A non-zero return (``hipErrorNotReady`` from a query included) stays on
    the thread as its last error, and torch's next check there would raise it.
    """
    if result[0] == hip.hipError_t.hipSuccess:
        return True
    hip.hipGetLastError()
    return False


class RocmEventIPCBackend(DefaultEventIPCBackend):
    """Interprocess events on ROCm.

    Args:
        event_module: See :class:`DefaultEventIPCBackend`.
        device_type: See :class:`DefaultEventIPCBackend`.
        hip_module: The ``rocm.bindings.hip`` module; imported on first use
            when omitted, injectable for testing.
    """

    def __init__(
        self,
        event_module: object | None = None,
        device_type: str | None = None,
        hip_module: object = _UNSET,
    ) -> None:
        super().__init__(event_module=event_module, device_type=device_type)
        self._hip_module = hip_module

    def _hip(self) -> Any:
        """Return ``rocm.bindings.hip``, or None when it is not installed."""
        if self._hip_module is _UNSET:
            try:
                # Third Party
                from rocm.bindings import hip

                self._hip_module = hip
            except ImportError:
                logger.warning(
                    "rocm.bindings.hip is not installed; importing IPC events "
                    "with torch (holds the GIL during the open)"
                )
                self._hip_module = None
        return self._hip_module

    def import_event_for_query(self, handle: bytes, device: object) -> object:
        """Import an event that is only queried on the host.

        The GIL is released during the open. The returned event must not be
        passed to :meth:`wait_event`; use :meth:`import_event` for that.

        Args:
            handle: Serialized event handle from another process.
            device: Device on which to import the event.

        Returns:
            A :class:`HipIpcEvent`, or the torch event if the handle could not
            be opened through ``rocm.bindings.hip``.
        """
        hip = self._hip()
        index = device if isinstance(device, int) else getattr(device, "index", None)
        if (
            hip is not None
            and index is not None
            and len(handle) == _HIP_IPC_HANDLE_SIZE
        ):
            if _hip_ok(hip, hip.hipSetDevice(int(index))):
                result = hip.hipIpcOpenEventHandle(handle)
                if _hip_ok(hip, result) and result[1] is not None:
                    return HipIpcEvent(hip, result[1])
        return self.import_event(handle, device)

    def query_event(self, event: object) -> bool:
        """Return whether ``event`` has completed."""
        if isinstance(event, HipIpcEvent):
            return _hip_ok(event._hip, event._hip.hipEventQuery(event.event))
        return super().query_event(event)
