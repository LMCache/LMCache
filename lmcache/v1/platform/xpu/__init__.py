# SPDX-License-Identifier: Apache-2.0
"""XPU (Intel SYCL) platform helpers."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend


class _XpuNoopEvent:
    """Minimal XPU event stand-in when real interprocess events are unavailable.

    The XPU backend exposes streams and devices but not the CUDA-like
    ``Event(interprocess=True)`` contract required by the LMCache-driven path.
    To avoid crashing the worker adapter on XPU startup, we intentionally model
    the event as a no-op completion object; the adapter can still create record
    and wait calls without invoking unsupported IPC semantics.
    """

    def __init__(self, device: object | None = None, *, interprocess: bool = False):
        self.device = device
        self.interprocess = interprocess

    def record(self, stream: object | None = None) -> None:
        return None

    def wait(self, stream: object | None = None) -> None:
        return None

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def ipc_handle(self) -> bytes:
        return b""

    @classmethod
    def from_ipc_handle(cls, device: object, handle: bytes) -> "_XpuNoopEvent":
        return cls(device=device, interprocess=True)


class _XpuEventIPCBackend:
    """XPU fallback backend for the common event-IPC call surface.

    This is intentionally conservative: it allows the LMCache worker adapter to
    create / record / wait / query events on XPU without requiring the
    unsupported CUDA-style cross-process event ABI.
    """

    device_type = "xpu"

    def check_event_support(self, device: object) -> None:
        return None

    def create_event(self, device: object) -> _XpuNoopEvent:
        return _XpuNoopEvent(device=device, interprocess=True)

    def export_event(self, event: object, device: object) -> bytes:
        if hasattr(event, "ipc_handle"):
            return event.ipc_handle()
        return b""

    def import_event(self, handle: bytes, device: object) -> _XpuNoopEvent:
        return _XpuNoopEvent(device=device, interprocess=True)

    def record_event(self, event: object, stream: object) -> None:
        if hasattr(event, "record"):
            event.record(stream)

    def wait_event(self, event: object, stream: object) -> None:
        if hasattr(event, "wait"):
            event.wait(stream)

    def query_event(self, event: object) -> bool:
        if hasattr(event, "query"):
            return bool(event.query())
        return True

    def synchronize_event(self, event: object, device: object) -> None:
        if hasattr(event, "synchronize"):
            event.synchronize()

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class XpuDeviceSpec(DeviceSpec):
    """XPU device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "xpu"

    @property
    def torch_module_name(self) -> str:
        return "xpu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.xpu.device_ops import XpuDeviceOps

        return XpuDeviceOps

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        backend = self._event_backend_cache
        if backend is None:
            backend = _XpuEventIPCBackend()
            self._event_backend_cache = backend
        return backend

    def is_handle_transfer_available(self) -> bool:
        """XPU does not currently provide a real IPC event backend.

        Keep the device in the engine-driven-only path unless a future SYCL IPC
        implementation adds a proper interprocess event ABI.
        """
        return False

    def is_available(self) -> bool:
        """Check XPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False
