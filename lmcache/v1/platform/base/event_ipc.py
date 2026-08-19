# SPDX-License-Identifier: Apache-2.0
"""Platform device-event IPC abstraction.

The ``lmcache_driven`` multiprocess handle path orders cross-process KV-cache
transfers with device events. Accelerators expose different event IPC ABIs.
This module defines a small backend-neutral interface so generic MP code never
imports a specific backend or branches on device type.

Event, device, and stream values are intentionally typed as opaque ``object``
handles: their concrete types differ per backend and generic code must not
depend on any one of them. See
``docs/design/v1/platform/base/event_ipc_abstraction.md``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Protocol, runtime_checkable
import inspect

# First Party
from lmcache import torch_dev, torch_device_type


def _accepts_interprocess(event_cls: object) -> bool:
    """Return whether ``event_cls`` accepts an ``interprocess`` parameter."""
    for obj in (
        event_cls,
        getattr(event_cls, "__init__", None),
        getattr(event_cls, "__new__", None),
    ):
        if obj is None:
            continue
        try:
            signature = inspect.signature(obj)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if "interprocess" in signature.parameters:
            return True
    return False


def _resolve_device_type(device: object) -> str:
    """Resolve a device-type string from a device-like object.

    Args:
        device: A ``torch.device``-like object (has ``.type``), a device-type
            string, or an integer device index.

    Returns:
        The device-type string. Integer indices use the active
        ``torch_device_type`` for backend selection; the original integer is
        still passed unchanged to backend operations by the caller.
    """
    if isinstance(device, int):
        return torch_device_type
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    if isinstance(device, str):
        return device.split(":", maxsplit=1)[0]
    return str(device)


@runtime_checkable
class EventIPCBackend(Protocol):
    """Backend-neutral device-event IPC operations.

    Implementations serialize/deserialize device events across processes and
    order stream work against them. ``check_event_support`` must raise before
    any cross-process memory transfer when the backend cannot provide IPC
    ordering (fail closed).
    """

    device_type: str

    def check_event_support(self, device: object) -> None:
        """Validate that event IPC is available for ``device``.

        Args:
            device: Device that will create or import IPC events.

        Raises:
            RuntimeError: If the backend cannot provide event IPC support.
        """
        ...

    def create_event(self, device: object) -> object:
        """Create an interprocess event on ``device``.

        Args:
            device: Device on which to create the event.

        Returns:
            A backend-native event object.
        """
        ...

    def export_event(self, event: object, device: object) -> bytes:
        """Serialize ``event`` for import by another process.

        Args:
            event: Backend-native event to export.
            device: Device that owns the event.

        Returns:
            Serialized event handle.
        """
        ...

    def import_event(self, handle: bytes, device: object) -> object:
        """Import a serialized event handle on ``device``.

        Args:
            handle: Serialized event handle from another process.
            device: Device on which to import the event.

        Returns:
            The imported backend-native event.
        """
        ...

    def record_event(self, event: object, stream: object) -> None:
        """Record ``event`` on ``stream``.

        Args:
            event: Backend-native event to record.
            stream: Stream that should record the event.
        """
        ...

    def wait_event(self, event: object, stream: object) -> None:
        """Make ``stream`` wait for ``event``.

        Args:
            event: Backend-native event to wait for.
            stream: Stream that should wait for the event.
        """
        ...

    def query_event(self, event: object) -> bool:
        """Return whether ``event`` has completed.

        Args:
            event: Backend-native event to query.

        Returns:
            ``True`` when the event has completed; otherwise ``False``.
        """
        ...

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until ``event`` completes.

        Args:
            event: Backend-native event to synchronize.
            device: Device that owns the event.
        """
        ...


class DefaultEventIPCBackend:
    """CUDA-style event IPC backend over an injectable event module.

    Works for any backend whose ``Event`` class supports interprocess handles,
    including the ``StubEvent`` used in CPU-only environments.

    Args:
        event_module: Object exposing an ``Event`` class that supports
            ``Event(interprocess=True)``, ``ipc_handle()``,
            ``from_ipc_handle(device, handle)``, ``record(stream)``,
            ``wait(stream)``, ``query()`` and ``synchronize()``. Defaults to the
            active ``torch_dev`` module; injectable for testing.
        device_type: Device-type label used in error messages. Defaults to the
            active ``torch_device_type``.
    """

    def __init__(
        self,
        event_module: object | None = None,
        device_type: str | None = None,
    ) -> None:
        self._event_module = event_module if event_module is not None else torch_dev
        self.device_type = device_type if device_type is not None else torch_device_type

    def check_event_support(self, device: object) -> None:
        """Raise ``RuntimeError`` if interprocess events are unsupported.

        Args:
            device: The target device (used only for the error message here).

        Raises:
            RuntimeError: If no ``Event`` class is available, it does not accept
                ``interprocess=True``, or ``from_ipc_handle`` is missing.
        """
        event_cls = getattr(self._event_module, "Event", None)
        if event_cls is None:
            raise RuntimeError(
                f"Device backend '{self.device_type}' does not support "
                "interprocess events: no Event class is available."
            )
        if not _accepts_interprocess(event_cls):
            raise RuntimeError(
                f"Device backend '{self.device_type}' does not support "
                "interprocess events: Event does not accept interprocess=True."
            )
        if not hasattr(event_cls, "from_ipc_handle"):
            raise RuntimeError(
                f"Device backend '{self.device_type}' does not support "
                "interprocess event handles: Event.from_ipc_handle is missing."
            )

    def create_event(self, device: object) -> object:
        """Create a new interprocess-capable event."""
        return self._event_module.Event(interprocess=True)  # type: ignore[union-attr]

    def export_event(self, event: object, device: object) -> bytes:
        """Serialize ``event`` into a process-portable IPC handle."""
        return event.ipc_handle()  # type: ignore[attr-defined]

    def import_event(self, handle: bytes, device: object) -> object:
        """Reconstruct an event from an IPC handle on ``device``."""
        return self._event_module.Event.from_ipc_handle(  # type: ignore[union-attr]
            device, handle
        )

    def record_event(self, event: object, stream: object) -> None:
        """Record ``event`` on ``stream``."""
        event.record(stream)  # type: ignore[attr-defined]

    def wait_event(self, event: object, stream: object) -> None:
        """Make ``stream`` wait for ``event``."""
        event.wait(stream)  # type: ignore[attr-defined]

    def query_event(self, event: object) -> bool:
        """Return whether ``event`` has completed."""
        return bool(event.query())  # type: ignore[attr-defined]

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until ``event`` completes."""
        event.synchronize()  # type: ignore[attr-defined]


# Maximum number of events in the pool.  Covers all current models
# (max ~128 layers) with headroom for future scaling.
EVENT_POOL_SIZE = 256


class EventPool:
    """Pre-allocated pool of interprocess events for a (context, worker) pair.

    Created once at registration time on the server, exported once during
    handshake.  The worker imports all handles at registration and caches
    them by index.  During layerwise transfer, partial frames carry only a
    pool index (int) instead of a full IPC handle (~64 bytes), eliminating
    all per-request cudaIpcOpenEventHandle / cudaIpcGetEventHandle calls
    from the forward pass.
    """

    def __init__(
        self,
        backend: EventIPCBackend,
        device: object,
        size: int = EVENT_POOL_SIZE,
    ) -> None:
        self._backend = backend
        self._device = device
        self._size = size
        # Ensure events are created on the correct device — the server
        # process may have a different default CUDA device.
        with torch_dev.device(device):
            self._events: list[object] = [
                backend.create_event(device) for _ in range(size)
            ]
            self._handles: list[bytes] = [
                backend.export_event(evt, device) for evt in self._events
            ]

    @property
    def size(self) -> int:
        return self._size

    @property
    def handles(self) -> list[bytes]:
        """Exported IPC handles — sent once during registration handshake."""
        return self._handles

    def event_at(self, index: int) -> object:
        """Return the event at *index* (for server-side record)."""
        return self._events[index]

    def record(self, index: int, stream: object) -> None:
        """Record the event at *index* on *stream*."""
        self._backend.record_event(self._events[index], stream)

    @classmethod
    def import_pool(
        cls,
        backend: EventIPCBackend,
        device: object,
        handles: list[bytes],
    ) -> "EventPool":
        """Worker-side: import pre-exported handles into a local pool.

        Unlike the server constructor, this does NOT create new events —
        it imports existing ones from IPC handles (one-time cost at
        registration, not on the forward path).
        """
        pool = object.__new__(cls)
        pool._backend = backend
        pool._device = device
        pool._size = len(handles)
        pool._events = [backend.import_event(h, device) for h in handles]
        pool._handles = handles
        return pool


def get_event_ipc_backend(device: object) -> EventIPCBackend:
    """Return the event IPC backend for ``device``'s device type.

    Args:
        device: A ``torch.device``-like object, a device-type string, or an
            integer device index. Integer indices use the active
            ``torch_device_type``.

    Returns:
        The registered ``EventIPCBackend`` for the resolved device type.

    Raises:
        RuntimeError: If the device type has no registered ``DeviceSpec`` or
            its specification does not provide an event IPC backend.
    """
    device_type = _resolve_device_type(device)
    # Platform subclass discovery can import this module before
    # platform.__init__ has defined get_device_spec, so this import must stay
    # lazy to avoid a circular-initialization failure.
    # First Party
    from lmcache.v1.platform import get_device_spec

    spec = get_device_spec(device_type)
    if spec is None:
        raise RuntimeError(f"No DeviceSpec registered for device type {device_type!r}.")

    backend = spec.event_ipc_backend
    if backend is None:
        raise RuntimeError(f"Device type {device_type!r} does not support event IPC.")
    return backend
