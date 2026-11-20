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
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable
import atexit
import inspect
import os
import queue
import threading
import time

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

    #: How many times one interprocess event may be recorded while the handles
    #: exported for it keep tracking its latest record, so that callers may
    #: reuse it (see :class:`EventRecordPool`). ``1`` means an exported handle
    #: stands only for the record made before the export: callers create a new
    #: event per operation. ``None`` means no limit.
    max_event_records: int | None = 1

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


# Persist the reference to the IPC events for a while so that the inference engine
# won't access the dangling references to the events.
_EXPORTED_EVENT_RING_SIZE = 2048


class DefaultEventIPCBackend(EventIPCBackend):
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
        max_event_records: See :attr:`EventIPCBackend.max_event_records`.
            The default ``1`` keeps one event per operation; a device spec
            whose interprocess events track re-records raises it.
    """

    def __init__(
        self,
        event_module: object | None = None,
        device_type: str | None = None,
        max_event_records: int | None = 1,
    ) -> None:
        self._event_module = event_module if event_module is not None else torch_dev
        self.device_type = device_type if device_type is not None else torch_device_type
        self.max_event_records = max_event_records
        self._exported_events: deque[object] = deque(maxlen=_EXPORTED_EVENT_RING_SIZE)

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
        """Serialize ``event`` into a process-portable IPC handle.

        Args:
            event: The interprocess-capable event to export.
            device: Device that owns the event (unused; part of the protocol
                signature).

        Returns:
            The IPC handle bytes for ``event``.
        """
        handle = event.ipc_handle()  # type: ignore[attr-defined]
        if len(self._exported_events) == _EXPORTED_EVENT_RING_SIZE:
            # Destroying an event can block; drop the oldest off this thread.
            _background_releaser.release(self._exported_events.popleft())
        self._exported_events.append(event)
        return handle

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


def _max_event_records(backend: object) -> int | None:
    """Return ``backend.max_event_records``; ``1`` if it declares none."""
    max_records = getattr(backend, "max_event_records", 1)
    if max_records is None or (isinstance(max_records, int) and max_records > 1):
        return max_records
    return 1


@dataclass(eq=False)
class _PooledEvent:
    """One event of an :class:`EventRecordPool` and its export state."""

    event: object
    records: int = 0
    handle: bytes = b""


class EventRecordPool:
    """Interprocess events recorded again instead of created per operation.

    Creating, exporting, importing and destroying an interprocess event for
    every operation costs device-runtime calls in both processes, and
    destroying an imported event can block the destroying thread. Where the
    backend's exported handles track later records (``max_event_records``
    other than ``1``), :meth:`acquire` hands out pooled events round-robin per
    stream and :meth:`export` exports each one once.

    Recording an event again only moves it later on the same stream, so a
    process that waits on or queries an older record of it waits for a newer
    one instead: it may wait longer, never less. Events are therefore pooled
    per stream, and an event is replaced once it has been recorded
    ``max_event_records`` times. Otherwise every :meth:`acquire` creates a new
    event and :meth:`export` exports it, as without the pool.

    Thread-safe.

    Args:
        backend: Event backend that creates, records and exports the events.
        device: Device that owns the events.
        slots: Pooled events per stream. An event is handed out again after
            ``slots`` further acquisitions for its stream.
    """

    def __init__(self, backend: EventIPCBackend, device: object, slots: int) -> None:
        if slots < 1:
            raise ValueError(f"slots must be positive, got {slots}")
        self._backend = backend
        self._device = device
        self._slots = slots
        self._max_records = _max_event_records(backend)
        # Per stream: the pooled events and the slot handed out next.
        self._pools: dict[object, tuple[list[_PooledEvent | None], list[int]]] = {}
        # Pooled events by id(), to find an exported event's cached handle.
        self._by_id: dict[int, _PooledEvent] = {}
        self._lock = threading.Lock()

    def acquire(self, stream: object) -> object:
        """Return the event to record next on ``stream``.

        The caller records the event on ``stream`` (and nowhere else) before
        exporting it; the pool counts that record.

        Args:
            stream: Stream on which the caller records the event.

        Returns:
            A backend-native event, possibly one recorded before on ``stream``.
        """
        if self._max_records == 1:
            return self._backend.create_event(self._device)
        with self._lock:
            slots, next_slot = self._pools.setdefault(
                stream, ([None] * self._slots, [0])
            )
            index = next_slot[0]
            next_slot[0] = (index + 1) % self._slots
            pooled = slots[index]
            if pooled is None or (
                self._max_records is not None and pooled.records >= self._max_records
            ):
                if pooled is not None:
                    del self._by_id[id(pooled.event)]
                pooled = _PooledEvent(self._backend.create_event(self._device))
                slots[index] = pooled
                self._by_id[id(pooled.event)] = pooled
            pooled.records += 1
            return pooled.event

    def export(self, event: object) -> bytes:
        """Return the IPC handle of ``event``, exporting a pooled event once.

        Args:
            event: An event from :meth:`acquire`, or any other event of the
                backend, which is exported as is.

        Returns:
            Serialized event handle.
        """
        with self._lock:
            pooled = self._by_id.get(id(event))
            if pooled is not None and pooled.event is event:
                if not pooled.handle:
                    pooled.handle = self._backend.export_event(event, self._device)
                return pooled.handle
        return self._backend.export_event(event, self._device)

    def close(self) -> None:
        """Drop the pooled events; later acquisitions start new ones."""
        with self._lock:
            self._pools.clear()
            self._by_id.clear()


# Seconds an evicted import is kept on the release thread before it is dropped.
_RELEASE_GRACE_S = 0.1


class _BackgroundReleaser:
    """Drops references on a daemon thread, started on first use.

    Each object is kept for ``_RELEASE_GRACE_S`` after its hand-over, so the
    thread that handed it over has dropped its own references by then and
    the last one is dropped here. At interpreter exit the thread drops what
    it still holds and is joined, so no object is destroyed on it while the
    process tears the device runtime down.
    """

    _STOP = object()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._pid = 0
        # Process whose release thread was stopped; it drops objects inline.
        self._stopped_pid = 0
        atexit.register(self.stop)

    def release(self, obj: object) -> None:
        """Hand ``obj`` over to be dropped on the release thread."""
        with self._lock:
            if self._stopped_pid == os.getpid():
                return
            if self._pid != os.getpid():
                # First use, or a forked child without the parent's thread.
                self._queue = queue.SimpleQueue()
                self._pid = os.getpid()
                self._thread = threading.Thread(
                    target=self._run,
                    args=(self._queue,),
                    daemon=True,
                    name="lmcache-event-release",
                )
                self._thread.start()
            self._queue.put(obj)

    def stop(self) -> None:
        """Drop everything handed over so far and stop the release thread.

        Objects handed over afterwards are dropped by the caller.
        """
        with self._lock:
            self._stopped_pid = os.getpid()
            thread = self._thread if self._pid == os.getpid() else None
            self._thread = None
            if thread is None:
                return
            self._queue.put(self._STOP)
        thread.join()

    @classmethod
    def _run(cls, items: "queue.SimpleQueue[object]") -> None:
        held: deque[tuple[float, object]] = deque()
        while True:
            timeout = (
                max(0.0, held[0][0] + _RELEASE_GRACE_S - time.monotonic())
                if held
                else None
            )
            try:
                obj = items.get(timeout=timeout)
                if obj is cls._STOP:
                    return
                held.append((time.monotonic(), obj))
                del obj
            except queue.Empty:
                pass
            now = time.monotonic()
            while held and now - held[0][0] >= _RELEASE_GRACE_S:
                held.popleft()


_background_releaser = _BackgroundReleaser()


class CachedImportEventBackend(EventIPCBackend):
    """Event backend wrapper that imports each event handle once.

    A peer that reuses its events (see :class:`EventRecordPool`) sends the
    same handle again; this wrapper returns the event imported the first
    time instead of importing it again and destroying the copy afterwards.
    At most ``capacity`` imports are kept, least recently used out first; an
    evicted import is released on a background thread, since destroying an
    imported event can block the destroying thread. A peer that creates an
    event per operation sends only new handles: each is imported once and
    evicted later, as before, but released off the caller's thread.

    Backends whose exported handles do not track later records
    (``max_event_records == 1``) are passed through unchanged, and so are all
    other operations. Thread-safe.

    Args:
        backend: The wrapped event backend.
        capacity: Imported events kept at most.
        release: Takes an evicted import; defaults to dropping it on a
            background thread. Injectable for tests.
    """

    def __init__(
        self,
        backend: EventIPCBackend,
        capacity: int,
        release: Callable[[object], None] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._backend = backend
        self._capacity = capacity
        self._release = release if release is not None else _background_releaser.release
        self._imports: OrderedDict[tuple[bytes, object], object] = OrderedDict()
        self._lock = threading.Lock()
        self.device_type = backend.device_type
        self.max_event_records = _max_event_records(backend)
        self._cache = self.max_event_records != 1

    def check_event_support(self, device: object) -> None:
        """Delegate to the wrapped backend."""
        self._backend.check_event_support(device)

    def create_event(self, device: object) -> object:
        """Delegate to the wrapped backend."""
        return self._backend.create_event(device)

    def export_event(self, event: object, device: object) -> bytes:
        """Delegate to the wrapped backend."""
        return self._backend.export_event(event, device)

    def record_event(self, event: object, stream: object) -> None:
        """Delegate to the wrapped backend."""
        self._backend.record_event(event, stream)

    def wait_event(self, event: object, stream: object) -> None:
        """Delegate to the wrapped backend."""
        self._backend.wait_event(event, stream)

    def query_event(self, event: object) -> bool:
        """Delegate to the wrapped backend."""
        return self._backend.query_event(event)

    def synchronize_event(self, event: object, device: object) -> None:
        """Delegate to the wrapped backend."""
        self._backend.synchronize_event(event, device)

    def import_event(self, handle: bytes, device: object) -> object:
        """Return the event imported for ``handle``, importing it once."""
        if not self._cache:
            return self._backend.import_event(handle, device)
        key = (handle, device)
        with self._lock:
            event = self._imports.get(key)
            if event is not None:
                self._imports.move_to_end(key)
                return event
        imported = self._backend.import_event(handle, device)
        evicted: list[object] = []
        with self._lock:
            # Another thread may have imported the same handle meanwhile.
            event = self._imports.setdefault(key, imported)
            self._imports.move_to_end(key)
            while len(self._imports) > self._capacity:
                evicted.append(self._imports.popitem(last=False)[1])
        if event is not imported:
            evicted.append(imported)
        del imported
        for old in evicted:
            self._release(old)
        return event

    def close(self) -> None:
        """Release every kept import."""
        with self._lock:
            imports = list(self._imports.values())
            self._imports.clear()
        for event in imports:
            self._release(event)


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
