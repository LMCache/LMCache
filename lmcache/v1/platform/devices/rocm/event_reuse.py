# SPDX-License-Identifier: Apache-2.0
"""Interprocess event reuse on ROCm.

``hipIpcOpenEventHandle`` can block for seconds, and without reuse a consumer
pays it on every transfer: producers create one interprocess event per
transfer. Here a producer records its transfers on a small set of long-lived
events, and a consumer keeps the events it imported from one peer, so each
handle is opened once per registration.

Producer side (:class:`ReusingEventIPCBackend`): ``create_event`` hands out an
event whose last record has finished. The event leaves the pool there and
returns to it in ``record_event``, so it cannot be handed to a second transfer
before the first one recorded it. An event that is created and never recorded
(an error path) simply never returns.

Consumer side (:meth:`ReusingEventIPCBackend.for_peer`): the imports are kept
by a view that belongs to one registration of one peer. Registering again
(a restarted peer) starts from an empty view, so a handle of the previous
process is never served from it.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict, deque
from typing import Any
import threading
import weakref

# First Party
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend

# Pooled events queried per ``create_event`` before a new one is created.
_POOL_PROBES = 8
# Imports one peer view keeps (least recently used first out).
IMPORTED_EVENT_CACHE_SIZE = 8192


class ReusingEventIPCBackend(DefaultEventIPCBackend):
    """Event IPC backend that reuses the events it creates.

    Per device, the pool holds the events that were recorded since they were
    last handed out, oldest record first. It grows to the number of records in
    flight at once. A pooled event is never replaced: it lives as long as the
    backend unless it is handed out and never recorded.
    """

    def __init__(
        self,
        event_module: object | None = None,
        device_type: str | None = None,
    ) -> None:
        super().__init__(event_module=event_module, device_type=device_type)
        self._pool_lock = threading.Lock()
        self._pools: dict[str, deque[Any]] = {}
        # Handed out and not recorded yet -> its pool. Weak: an event that is
        # never recorded is dropped with its last reference.
        self._unrecorded: weakref.WeakKeyDictionary[Any, str] = (
            weakref.WeakKeyDictionary()
        )

    def create_event(self, device: object) -> object:
        """Return an interprocess event no pending transfer is recorded on.

        Args:
            device: Device on which the event will be recorded.

        Returns:
            A pooled event whose last record has finished, else a new event.
        """
        key = str(device)
        with self._pool_lock:
            pool = self._pools.setdefault(key, deque())
            event = None
            for _ in range(min(len(pool), _POOL_PROBES)):
                candidate = pool.popleft()
                # A consumer of the pending record must not see a later one.
                if candidate.query():
                    event = candidate
                    break
                pool.append(candidate)
            if event is None:
                event = super().create_event(device)
            try:
                self._unrecorded[event] = key
            except TypeError:
                pass  # not weakly referable: used once, as without the pool
            return event

    def record_event(self, event: object, stream: object) -> None:
        """Record ``event`` on ``stream`` and make it reusable once it finishes."""
        super().record_event(event, stream)
        with self._pool_lock:
            try:
                key = self._unrecorded.pop(event, None)
            except TypeError:
                key = None
            if key is not None:
                self._pools[key].append(event)

    def for_peer(self) -> "PeerEventIPCBackend":
        """Return the backend to use for one registration of one peer."""
        return PeerEventIPCBackend(self)


class PeerEventIPCBackend:
    """A backend for one registered peer that opens each of its handles once.

    Everything but the imports is delegated to the shared backend.

    Args:
        backend: The device's event IPC backend.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._imports: OrderedDict[tuple[bytes, str], object] = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def import_event(self, handle: bytes, device: object) -> object:
        """Return the event behind ``handle``, importing it on first use."""
        key = (bytes(handle), str(device))
        with self._lock:
            event = self._imports.get(key)
            if event is not None:
                self._imports.move_to_end(key)
                return event
        event = self._backend.import_event(handle, device)
        with self._lock:
            self._imports[key] = event
            while len(self._imports) > IMPORTED_EVENT_CACHE_SIZE:
                self._imports.popitem(last=False)
        return event
