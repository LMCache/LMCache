# SPDX-License-Identifier: Apache-2.0
"""Pre-allocated pool of interprocess device events.

Kept separate from :mod:`lmcache.v1.platform.base.event_ipc` so the default
handle path never imports pool machinery; it is only used by the layer-wise
transfer path, which signals an event by pool index instead of exporting a
fresh IPC handle per layer.
"""

# Future
from __future__ import annotations

# First Party
from lmcache import torch_dev
from lmcache.v1.platform.base.event_ipc import EventIPCBackend

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
