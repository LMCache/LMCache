# SPDX-License-Identifier: Apache-2.0
"""Server-side lifetimes of IPC events shared with workers."""

# Standard
from collections import deque
import threading


class IPCEventRegistry:
    """Hold IPC events the server has exported to or imported from workers.

    Both maps are keyed by the IPC handle bytes. An exported completion event
    stays held until the worker releases its handle (``RELEASE_EVENT``), then
    returns to its context's pool. An imported worker event stays held until
    the host callback queued on the transfer stream right after its wait
    fires, which proves the wait has been consumed. The same worker handle is
    imported once per transfer that waits on it, so imports are counted per
    handle and one is dropped per callback. Both are released on facts, not
    inference.
    """

    def __init__(self) -> None:
        self._exported: dict[bytes, tuple[deque[object], object]] = {}
        self._imported: dict[bytes, list[object]] = {}
        self._lock = threading.Lock()

    def track_exported(self, handle: bytes, event: object, pool: deque[object]) -> None:
        """Hold ``event`` until ``handle`` is released, then return it to ``pool``.

        Args:
            handle: Serialized handle sent to the worker.
            event: The exported completion event.
            pool: Idle pool the event returns to on release.
        """
        with self._lock:
            self._exported[handle] = (pool, event)

    def release_exported(self, handle: bytes) -> None:
        """Return the event behind ``handle`` to its pool.

        Args:
            handle: Handle the worker has finished with. Unknown handles, such
                as those of an unregistered context, are ignored.
        """
        with self._lock:
            held = self._exported.pop(handle, None)
        if held is not None:
            pool, event = held
            pool.append(event)

    def track_imported(self, handle: bytes, event: object) -> None:
        """Hold an imported worker event until :meth:`release_imported`.

        Args:
            handle: The worker's handle the event was imported from.
            event: The imported event.
        """
        with self._lock:
            self._imported.setdefault(handle, []).append(event)

    def release_imported(self, handle: bytes) -> None:
        """Drop one imported event for ``handle``; its stream wait has drained.

        Args:
            handle: The worker's handle. Unknown handles are ignored.
        """
        with self._lock:
            events = self._imported.get(handle)
            if events:
                events.pop()
                if not events:
                    del self._imported[handle]

    def forget_pool(self, pool: deque[object]) -> None:
        """Drop exported events belonging to ``pool`` of an unregistered context.

        Args:
            pool: The context's idle pool.
        """
        with self._lock:
            stale = [h for h, (p, _) in self._exported.items() if p is pool]
            for handle in stale:
                del self._exported[handle]

    @property
    def exported_count(self) -> int:
        """Number of exported events not yet released by a worker."""
        with self._lock:
            return len(self._exported)

    @property
    def imported_count(self) -> int:
        """Number of imported events whose stream wait has not yet drained."""
        with self._lock:
            return sum(len(events) for events in self._imported.values())
