# SPDX-License-Identifier: Apache-2.0
"""Server-side lifetimes of IPC events shared with workers."""

# Standard
import threading


class IPCEventRegistry:
    """Hold the IPC events a cache context shares with its worker until released."""

    def __init__(self) -> None:
        self._exported: dict[bytes, object] = {}
        self._imported: dict[int, object] = {}
        self._next_import_token = 0
        self._lock = threading.Lock()

    def hold_exported(self, handle: bytes, event: object) -> None:
        """Hold ``event`` until ``handle`` is released.

        Args:
            handle: Serialized handle sent to the worker.
            event: The exported completion event.
        """
        with self._lock:
            self._exported[handle] = event

    def release_exported(self, handle: bytes) -> bool:
        """Drop the event behind ``handle``; the worker is done with it.

        Args:
            handle: Handle the worker has finished with.

        Returns:
            Whether an event was held for ``handle``.
        """
        with self._lock:
            return self._exported.pop(handle, None) is not None

    def hold_imported(self, event: object) -> int:
        """Hold an imported worker event until :meth:`release_imported`.

        Args:
            event: The imported event.

        Returns:
            The token that releases this import.
        """
        with self._lock:
            token = self._next_import_token
            self._next_import_token += 1
            self._imported[token] = event
            return token

    def release_imported(self, token: int) -> bool:
        """Drop the imported event behind ``token``; its stream wait has drained.

        Args:
            token: Token returned by :meth:`hold_imported`.

        Returns:
            Whether an import was held for ``token``.
        """
        with self._lock:
            return self._imported.pop(token, None) is not None
