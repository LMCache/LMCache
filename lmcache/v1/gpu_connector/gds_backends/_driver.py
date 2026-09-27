# SPDX-License-Identifier: Apache-2.0
"""Ownership of a native driver's process-wide session."""

# Standard
from collections.abc import Callable
import threading


class SharedDriver:
    """Keep a session alive until its last backend releases it.

    Each native backend defines one class-level SharedDriver. Owners must finish
    IO and deregister their own resources before releasing their ownership.
    """

    def __init__(self) -> None:
        self._owners: set[object] = set()
        self._lock = threading.Lock()

    def acquire(self, owner: object, open_driver: Callable[[], None]) -> None:
        """Acquire once per owner; a failed open does not acquire ownership."""
        with self._lock:
            if owner in self._owners:
                return
            if not self._owners:
                open_driver()
            self._owners.add(owner)

    def release(self, owner: object, close_driver: Callable[[], None]) -> None:
        """Release once and close on the last owner; failed closes stay released."""
        with self._lock:
            if owner not in self._owners:
                return
            self._owners.remove(owner)
            if not self._owners:
                close_driver()
