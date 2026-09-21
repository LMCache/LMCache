# SPDX-License-Identifier: Apache-2.0
"""Shared lifetime accounting for a native driver's process-wide state."""

# Standard
from collections.abc import Callable
import threading


class SharedDriver:
    """Keep a driver open until its last backend releases it.

    Each concrete backend class owns one SharedDriver. This is native resource
    accounting, not a selected backend singleton. Owners must finish their DMA
    and deregister their own resources before releasing the driver.
    """

    def __init__(self) -> None:
        self._owners: set[object] = set()
        self._lock = threading.Lock()

    def acquire(self, owner: object, open_driver: Callable[[], None]) -> None:
        """Add an owner, opening once; a failed open does not acquire ownership."""
        with self._lock:
            if owner in self._owners:
                return
            if not self._owners:
                open_driver()
            self._owners.add(owner)

    def release(self, owner: object, close_driver: Callable[[], None]) -> None:
        """Remove an owner and close only after the last owner releases it."""
        with self._lock:
            if owner not in self._owners:
                return
            self._owners.remove(owner)
            if not self._owners:
                close_driver()
