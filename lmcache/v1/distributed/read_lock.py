# SPDX-License-Identifier: Apache-2.0
"""Owner-aware read reservations for resident L1 objects."""

# Standard
from collections.abc import Callable
import threading
import time


class ReadLock:
    """Keep independently expiring counts for each read reservation owner."""

    def __init__(
        self, ttl_seconds: int, clock: Callable[[], float] | None = None
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = time.monotonic if clock is None else clock
        self._owners: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def lock(self, owner: str, count: int = 1) -> None:
        """Reserve ``count`` reads for ``owner`` and renew only its expiry."""
        if count <= 0:
            raise ValueError("count must be positive")
        with self._lock:
            now = self._clock()
            self._prune(now)
            current = self._owners.get(owner, (0, now))[0]
            self._owners[owner] = (current + count, now + self._ttl_seconds)

    def unlock(self, owner: str, count: int = 1) -> bool:
        """Release only this owner's live reads; refuse missing or expired ones."""
        if count <= 0:
            raise ValueError("count must be positive")
        with self._lock:
            self._prune(self._clock())
            current = self._owners.get(owner)
            if current is None or current[0] < count:
                return False
            if current[0] == count:
                del self._owners[owner]
            else:
                self._owners[owner] = (current[0] - count, current[1])
            return True

    def is_locked(self, owner: str | None = None) -> bool:
        """Check whether ``owner`` (or any owner) still holds a live read."""
        with self._lock:
            self._prune(self._clock())
            return owner in self._owners if owner is not None else bool(self._owners)

    def legacy_owner(self) -> str | None:
        """Return the only live owner for a legacy anonymous operation.

        Anonymous callers can be kept compatible when there is no ambiguity.
        Multiple live owners deliberately return ``None`` so an old caller
        cannot release another reader's reservation.
        """
        with self._lock:
            self._prune(self._clock())
            if len(self._owners) != 1:
                return None
            return next(iter(self._owners))

    def _prune(self, now: float) -> None:
        for owner, (_, expiry) in list(self._owners.items()):
            if now >= expiry:
                del self._owners[owner]
