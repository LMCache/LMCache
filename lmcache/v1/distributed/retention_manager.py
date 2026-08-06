# SPDX-License-Identifier: Apache-2.0
"""
Per-key retention registry for explicitly retained KV chunks.

A store request may carry a retention ttl (see
``IPCCacheServerKey.retention_ttl_sec``). The chunks written by such a
store are registered here with a deadline; eviction paths consult
``is_evictable`` and skip keys whose deadline has not passed. Expired
keys are dropped by ``sweep`` and thereby rejoin the normal LRU pool --
expiry never deletes data.

Deadlines are extend-only: re-storing a key with a shorter ttl never
shortens its existing window.

A byte budget caps how much of the cache retention can shield, so the
eviction loop always has evictable keys left. Keys that would push the
retained total over the budget are simply not registered -- the store
itself proceeds and the data stays subject to normal LRU eviction.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
import operator
import threading
import time

# Third Party
from sortedcontainers import SortedKeyList

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey

logger = init_logger(__name__)


class RetentionManager:
    """Thread-safe map of ObjectKey -> retention deadline.

    Args:
        max_retained_bytes: Byte budget for retained data. 0 rejects
            every registration (retention effectively disabled).
        clock: Monotonic time source, injectable for tests.
    """

    def __init__(
        self,
        max_retained_bytes: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._max_retained_bytes = max_retained_bytes
        # ObjectKey -> (deadline, size_bytes)
        self._entries: dict[ObjectKey, tuple[float, int]] = {}
        # (deadline, key) ordered by deadline; in lockstep with _entries.
        self._deadlines = SortedKeyList(key=operator.itemgetter(0))
        self._retained_bytes = 0
        self._num_stamps = 0
        self._num_extends = 0
        self._num_expirations = 0
        self._num_budget_rejections = 0

    def note_stored(
        self,
        keys: list[ObjectKey],
        sizes: list[int],
        ttl_sec: int,
    ) -> int:
        """Register (or extend) retention for freshly stored keys.

        Extending an already-retained key is always allowed and costs no
        budget. New keys are admitted while the retained total stays
        within the budget; the rest are skipped and counted as
        rejections.

        Returns the number of keys registered or extended.
        """
        if ttl_sec <= 0 or not keys:
            return 0
        deadline = self._clock() + ttl_sec
        accepted = 0
        with self._lock:
            for key, size in zip(keys, sizes, strict=True):
                entry = self._entries.get(key)
                if entry is not None:
                    if deadline > entry[0]:
                        self._deadlines.remove((entry[0], key))
                        self._entries[key] = (deadline, entry[1])
                        self._deadlines.add((deadline, key))
                    self._num_extends += 1
                    accepted += 1
                    continue
                if self._retained_bytes + size > self._max_retained_bytes:
                    self._num_budget_rejections += 1
                    continue
                self._entries[key] = (deadline, size)
                self._deadlines.add((deadline, key))
                self._retained_bytes += size
                self._num_stamps += 1
                accepted += 1
            retained_bytes = self._retained_bytes
        if accepted < len(keys):
            logger.warning(
                "Retention budget exhausted: admitted %d of %d keys "
                "(retained_bytes=%d, budget=%d)",
                accepted,
                len(keys),
                retained_bytes,
                self._max_retained_bytes,
            )
        return accepted

    @property
    def max_retained_bytes(self) -> int:
        """The configured retention byte budget (0 = retention disabled)."""
        return self._max_retained_bytes

    def is_evictable(self, key: ObjectKey) -> bool:
        """False while the key's retention window is still open."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return True
            return entry[0] <= self._clock()

    def forget(self, keys: list[ObjectKey]) -> None:
        """Drop entries for keys deleted outside the eviction loop."""
        with self._lock:
            for key in keys:
                entry = self._entries.pop(key, None)
                if entry is not None:
                    self._deadlines.remove((entry[0], key))
                    self._retained_bytes -= entry[1]

    def sweep(self) -> int:
        """Drop expired entries so their keys rejoin the LRU pool."""
        now = self._clock()
        expired = 0
        with self._lock:
            while self._deadlines and self._deadlines[0][0] <= now:
                _, key = self._deadlines.pop(0)
                self._retained_bytes -= self._entries.pop(key)[1]
                expired += 1
            self._num_expirations += expired
        return expired

    def report_status(self) -> dict:
        with self._lock:
            return {
                "retained_keys": len(self._entries),
                "retained_bytes": self._retained_bytes,
                "max_retained_bytes": self._max_retained_bytes,
                "stamps": self._num_stamps,
                "extends": self._num_extends,
                "expirations": self._num_expirations,
                "budget_rejections": self._num_budget_rejections,
            }
