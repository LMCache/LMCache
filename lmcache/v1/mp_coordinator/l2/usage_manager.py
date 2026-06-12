# SPDX-License-Identifier: Apache-2.0
"""Per-``cache_salt`` L2 usage manager for the MP coordinator.

Maintains running byte totals per tenant, updated by store events
reported by MP servers. Eviction (byte subtraction) is driven by
the coordinator itself, not by MP servers. Also owns the per-key size
ledger so re-stores (same key, same or different size) don't
double-count and so coordinator-initiated evictions can recover the
exact bytes to subtract.

Thread-safe and dependency-free apart from
:class:`~lmcache.v1.distributed.api.ObjectKey`.
"""

# Future
from __future__ import annotations

# Standard
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey

logger = init_logger(__name__)


class L2UsageManager:
    """Thread-safe in-memory manager of L2 byte usage per ``cache_salt``.

    Two views over the same state:

    - Per-tenant byte totals (``_bytes_by_salt``) used by quota
      enforcement.
    - Per-key size ledger (``_key_sizes``) used to make re-stores
      idempotent and to give the eviction path the bytes it freed.

    Both are kept in sync under a single lock. Byte counters are
    clamped at zero on underflow.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bytes_by_salt: dict[str, int] = {}
        self._total_bytes: int = 0
        self._key_sizes: dict[ObjectKey, int] = {}

    def has_key(self, key: ObjectKey) -> bool:
        """Return ``True`` if ``key`` has a recorded size.

        Useful for callers that want to skip duplicate work when a key
        is already tracked (e.g. the resync pass deciding whether to
        propagate a store event into the eviction LRU).
        """
        with self._lock:
            return key in self._key_sizes

    def get_key_size(self, key: ObjectKey) -> int | None:
        """Return the bytes tracked for ``key``, or ``None`` if unknown."""
        with self._lock:
            return self._key_sizes.get(key)

    def record_stored(self, key: ObjectKey, num_bytes: int) -> None:
        """Record that ``key`` is now resident on L2 at ``num_bytes``.

        If ``key`` was already tracked, this **replaces** its size and
        adjusts the per-salt + global totals by the size delta — so a
        re-store at the same size is a no-op and a re-store at a
        different size correctly reflects the new value (no
        double-counting).

        Args:
            key: The object key. ``key.cache_salt`` determines which
                bucket the bytes contribute to.
            num_bytes: Bytes stored. Must be non-negative.

        Raises:
            ValueError: If ``num_bytes`` is negative.
        """
        if num_bytes < 0:
            raise ValueError(f"num_bytes must be non-negative (got {num_bytes})")
        with self._lock:
            existing = self._key_sizes.get(key)
            if existing is not None:
                delta = num_bytes - existing
            else:
                delta = num_bytes
            if delta == 0:
                # Already tracked at the same size — keep the entry,
                # nothing to adjust.
                self._key_sizes[key] = num_bytes
                return
            self._key_sizes[key] = num_bytes
            salt = key.cache_salt
            new_salt_total = self._bytes_by_salt.get(salt, 0) + delta
            if new_salt_total <= 0:
                self._bytes_by_salt.pop(salt, None)
            else:
                self._bytes_by_salt[salt] = new_salt_total
            self._total_bytes = max(0, self._total_bytes + delta)

    def record_evicted(self, key: ObjectKey) -> int:
        """Record that the coordinator evicted ``key`` from L2.

        Removes ``key`` from the size ledger and subtracts its
        recorded bytes from the per-salt and global totals. A no-op
        for keys that were never recorded.

        Args:
            key: The object key being evicted.

        Returns:
            Bytes freed (the size that was recorded for ``key``, or
            ``0`` when the key was unknown).
        """
        with self._lock:
            size = self._key_sizes.pop(key, None)
            if size is None or size == 0:
                return size or 0
            salt = key.cache_salt
            current = self._bytes_by_salt.get(salt, 0)
            new_val = current - size
            if new_val < 0:
                logger.warning(
                    "Usage underflow for cache_salt=%r on evict: %d - %d = %d",
                    salt,
                    current,
                    size,
                    new_val,
                )
                new_val = 0
            if new_val == 0:
                self._bytes_by_salt.pop(salt, None)
            else:
                self._bytes_by_salt[salt] = new_val
            self._total_bytes = max(0, self._total_bytes - size)
            return size

    def get(self, cache_salt: str) -> int:
        """Return the current byte usage for ``cache_salt``.

        Args:
            cache_salt: The tenant identifier.

        Returns:
            Bytes currently tracked, or 0 if no usage recorded.
        """
        with self._lock:
            return self._bytes_by_salt.get(cache_salt, 0)

    def get_all(self) -> dict[str, int]:
        """Return a snapshot of per-salt byte usage.

        Returns:
            A copy of the internal mapping (salt -> bytes).
        """
        with self._lock:
            return dict(self._bytes_by_salt)

    def get_total(self) -> int:
        """Return the total bytes tracked across all salts.

        Returns:
            Total byte usage.
        """
        with self._lock:
            return self._total_bytes
