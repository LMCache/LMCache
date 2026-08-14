# SPDX-License-Identifier: Apache-2.0
"""Per-``cache_salt`` L2 usage view for the MP coordinator.

Per-salt byte totals derived from the same admitted cache-event stream
that builds the key directory. Owned and fed by
``FleetEvictionController``, the only thing that acts on usage.
"""

# Future
from __future__ import annotations

# Standard
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType

logger = init_logger(__name__)

# One tracked L2 placement: ``(key, owner, backend)`` where ``owner`` is
# the reporting instance for private placements and ``""`` for shared
# pools (fleet-scoped, one pool per backend type) — the same identity
# the key directory upserts on.
_PlacementId = tuple[ObjectKey, str, str]


class L2UsageManager:
    """Thread-safe per-``cache_salt`` L2 byte usage view."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._placement_sizes: dict[_PlacementId, int] = {}
        self._key_sizes: dict[ObjectKey, int] = {}
        self._bytes_by_salt: dict[str, int] = {}
        self._total_bytes: int = 0

    def consume(self, batch: CacheEventBatch) -> None:
        """Account one gate-admitted batch: L2 ``STORE`` upserts
        placement bytes (delta on re-store), L2 ``DELETE`` removes them.

        Args:
            batch: The admitted batch; other tiers and ``ACCESS`` are
                ignored.
        """
        if batch.tier != Tier.L2:
            return
        owner = "" if batch.shared else batch.instance_id
        with self._lock:
            for entry in batch.entries:
                key = entry.key.to_object_key()
                placement_id = (key, owner, batch.backend)
                if batch.event_type == CacheEventType.STORE:
                    old = self._placement_sizes.get(placement_id, 0)
                    self._placement_sizes[placement_id] = entry.size_bytes
                    self._adjust_locked(key, entry.size_bytes - old)
                elif batch.event_type == CacheEventType.DELETE:
                    size = self._placement_sizes.pop(placement_id, 0)
                    self._adjust_locked(key, -size)

    def get(self, cache_salt: str) -> int:
        """Return the current L2 byte usage for ``cache_salt``."""
        with self._lock:
            return self._bytes_by_salt.get(cache_salt, 0)

    def get_all(self) -> dict[str, int]:
        """Return a snapshot copy of per-salt L2 byte usage."""
        with self._lock:
            return dict(self._bytes_by_salt)

    def get_total(self) -> int:
        """Return total L2 bytes tracked across all salts."""
        with self._lock:
            return self._total_bytes

    def get_key_size(self, key: ObjectKey) -> int:
        """Return the L2 bytes held for ``key`` across all its tracked
        placements (``0`` when the key has none)."""
        with self._lock:
            return self._key_sizes.get(key, 0)

    def _adjust_locked(self, key: ObjectKey, delta: int) -> None:
        """Apply ``delta`` bytes to the per-key/salt/total counters."""
        if delta == 0:
            return
        new_key_total = self._key_sizes.get(key, 0) + delta
        if new_key_total <= 0:
            self._key_sizes.pop(key, None)
        else:
            self._key_sizes[key] = new_key_total
        salt = key.cache_salt
        new_salt_total = self._bytes_by_salt.get(salt, 0) + delta
        if new_salt_total <= 0:
            if new_salt_total < 0:
                logger.warning(
                    "L2 usage underflow for cache_salt=%r (delta %d); clamping to 0",
                    salt,
                    delta,
                )
            self._bytes_by_salt.pop(salt, None)
        else:
            self._bytes_by_salt[salt] = new_salt_total
        self._total_bytes = max(0, self._total_bytes + delta)
