# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side eviction manager with per-``cache_salt`` LRU.

Mirrors the structure of
:class:`~lmcache.v1.distributed.eviction_policy.isolated_lru.IsolatedLRUEvictionPolicy`
but runs inside the coordinator process and uses a lightweight
:class:`CacheKey` instead of :class:`ObjectKey` (which pulls in
``torch``).

The manager periodically checks per-salt usage
(from :class:`L2UsageManager`) against limits
(from :class:`L2QuotaManager`).
When a salt exceeds its quota, it selects LRU victims and **logs**
them — actual deletion is not implemented yet.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.l2.quota_manager import L2QuotaManager
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.schemas import CacheKey

logger = init_logger(__name__)


class L2EvictionManager:
    """Per-``cache_salt`` LRU eviction manager for the coordinator.

    Maintains one ``OrderedDict`` per ``cache_salt``, ordered from
    least-recently-used (front) to most-recently-used (end). Also
    tracks per-key byte sizes so eviction can be byte-aware.

    Thread-safety: every public method acquires ``_lock``.

    Args:
        quota_manager: The shared quota registry.
        usage_manager: The shared usage manager.
        eviction_ratio: Fraction of over-quota bytes to target for
            eviction each cycle.
    """

    def __init__(
        self,
        quota_manager: L2QuotaManager,
        usage_manager: L2UsageManager,
        eviction_ratio: float = 0.5,
    ) -> None:
        self._lock = threading.Lock()
        self._quota_manager = quota_manager
        self._usage_manager = usage_manager
        self._eviction_ratio = max(0.0, min(1.0, eviction_ratio))
        self._per_salt_order: dict[str, OrderedDict[CacheKey, None]] = {}
        self._key_sizes: dict[CacheKey, int] = {}

    def on_store(self, key: CacheKey, size_bytes: int) -> None:
        """Record that a key was stored.

        Inserts into (or refreshes) the LRU for the key's
        ``cache_salt``, and records the per-key byte size.

        Args:
            key: The cache key that was stored.
            size_bytes: Number of bytes stored for this key.
        """
        with self._lock:
            order = self._per_salt_order.get(key.cache_salt)
            if order is None:
                order = OrderedDict()
                self._per_salt_order[key.cache_salt] = order
            if key in order:
                order.move_to_end(key)
            else:
                order[key] = None
            self._key_sizes[key] = size_bytes

    def on_lookup(self, key: CacheKey) -> None:
        """Record that a key was looked up (touch — move to MRU end).

        Args:
            key: The cache key that was looked up.
        """
        with self._lock:
            order = self._per_salt_order.get(key.cache_salt)
            if order is not None and key in order:
                order.move_to_end(key)

    def on_remove(self, keys: list[CacheKey]) -> None:
        """Remove keys from LRU tracking (after eviction is executed).

        Args:
            keys: The cache keys that were removed.
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                order = self._per_salt_order.get(key.cache_salt)
                if order is None:
                    continue
                order.pop(key, None)
                if not order:
                    del self._per_salt_order[key.cache_salt]
                self._key_sizes.pop(key, None)

    def execute_evictions(self) -> dict[str, list[CacheKey]]:
        """Check all tracked salts against their quotas and log eviction candidates.

        Salts with no quota or a zero quota are fully evicted. Salts
        over quota have LRU keys selected targeting ``eviction_ratio``
        of the over-quota bytes. Keys are logged but not actually
        deleted.

        Returns:
            A mapping of ``cache_salt`` to the list of keys selected
            for eviction.
        """
        quotas = {e.cache_salt: e.limit_bytes for e in self._quota_manager.list_all()}
        with self._lock:
            tracked_salts = list(self._per_salt_order.keys())

        eviction_plan: dict[str, list[CacheKey]] = {}

        for cache_salt in tracked_salts:
            limit_bytes = quotas.get(cache_salt, 0)
            current_bytes = self._usage_manager.get(cache_salt)
            if current_bytes <= limit_bytes:
                continue

            over_bytes = current_bytes - limit_bytes
            target_bytes = int(over_bytes * self._eviction_ratio)
            if target_bytes == 0 and over_bytes > 0:
                target_bytes = over_bytes

            keys_to_evict = self._select_keys_to_evict(cache_salt, target_bytes)
            if keys_to_evict:
                eviction_plan[cache_salt] = keys_to_evict
                evict_bytes = sum(self._key_sizes.get(k, 0) for k in keys_to_evict)
                logger.info(
                    "Eviction plan for cache_salt=%r: %d keys "
                    "(%d bytes) to free; usage=%d, quota=%d, "
                    "over_by=%d",
                    cache_salt,
                    len(keys_to_evict),
                    evict_bytes,
                    current_bytes,
                    limit_bytes,
                    over_bytes,
                )
                for k in keys_to_evict:
                    logger.info(
                        "  -> evict key: model=%s, kv_rank=%d, hash=%s, size=%d",
                        k.model_name,
                        k.kv_rank,
                        k.chunk_hash_hex,
                        self._key_sizes.get(k, 0),
                    )

        # TODO: once eviction is wired end-to-end, call on_remove()
        # for each salt's victims after the MP server confirms deletion.
        return eviction_plan

    def _select_keys_to_evict(
        self, cache_salt: str, target_bytes: int
    ) -> list[CacheKey]:
        """Select LRU victims from a salt's bucket to free ``target_bytes``.

        Args:
            cache_salt: The salt to evict from.
            target_bytes: Target number of bytes to free.

        Returns:
            List of keys in LRU order (oldest first).
        """
        with self._lock:
            order = self._per_salt_order.get(cache_salt)
            if not order:
                return []

            keys_to_evict: list[CacheKey] = []
            freed = 0
            for key in order:
                keys_to_evict.append(key)
                freed += self._key_sizes.get(key, 0)
                if freed >= target_bytes:
                    break

            return keys_to_evict
