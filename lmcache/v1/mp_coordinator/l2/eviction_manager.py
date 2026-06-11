# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side eviction manager with per-``cache_salt`` LRU.

Wraps :class:`IsolatedLRUEvictionPolicy` for LRU key ordering,
matching the eviction logic in
:class:`~lmcache.v1.distributed.storage_controllers.eviction_controller.L2EvictionController`.

The manager periodically checks per-salt usage
(from :class:`L2UsageManager`) against ``watermark * quota``
(from :class:`QuotaManager`).
When a salt exceeds its threshold, it selects LRU keys and **logs**
them — actual deletion is not implemented yet.
"""

# Future
from __future__ import annotations

# Standard
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager

logger = init_logger(__name__)


class L2EvictionManager:
    """Per-``cache_salt`` LRU eviction manager for the coordinator.

    Delegates LRU ordering to :class:`IsolatedLRUEvictionPolicy`.
    Mirrors the trigger and ratio logic of
    :class:`L2EvictionController._check_and_evict_by_cache_salt`:
    eviction fires when ``usage >= watermark * quota``, and
    ``eviction_ratio`` is passed directly to the policy as a
    fraction of keys by count.

    Thread-safety: ``_key_sizes`` is guarded by ``_lock``;
    the policy has its own internal lock.

    Args:
        quota_manager: The shared quota registry.
        usage_manager: The shared usage manager.
        eviction_ratio: Fraction of tracked keys to evict per
            cycle (by count). Passed to the policy.
        trigger_watermark: Eviction fires when usage reaches
            this fraction of the quota.
    """

    def __init__(
        self,
        quota_manager: QuotaManager,
        usage_manager: L2UsageManager,
        eviction_ratio: float = 0.5,
        trigger_watermark: float = 1.0,
    ) -> None:
        self._lock = threading.Lock()
        self._quota_manager = quota_manager
        self._usage_manager = usage_manager
        self._eviction_ratio = max(0.0, min(1.0, eviction_ratio))
        self._trigger_watermark = trigger_watermark
        self._policy = IsolatedLRUEvictionPolicy()
        self._key_sizes: dict[ObjectKey, int] = {}

    def on_store(self, key: ObjectKey, size_bytes: int) -> None:
        """Record that a key was stored.

        Args:
            key: The object key that was stored.
            size_bytes: Number of bytes stored for this key.
        """
        self._policy.on_keys_created([key])
        with self._lock:
            self._key_sizes[key] = size_bytes

    def on_lookup(self, key: ObjectKey) -> None:
        """Record that a key was looked up (touch — move to MRU end).

        Args:
            key: The object key that was looked up.
        """
        self._policy.on_keys_touched([key])

    def on_remove(self, keys: list[ObjectKey]) -> None:
        """Remove keys from LRU tracking (after eviction is executed).

        Args:
            keys: The object keys that were removed.
        """
        if not keys:
            return
        self._policy.on_keys_removed(keys)
        with self._lock:
            for key in keys:
                self._key_sizes.pop(key, None)

    def execute_evictions(self) -> dict[str, list[ObjectKey]]:
        """Check all tracked salts against their quotas and log eviction candidates.

        For every tracked salt, compare usage against
        ``watermark * quota``. Salts over threshold get eviction
        scoped to their own LRU list. Salts with no quota or zero
        quota get a full eviction (ratio=1.0).

        Returns:
            A mapping of ``cache_salt`` to the list of keys selected
            for eviction.
        """
        tracked_salts = self._policy.get_tracked_salts()
        eviction_plan: dict[str, list[ObjectKey]] = {}

        for cache_salt in tracked_salts:
            current_bytes = self._usage_manager.get(cache_salt)
            if current_bytes <= 0:
                continue
            limit = self._quota_manager.get_limit_bytes(cache_salt)
            if current_bytes < self._trigger_watermark * limit:
                continue

            effective_ratio = 1.0 if limit == 0 else self._eviction_ratio
            actions = self._policy.get_eviction_actions(
                effective_ratio, cache_salt=cache_salt
            )
            keys_to_evict: list[ObjectKey] = []
            for action in actions:
                keys_to_evict.extend(action.keys)

            if keys_to_evict:
                eviction_plan[cache_salt] = keys_to_evict
                with self._lock:
                    sizes = [self._key_sizes.get(k, 0) for k in keys_to_evict]
                evict_bytes = sum(sizes)
                logger.info(
                    "Eviction plan for cache_salt=%r: %d keys "
                    "(%d bytes) to free; usage=%d, quota=%d, "
                    "watermark=%.2f, ratio=%.2f",
                    cache_salt,
                    len(keys_to_evict),
                    evict_bytes,
                    current_bytes,
                    limit,
                    self._trigger_watermark,
                    effective_ratio,
                )
                for k, size in zip(keys_to_evict, sizes, strict=True):
                    logger.info(
                        "  -> evict key: model=%s, kv_rank=%d, hash=%s, size=%d",
                        k.model_name,
                        k.kv_rank,
                        k.chunk_hash.hex(),
                        size,
                    )

        # TODO: once eviction is wired end-to-end, call on_remove()
        # for each salt's victims after the MP server confirms deletion.
        return eviction_plan
