# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side eviction manager with per-``cache_salt`` LRU.

Wraps :class:`IsolatedLRUEvictionPolicy` for LRU key ordering,
matching the eviction logic in
:class:`~lmcache.v1.distributed.storage_controllers.eviction_controller.L2EvictionController`.

The manager periodically checks per-salt usage
(from :class:`L2UsageManager`) against ``watermark * quota``
(from :class:`QuotaManager`). When a salt exceeds its threshold, the
manager selects LRU keys, dispatches a ``POST /l2/keys:evict`` to one
registered MP server, and updates its local LRU tracking after the
delete returns. The MP server in turn calls the underlying L2
adapter (S3 today), so a single dispatch is enough — all coordinators
share the same backing bucket.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.registry import InstanceRegistry

logger = init_logger(__name__)


class L2EvictionManager:
    """Per-``cache_salt`` LRU eviction manager for the coordinator.

    Delegates LRU ordering to :class:`IsolatedLRUEvictionPolicy` and
    the per-key size ledger to the shared :class:`L2UsageManager`.
    Mirrors the trigger and ratio logic of
    :class:`L2EvictionController._check_and_evict_by_cache_salt`:
    eviction fires when ``usage >= watermark * quota``, and
    ``eviction_ratio`` is passed directly to the policy as a
    fraction of keys by count.

    Args:
        quota_manager: The shared quota registry.
        usage_manager: The shared usage manager; owns the per-key
            size ledger that this class reads for logging and that
            :meth:`on_remove` updates as part of the eviction
            bookkeeping.
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
        self._quota_manager = quota_manager
        self._usage_manager = usage_manager
        self._eviction_ratio = max(0.0, min(1.0, eviction_ratio))
        self._trigger_watermark = trigger_watermark
        self._policy = IsolatedLRUEvictionPolicy()

    def on_store(self, key: ObjectKey) -> None:
        """Record that a key was stored — register it in the LRU policy.

        Sizes / per-tenant byte accounting are owned by
        :class:`L2UsageManager`; callers should invoke
        ``usage_manager.record_stored(key, num_bytes)`` separately.
        """
        self._policy.on_keys_created([key])

    def on_lookup(self, key: ObjectKey) -> None:
        """Record that a key was looked up (touch — move to MRU end)."""
        self._policy.on_keys_touched([key])

    def on_remove(self, keys: list[ObjectKey]) -> None:
        """Remove keys from LRU tracking and the size ledger.

        Calls :meth:`L2UsageManager.record_evicted` for each key so
        per-tenant byte totals reflect the freed bytes.
        """
        if not keys:
            return
        self._policy.on_keys_removed(keys)
        for key in keys:
            self._usage_manager.record_evicted(key)

    def compute_eviction_plan(self) -> dict[str, list[ObjectKey]]:
        """Check all tracked salts against their quotas and select
        eviction candidates per salt.

        For every tracked salt, compare usage against
        ``watermark * quota``. Salts over threshold get eviction
        scoped to their own LRU list. Salts with no quota or zero
        quota get a full eviction (ratio=1.0).

        Pure: no network calls, no state mutation beyond logging. The
        caller (:meth:`execute_evictions`) is responsible for applying
        the plan against the fleet and updating the LRU.

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
                sizes = [
                    self._usage_manager.get_key_size(k) or 0 for k in keys_to_evict
                ]
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

        return eviction_plan

    async def execute_evictions(
        self,
        registry: InstanceRegistry,
        http_client: httpx.AsyncClient,
        request_timeout: float = 30.0,
    ) -> dict[str, list[ObjectKey]]:
        """Compute the eviction plan and apply it via the MP fleet.

        Picks any one MP server from ``registry`` and dispatches the
        full set of victim keys to its ``POST /l2/keys:evict``
        endpoint. Since every MP server in the fleet shares the same
        backing L2 (e.g. one S3 bucket), a single dispatch evicts the
        keys for all of them — there is no need to broadcast.

        On a successful HTTP response, removes the dispatched keys
        from the local LRU via :meth:`on_remove` so the coordinator's
        tracking matches the fleet's actual state. On any failure
        (no registered instances, network error, non-2xx response)
        the LRU is **not** updated — the next eviction cycle will
        re-select the same keys and retry.

        Args:
            registry: Live MP server registry. Eviction is a no-op
                when empty.
            http_client: Shared async HTTP client owned by the
                coordinator app lifespan.
            request_timeout: Per-request timeout in seconds passed to
                ``httpx``.

        Returns:
            The eviction plan that was attempted (same shape as
            :meth:`compute_eviction_plan`). A non-empty return does
            NOT imply the dispatch succeeded — check the logs.
        """
        plan = self.compute_eviction_plan()
        if not plan:
            return plan

        instances = registry.all_instances()
        if not instances:
            logger.warning(
                "Eviction plan computed (%d salts) but no MP servers are "
                "registered; skipping dispatch",
                len(plan),
            )
            return plan

        target = instances[0]
        url = f"http://{target.ip}:{target.http_port}/l2/keys:evict"
        all_keys: list[ObjectKey] = [k for keys in plan.values() for k in keys]
        body = {"keys": [asdict(k.to_encoded_object_key()) for k in all_keys]}

        try:
            resp = await http_client.post(url, json=body, timeout=request_timeout)
            resp.raise_for_status()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "Eviction dispatch to %s (%d keys) failed: %s; LRU unchanged",
                target.instance_id,
                len(all_keys),
                e,
            )
            return plan

        logger.info(
            "Eviction dispatched to %s: %d keys across %d salts",
            target.instance_id,
            len(all_keys),
            len(plan),
        )
        self.on_remove(all_keys)
        return plan
