# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side eviction manager with per-``cache_salt`` LRU.

Wraps :class:`IsolatedLRUEvictionPolicy` for LRU key ordering,
matching the eviction logic in
:class:`~lmcache.v1.distributed.storage_controllers.eviction_controller.L2EvictionController`.

The manager periodically checks per-salt usage
(from :class:`L2UsageManager`) against ``watermark * quota``
(from :class:`QuotaManager`). When a salt exceeds its threshold, the
manager selects LRU keys, dispatches a ``DELETE /l2`` to one
registered MP server, and updates its local LRU tracking after the
delete returns. The MP server in turn calls the underlying L2
adapter (S3 today), so a single dispatch is enough — all coordinators
share the same backing bucket.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict
import asyncio

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
            size ledger that this class reads for logging. Writes to
            the ledger (``record_stored`` / ``record_evicted``) are
            the caller's responsibility — paired with
            :meth:`on_store` / :meth:`on_remove`.
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

    def on_remove(self, key: ObjectKey) -> None:
        """Drop a single key from the LRU policy.

        Callers must also invoke :meth:`L2UsageManager.record_evicted`
        for the same key so the per-salt byte totals + per-key size
        ledger stay consistent.
        """
        self._policy.on_keys_removed([key])

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
    ) -> dict[str, list[ObjectKey]]:
        """Compute the eviction plan and fire a fan-out DELETE.

        Picks a uniformly random MP server from ``registry`` (via
        :meth:`InstanceRegistry.random_instance`) and **fire-and-forget**
        dispatches the full set of victim keys to its ``DELETE /l2``
        endpoint. Since every MP server in the fleet shares the same
        backing L2 (e.g. one S3 bucket), a single dispatch evicts the
        keys for all of them — there is no need to broadcast. Random
        selection spreads eviction-RPC load across the fleet.

        The LRU + per-salt usage updates are **not** applied here.
        After the MP server's L2 adapter finishes the deletion, it
        fires ``on_l2_keys_deleted`` on its registered listeners —
        :class:`L2EventListener` then ships ``DELETE`` events back
        through ``POST /l2/events``, and the coordinator's
        ``/l2/events`` handler calls :meth:`on_remove` on this
        manager. That round-trip is the authoritative signal that the
        keys are gone from the bucket.

        If dispatch fails (network error, no instances registered) the
        next eviction cycle will re-pick the same keys (the LRU still
        carries them) and try again — at-least-once semantics, safe
        because the lmcache delete is itself idempotent.

        Args:
            registry: Live MP server registry. Eviction is a no-op
                when empty.
            http_client: Shared async HTTP client owned by the
                coordinator app lifespan.

        Returns:
            The eviction plan that was scheduled. ``execute_evictions``
            returns as soon as the background dispatch task is
            spawned; a non-empty return does NOT imply the dispatch
            even left the process.
        """
        plan = self.compute_eviction_plan()
        if not plan:
            return plan

        target = registry.random_instance()
        if target is None:
            logger.warning(
                "Eviction plan computed (%d salts) but no MP servers are "
                "registered; skipping dispatch",
                len(plan),
            )
            return plan

        url = f"http://{target.ip}:{target.http_port}/l2"
        all_keys: list[ObjectKey] = [k for keys in plan.values() for k in keys]
        body = {"keys": [asdict(k.to_encoded_object_key()) for k in all_keys]}

        asyncio.create_task(  # noqa: RUF006
            self._dispatch_eviction(
                http_client=http_client,
                url=url,
                body=body,
                instance_id=target.instance_id,
                key_count=len(all_keys),
                salt_count=len(plan),
            )
        )
        return plan

    @staticmethod
    async def _dispatch_eviction(
        http_client: httpx.AsyncClient,
        url: str,
        body: dict,
        instance_id: str,
        key_count: int,
        salt_count: int,
    ) -> None:
        """Background task: send the DELETE and log the outcome.

        Failures are logged but not retried here — the next eviction
        cycle's :meth:`compute_eviction_plan` will pick the same keys
        again (because the LRU is only cleared by ``DELETE`` events,
        not by a successful dispatch).
        """
        try:
            # ``httpx.AsyncClient.delete`` doesn't accept ``json=`` —
            # ``request("DELETE", ...)`` is the supported form for
            # DELETE-with-body.
            resp = await http_client.request("DELETE", url, json=body)
            resp.raise_for_status()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "Eviction dispatch to %s (%d keys) failed: %s",
                instance_id,
                key_count,
                e,
            )
            return
        logger.info(
            "Eviction dispatched to %s: %d keys across %d salts",
            instance_id,
            key_count,
            salt_count,
        )
