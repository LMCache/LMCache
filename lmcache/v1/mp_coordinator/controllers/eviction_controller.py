# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide per-``cache_salt`` L2 eviction control loop.

See ``docs/design/v1/mp_coordinator/l2_usage_and_eviction.md``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict
from typing import TYPE_CHECKING
import asyncio

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import Tier
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType
from lmcache.v1.mp_coordinator.controllers.usage_manager import L2UsageManager
from lmcache.v1.multiprocess.cache_control.object_service import (
    MAX_DELETE_BATCH,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.mp_coordinator.registry import InstanceRegistry

logger = init_logger(__name__)


class FleetEvictionController:
    """Per-``cache_salt`` L2 eviction controller for the fleet.

    Owns the quota registry and usage view it enforces against, both
    exposed for the ``/quota`` endpoints and both fed from
    :meth:`consume`. :meth:`run` is the loop, :meth:`execute_evictions`
    one pass of it.

    Args:
        eviction_ratio: Fraction of tracked keys to evict per cycle.
        trigger_watermark: Eviction fires when usage reaches this
            fraction of the quota.
    """

    def __init__(
        self,
        eviction_ratio: float = 0.5,
        trigger_watermark: float = 1.0,
    ) -> None:
        self._quota_manager = QuotaManager()
        self._usage_manager = L2UsageManager()
        self._eviction_ratio = max(0.0, min(1.0, eviction_ratio))
        self._trigger_watermark = trigger_watermark
        self._policy = IsolatedLRUEvictionPolicy()
        self._in_flight_dispatches: set[asyncio.Task] = set()
        # Reference-counted L2 pins: a key is excluded from eviction plans while
        # its count is > 0. Not persisted across coordinator restarts.
        self._pin_counts: dict[ObjectKey, int] = {}

    @property
    def quota(self) -> QuotaManager:
        """The budgets this controller enforces."""
        return self._quota_manager

    @property
    def usage(self) -> L2UsageManager:
        """The usage view this controller enforces against."""
        return self._usage_manager

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one gate-admitted batch to usage, then the LRU.

        A delete drops the key from the LRU only once its **last** L2
        placement is gone: usage is per placement, so while another copy
        still holds bytes the key must stay evictable, or those bytes
        could exceed quota with nothing for the planner to select. Usage
        consuming first is what makes that size read correct.

        Args:
            batch: The admitted batch.
        """
        if batch.tier != Tier.L2:
            return
        self._usage_manager.consume(batch)
        for entry in batch.entries:
            key = entry.key.to_object_key()
            if batch.event_type == CacheEventType.STORE:
                self.on_store(key)
            elif batch.event_type == CacheEventType.ACCESS:
                self.on_lookup(key)
            elif batch.event_type == CacheEventType.DELETE:
                if self._usage_manager.get_key_size(key) == 0:
                    self.on_remove(key)

    def fence_instance(self, instance_id: str) -> None:
        """No-op: fencing voids L1 only, and the L2 bytes this
        controller accounts outlive the reporting process.

        Args:
            instance_id: The restarted or departed instance (unused).
        """

    def on_store(self, key: ObjectKey) -> None:
        """Register a stored key in the LRU (bytes go to the usage view)."""
        self._policy.on_keys_created([key])

    def on_lookup(self, key: ObjectKey) -> None:
        """Touch ``key`` in the LRU (move to MRU end)."""
        self._policy.on_keys_touched([key])

    def on_remove(self, key: ObjectKey) -> None:
        """Drop ``key`` from the LRU (bytes go to the usage view)."""
        self._policy.on_keys_removed([key])

    def pin(self, keys: list[ObjectKey]) -> None:
        """Increment each key's pin count, excluding it from eviction."""
        for key in keys:
            self._pin_counts[key] = self._pin_counts.get(key, 0) + 1

    def unpin(self, keys: list[ObjectKey]) -> None:
        """Decrement each key's pin count, floored at 0."""
        for key in keys:
            count = self._pin_counts.get(key, 0)
            if count <= 1:
                self._pin_counts.pop(key, None)
            else:
                self._pin_counts[key] = count - 1

    def filter_unpinned(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Return the subset of ``keys`` with no active L2 pin, in input order.

        Used by non-force delete to skip L2-pinned keys.
        """
        return [key for key in keys if key not in self._pin_counts]

    def drop_pins(self, keys: list[ObjectKey]) -> None:
        """Remove each key from the L2 pin set (used by force delete; idempotent)."""
        for key in keys:
            self._pin_counts.pop(key, None)

    def compute_eviction_plan(self) -> dict[str, list[ObjectKey]]:
        """Select eviction candidates per ``cache_salt``.

        Salts over ``watermark * quota`` get ``eviction_ratio`` of
        their LRU keys; a quota of ``0`` means full eviction. Salts
        without an explicit quota use the registry's default limit
        (``QuotaManager.effective_limit_bytes``): until the external
        quota controller sets one (``PUT /quota/config``), the
        coordinator will not start evicting unquota'd salts.
        """
        tracked_salts = self._policy.get_tracked_salts()
        eviction_plan: dict[str, list[ObjectKey]] = {}

        for cache_salt in tracked_salts:
            current_bytes = self._usage_manager.get(cache_salt)
            if current_bytes <= 0:
                continue
            limit = self._quota_manager.effective_limit_bytes(cache_salt)
            if limit is None:
                # No explicit quota and no default configured yet —
                # exempt until the quota controller arms enforcement.
                continue
            if current_bytes < self._trigger_watermark * limit:
                continue

            effective_ratio = 1.0 if limit == 0 else self._eviction_ratio
            actions = self._policy.get_eviction_actions(
                effective_ratio,
                cache_salt=cache_salt,
                key_eligible_filter=lambda key: key not in self._pin_counts,
            )
            keys_to_evict: list[ObjectKey] = []
            for action in actions:
                keys_to_evict.extend(action.keys)

            if keys_to_evict:
                eviction_plan[cache_salt] = keys_to_evict
                evict_bytes = sum(
                    self._usage_manager.get_key_size(k) for k in keys_to_evict
                )
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

    async def run(
        self,
        registry: InstanceRegistry,
        http_client: httpx.AsyncClient,
        check_interval: float,
    ) -> None:
        """Run the control loop until cancelled, sleeping first.

        Args:
            registry: Fleet membership; supplies the dispatch target.
            http_client: Client for the outbound DELETE requests.
            check_interval: Seconds between passes; must be positive.

        Raises:
            ValueError: If ``check_interval`` is not positive.
        """
        if check_interval <= 0:
            raise ValueError(f"check_interval must be > 0 (got {check_interval})")
        while True:
            await asyncio.sleep(check_interval)
            await self.execute_evictions(registry, http_client)

    async def execute_evictions(
        self,
        registry: InstanceRegistry,
        http_client: httpx.AsyncClient,
    ) -> dict[str, list[ObjectKey]]:
        """Compute the plan and fire-and-forget ``DELETE /cache/objects``
        to one random registered MP server.

        Keys are chunked at ``MAX_DELETE_BATCH`` because the MP endpoint
        rejects a larger single request with HTTP 400. Returns as soon as
        the dispatch tasks are spawned; the LRU clears only when the
        matching ``delete`` event comes back on the cache-event stream.
        At-least-once, safe because the delete is idempotent.
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

        url = f"http://{target.ip}:{target.http_port}/cache/objects"
        all_keys: list[ObjectKey] = [k for keys in plan.values() for k in keys]

        for start in range(0, len(all_keys), MAX_DELETE_BATCH):
            chunk = all_keys[start : start + MAX_DELETE_BATCH]
            body = {"keys": [asdict(k.to_encoded_object_key()) for k in chunk]}
            task = asyncio.create_task(
                self._dispatch_eviction(
                    http_client=http_client,
                    url=url,
                    body=body,
                    instance_id=target.instance_id,
                    key_count=len(chunk),
                    salt_count=len({k.cache_salt for k in chunk}),
                )
            )
            self._in_flight_dispatches.add(task)
            task.add_done_callback(self._in_flight_dispatches.discard)
        return plan

    async def wait_for_in_flight_dispatches(self) -> None:
        """Await every outstanding fire-and-forget dispatch."""
        await asyncio.gather(*self._in_flight_dispatches, return_exceptions=True)

    @staticmethod
    async def _dispatch_eviction(
        http_client: httpx.AsyncClient,
        url: str,
        body: dict,
        instance_id: str,
        key_count: int,
        salt_count: int,
    ) -> None:
        """Send the DELETE and log the outcome. Failures are not retried."""
        try:
            # ``httpx.AsyncClient.delete`` doesn't accept ``json=``;
            # ``request("DELETE", ...)`` is the supported form.
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
