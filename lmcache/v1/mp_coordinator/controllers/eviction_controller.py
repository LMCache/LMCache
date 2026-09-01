# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide per-``cache_salt`` L2 eviction control loop.

Dispatch is placement-aware: a shared pool is reachable through any
registered server, a local one only through its owner.

See ``docs/design/v1/mp_coordinator/usage_and_eviction.md``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast
import asyncio

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import EncodedObjectKey, Tier
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType
from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager
from lmcache.v1.multiprocess.cache_control.object_service import (
    MAX_DELETE_BATCH,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry
    from lmcache.v1.mp_coordinator.registry import InstanceRegistry
    from lmcache.v1.mp_coordinator.views.base import View

logger = init_logger(__name__)


@dataclass(frozen=True)
class _DeleteRoute:
    """Where one live copy of a victim key has to be deleted.

    Attributes:
        instance_id: MP server owning the copy; ``""`` means any registered
            one will do, resolved against the registry at dispatch time
            rather than at plan time.
        tier: Tier the copy lives on, sent as the request's ``tier``.
        backend: Backend within that tier holding the copy, sent as the
            request's ``adapter``; ``""`` selects the target's default.
    """

    instance_id: str
    tier: Tier
    backend: str


@dataclass(frozen=True)
class _EvictionPlan:
    """What one cycle decided to evict, and where each victim lives.

    Attributes:
        victims: Keys to evict, per ``cache_salt``.
        routes: Every victim's live copies, keyed by victim. A victim
            always has at least one — a key with nowhere to delete it is
            not selected.
    """

    victims: dict[str, list[ObjectKey]]
    routes: dict[ObjectKey, list[_DeleteRoute]]

    def keys_by_route(self) -> dict[_DeleteRoute, list[ObjectKey]]:
        """Group the victims by the route their delete takes, so each
        route's keys travel in one request."""
        grouped: dict[_DeleteRoute, list[ObjectKey]] = {}
        for keys in self.victims.values():
            for key in keys:
                for route in self.routes[key]:
                    grouped.setdefault(route, []).append(key)
        return grouped


class FleetEvictionController(Controller):
    """Per-``cache_salt`` L2 eviction controller for the fleet.

    Owns the quota registry it enforces (exposed for the ``/quota``
    endpoints) and reads the fleet usage view on the ``l2`` tier.
    :meth:`run` is the loop, :meth:`execute_evictions` one pass of it.

    Args:
        usage_manager: The fleet usage view. A consumer in its own
            right, registered on the broadcaster **before** this
            controller so it has accounted a batch by the time
            :meth:`consume` reads sizes from it.
        key_directory: The fleet key directory view; supplies each
            victim's L2 placements, which decide where its delete goes.
        eviction_ratio: Fraction of tracked keys to evict per cycle.
        trigger_watermark: Eviction fires when usage reaches this
            fraction of the quota.
    """

    def __init__(
        self,
        usage_manager: CacheUsageManager,
        key_directory: KeyDirectory,
        eviction_ratio: float = 0.5,
        trigger_watermark: float = 1.0,
    ) -> None:
        self._quota_manager = QuotaManager()
        self._usage_manager = usage_manager
        self._key_directory = key_directory
        self._eviction_ratio = max(0.0, min(1.0, eviction_ratio))
        self._trigger_watermark = trigger_watermark
        self._policy = IsolatedLRUEvictionPolicy()
        self._in_flight_dispatches: set[asyncio.Task] = set()
        self._pin_counts: dict[ObjectKey, int] = {}

    @property
    def quota(self) -> QuotaManager:
        """The budgets this controller enforces."""
        return self._quota_manager

    @property
    def policy(self) -> EvictionPolicy:
        """The eviction policy, persisted as a durable component.

        What it stores is the policy's business: an ordering-based policy
        carries its order, one that derives everything from the keys
        carries nothing.
        """
        return self._policy

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one gate-admitted batch to the LRU.

        A delete drops the key from the LRU only once its **last** L2
        placement is gone: usage is per placement, so while another copy
        still holds bytes the key must stay evictable, or those bytes
        could exceed quota with nothing for the planner to select. That
        size read is correct because the usage view consumed the same
        batch first, which registration order in ``create_app``
        guarantees.

        Args:
            batch: The admitted batch; other tiers are ignored.
        """
        if batch.tier != Tier.L2:
            return
        for entry in batch.entries:
            key = entry.key.to_object_key()
            if batch.event_type == CacheEventType.STORE:
                self.on_store(key)
            elif batch.event_type == CacheEventType.ACCESS:
                self.on_lookup(key)
            elif batch.event_type == CacheEventType.DELETE:
                if self._usage_manager.get_key_bytes(Tier.L2, key) == 0:
                    self.on_remove(key)

    def fence_instance(self, instance_id: str) -> None:
        """No-op: fencing voids L1 only, and this controller's LRU
        tracks L2 keys, which outlive the reporting process.

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

    @classmethod
    def from_config(
        cls,
        config: "MPCoordinatorConfig",
        views: "Registry[View]",
        controllers: "Registry[Controller]",
    ) -> "FleetEvictionController":
        """Build the controller from configuration and the fleet's views.

        The views come from the registry rather than being made here:
        the coordinator has exactly one of each, and the eviction plan is
        only correct if it reads the same bytes and placements the fleet
        reported.

        Args:
            config: The coordinator configuration.
            views: The fleet's read models.
            controllers: Unused; this controller depends on no peer.
        """
        return cls(
            usage_manager=views.get(CacheUsageManager),
            key_directory=views.get(KeyDirectory),
            eviction_ratio=config.eviction_ratio,
            trigger_watermark=config.trigger_watermark,
        )

    def get_durable_components(self) -> tuple[DurableComponent, ...]:
        """Return the state this controller owns that must outlive the process.

        Each carries its own ``persistence_type``, so a caller routes
        them without knowing what this controller is made of.

        Returns:
            The pin table (this object), the quota limits, and the policy.
        """
        return (self, self._quota_manager, self._policy)

    @property
    def persistence_type(self) -> PersistenceType:
        """Pins are operator intent; nothing else can reconstruct them."""
        return PersistenceType.METADATA

    @property
    def name(self) -> str:
        """Name of the pin table's section in the metadata document."""
        return "pins"

    def capture(self) -> Mapping[str, object]:
        """Return the pin table in its durable form.

        Returns:
            ``{"entries": [{"key": <EncodedObjectKey fields>, "count":
            int}]}``.
        """
        return {
            "entries": [
                {"key": asdict(key.to_encoded_object_key()), "count": count}
                for key, count in self._pin_counts.items()
            ]
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Replace the pin table with a captured one.

        The document is the coordinator's own, so values are taken as
        written.

        Args:
            state: A :meth:`capture` value, as decoded from the
                metadata document — see there for the shape; non-positive
                counts are dropped.
        """
        entries = cast("list[Mapping[str, object]]", state["entries"])
        restored = (_decode_pin(entry) for entry in entries)
        self._pin_counts = {key: count for key, count in restored if count > 0}

    def filter_unpinned(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Return the subset of ``keys`` with no active L2 pin, in input order.

        Used by non-force delete to skip L2-pinned keys.
        """
        return [key for key in keys if key not in self._pin_counts]

    def drop_pins(self, keys: list[ObjectKey]) -> None:
        """Remove each key from the L2 pin set (used by force delete; idempotent)."""
        for key in keys:
            self._pin_counts.pop(key, None)

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
        """Compute the plan and fire-and-forget the ``DELETE /cache/objects``
        requests that carry it out.

        One request per ``(MP server, tier, backend)`` holding a victim, so a
        key stored privately on three instances costs three deletes. Keys are
        chunked at ``MAX_DELETE_BATCH`` because the MP endpoint rejects a
        larger single request with HTTP 400.

        Args:
            registry: Fleet membership; resolves a route to a live address.
            http_client: Client for the outbound DELETE requests.

        Returns:
            The plan, as soon as the dispatch tasks are spawned; the LRU
            clears only when the matching ``delete`` event comes back on the
            cache-event stream. At-least-once, safe because the delete is
            idempotent.
        """
        plan = self._compute_eviction_plan(registry)
        if not plan.victims:
            return plan.victims

        for route, keys in plan.keys_by_route().items():
            target = (
                registry.get(route.instance_id)
                if route.instance_id
                else registry.random_instance()
            )
            if target is None:
                logger.warning(
                    "Eviction plan holds %d keys on %s; skipping dispatch",
                    len(keys),
                    f"{route.instance_id}, which left the fleet mid-cycle"
                    if route.instance_id
                    else "a shared pool, but no MP server is registered",
                )
                continue
            url = f"http://{target.ip}:{target.http_port}/cache/objects"
            for start in range(0, len(keys), MAX_DELETE_BATCH):
                chunk = keys[start : start + MAX_DELETE_BATCH]
                body: dict[str, object] = {
                    "keys": [asdict(k.to_encoded_object_key()) for k in chunk],
                    # Stated rather than left to the endpoint's default, so
                    # the request says which tier it is deleting from.
                    "tier": route.tier.value,
                }
                if route.backend:
                    body["adapter"] = route.backend
                task = asyncio.create_task(
                    self._dispatch_eviction(
                        http_client=http_client,
                        url=url,
                        body=body,
                        instance_id=target.instance_id,
                        tier=route.tier,
                        backend=route.backend,
                        key_count=len(chunk),
                        salt_count=len({k.cache_salt for k in chunk}),
                    )
                )
                self._in_flight_dispatches.add(task)
                task.add_done_callback(self._in_flight_dispatches.discard)
        return plan.victims

    async def wait_for_in_flight_dispatches(self) -> None:
        """Await every outstanding fire-and-forget dispatch."""
        await asyncio.gather(*self._in_flight_dispatches, return_exceptions=True)

    # -- Internals ------------------------------------------------------------

    def _compute_eviction_plan(self, registry: InstanceRegistry) -> _EvictionPlan:
        """Plan this cycle's eviction: what to drop, and where from.

        A salt fires at ``watermark * quota`` and gives up ``eviction_ratio``
        of its LRU keys (quota ``0`` ⇒ all of them); one without an explicit
        quota falls back to the default limit, and is exempt while that is
        unset. Pinned keys and keys no live server can delete are skipped.

        Args:
            registry: Fleet membership; decides which copies are reachable.
        """
        tracked_salts = self._policy.get_tracked_salts()
        eviction_plan: dict[str, list[ObjectKey]] = {}
        # Routing is a by-product of the eligibility check, which has to
        # resolve it anyway; only the selected keys' routes are kept.
        resolved: dict[ObjectKey, list[_DeleteRoute]] = {}

        for cache_salt in tracked_salts:
            current_bytes = self._usage_manager.get_salt_bytes(Tier.L2, cache_salt)
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
                key_eligible_filter=lambda key: self._is_evictable(
                    key, registry, Tier.L2, resolved
                ),
            )
            keys_to_evict: list[ObjectKey] = []
            for action in actions:
                keys_to_evict.extend(action.keys)

            if keys_to_evict:
                eviction_plan[cache_salt] = keys_to_evict
                evict_bytes = sum(
                    self._usage_manager.get_key_bytes(Tier.L2, k) for k in keys_to_evict
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

        return _EvictionPlan(
            victims=eviction_plan,
            routes={
                key: resolved[key] for keys in eviction_plan.values() for key in keys
            },
        )

    def _is_evictable(
        self,
        key: ObjectKey,
        registry: InstanceRegistry,
        tier: Tier,
        routes: dict[ObjectKey, list[_DeleteRoute]],
    ) -> bool:
        """Whether ``key`` may be selected as a victim this cycle.

        True when it carries no pin and at least one ``tier`` copy is
        reachable. ``routes`` memoizes the resolution for the caller.

        Runs under the LRU policy's lock and takes the key directory's, an
        order nothing reverses — the directory never reaches into eviction.
        """
        if key in self._pin_counts:
            return False
        key_routes = routes.get(key)
        if key_routes is None:
            key_routes = self._resolve_routes(key, registry, tier)
            routes[key] = key_routes
        return bool(key_routes)

    def _resolve_routes(
        self, key: ObjectKey, registry: InstanceRegistry, tier: Tier
    ) -> list[_DeleteRoute]:
        """Resolve where every live ``tier`` copy of ``key`` must be deleted.

        A shared placement is one pool the whole fleet mounts, so any
        registered server deletes it — preferably its reporter, which
        certainly mounts that backend. A private placement is the reporter's
        own storage: only that instance can delete it, and once it leaves the
        fleet the copy is unreachable.

        Args:
            key: The victim to route.
            registry: Fleet membership; decides which owners are live.
            tier: The tier being evicted; placements on others are ignored.

        Returns:
            One route per deletable copy, deduplicated; one best-effort route
            to any registered server when the directory knows no placement on
            ``tier``, so a directory blind spot cannot stall eviction; empty
            when every known copy is stranded on a departed instance.
        """
        placements = [p for p in self._key_directory.lookup([key])[0] if p.tier == tier]
        if not placements:
            return [_DeleteRoute(instance_id="", tier=tier, backend="")]
        routes: list[_DeleteRoute] = []
        for placement in placements:
            if placement.shared:
                instance_id = (
                    placement.instance_id
                    if registry.contains(placement.instance_id)
                    else ""
                )
            elif registry.contains(placement.instance_id):
                instance_id = placement.instance_id
            else:
                continue
            route = _DeleteRoute(
                instance_id=instance_id, tier=tier, backend=placement.backend
            )
            if route not in routes:
                routes.append(route)
        return routes

    @staticmethod
    async def _dispatch_eviction(
        http_client: httpx.AsyncClient,
        url: str,
        body: dict[str, object],
        instance_id: str,
        tier: Tier,
        backend: str,
        key_count: int,
        salt_count: int,
    ) -> None:
        """Send the DELETE and log the outcome. Failures are not retried."""
        target = f"{instance_id}/{tier.value}/{backend or 'default backend'}"
        try:
            # ``httpx.AsyncClient.delete`` doesn't accept ``json=``;
            # ``request("DELETE", ...)`` is the supported form.
            resp = await http_client.request("DELETE", url, json=body)
            resp.raise_for_status()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "Eviction dispatch to %s (%d keys) failed: %s",
                target,
                key_count,
                e,
            )
            return
        logger.info(
            "Eviction dispatched to %s: %d keys across %d salts",
            target,
            key_count,
            salt_count,
        )


def _decode_pin(entry: Mapping[str, object]) -> tuple[ObjectKey, int]:
    """Rebuild one pin from the form :meth:`capture` produced.

    Args:
        entry: One ``entries`` element of the pin section.

    Returns:
        The key and its pin count.
    """
    fields = cast("Mapping[str, object]", entry["key"])
    encoded = EncodedObjectKey(
        chunk_hash_hex=str(fields["chunk_hash_hex"]),
        model_name=str(fields["model_name"]),
        kv_rank=cast(int, fields["kv_rank"]),
        object_group_id=cast(int, fields["object_group_id"]),
        cache_salt=str(fields["cache_salt"]),
    )
    return encoded.to_object_key(), cast(int, entry["count"])
