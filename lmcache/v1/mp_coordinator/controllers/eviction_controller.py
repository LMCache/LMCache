# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide per-``cache_salt`` eviction, on both cache tiers.

One controller, because per-tenant quota enforcement is one job: a budget
per salt, the bytes spent against it, and an LRU that picks victims when
a salt is over. That machine is identical for L1 and L2 — only two
questions have different answers, and both follow from **where the bytes
live**:

============  ==================================  =========================
Question      L2 (shared storage)                 L1 (a node's own memory)
============  ==================================  =========================
Who deletes?  Any member; one request evicts      Only the node holding the
              the fleet.                          key, so one per holder.
Restart?      Nothing; the bytes outlive the      Its bytes died with it;
              reporter.                           drop what it last held.
============  ==================================  =========================

Everything else — the quota table, the ordering, the sweep, the chunked
dispatch — is shared, and the tier is a parameter rather than a subclass.
Each tier keeps its own quota registry and its own LRU, so a key resident
in both is budgeted twice and ordered separately.

Lifetime comes from :class:`Controller`: :meth:`run` is entered once by
the app's lifespan, so nothing here is started or named by ``create_app``.

See ``docs/design/v1/mp_coordinator/usage_and_eviction.md``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, cast
import asyncio
import contextlib

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
from lmcache.v1.mp_coordinator.controllers.base import (
    Controller,
    ControllerRuntime,
)
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.utils.encoding import decode_key, encode_key
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
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
    from lmcache.v1.mp_coordinator.views.base import View
    from lmcache.v1.mp_coordinator.views.instance_registry import MPInstance

logger = init_logger(__name__)

# The tiers this controller budgets, in sweep order. ``all`` is not one of
# them: it names both, and a key resident in both holds bytes in both.
ENFORCED_TIERS = (Tier.L1, Tier.L2)

# Each tier's durable section names. Distinct per tier because a document
# cannot hold two sections under one name.
_QUOTA_SECTIONS = {Tier.L1: "l1_quotas", Tier.L2: "quotas"}
_LRU_SECTIONS = {Tier.L1: "l1_lru_order", Tier.L2: "lru_order"}


@dataclass(frozen=True)
class EvictionDispatch:
    """One outbound delete: the victims, and the server that holds them.

    Attributes:
        instance: The MP server to send the delete to.
        keys: The keys it should drop, in eviction order. Chunked at
            ``MAX_DELETE_BATCH`` by the caller, so this may exceed one
            request's worth.
    """

    instance: MPInstance
    keys: list[ObjectKey] = field(default_factory=list)


class FleetEvictionController(Controller):
    """Per-``cache_salt`` quota enforcement for the fleet, on both tiers.

    Owns the quota registries these budgets come from (exposed for the
    ``/quota`` endpoints) and the per-tier LRUs that order their keys;
    reads the bytes it compares against off the shared usage view, the
    holders of an L1 key off the key directory, and addresses off the
    instance registry.

    Args:
        usage_manager: The fleet usage view. A consumer in its own
            right, registered on the broadcaster **before** this
            controller so it has accounted a batch by the time
            :meth:`consume` reads sizes from it.
        key_directory: The fleet placement view, which resolves an L1
            victim to the nodes holding it.
        registry: Fleet membership; turns a holder into an address.
        eviction_ratio: Fraction of a salt's tracked keys to evict per
            cycle, clamped to ``[0.0, 1.0]``.
        trigger_watermark: Eviction fires when usage reaches this
            fraction of the quota.
        check_interval: Seconds between sweeps; ``0`` runs no loop.
    """

    def __init__(
        self,
        usage_manager: CacheUsageManager,
        key_directory: KeyDirectory,
        registry: InstanceRegistry,
        eviction_ratio: float = 0.5,
        trigger_watermark: float = 1.0,
        check_interval: float = 0.0,
    ) -> None:
        self._usage_manager = usage_manager
        self._key_directory = key_directory
        self._registry = registry
        self._eviction_ratio = max(0.0, min(1.0, eviction_ratio))
        self._trigger_watermark = trigger_watermark
        self._check_interval = check_interval
        self._quotas = {
            tier: QuotaManager(section_name=_QUOTA_SECTIONS[tier])
            for tier in ENFORCED_TIERS
        }
        self._policies = {
            tier: IsolatedLRUEvictionPolicy(section_name=_LRU_SECTIONS[tier])
            for tier in ENFORCED_TIERS
        }
        # L2 only: operator pins, which hold a key back from selection.
        # L1's equivalent is the node's own read/write lock, checked at
        # delete time, so there is nothing to track here for it.
        self._pin_counts: dict[ObjectKey, int] = {}
        # L1 only: who holds what, so a fence knows which keys just lost
        # their last copy.
        self._l1_owners = _L1PlacementOwners()
        self._in_flight_dispatches: set[asyncio.Task] = set()

    def quota(self, tier: Tier) -> QuotaManager:
        """Return the budgets enforced on ``tier``.

        Args:
            tier: The tier to read. Each has its own registry: the two
                govern different bytes, so one's numbers are meaningless
                for the other.

        Returns:
            That tier's quota registry.

        Raises:
            KeyError: If ``tier`` is not one this controller enforces.
        """
        return self._quotas[tier]

    def policy(self, tier: Tier) -> EvictionPolicy:
        """Return the eviction ordering for ``tier``.

        Args:
            tier: The tier to read.

        Returns:
            That tier's policy, persisted as a durable component.

        Raises:
            KeyError: If ``tier`` is not one this controller enforces.
        """
        return self._policies[tier]

    # -- Event stream ---------------------------------------------------------

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one gate-admitted batch to its tier's LRU.

        A delete drops the key from the LRU only once its **last**
        placement on that tier is gone: usage is per placement, so while
        another copy still holds bytes the key must stay evictable, or
        those bytes could exceed quota with nothing for the planner to
        select. That size read is correct because the usage view consumed
        the same batch first, which registration order in ``create_app``
        guarantees.

        Args:
            batch: The admitted batch; tiers with no budget are ignored.
        """
        if batch.tier not in self._policies:
            return
        # A shared L1 pool is owned by the fleet, not by the member that
        # reported it, so no one reporter's fencing may drop it.
        owner = "" if batch.shared else batch.instance_id
        track_owner = batch.tier == Tier.L1 and bool(owner)
        for entry in batch.entries:
            key = entry.key.to_object_key()
            if batch.event_type == CacheEventType.STORE:
                self.on_store(batch.tier, key)
                if track_owner:
                    self._l1_owners.add(owner, key, batch.backend)
            elif batch.event_type == CacheEventType.ACCESS:
                self.on_lookup(batch.tier, key)
            elif batch.event_type == CacheEventType.DELETE:
                if track_owner:
                    self._l1_owners.discard(owner, key, batch.backend)
                if self._usage_manager.get_key_bytes(batch.tier, key) == 0:
                    self.on_remove(batch.tier, key)

    def fence_instance(self, instance_id: str) -> None:
        """Drop every key ``instance_id`` was the last L1 holder of.

        Its L1 bytes were that process's memory and died with it. Its L2
        keys stay: they live on storage the fleet shares and leave only
        via ``DELETE`` events. A key another node still holds stays too —
        the usage view has already subtracted the fenced placements by
        the time this runs, so a remaining byte count is a remaining copy.

        Args:
            instance_id: The instance whose reported L1 state is void.
        """
        for key, _ in self._l1_owners.pop(instance_id):
            if self._usage_manager.get_key_bytes(Tier.L1, key) == 0:
                self.on_remove(Tier.L1, key)

    def on_store(self, tier: Tier, key: ObjectKey) -> None:
        """Register a stored key in ``tier``'s LRU (bytes go to the usage view)."""
        self._policies[tier].on_keys_created([key])

    def on_lookup(self, tier: Tier, key: ObjectKey) -> None:
        """Touch ``key`` in ``tier``'s LRU (move to MRU end)."""
        self._policies[tier].on_keys_touched([key])

    def on_remove(self, tier: Tier, key: ObjectKey) -> None:
        """Drop ``key`` from ``tier``'s LRU (bytes go to the usage view)."""
        self._policies[tier].on_keys_removed([key])

    # -- Pins (L2) ------------------------------------------------------------

    def pin(self, keys: list[ObjectKey]) -> None:
        """Increment each key's L2 pin count, excluding it from eviction."""
        for key in keys:
            self._pin_counts[key] = self._pin_counts.get(key, 0) + 1

    def unpin(self, keys: list[ObjectKey]) -> None:
        """Decrement each key's L2 pin count, floored at 0."""
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

    def is_evictable(self, tier: Tier, key: ObjectKey) -> bool:
        """Whether ``key`` may be selected as a victim on ``tier``.

        Args:
            tier: The tier being planned.
            key: The candidate.

        Returns:
            ``True`` unless an operator holds an L2 pin on it. L1 has no
            pins: a node refuses a locked object at delete time instead,
            which no coordinator-side table could anticipate.
        """
        if tier == Tier.L2:
            return key not in self._pin_counts
        return True

    # -- Planning -------------------------------------------------------------

    def compute_eviction_plan(self, tier: Tier) -> dict[str, list[ObjectKey]]:
        """Select ``tier``'s eviction candidates per ``cache_salt``.

        Salts over ``watermark * quota`` get ``eviction_ratio`` of their
        LRU keys; a quota of ``0`` means full eviction. Salts without an
        explicit quota use that tier's default limit
        (``QuotaManager.effective_limit_bytes``): until the external
        quota controller sets one (``PUT /quota/config``), the
        coordinator will not start evicting unquota'd salts.

        Args:
            tier: The tier to plan for.

        Returns:
            Victims per salt, in eviction order. Pure — no network, no
            mutation; salts under their budget are absent.

        Raises:
            KeyError: If ``tier`` is not one this controller enforces.
        """
        policy = self._policies[tier]
        quota = self._quotas[tier]
        eviction_plan: dict[str, list[ObjectKey]] = {}

        for cache_salt in policy.get_tracked_salts():
            current_bytes = self._usage_manager.get_salt_bytes(tier, cache_salt)
            if current_bytes <= 0:
                continue
            limit = quota.effective_limit_bytes(cache_salt)
            if limit is None:
                # No explicit quota and no default configured yet —
                # exempt until the quota controller arms enforcement.
                continue
            if current_bytes < self._trigger_watermark * limit:
                continue

            effective_ratio = 1.0 if limit == 0 else self._eviction_ratio
            actions = policy.get_eviction_actions(
                effective_ratio,
                cache_salt=cache_salt,
                key_eligible_filter=lambda key: self.is_evictable(tier, key),
            )
            keys_to_evict: list[ObjectKey] = []
            for action in actions:
                keys_to_evict.extend(action.keys)

            if keys_to_evict:
                eviction_plan[cache_salt] = keys_to_evict
                evict_bytes = sum(
                    self._usage_manager.get_key_bytes(tier, k) for k in keys_to_evict
                )
                logger.info(
                    "%s eviction plan for cache_salt=%r: %d keys "
                    "(%d bytes) to free; usage=%d, quota=%d, "
                    "watermark=%.2f, ratio=%.2f",
                    tier.value,
                    cache_salt,
                    len(keys_to_evict),
                    evict_bytes,
                    current_bytes,
                    limit,
                    self._trigger_watermark,
                    effective_ratio,
                )

        return eviction_plan

    def plan_dispatches(
        self, tier: Tier, plan: dict[str, list[ObjectKey]]
    ) -> list[EvictionDispatch]:
        """Route ``plan``'s victims to the servers that can delete them.

        The one place the tiers diverge. L2 bytes sit on storage the
        fleet shares, so one uniformly chosen member evicts for everyone.
        L1 bytes are one node's memory, so each victim is resolved
        through the key directory and sent to every node holding it — a
        key with copies on several nodes is deleted on all of them,
        because every copy spends against the tenant's budget.

        Args:
            tier: The tier being dispatched.
            plan: The victims per salt.

        Returns:
            One entry per target server, empty when none can be reached.
            For L1, victims the directory has no placement for, and
            holders no longer registered, are skipped and logged — the
            next delete event repairs the disagreement.
        """
        victims = [key for keys in plan.values() for key in keys]
        if tier == Tier.L2:
            target = self._registry.random_instance()
            if target is None:
                return []
            return [EvictionDispatch(instance=target, keys=victims)]

        keys_by_instance: dict[str, list[ObjectKey]] = {}
        unplaced = 0
        placements_per_key = self._key_directory.lookup(victims)
        for key, placements in zip(victims, placements_per_key, strict=True):
            holders = {p.instance_id for p in placements if p.tier == Tier.L1}
            if not holders:
                unplaced += 1
                continue
            for instance_id in holders:
                keys_by_instance.setdefault(instance_id, []).append(key)
        if unplaced:
            logger.warning(
                "Skipping %d L1 eviction victim(s) with no placement in the "
                "key directory",
                unplaced,
            )

        dispatches: list[EvictionDispatch] = []
        for instance_id, keys in keys_by_instance.items():
            instance = self._registry.get(instance_id)
            if instance is None:
                logger.warning(
                    "Skipping %d L1 eviction victim(s) held by unregistered "
                    "instance %s",
                    len(keys),
                    instance_id,
                )
                continue
            dispatches.append(EvictionDispatch(instance=instance, keys=keys))
        return dispatches

    # -- Lifetime -------------------------------------------------------------

    @asynccontextmanager
    async def run(self, runtime: ControllerRuntime) -> AsyncIterator[None]:
        """Sweep on a cadence while the app serves, then drain.

        A dispatch is fire-and-forget, so one the last sweep launched
        would otherwise die with the process; the exit half waits for it.

        Args:
            runtime: Supplies the client for the outbound DELETEs.
        """
        task: asyncio.Task | None = None
        if self._check_interval > 0:
            task = asyncio.create_task(self._sweep_forever(runtime.http_client))
        else:
            logger.debug("Eviction loop disabled (check_interval=0)")
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self.wait_for_in_flight_dispatches()

    async def execute_evictions(
        self,
        http_client: httpx.AsyncClient,
    ) -> dict[Tier, dict[str, list[ObjectKey]]]:
        """Run one pass over every enforced tier.

        Each tier's plan is routed through :meth:`plan_dispatches` and
        **fire-and-forget** ``DELETE /cache/objects`` sent to each target;
        the LRU clears only when the matching delete event comes back on
        the cache-event stream. At-least-once, safe because the delete is
        idempotent. Keys are chunked at ``MAX_DELETE_BATCH`` because the
        MP endpoint rejects a larger single request with HTTP 400.

        Args:
            http_client: Client for the outbound DELETE requests.

        Returns:
            The computed plan per tier, tiers with nothing over quota
            omitted. A present plan does **not** mean anything was
            dispatched: it is returned unchanged when no target could be
            reached, which is logged rather than signalled here.
        """
        plans: dict[Tier, dict[str, list[ObjectKey]]] = {}
        for tier in ENFORCED_TIERS:
            plan = self.compute_eviction_plan(tier)
            if not plan:
                continue
            plans[tier] = plan
            dispatches = self.plan_dispatches(tier, plan)
            if not dispatches:
                logger.warning(
                    "%s eviction plan computed (%d salts) but no MP server "
                    "holding the victims is registered; skipping dispatch",
                    tier.value,
                    len(plan),
                )
                continue
            self._dispatch_all(http_client, tier, dispatches)
        return plans

    async def wait_for_in_flight_dispatches(self) -> None:
        """Await every outstanding fire-and-forget dispatch."""
        await asyncio.gather(*self._in_flight_dispatches, return_exceptions=True)

    # -- Construction and durability ------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "MPCoordinatorConfig",
        views: "Registry[View]",
    ) -> "FleetEvictionController":
        """Build the controller from configuration and the fleet's views.

        Every view comes from the registry rather than being made here:
        the coordinator has exactly one of each, and a plan is only
        correct if it reads the bytes and placements the fleet reported.

        Args:
            config: The coordinator configuration.
            views: The fleet's read models.
        """
        return cls(
            usage_manager=views.get(CacheUsageManager),
            key_directory=views.get(KeyDirectory),
            registry=views.get(InstanceRegistry),
            eviction_ratio=config.eviction_ratio,
            trigger_watermark=config.trigger_watermark,
            check_interval=config.eviction_check_interval,
        )

    def get_durable_components(self) -> tuple[DurableComponent, ...]:
        """Return the state this controller owns that must outlive the process.

        Each carries its own ``persistence_type``, so a caller routes
        them without knowing what this controller is made of.

        Returns:
            The pin table (this object), the L1 owner index, and both
            tiers' quota limits and orderings.
        """
        return (
            self,
            self._l1_owners,
            *self._quotas.values(),
            *self._policies.values(),
        )

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

    # -- Internals ----------------------------------------------------------------

    def _dispatch_all(
        self,
        http_client: httpx.AsyncClient,
        tier: Tier,
        dispatches: list[EvictionDispatch],
    ) -> None:
        """Spawn one fire-and-forget DELETE per chunk of each dispatch.

        Args:
            http_client: Client for the outbound requests.
            tier: The tier being evicted.
            dispatches: Targets and their victims.
        """
        for dispatch in dispatches:
            instance = dispatch.instance
            url = f"http://{instance.ip}:{instance.http_port}/cache/objects"
            for start in range(0, len(dispatch.keys), MAX_DELETE_BATCH):
                chunk = dispatch.keys[start : start + MAX_DELETE_BATCH]
                body: dict[str, object] = {
                    "keys": [asdict(k.to_encoded_object_key()) for k in chunk],
                    "tier": tier.value,
                    # Locked L1 objects are in use; skipping them costs one
                    # cycle, force-dropping them corrupts a live transfer.
                    "force": False,
                }
                task = asyncio.create_task(
                    self._dispatch_eviction(
                        http_client=http_client,
                        url=url,
                        body=body,
                        tier=tier,
                        instance_id=instance.instance_id,
                        key_count=len(chunk),
                        salt_count=len({k.cache_salt for k in chunk}),
                    )
                )
                self._in_flight_dispatches.add(task)
                task.add_done_callback(self._in_flight_dispatches.discard)

    async def _sweep_forever(self, http_client: httpx.AsyncClient) -> None:
        """Evict on a cadence until cancelled, sleeping first."""
        while True:
            await asyncio.sleep(self._check_interval)
            await self.execute_evictions(http_client)

    @staticmethod
    async def _dispatch_eviction(
        http_client: httpx.AsyncClient,
        url: str,
        body: Mapping[str, object],
        tier: Tier,
        instance_id: str,
        key_count: int,
        salt_count: int,
    ) -> None:
        """Send the DELETE and log the outcome. Failures are not retried.

        Args:
            http_client: Client for the request.
            url: The target server's ``/cache/objects`` endpoint.
            body: The MP server's ``DeleteObjectsRequest`` shape --
                ``{"keys": [<EncodedObjectKey fields>], "tier": str,
                "force": bool}``.
            tier: The tier being evicted; for the log line only.
            instance_id: The target, for the log line only.
            key_count: Keys in this chunk, for the log line only.
            salt_count: Distinct salts in this chunk, for the log line only.
        """
        try:
            # ``httpx.AsyncClient.delete`` doesn't accept ``json=``;
            # ``request("DELETE", ...)`` is the supported form.
            resp = await http_client.request("DELETE", url, json=body)
            resp.raise_for_status()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "%s eviction dispatch to %s (%d keys) failed: %s",
                tier.value,
                instance_id,
                key_count,
                e,
            )
            return
        logger.info(
            "%s eviction dispatched to %s: %d keys across %d salts",
            tier.value,
            instance_id,
            key_count,
            salt_count,
        )


class _L1PlacementOwners:
    """Which instance holds which L1 placement, for fencing.

    Mirrors the usage view's own index rather than reading it: by the
    time the controller is fenced the view has already dropped the
    instance, which is exactly what makes the byte reads that follow say
    "gone". Kept per ``(key, backend)`` so a key on two backends of one
    node survives losing one of them.

    Checkpointed, because it is the only thing that can answer "was that
    the last copy?" after a restart. A restored ordering without it would
    keep the keys of any instance that restarted afterwards, and a plan
    built from those would dispatch deletes that free nothing.
    """

    def __init__(self) -> None:
        self._by_instance: dict[str, set[tuple[ObjectKey, str]]] = {}

    def add(self, owner: str, key: ObjectKey, backend: str) -> None:
        """Record that ``owner`` holds ``key`` on ``backend``."""
        self._by_instance.setdefault(owner, set()).add((key, backend))

    def discard(self, owner: str, key: ObjectKey, backend: str) -> None:
        """Forget one placement; a no-op if it was not recorded."""
        held = self._by_instance.get(owner)
        if held is None:
            return
        held.discard((key, backend))
        if not held:
            del self._by_instance[owner]

    def pop(self, owner: str) -> set[tuple[ObjectKey, str]]:
        """Remove and return everything ``owner`` held."""
        return self._by_instance.pop(owner, set())

    @property
    def name(self) -> str:
        """Name of this index's section in the checkpoint."""
        return "l1_placement_owners"

    @property
    def persistence_type(self) -> PersistenceType:
        """Derived from the event stream, so it rides with the checkpoint
        that carries the ordering it answers for."""
        return PersistenceType.CHECKPOINT

    def capture(self) -> Mapping[str, object]:
        """Return the index in durable form.

        Returns:
            ``{"placements": [(instance_id, key, backend), ...]}``.
        """
        return {
            "placements": [
                (instance_id, encode_key(key), backend)
                for instance_id, held in self._by_instance.items()
                for key, backend in held
            ]
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Replace the index with a captured one.

        Call once at startup.

        Args:
            state: A :meth:`capture` value, as decoded from the
                checkpoint holding it — see there for the shape.

        Note:
            A wholesale replacement, like the ordering it rides beside:
            restoring twice cannot double-count, so it carries no
            load-once guard of its own.
        """
        placements = cast("list[tuple[str, object, str]]", state["placements"])
        restored: dict[str, set[tuple[ObjectKey, str]]] = {}
        for instance_id, encoded_key, backend in placements:
            restored.setdefault(instance_id, set()).add(
                (decode_key(encoded_key), backend)
            )
        self._by_instance = restored


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
