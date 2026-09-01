# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's fleet L1 eviction controller.

The quota → usage → evict selection is shared with L2 and covered in
``test_eviction_controller.py``. What is exercised here is what L1 does
differently, and both halves follow from L1 bytes being one node's
process memory: victims are routed to the nodes the key directory says
hold them, and a fenced reporter's keys leave the ordering.
"""

# Standard
import json
import time

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    PersistenceType,
)
from lmcache.v1.mp_coordinator.views.instance_registry import (
    InstanceRegistry,
    MPInstance,
)
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager
import lmcache.v1.mp_coordinator.controllers.eviction_controller as eviction_controller


def _make_key(salt: str, h: str = "aa", model: str = "m", rank: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes.fromhex(h),
        model_name=model,
        kv_rank=rank,
        cache_salt=salt,
    )


class _Fleet:
    """The three collaborators wired the way ``create_app`` wires them.

    Batches go to the views first and the controller second, which is
    the order the broadcaster uses and what makes the controller's byte
    reads see the batch it is consuming.
    """

    def __init__(
        self,
        eviction_ratio: float = 1.0,
        default_limit_bytes: int | None = 0,
        instances: tuple[MPInstance, ...] = (),
    ) -> None:
        self.usage = CacheUsageManager()
        self.directory = KeyDirectory()
        self.registry = _registry(*instances)
        self.controller = FleetEvictionController(
            usage_manager=self.usage,
            key_directory=self.directory,
            registry=self.registry,
            eviction_ratio=eviction_ratio,
        )
        self.controller.quota(Tier.L1).set_default_limit_bytes(default_limit_bytes)
        self._seq = 0

    def emit(
        self,
        event_type: CacheEventType,
        key: ObjectKey,
        instance_id: str,
        size: int = 0,
        tier: Tier = Tier.L1,
        backend: str = "dram",
        shared: bool = False,
    ) -> None:
        """Broadcast one single-entry batch to every consumer."""
        self._seq += 1
        batch = CacheEventBatch(
            instance_id=instance_id,
            incarnation=1,
            seq=self._seq,
            event_type=event_type,
            tier=tier,
            backend=backend,
            entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size)],
            shared=shared,
        )
        self.usage.consume(batch)
        self.directory.consume(batch)
        self.controller.consume(batch)

    def store(self, key: ObjectKey, instance_id: str, size: int, **kw) -> None:
        self.emit(CacheEventType.STORE, key, instance_id, size, **kw)

    def delete(self, key: ObjectKey, instance_id: str, **kw) -> None:
        self.emit(CacheEventType.DELETE, key, instance_id, **kw)

    def fence(self, instance_id: str) -> None:
        """Fence one instance across every consumer, views first."""
        self.usage.fence_instance(instance_id)
        self.directory.fence_instance(instance_id)
        self.controller.fence_instance(instance_id)


def _instance(instance_id: str, ip: str = "10.0.0.1", port: int = 8000) -> MPInstance:
    now = time.time()
    return MPInstance(
        instance_id=instance_id,
        ip=ip,
        http_port=port,
        registration_time=now,
        last_heartbeat_time=now,
    )


def _registry(*instances: MPInstance) -> InstanceRegistry:
    registry = InstanceRegistry()
    for instance in instances:
        registry.register(instance)
    return registry


# -- Selection ---------------------------------------------------------------


def test_l1_stores_are_tracked_and_l2_stores_are_not():
    """The controller enforces one tier; L2 bytes are another
    controller's budget and must not appear in this plan."""
    fleet = _Fleet(
        instances=(
            _instance("mp-1", ip="10.0.0.1"),
            _instance("mp-2", ip="10.0.0.2", port=8765),
        )
    )
    l1_key = _make_key("alice", h="01")
    l2_key = _make_key("alice", h="02")
    fleet.store(l1_key, "mp-1", 100)
    fleet.store(l2_key, "mp-1", 100, tier=Tier.L2, backend="fs")

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [l1_key]}


def test_over_quota_salt_is_selected_in_lru_order():
    fleet = _Fleet(eviction_ratio=0.5)
    old = _make_key("alice", h="01")
    new = _make_key("alice", h="02")
    fleet.store(old, "mp-1", 100)
    fleet.store(new, "mp-1", 100)
    fleet.controller.quota(Tier.L1).set_quota("alice", 150)

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [old]}


def test_under_quota_salt_is_left_alone():
    fleet = _Fleet(instances=(_instance("mp-1", ip="10.0.0.1"),))
    fleet.store(_make_key("alice"), "mp-1", 100)
    fleet.controller.quota(Tier.L1).set_quota("alice", 1000)

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {}


def test_quota_is_the_fleet_wide_total_not_one_node():
    """Two nodes each holding half a tenant's bytes are over a budget
    neither exceeds alone — the reason this lives on the coordinator."""
    fleet = _Fleet()
    on_a = _make_key("alice", h="01")
    on_b = _make_key("alice", h="02")
    fleet.store(on_a, "mp-1", 100)
    fleet.store(on_b, "mp-2", 100)
    fleet.controller.quota(Tier.L1).set_quota("alice", 150)

    assert set(fleet.controller.compute_eviction_plan(Tier.L1)["alice"]) == {on_a, on_b}


def test_l1_quota_is_independent_of_the_l2_registry():
    """Separate registries: an L1 salt with no entry falls to the L1
    default, whatever L2 was told."""
    fleet = _Fleet(default_limit_bytes=None)
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100)

    # Exempt while the L1 default is unset, even though bytes exist.
    assert fleet.controller.compute_eviction_plan(Tier.L1) == {}

    fleet.controller.quota(Tier.L1).set_default_limit_bytes(0)
    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [key]}


# -- Fencing -----------------------------------------------------------------


def test_fencing_drops_the_keys_only_that_node_held():
    """A restart voids L1 memory. Keys whose last copy was on the fenced
    node leave the ordering, so the planner cannot pick a ghost that no
    delete would ever free."""
    fleet = _Fleet()
    gone = _make_key("alice", h="01")
    survives = _make_key("alice", h="02")
    fleet.store(gone, "mp-1", 100)
    fleet.store(survives, "mp-2", 100)

    fleet.fence("mp-1")

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [survives]}


def test_fencing_keeps_a_key_another_node_still_holds():
    fleet = _Fleet()
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100)
    fleet.store(key, "mp-2", 100)

    fleet.fence("mp-1")

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [key]}


def test_fencing_spares_a_shared_l1_pool():
    """A shared pool outlives any one member that reported it, so no
    reporter's fencing may drop it."""
    fleet = _Fleet()
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100, shared=True, backend="pool")

    fleet.fence("mp-1")

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [key]}


def test_a_deleted_key_is_not_resurrected_by_a_later_fence():
    """The owner index tracks placements, not history: deleting the only
    copy and then fencing its holder must not touch anything."""
    fleet = _Fleet()
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100)
    fleet.delete(key, "mp-1")

    fleet.fence("mp-1")

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {}


# -- Dispatch ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_goes_to_the_node_that_holds_the_key():
    """Only the holder can free L1 bytes, so the delete is addressed to
    it rather than to any registered member."""
    fleet = _Fleet(
        instances=(
            _instance("mp-1", ip="10.0.0.1"),
            _instance("mp-2", ip="10.0.0.2", port=8765),
        )
    )
    key = _make_key("alice")
    fleet.store(key, "mp-2", 100)

    captured: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((str(request.url), json.loads(request.read() or b"{}")))
        return httpx.Response(200, json={"deleted": 1, "skipped": 0, "ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plan = await fleet.controller.execute_evictions(client)
        await fleet.controller.wait_for_in_flight_dispatches()

    assert plan == {Tier.L1: {"alice": [key]}}
    assert len(captured) == 1
    url, body = captured[0]
    assert url == "http://10.0.0.2:8765/cache/objects"
    assert body == {
        "keys": [
            {
                "chunk_hash_hex": "aa",
                "model_name": "m",
                "kv_rank": 0,
                "object_group_id": 0,
                "cache_salt": "alice",
            }
        ],
        "tier": "l1",
        # Locked objects are in use; skipping them costs one cycle,
        # force-dropping them corrupts a live transfer.
        "force": False,
    }


@pytest.mark.asyncio
async def test_dispatch_reaches_every_holder_of_a_replicated_key():
    """Every copy spends against the tenant's budget, so every copy is
    deleted -- one request per holding node."""
    fleet = _Fleet(
        instances=(_instance("mp-1", ip="10.0.0.1"), _instance("mp-2", ip="10.0.0.2"))
    )
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100)
    fleet.store(key, "mp-2", 100)

    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"deleted": 1, "skipped": 0, "ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fleet.controller.execute_evictions(client)
        await fleet.controller.wait_for_in_flight_dispatches()

    assert sorted(urls) == [
        "http://10.0.0.1:8000/cache/objects",
        "http://10.0.0.2:8000/cache/objects",
    ]


@pytest.mark.asyncio
async def test_dispatch_skips_a_holder_that_is_no_longer_registered():
    """A departed node's bytes are already gone; addressing it would
    only fail. The registered holder is still served."""
    # ``mp-gone`` is deliberately absent from the registry.
    fleet = _Fleet(instances=(_instance("mp-1"),))
    held_by_both = _make_key("alice", h="01")
    fleet.store(held_by_both, "mp-1", 100)
    fleet.store(held_by_both, "mp-gone", 100)

    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"deleted": 1, "skipped": 0, "ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fleet.controller.execute_evictions(client)
        await fleet.controller.wait_for_in_flight_dispatches()

    assert urls == ["http://10.0.0.1:8000/cache/objects"]


@pytest.mark.asyncio
async def test_dispatch_is_skipped_when_no_holder_is_registered():
    """Nothing to send to, and the plan is kept so the next cycle retries."""
    fleet = _Fleet(instances=(_instance("mp-1"),))
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: pytest.fail("must not be called")  # type: ignore[arg-type]
        )
    ) as client:
        plan = await fleet.controller.execute_evictions(client)
        await fleet.controller.wait_for_in_flight_dispatches()

    assert plan == {Tier.L1: {"alice": [key]}}
    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [key]}


@pytest.mark.asyncio
async def test_dispatch_does_not_clear_the_lru_until_the_delete_event():
    """At-least-once: the ordering and the bytes clear only when the MP
    server reports the deletion back on the cache-event stream."""
    fleet = _Fleet()
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"deleted": 1, "skipped": 0, "ok": True}
            )
        )
    ) as client:
        await fleet.controller.execute_evictions(client)
        await fleet.controller.wait_for_in_flight_dispatches()

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [key]}

    fleet.delete(key, "mp-1")

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {}
    assert fleet.usage.get_salt_bytes(Tier.L1, "alice") == 0


@pytest.mark.asyncio
async def test_dispatch_chunks_a_plan_larger_than_the_endpoint_cap(monkeypatch):
    """The MP endpoint rejects an oversized single delete with HTTP 400,
    so one holder's victims are split across requests."""
    monkeypatch.setattr(eviction_controller, "MAX_DELETE_BATCH", 2)

    fleet = _Fleet(instances=(_instance("mp-1"),))
    keys = [_make_key("alice", h=f"{i:02x}") for i in range(5)]
    for key in keys:
        fleet.store(key, "mp-1", 100)

    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read() or b"{}")
        batch_sizes.append(len(body["keys"]))
        return httpx.Response(200, json={"deleted": len(body["keys"]), "ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fleet.controller.execute_evictions(client)
        await fleet.controller.wait_for_in_flight_dispatches()

    assert sorted(batch_sizes, reverse=True) == [2, 2, 1]


@pytest.mark.asyncio
async def test_http_failure_keeps_the_plan_for_the_next_cycle():
    fleet = _Fleet(instances=(_instance("mp-1"),))
    key = _make_key("alice")
    fleet.store(key, "mp-1", 100)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "boom"})
        )
    ) as client:
        await fleet.controller.execute_evictions(client)
        await fleet.controller.wait_for_in_flight_dispatches()

    assert fleet.controller.compute_eviction_plan(Tier.L1) == {"alice": [key]}


# -- Durability --------------------------------------------------------------


def test_each_tier_gets_its_own_durable_sections():
    """Both tiers persist a quota table and an LRU ordering; a shared
    section name would make one silently overwrite the other."""
    fleet = _Fleet()

    names = [c.name for c in fleet.controller.get_durable_components()]

    assert len(names) == len(set(names)), f"duplicate section name in {names}"
    assert {"quotas", "l1_quotas", "lru_order", "l1_lru_order"} <= set(names)


def test_a_restored_coordinator_can_still_fence():
    """The owner index rides in the checkpoint beside the ordering it
    answers for. Without it, a node that restarts after the coordinator
    did would leave its keys in the LRU with no bytes behind them, and
    every later plan would dispatch deletes that free nothing."""
    before = _Fleet()
    key = _make_key("alice")
    before.store(key, "mp-1", 100)

    after = _Fleet()
    _carry_checkpoint(before, after)
    after.usage.restore(before.usage.capture())
    assert after.controller.compute_eviction_plan(Tier.L1) == {"alice": [key]}

    # The restored coordinator still knows mp-1 held the only copy.
    after.fence("mp-1")

    assert after.controller.compute_eviction_plan(Tier.L1) == {}


def _carry_checkpoint(src: _Fleet, dst: _Fleet) -> None:
    """Move ``src``'s checkpointed controller state into ``dst``.

    Matches sections by name, the way a real checkpoint round-trip does.
    """
    by_name = {c.name: c for c in dst.controller.get_durable_components()}
    for component in src.controller.get_durable_components():
        if component.persistence_type is PersistenceType.CHECKPOINT:
            by_name[component.name].restore(component.capture())


def test_quota_limits_survive_a_capture_and_restore():
    fleet = _Fleet()
    fleet.controller.quota(Tier.L1).set_quota("alice", 4096)
    captured = fleet.controller.quota(Tier.L1).capture()

    restored = _Fleet()
    restored.controller.quota(Tier.L1).restore(captured)

    assert restored.controller.quota(Tier.L1).get_limit_bytes("alice") == 4096
