# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator eviction controller."""

# Standard
import asyncio
import time

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.controllers.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance


def _make_key(salt: str, model: str = "m", rank: int = 0, h: str = "aa") -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes.fromhex(h),
        model_name=model,
        kv_rank=rank,
        cache_salt=salt,
    )


def _setup(
    eviction_ratio: float = 0.5,
    trigger_watermark: float = 1.0,
    default_limit_bytes: int | None = 0,
) -> tuple[FleetEvictionController, QuotaManager, L2UsageManager]:
    """Build the manager plus the quota registry and usage view it owns.

    ``default_limit_bytes=0`` (the helper default) arms strict allowlist
    enforcement — the steady state a quota controller configures via
    ``PUT /quota/config`` after re-syncing quotas — so most tests exercise
    armed behavior. Pass ``default_limit_bytes=None`` to exercise the
    exempt boot state instead.
    """
    ctrl = FleetEvictionController(
        eviction_ratio=eviction_ratio,
        trigger_watermark=trigger_watermark,
    )
    ctrl.quota.set_default_limit_bytes(default_limit_bytes)
    return ctrl, ctrl.quota, ctrl.usage


def _l2_batch(
    event_type: CacheEventType, key: ObjectKey, size: int = 0
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=1,
        event_type=event_type,
        tier=Tier.L2,
        backend="fs",
        entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size)],
    )


def _store(ctrl: FleetEvictionController, key: ObjectKey, size: int) -> None:
    """Helper: feed one store event in, exactly as the broadcaster does."""
    ctrl.consume(_l2_batch(CacheEventType.STORE, key, size))


def _remove(ctrl: FleetEvictionController, key: ObjectKey) -> None:
    """Helper: feed one delete event in, exactly as the broadcaster does."""
    ctrl.consume(_l2_batch(CacheEventType.DELETE, key))


def test_on_store_tracks_key():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 100)
    result = ctrl.compute_eviction_plan()
    assert result["a"] == [k]


def test_on_lookup_touches_key():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, k1, 100)
    _store(ctrl, k2, 100)
    ctrl.on_lookup(k1)
    result = ctrl.compute_eviction_plan()
    assert result["a"][0] == k2


def test_on_lookup_unknown_key_is_noop():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    ctrl.on_lookup(k)
    # Lookup without prior store ⇒ key never tracked. Add an
    # unrelated key so the salt has some usage, but the unknown key
    # mustn't show up in the plan.
    other = _make_key("a", h="ff")
    _store(ctrl, other, 100)
    result = ctrl.compute_eviction_plan()
    assert k not in result.get("a", [])


def test_on_remove():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, k1, 100)
    _store(ctrl, k2, 200)
    _remove(ctrl, k1)
    # _remove drops k1 from both the LRU and the directory's ledger.
    assert kd.get_key_size(k1) == 0
    assert kd.get_key_size(k2) > 0
    result = ctrl.compute_eviction_plan()
    assert result["a"] == [k2]


def test_on_remove_subtracts_bytes_from_usage():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, k1, 100)
    _store(ctrl, k2, 200)
    assert kd.get("a") == 300
    _remove(ctrl, k1)
    # Bucket loses exactly k1's bytes.
    assert kd.get("a") == 200


def test_on_remove_cleans_empty_bucket():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 100)
    _remove(ctrl, k)
    assert kd.get("a") == 0
    result = ctrl.compute_eviction_plan()
    assert result == {}


def test_no_quota_evicts_all_when_default_armed():
    """Armed default (0): a salt with no explicit quota is fully evicted."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 1000)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert result["a"] == [k]


# ============================================================================
# Default-limit gating (exempt boot state vs armed allowlist)
# ============================================================================


def test_unquotad_salt_exempt_until_default_set():
    """Boot state (default None): unquota'd salts are skipped entirely —
    a restarted coordinator with an empty quota table must not plan a
    mass eviction of unknown tenants."""
    ctrl, _, kd = _setup(eviction_ratio=1.0, default_limit_bytes=None)
    _store(ctrl, _make_key("a", h="01"), 1000)
    _store(ctrl, _make_key("b", h="02"), 2000)
    assert ctrl.compute_eviction_plan() == {}


def test_explicit_quota_enforced_even_while_default_unset():
    """Per-salt quotas take effect as soon as they are registered, even
    before the controller arms the default — over-quota tenants are
    evicted while unquota'd tenants stay exempt."""
    ctrl, qs, kd = _setup(eviction_ratio=1.0, default_limit_bytes=None)
    ka = _make_key("a", h="01")
    kb = _make_key("b", h="02")
    _store(ctrl, ka, 1000)
    _store(ctrl, kb, 1000)
    qs.set_quota("a", 500)  # over quota
    result = ctrl.compute_eviction_plan()
    assert result.get("a") == [ka]
    assert "b" not in result  # unquota'd, default unset ⇒ exempt


def test_setting_default_zero_arms_allowlist_eviction():
    """The controller's ``PUT /quota/config`` signal: flipping the default
    from None to 0 makes previously-exempt unquota'd bytes evictable."""
    ctrl, qs, kd = _setup(eviction_ratio=1.0, default_limit_bytes=None)
    k = _make_key("a")
    _store(ctrl, k, 1000)
    assert ctrl.compute_eviction_plan() == {}

    qs.set_default_limit_bytes(0)
    result = ctrl.compute_eviction_plan()
    assert result.get("a") == [k]


def test_positive_default_acts_as_budget_for_unquotad_salts():
    """A positive default gives unquota'd salts a real byte budget."""
    ctrl, qs, kd = _setup(eviction_ratio=1.0, default_limit_bytes=None)
    qs.set_default_limit_bytes(1500)
    under = _make_key("a", h="01")
    over = _make_key("b", h="02")
    _store(ctrl, under, 1000)  # under the 1500 default
    _store(ctrl, over, 2000)  # over it
    result = ctrl.compute_eviction_plan()
    assert "a" not in result
    assert result.get("b") == [over]


def test_under_quota():
    ctrl, qs, kd = _setup()
    qs.set_quota("a", 2000)
    _store(ctrl, _make_key("a"), 1000)
    result = ctrl.compute_eviction_plan()
    assert result == {}


def test_over_quota():
    ctrl, qs, kd = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 500)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, k1, 400)
    _store(ctrl, k2, 600)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert k1 in result["a"]
    assert k2 in result["a"]


def test_eviction_ratio():
    ctrl, qs, kd = _setup(eviction_ratio=0.5)
    qs.set_quota("a", 500)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, k1, 200)
    _store(ctrl, k2, 800)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert len(result["a"]) == 1
    assert result["a"][0] == k1


def test_zero_quota_evicts_all():
    ctrl, qs, kd = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 0)
    k = _make_key("a")
    _store(ctrl, k, 1000)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert result["a"] == [k]


def test_multiple_salts_independent():
    ctrl, qs, kd = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 100)
    qs.set_quota("b", 5000)
    ka = _make_key("a", h="01")
    kb = _make_key("b", h="02")
    _store(ctrl, ka, 500)
    _store(ctrl, kb, 1000)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert "b" not in result


def test_watermark_below_threshold_skips():
    ctrl, qs, kd = _setup(trigger_watermark=0.8)
    qs.set_quota("a", 1000)
    _store(ctrl, _make_key("a"), 700)
    result = ctrl.compute_eviction_plan()
    assert result == {}


def test_watermark_above_threshold_evicts():
    ctrl, qs, kd = _setup(eviction_ratio=1.0, trigger_watermark=0.8)
    qs.set_quota("a", 1000)
    k = _make_key("a")
    _store(ctrl, k, 900)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert result["a"] == [k]


# ============================================================================
# execute_evictions (async dispatch)
# ============================================================================


def _make_registry(*instances: MPInstance) -> InstanceRegistry:
    reg = InstanceRegistry()
    for inst in instances:
        reg.register(inst)
    return reg


def _instance(instance_id: str, ip: str = "10.0.0.1", port: int = 8000) -> MPInstance:
    now = time.time()
    return MPInstance(
        instance_id=instance_id,
        ip=ip,
        http_port=port,
        registration_time=now,
        last_heartbeat_time=now,
    )


@pytest.mark.asyncio
async def test_execute_evictions_dispatches_to_registered_instance():
    """Computed plan must DELETE /cache/objects to a registered MP server with
    the right body shape. The LRU is NOT cleared by ``execute_evictions``
    itself — that happens later via the coordinator's cache-event stream
    handler when the MP server reports the deletion back."""
    ctrl, qs, kd = _setup(eviction_ratio=1.0)
    k = _make_key("alice", h="aa")
    _store(ctrl, k, 100)
    qs.set_quota("alice", 0)  # ratio=1.0 → full eviction

    registry = _make_registry(_instance("mp-1", ip="10.0.0.7", port=8765))

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = (request.read() or b"").decode()
        return httpx.Response(200, json={"requested": 1, "adapter": "s3", "ok": True})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        plan = await ctrl.execute_evictions(registry, client)
        # Dispatch is fire-and-forget — wait for the background task
        # to actually issue the HTTP call before the client closes.
        await ctrl.wait_for_in_flight_dispatches()

    assert plan == {"alice": [k]}
    # Hit the single registered instance.
    assert captured["url"] == "http://10.0.0.7:8765/cache/objects"
    # Body shape matches the MP endpoint's contract.
    # Standard
    import json as _json

    body = _json.loads(captured["json"])
    assert body == {
        "keys": [
            {
                "chunk_hash_hex": "aa",
                "model_name": "m",
                "kv_rank": 0,
                "object_group_id": 0,
                "cache_salt": "alice",
            }
        ]
    }
    # LRU + usage are UNCHANGED at this point — the DELETE event hasn't
    # arrived yet. Cleanup happens once the MP server flushes its
    # ``on_l2_keys_deleted`` events back through the cache-event stream.
    assert ctrl.compute_eviction_plan() == {"alice": [k]}
    assert kd.get("alice") == 100


@pytest.mark.asyncio
async def test_execute_evictions_no_instances_skips_dispatch_and_keeps_lru():
    """No registered MP servers ⇒ the plan is logged but neither
    dispatched nor cleared from the LRU."""
    ctrl, qs, kd = _setup(eviction_ratio=1.0)
    k = _make_key("alice", h="bb")
    _store(ctrl, k, 100)
    qs.set_quota("alice", 0)

    registry = _make_registry()  # empty

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: pytest.fail("must not be called")  # type: ignore[arg-type]
        )
    ) as client:
        plan = await ctrl.execute_evictions(registry, client)
        await ctrl.wait_for_in_flight_dispatches()

    assert plan == {"alice": [k]}
    # LRU UNCHANGED — the same plan should re-emerge next cycle.
    assert ctrl.compute_eviction_plan() == {"alice": [k]}


@pytest.mark.asyncio
async def test_execute_evictions_http_failure_keeps_lru():
    """A non-2xx (or transport error) from the MP server must NOT
    clear the LRU — the next cycle should retry."""
    ctrl, qs, kd = _setup(eviction_ratio=1.0)
    k = _make_key("alice", h="cc")
    _store(ctrl, k, 100)
    qs.set_quota("alice", 0)

    registry = _make_registry(_instance("mp-1"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plan = await ctrl.execute_evictions(registry, client)
        await ctrl.wait_for_in_flight_dispatches()

    assert plan == {"alice": [k]}
    # LRU UNCHANGED — retry on the next cycle.
    assert ctrl.compute_eviction_plan() == {"alice": [k]}


@pytest.mark.asyncio
async def test_execute_evictions_chunks_large_plan(monkeypatch):
    """A plan larger than ``MAX_KEYS_PER_DELETE`` is split into multiple DELETE
    requests, each within the cap, together covering every key. Guards against
    the MP endpoint's per-request key limit (object_service.MAX_DELETE_BATCH),
    which rejects an oversized single request with HTTP 400."""
    # First Party
    import lmcache.v1.mp_coordinator.controllers.eviction_controller as em

    monkeypatch.setattr(em, "MAX_DELETE_BATCH", 2)

    ctrl, qs, kd = _setup(eviction_ratio=1.0)
    keys = [_make_key("alice", h=f"{i:02x}") for i in range(5)]
    for k in keys:
        _store(ctrl, k, 100)
    qs.set_quota("alice", 0)  # ratio=1.0 → full eviction of all 5 keys

    registry = _make_registry(_instance("mp-1"))

    # Standard
    import json as _json

    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads((request.read() or b"").decode())
        batch_sizes.append(len(body["keys"]))
        return httpx.Response(200, json={"requested": len(body["keys"]), "ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plan = await ctrl.execute_evictions(registry, client)
        await ctrl.wait_for_in_flight_dispatches()

    assert plan == {"alice": keys}
    # 5 keys, cap 2 → three requests of 2, 2, 1; none exceeds the cap.
    assert sum(batch_sizes) == 5
    assert all(n <= 2 for n in batch_sizes)
    assert sorted(batch_sizes, reverse=True) == [2, 2, 1]


@pytest.mark.asyncio
async def test_execute_evictions_empty_plan_is_noop():
    """No salts over threshold ⇒ no HTTP dispatch."""
    ctrl, _, _ = _setup()

    registry = _make_registry(_instance("mp-1"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: pytest.fail("must not be called")  # type: ignore[arg-type]
        )
    ) as client:
        plan = await ctrl.execute_evictions(registry, client)
        await ctrl.wait_for_in_flight_dispatches()

    assert plan == {}


# =============================================================================
# L2 pin / unpin (eviction exclusion)
# =============================================================================


def test_pin_excludes_key_from_eviction_plan():
    """A pinned key is never selected for eviction, even over-quota."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, k1, 100)
    _store(ctrl, k2, 100)

    ctrl.pin([k1])
    plan = ctrl.compute_eviction_plan()
    assert k1 not in plan.get("a", [])
    assert k2 in plan["a"]


def test_unpin_restores_eviction_eligibility():
    """After unpin, a previously pinned key can be evicted again."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 100)

    ctrl.pin([k])
    assert ctrl.compute_eviction_plan() == {}

    ctrl.unpin([k])
    assert ctrl.compute_eviction_plan()["a"] == [k]


def test_pin_is_reference_counted():
    """Two pins require two unpins before the key can be evicted."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 100)

    ctrl.pin([k])
    ctrl.pin([k])
    assert ctrl.compute_eviction_plan() == {}

    ctrl.unpin([k])
    assert ctrl.compute_eviction_plan() == {}  # still pinned once

    ctrl.unpin([k])
    assert ctrl.compute_eviction_plan()["a"] == [k]


def test_unpin_unknown_key_is_noop():
    """Unpinning a key that was never pinned does not go negative."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 100)

    ctrl.unpin([_make_key("a", h="ff")])  # never pinned
    assert ctrl.compute_eviction_plan()["a"] == [k]


# =============================================================================
# L2 delete helpers (filter_unpinned / drop_pins)
# =============================================================================


def test_filter_unpinned_returns_only_unpinned_keys():
    """filter_unpinned keeps unpinned keys and drops pinned ones, in order."""
    ctrl, _, _ = _setup()
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    k3 = _make_key("a", h="03")
    ctrl.pin([k2])

    assert ctrl.filter_unpinned([k1, k2, k3]) == [k1, k3]


def test_drop_pins_purges_pin_regardless_of_count():
    """drop_pins removes a key from the pin set even if pinned multiple times."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 100)
    ctrl.pin([k])
    ctrl.pin([k])  # pinned twice

    ctrl.drop_pins([k])
    # A single drop clears all pin counts: the key is evictable again.
    assert ctrl.compute_eviction_plan()["a"] == [k]
    assert ctrl.filter_unpinned([k]) == [k]


def test_drop_pins_unknown_key_is_noop():
    """drop_pins on a never-pinned key does not raise."""
    ctrl, _, _ = _setup()
    ctrl.drop_pins([_make_key("a", h="ff")])  # no error


# =============================================================================
# consume / fence_instance (CacheEventBroadcaster consumer hooks)
# =============================================================================


def test_consume_maps_l2_events_onto_the_lru():
    """STORE registers, ACCESS touches, DELETE drops."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    for k in (k1, k2):
        ctrl.consume(_l2_batch(CacheEventType.STORE, k, 100))
    ctrl.consume(_l2_batch(CacheEventType.ACCESS, k1))
    plan = ctrl.compute_eviction_plan()
    assert plan["a"][0] == k2  # k1 touched to MRU

    ctrl.consume(_l2_batch(CacheEventType.DELETE, k2))
    assert ctrl.compute_eviction_plan()["a"] == [k1]


def test_consume_keeps_key_in_lru_while_another_placement_remains():
    """Deleting one of a key's L2 placements must not drop it from the
    LRU: the remaining copy's bytes still count against quota, so the
    key must stay selectable by the planner."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")

    private = _l2_batch(CacheEventType.STORE, k, 100)
    shared = CacheEventBatch(
        instance_id="node-b",
        incarnation=1,
        seq=1,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="s3",
        shared=True,
        entries=[CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=100)],
    )
    for b in (private, shared):
        ctrl.consume(b)

    # Delete the private copy; the shared copy remains.
    ctrl.consume(_l2_batch(CacheEventType.DELETE, k))
    assert kd.get_key_size(k) == 100
    assert ctrl.compute_eviction_plan()["a"] == [k]  # still evictable

    # Delete the last placement: now the LRU lets go.
    delete_shared = CacheEventBatch(
        instance_id="node-b",
        incarnation=1,
        seq=2,
        event_type=CacheEventType.DELETE,
        tier=Tier.L2,
        backend="s3",
        shared=True,
        entries=[CacheEventEntry(key=k.to_encoded_object_key())],
    )
    ctrl.consume(delete_shared)
    assert kd.get_key_size(k) == 0
    assert ctrl.compute_eviction_plan() == {}


def test_consume_ignores_l1_batches():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    batch = CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=1,
        event_type=CacheEventType.STORE,
        tier=Tier.L1,
        backend="dram",
        entries=[CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=100)],
    )
    ctrl.consume(batch)
    assert ctrl.compute_eviction_plan() == {}


def test_consume_ignores_l1_batches_for_usage_too():
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    ctrl.consume(
        CacheEventBatch(
            instance_id="node-a",
            incarnation=1,
            seq=1,
            event_type=CacheEventType.STORE,
            tier=Tier.L1,
            backend="dram",
            entries=[CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=100)],
        )
    )
    assert kd.get("a") == 0


def test_consume_feeds_the_owned_usage_view():
    """One ``consume`` call updates both halves the manager owns."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")

    _store(ctrl, k, 100)

    assert kd.get("a") == 100
    assert ctrl.compute_eviction_plan()["a"] == [k]


def test_fence_instance_keeps_l2_usage_and_lru():
    """A reporter restart voids only L1 state; the L2 bytes it reported
    outlive the process, so neither the usage view nor the LRU forgets."""
    ctrl, _, kd = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, k, 100)

    ctrl.fence_instance("node-a")

    assert kd.get("a") == 100
    assert ctrl.compute_eviction_plan()["a"] == [k]


# ============================================================================
# run (the control loop)
# ============================================================================


@pytest.mark.asyncio
async def test_run_rejects_non_positive_check_interval():
    ctrl, _, _ = _setup()
    registry = _make_registry()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    ) as client:
        with pytest.raises(ValueError, match="check_interval"):
            await ctrl.run(registry, client, 0)
        with pytest.raises(ValueError, match="check_interval"):
            await ctrl.run(registry, client, -1.0)


@pytest.mark.asyncio
async def test_run_evicts_on_each_tick_until_cancelled():
    """The loop dispatches an over-quota salt's victims, and stops when the
    task is cancelled (how the app lifespan shuts it down)."""
    ctrl, qs, _ = _setup(eviction_ratio=1.0)
    k = _make_key("alice", h="aa")
    _store(ctrl, k, 100)
    qs.set_quota("alice", 0)  # ratio=1.0 → full eviction
    registry = _make_registry(_instance("mp-1"))

    dispatched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        dispatched.append(str(request.url))
        return httpx.Response(200, json={"requested": 1, "adapter": "s3", "ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(ctrl.run(registry, client, 0.01))
        # Long enough for several ticks, short enough to keep the test fast.
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await ctrl.wait_for_in_flight_dispatches()

    assert dispatched, "the loop never dispatched an eviction"
    assert dispatched[0] == "http://10.0.0.1:8000/cache/objects"


@pytest.mark.asyncio
async def test_run_sleeps_before_the_first_check():
    """Construction must not race the first pass: nothing is dispatched
    before one interval has elapsed."""
    ctrl, qs, _ = _setup(eviction_ratio=1.0)
    _store(ctrl, _make_key("alice", h="aa"), 100)
    qs.set_quota("alice", 0)
    registry = _make_registry(_instance("mp-1"))

    dispatched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        dispatched.append(str(request.url))
        return httpx.Response(200, json={"requested": 1, "adapter": "s3", "ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(ctrl.run(registry, client, 30.0))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert dispatched == []
