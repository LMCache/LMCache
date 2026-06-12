# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator eviction manager."""

# Standard
import time

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.l2.eviction_manager import (
    L2EvictionManager,
)
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager
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
) -> tuple[L2EvictionManager, QuotaManager, L2UsageManager]:
    qs = QuotaManager()
    ut = L2UsageManager()
    ctrl = L2EvictionManager(
        qs,
        ut,
        eviction_ratio=eviction_ratio,
        trigger_watermark=trigger_watermark,
    )
    return ctrl, qs, ut


def _store(
    ctrl: L2EvictionManager,
    ut: L2UsageManager,
    key: ObjectKey,
    size: int,
) -> None:
    """Helper: record the bytes against the usage ledger and register
    the key in the eviction LRU — the two calls that the production
    ``/l2/events`` handler issues for a single store event."""
    ut.record_stored(key, size)
    ctrl.on_store(key)


def test_on_store_tracks_key():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, ut, k, 100)
    result = ctrl.compute_eviction_plan()
    assert result["a"] == [k]


def test_on_lookup_touches_key():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, ut, k1, 100)
    _store(ctrl, ut, k2, 100)
    ctrl.on_lookup(k1)
    result = ctrl.compute_eviction_plan()
    assert result["a"][0] == k2


def test_on_lookup_unknown_key_is_noop():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    ctrl.on_lookup(k)
    # Lookup without prior store ⇒ key never tracked. Add an
    # unrelated key so the salt has some usage, but the unknown key
    # mustn't show up in the plan.
    other = _make_key("a", h="ff")
    _store(ctrl, ut, other, 100)
    result = ctrl.compute_eviction_plan()
    assert k not in result.get("a", [])


def test_on_remove():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, ut, k1, 100)
    _store(ctrl, ut, k2, 200)
    ctrl.on_remove([k1])
    # on_remove drops k1 from both LRU and the size ledger.
    assert not ut.has_key(k1)
    assert ut.has_key(k2)
    result = ctrl.compute_eviction_plan()
    assert result["a"] == [k2]


def test_on_remove_subtracts_bytes_from_usage():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, ut, k1, 100)
    _store(ctrl, ut, k2, 200)
    assert ut.get("a") == 300
    ctrl.on_remove([k1])
    # Bucket loses exactly k1's bytes.
    assert ut.get("a") == 200


def test_on_remove_cleans_empty_bucket():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, ut, k, 100)
    ctrl.on_remove([k])
    assert ut.get("a") == 0
    result = ctrl.compute_eviction_plan()
    assert result == {}


def test_on_remove_empty_list_is_noop():
    ctrl, _, _ = _setup()
    ctrl.on_remove([])
    result = ctrl.compute_eviction_plan()
    assert result == {}


def test_no_quotas_evicts_all():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    _store(ctrl, ut, k, 1000)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert result["a"] == [k]


def test_under_quota():
    ctrl, qs, ut = _setup()
    qs.set_quota("a", 2000)
    _store(ctrl, ut, _make_key("a"), 1000)
    result = ctrl.compute_eviction_plan()
    assert result == {}


def test_over_quota():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 500)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, ut, k1, 400)
    _store(ctrl, ut, k2, 600)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert k1 in result["a"]
    assert k2 in result["a"]


def test_eviction_ratio():
    ctrl, qs, ut = _setup(eviction_ratio=0.5)
    qs.set_quota("a", 500)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    _store(ctrl, ut, k1, 200)
    _store(ctrl, ut, k2, 800)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert len(result["a"]) == 1
    assert result["a"][0] == k1


def test_zero_quota_evicts_all():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 0)
    k = _make_key("a")
    _store(ctrl, ut, k, 1000)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert result["a"] == [k]


def test_multiple_salts_independent():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 100)
    qs.set_quota("b", 5000)
    ka = _make_key("a", h="01")
    kb = _make_key("b", h="02")
    _store(ctrl, ut, ka, 500)
    _store(ctrl, ut, kb, 1000)
    result = ctrl.compute_eviction_plan()
    assert "a" in result
    assert "b" not in result


def test_watermark_below_threshold_skips():
    ctrl, qs, ut = _setup(trigger_watermark=0.8)
    qs.set_quota("a", 1000)
    _store(ctrl, ut, _make_key("a"), 700)
    result = ctrl.compute_eviction_plan()
    assert result == {}


def test_watermark_above_threshold_evicts():
    ctrl, qs, ut = _setup(eviction_ratio=1.0, trigger_watermark=0.8)
    qs.set_quota("a", 1000)
    k = _make_key("a")
    _store(ctrl, ut, k, 900)
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
async def test_execute_evictions_dispatches_to_first_registered_instance():
    """Computed plan must POST to one registered MP server's
    ``/l2/keys:evict`` and clear those keys from the LRU on success."""
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    k = _make_key("alice", h="aa")
    _store(ctrl, ut, k, 100)
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

    assert plan == {"alice": [k]}
    # Picked first (only) instance — http://<ip>:<port>/l2/keys:evict.
    assert captured["url"] == "http://10.0.0.7:8765/l2/keys:evict"
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
    # LRU cleared and the usage manager subtracted the freed bytes.
    assert ctrl.compute_eviction_plan() == {}
    assert ut.get("alice") == 0


@pytest.mark.asyncio
async def test_execute_evictions_no_instances_skips_dispatch_and_keeps_lru():
    """No registered MP servers ⇒ the plan is logged but neither
    dispatched nor cleared from the LRU."""
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    k = _make_key("alice", h="bb")
    _store(ctrl, ut, k, 100)
    qs.set_quota("alice", 0)

    registry = _make_registry()  # empty

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: pytest.fail("must not be called")  # type: ignore[arg-type]
        )
    ) as client:
        plan = await ctrl.execute_evictions(registry, client)

    assert plan == {"alice": [k]}
    # LRU UNCHANGED — the same plan should re-emerge next cycle.
    assert ctrl.compute_eviction_plan() == {"alice": [k]}


@pytest.mark.asyncio
async def test_execute_evictions_http_failure_keeps_lru():
    """A non-2xx (or transport error) from the MP server must NOT
    clear the LRU — the next cycle should retry."""
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    k = _make_key("alice", h="cc")
    _store(ctrl, ut, k, 100)
    qs.set_quota("alice", 0)

    registry = _make_registry(_instance("mp-1"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plan = await ctrl.execute_evictions(registry, client)

    assert plan == {"alice": [k]}
    # LRU UNCHANGED — retry on the next cycle.
    assert ctrl.compute_eviction_plan() == {"alice": [k]}


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

    assert plan == {}
