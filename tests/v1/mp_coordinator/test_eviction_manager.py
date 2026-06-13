# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator eviction manager."""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.l2.eviction_manager import (
    L2EvictionManager,
)
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager


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


def test_on_store_tracks_key():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    ctrl.on_store(k, 100)
    ut.record_stored("a", 100)
    result = ctrl.execute_evictions()
    assert result["a"] == [k]


def test_on_lookup_touches_key():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    ctrl.on_store(k1, 100)
    ctrl.on_store(k2, 100)
    ctrl.on_lookup(k1)
    ut.record_stored("a", 200)
    result = ctrl.execute_evictions()
    assert result["a"][0] == k2


def test_on_lookup_unknown_key_is_noop():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    ctrl.on_lookup(k)
    ut.record_stored("a", 100)
    result = ctrl.execute_evictions()
    assert result == {}


def test_on_remove():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    ctrl.on_store(k1, 100)
    ctrl.on_store(k2, 200)
    ctrl.on_remove([k1])
    ut.record_stored("a", 200)
    result = ctrl.execute_evictions()
    assert result["a"] == [k2]


def test_on_remove_cleans_empty_bucket():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    ctrl.on_store(k, 100)
    ctrl.on_remove([k])
    ut.record_stored("a", 100)
    result = ctrl.execute_evictions()
    assert result == {}


def test_on_remove_empty_list_is_noop():
    ctrl, _, _ = _setup()
    ctrl.on_remove([])
    result = ctrl.execute_evictions()
    assert result == {}


def test_no_quotas_evicts_all():
    ctrl, _, ut = _setup(eviction_ratio=1.0)
    k = _make_key("a")
    ctrl.on_store(k, 1000)
    ut.record_stored("a", 1000)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert result["a"] == [k]


def test_under_quota():
    ctrl, qs, ut = _setup()
    qs.set_quota("a", 2000)
    ut.record_stored("a", 1000)
    ctrl.on_store(_make_key("a"), 1000)
    result = ctrl.execute_evictions()
    assert result == {}


def test_over_quota():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 500)
    ut.record_stored("a", 1000)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    ctrl.on_store(k1, 400)
    ctrl.on_store(k2, 600)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert k1 in result["a"]
    assert k2 in result["a"]


def test_eviction_ratio():
    ctrl, qs, ut = _setup(eviction_ratio=0.5)
    qs.set_quota("a", 500)
    ut.record_stored("a", 1000)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    ctrl.on_store(k1, 200)
    ctrl.on_store(k2, 800)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert len(result["a"]) == 1
    assert result["a"][0] == k1


def test_zero_quota_evicts_all():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 0)
    ut.record_stored("a", 1000)
    k = _make_key("a")
    ctrl.on_store(k, 1000)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert result["a"] == [k]


def test_multiple_salts_independent():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set_quota("a", 100)
    qs.set_quota("b", 5000)
    ut.record_stored("a", 500)
    ut.record_stored("b", 1000)
    ka = _make_key("a", h="01")
    kb = _make_key("b", h="02")
    ctrl.on_store(ka, 500)
    ctrl.on_store(kb, 1000)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert "b" not in result


def test_watermark_below_threshold_skips():
    ctrl, qs, ut = _setup(trigger_watermark=0.8)
    qs.set_quota("a", 1000)
    ut.record_stored("a", 700)
    ctrl.on_store(_make_key("a"), 700)
    result = ctrl.execute_evictions()
    assert result == {}


def test_watermark_above_threshold_evicts():
    ctrl, qs, ut = _setup(eviction_ratio=1.0, trigger_watermark=0.8)
    qs.set_quota("a", 1000)
    ut.record_stored("a", 900)
    k = _make_key("a")
    ctrl.on_store(k, 900)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert result["a"] == [k]
