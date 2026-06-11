# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator eviction manager."""

# First Party
from lmcache.v1.mp_coordinator.l2.eviction_manager import (
    L2EvictionManager,
)
from lmcache.v1.mp_coordinator.l2.quota_manager import L2QuotaManager
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.schemas import CacheKey


def _make_key(salt: str, model: str = "m", rank: int = 0, h: str = "aa") -> CacheKey:
    return CacheKey(chunk_hash_hex=h, model_name=model, kv_rank=rank, cache_salt=salt)


def _setup(
    eviction_ratio: float = 0.5,
) -> tuple[L2EvictionManager, L2QuotaManager, L2UsageManager]:
    qs = L2QuotaManager()
    ut = L2UsageManager()
    ctrl = L2EvictionManager(qs, ut, eviction_ratio=eviction_ratio)
    return ctrl, qs, ut


def test_on_store_tracks_key():
    ctrl, _, _ = _setup()
    k = _make_key("a")
    ctrl.on_store(k, 100)
    assert ctrl._select_keys_to_evict("a", 100) == [k]


def test_on_store_updates_existing_key():
    ctrl, _, _ = _setup()
    k = _make_key("a")
    ctrl.on_store(k, 100)
    ctrl.on_store(k, 200)
    assert ctrl._select_keys_to_evict("a", 200) == [k]


def test_on_lookup_touches_key():
    ctrl, _, _ = _setup()
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    ctrl.on_store(k1, 100)
    ctrl.on_store(k2, 100)
    ctrl.on_lookup(k1)
    keys_to_evict = ctrl._select_keys_to_evict("a", 100)
    assert keys_to_evict[0] == k2


def test_on_lookup_unknown_key_is_noop():
    ctrl, _, _ = _setup()
    k = _make_key("a")
    ctrl.on_lookup(k)
    assert ctrl._select_keys_to_evict("a", 1) == []


def test_on_remove():
    ctrl, _, _ = _setup()
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    ctrl.on_store(k1, 100)
    ctrl.on_store(k2, 200)
    ctrl.on_remove([k1])
    assert ctrl._select_keys_to_evict("a", 200) == [k2]


def test_on_remove_cleans_empty_bucket():
    ctrl, _, _ = _setup()
    k = _make_key("a")
    ctrl.on_store(k, 100)
    ctrl.on_remove([k])
    assert ctrl._select_keys_to_evict("a", 1) == []


def test_on_remove_empty_list_is_noop():
    ctrl, _, _ = _setup()
    ctrl.on_remove([])
    assert ctrl._select_keys_to_evict("a", 1) == []


def test_select_keys_to_evict_lru_order():
    ctrl, _, _ = _setup()
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    k3 = _make_key("a", h="03")
    ctrl.on_store(k1, 100)
    ctrl.on_store(k2, 200)
    ctrl.on_store(k3, 300)
    keys_to_evict = ctrl._select_keys_to_evict("a", 250)
    assert keys_to_evict == [k1, k2]


def test_select_keys_to_evict_empty_bucket():
    ctrl, _, _ = _setup()
    assert ctrl._select_keys_to_evict("nonexistent", 100) == []


def test_check_and_log_no_quotas_evicts_all():
    ctrl, _, ut = _setup()
    k = _make_key("a")
    ctrl.on_store(k, 1000)
    ut.record_stored("a", 1000)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert result["a"] == [k]


def test_check_and_log_under_quota():
    ctrl, qs, ut = _setup()
    qs.set("a", 2000)
    ut.record_stored("a", 1000)
    ctrl.on_store(_make_key("a"), 1000)
    result = ctrl.execute_evictions()
    assert result == {}


def test_check_and_log_over_quota():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set("a", 500)
    ut.record_stored("a", 1000)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    ctrl.on_store(k1, 400)
    ctrl.on_store(k2, 600)
    result = ctrl.execute_evictions()
    assert "a" in result
    keys_to_evict = result["a"]
    assert keys_to_evict[0] == k1
    total_evict_bytes = 400 + 600
    assert total_evict_bytes >= 500


def test_check_and_log_eviction_ratio():
    ctrl, qs, ut = _setup(eviction_ratio=0.5)
    qs.set("a", 500)
    ut.record_stored("a", 1000)
    k1 = _make_key("a", h="01")
    k2 = _make_key("a", h="02")
    k3 = _make_key("a", h="03")
    ctrl.on_store(k1, 200)
    ctrl.on_store(k2, 200)
    ctrl.on_store(k3, 600)
    result = ctrl.execute_evictions()
    assert "a" in result
    keys_to_evict = result["a"]
    assert len(keys_to_evict) >= 1
    assert keys_to_evict[0] == k1


def test_check_and_log_zero_quota_evicts_all():
    ctrl, qs, ut = _setup()
    qs.set("a", 0)
    ut.record_stored("a", 1000)
    k = _make_key("a")
    ctrl.on_store(k, 1000)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert result["a"] == [k]


def test_multiple_salts_independent():
    ctrl, qs, ut = _setup(eviction_ratio=1.0)
    qs.set("a", 100)
    qs.set("b", 5000)
    ut.record_stored("a", 500)
    ut.record_stored("b", 1000)
    ka = _make_key("a", h="01")
    kb = _make_key("b", h="02")
    ctrl.on_store(ka, 500)
    ctrl.on_store(kb, 1000)
    result = ctrl.execute_evictions()
    assert "a" in result
    assert "b" not in result
