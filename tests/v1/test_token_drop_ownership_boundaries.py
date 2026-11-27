# SPDX-License-Identifier: Apache-2.0
"""Ownership contracts at the TD integration boundaries."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import gc
import weakref

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    TokenDropQueryHooks,
)
from lmcache.integration.vllm.experimental.token_drop_kv import (
    KVReadView,
    PagedKVSequenceView,
)
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    ResidentKVState,
    TokenDropScheduler,
    bind_resident_kv_state,
    get_resident_kv_state,
)
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPWorkerMetadata,
)


def test_hook_lifecycle_needs_only_layer_names_and_callback(monkeypatch):
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter._check_vllm_version",
        lambda: None,
    )
    original = lambda *args, **kwargs: args[1]
    layer = SimpleNamespace(layer_name="a", impl=SimpleNamespace(forward=original))
    received = {}
    hooks = TokenDropQueryHooks()
    hooks.install(["a"], {"a": layer}, received.__setitem__)
    q = torch.arange(6).reshape(3, 2)
    try:
        assert (
            layer.impl.forward(
                layer, q, q, q, None, SimpleNamespace(num_actual_tokens=2)
            )
            is q
        )
        assert torch.equal(received["a"], q[:2])
    finally:
        hooks.remove()
    assert layer.impl.forward is original


class _Pool:
    pass


def test_resident_state_binding_isolated_and_idempotent():
    a, b, unbound = _Pool(), _Pool(), _Pool()
    first, second = ResidentKVState(), ResidentKVState()
    first.set("same", 16)
    second.set("same", 32)
    bind_resident_kv_state(a, first)
    bind_resident_kv_state(a, first)
    bind_resident_kv_state(b, second)
    assert get_resident_kv_state(a).get("same") == 16
    assert get_resident_kv_state(b).get("same") == 32
    assert get_resident_kv_state(unbound) is None
    with pytest.raises(RuntimeError, match="already bound"):
        bind_resident_kv_state(a, second)
    with pytest.raises(RuntimeError, match="already bound"):
        bind_resident_kv_state(unbound, first)
    first.clear("same")
    assert get_resident_kv_state(a).get("same") is None
    assert get_resident_kv_state(b).get("same") == 32


def test_resident_binding_does_not_keep_pool_or_state_alive():
    pool = _Pool()
    state = ResidentKVState()
    bind_resident_kv_state(pool, state)
    pool_ref, state_ref = weakref.ref(pool), weakref.ref(state)
    del state, pool
    gc.collect()
    assert pool_ref() is None
    assert state_ref() is None


def test_scheduler_generations_and_finish_state_are_instance_local():
    a, b = TokenDropScheduler(), TokenDropScheduler()
    pool_a, pool_b = _Pool(), _Pool()
    a.bind_block_pool(pool_a)
    b.bind_block_pool(pool_b)
    assert a.next_generation() == 1
    assert a.next_generation() == 2
    assert b.next_generation() == 1
    get_resident_kv_state(pool_a).set("same", 16)
    get_resident_kv_state(pool_b).set("same", 32)
    a.finish_request("same")
    assert get_resident_kv_state(pool_a).get("same") is None
    assert get_resident_kv_state(pool_b).get("same") == 32
    out = SimpleNamespace(preempted_req_ids=set(), num_scheduled_tokens={})
    meta_a, meta_b = LMCacheMPConnectorMetadata(), LMCacheMPConnectorMetadata()
    a.prepare_step(out, meta_a)
    b.prepare_step(out, meta_b)
    assert meta_a.token_drop_finished_ids == {"same"}
    assert meta_b.token_drop_finished_ids == set()
    fresh = LMCacheMPConnectorMetadata()
    a.prepare_step(out, fresh)
    assert fresh.token_drop_finished_ids == set()


def test_read_view_exposes_selection_only_and_preserves_native_kv():
    cache = torch.arange(24, dtype=torch.float32).reshape(2, 2, 3, 1, 2)
    original = cache.clone()
    view = PagedKVSequenceView(cache, torch.tensor([4, 0]))
    read_view = KVReadView(view)
    assert torch.equal(read_view.get_keys(), view.get_keys())
    assert torch.equal(read_view.get_values(), view.get_values())
    assert not hasattr(read_view, "compact_kv_in_place")
    read_view.get_keys().zero_()
    read_view.get_values().zero_()
    assert torch.equal(cache, original)


def test_scheduler_cannot_reclaim_from_another_pool():
    scheduler, other_scheduler = TokenDropScheduler(), TokenDropScheduler()
    pool, other_pool = _Pool(), _Pool()
    scheduler.bind_block_pool(pool)
    other_scheduler.bind_block_pool(other_pool)
    blocks = [SimpleNamespace(block_id=i, ref_cnt=1, block_hash=None) for i in (1, 2)]
    scheduler.register_allocation("same", SimpleNamespace(blocks=[blocks]))
    tracker = SimpleNamespace(
        token_drop_spec=object(),
        token_drop_generation=1,
        num_scheduled_tokens=32,
        allocated_block_ids={0: [1, 2]},
    )
    meta = LMCacheMPWorkerMetadata(
        completed_store_requests={}, resident_kv_updates={"same": (1, 16)}
    )
    with pytest.raises(RuntimeError, match="not bound"):
        scheduler.commit_completed_updates(meta, {"same": tracker}, other_pool, 16)
    assert [b.block_id for b in blocks] == [1, 2]
    assert tracker.allocated_block_ids == {0: [1, 2]}
    assert get_resident_kv_state(pool).get("same") is None


def test_owner_gc_while_pool_survives():
    pool = _Pool()
    owner = TokenDropScheduler()
    owner.bind_block_pool(pool)
    state = get_resident_kv_state(pool)
    state.set("same", 16)
    state_ref = weakref.ref(state)
    del state, owner
    gc.collect()
    assert state_ref() is None
    assert get_resident_kv_state(pool) is None
    replacement = TokenDropScheduler()
    replacement.bind_block_pool(pool)
    assert get_resident_kv_state(pool).get("same") is None


def test_connector_shutdown_unbinds_only_its_own_pool():
    pool, other_pool = _Pool(), _Pool()
    owner, other = TokenDropScheduler(), TokenDropScheduler()
    owner.bind_block_pool(pool)
    other.bind_block_pool(other_pool)
    state = get_resident_kv_state(pool)
    state.set("same", 16)
    other_state = get_resident_kv_state(other_pool)
    other_state.set("same", 32)
    connector = SimpleNamespace(
        scheduler_adapter=Mock(),
        lazy_offload=False,
        _token_drop_enabled=True,
        _token_drop_scheduler=owner,
        _gpu_block_pool=pool,
    )
    LMCacheMPConnector.shutdown(connector)
    assert get_resident_kv_state(pool) is None
    assert get_resident_kv_state(other_pool) is other_state
    replacement = TokenDropScheduler()
    replacement.bind_block_pool(pool)
    LMCacheMPConnector.shutdown(connector)
    assert get_resident_kv_state(pool).get("same") is None
    assert other_state.get("same") == 32


def test_shutdown_failure_still_releases_binding():
    pool = _Pool()
    owner = TokenDropScheduler()
    owner.bind_block_pool(pool)
    connector = SimpleNamespace(
        scheduler_adapter=Mock(),
        lazy_offload=False,
        _token_drop_enabled=True,
        _token_drop_scheduler=owner,
        _gpu_block_pool=pool,
    )
    connector.scheduler_adapter.shutdown.side_effect = RuntimeError("transport")
    with pytest.raises(RuntimeError, match="transport"):
        LMCacheMPConnector.shutdown(connector)
    assert get_resident_kv_state(pool) is None


def test_failed_pool_rebind_preserves_original_connector_binding():
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

    pool, other_pool = _Pool(), _Pool()
    owner = TokenDropScheduler()
    connector = SimpleNamespace(
        role=KVConnectorRole.SCHEDULER,
        _token_drop_enabled=True,
        _token_drop_scheduler=owner,
        _gpu_block_pool=None,
        lazy_offload=False,
        scheduler_adapter=Mock(),
    )
    LMCacheMPConnector.bind_gpu_block_pool(connector, pool)
    state = get_resident_kv_state(pool)
    with pytest.raises(RuntimeError, match="already bound"):
        LMCacheMPConnector.bind_gpu_block_pool(connector, other_pool)
    assert get_resident_kv_state(pool) is state
    assert get_resident_kv_state(other_pool) is None
    LMCacheMPConnector.shutdown(connector)
    assert get_resident_kv_state(pool) is None
