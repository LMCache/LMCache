# SPDX-License-Identifier: Apache-2.0

"""Maru-specific unit tests for MaruL1Manager, its allocator, and its control.

The cross-backend L1 contract shared by the stock ``L1Manager`` and
``MaruL1Manager`` (plus the Protocol structural guards) lives in
``test_l1_manager_conformance.py``. This file covers only maru-specific
behavior: CXL-pool allocation, config / startup guards, RPC-failure page/pin
safety, the TTL sweeper, the L2->L1 promote paths, maru-specific notification
nuances, and control integration through the stock tiering stack. Duplicates of
the shared contract are intentionally absent -- see the conformance suite.
"""

# Standard
from unittest.mock import MagicMock
import sys
import time

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        MaruL1Config,
        StorageManagerConfig,
        parse_args,
    )
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
    from lmcache.v1.distributed.maru_l1_manager import (
        MaruL1Manager,
        _PendingRead,
        object_key_to_string,
    )
    from lmcache.v1.distributed.memory_manager.maru_memory_allocator import (
        MaruMemoryAllocator,
        _to_tcp,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.mp_observability.event import EventType

    # Local
    from .maru_fakes import (
        FakeCxlAdapter,
        FakeMaruHandler,
        RecordingListener,
        make_maru_manager,
    )
except ImportError:
    pytest.skip("maru manager deps unavailable", allow_module_level=True)

_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([4, 8])], dtypes=[torch.float16])


def _key(idx: int, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=idx.to_bytes(4, "big"),
        model_name="test-model",
        kv_rank=0xABCD,
        cache_salt=salt,
    )


def _seed(handler, key: ObjectKey, rid: int = 7, pid: int = 3, size: int = 64):
    """Register a key in the fake directory (as if another instance stored it)."""
    handler.store_map[object_key_to_string(key)] = (rid, pid, size)


# =========================================================================
# allocator (MaruMemoryAllocator): URL normalization, init, delegation
# (absorbed from the former test_maru_memory_allocator.py, mocked runtime)
# =========================================================================


def _alloc_cfg() -> MaruL1Config:
    return MaruL1Config(server_url="maru://localhost:9000", pool_size_bytes=1 << 30)


# 1 group, shape (4 tokens, 8 feats) fp16 -> 4*8*2 = 64 bytes; 4 tokens/chunk.
_ALLOC_LAYOUT = ([torch.Size([4, 8])], [torch.float16], MemoryFormat.KV_2LTD, 4)


@pytest.fixture
def maru_mocks(monkeypatch):
    """Inject fake maru / maru_lmcache modules for init_layout's lazy imports."""
    handler = MagicMock()
    handler.connect.return_value = True
    handler.get_chunk_size.return_value = 64
    adapter = MagicMock()
    maru_mod = MagicMock()
    maru_mod.MaruHandler.return_value = handler
    lmcache_mod = MagicMock()
    lmcache_mod.CxlMemoryAdapter.return_value = adapter
    monkeypatch.setitem(sys.modules, "maru", maru_mod)
    monkeypatch.setitem(sys.modules, "maru_lmcache", lmcache_mod)
    return handler, adapter, maru_mod


def test_to_tcp():
    assert _to_tcp("maru://h:1") == "tcp://h:1"
    assert _to_tcp("tcp://h:1") == "tcp://h:1"


def test_methods_before_init_raise():
    alloc = MaruMemoryAllocator(_alloc_cfg())
    assert not alloc.is_initialized
    with pytest.raises(RuntimeError):
        alloc.allocate([torch.Size([4, 8])], [torch.float16])
    with pytest.raises(RuntimeError):
        _ = alloc.handler
    with pytest.raises(RuntimeError):
        _ = alloc.single_token_size


def test_init_layout_builds_pool(maru_mocks):
    handler, adapter, maru_mod = maru_mocks
    alloc = MaruMemoryAllocator(_alloc_cfg())
    alloc.init_layout(*_ALLOC_LAYOUT)

    assert alloc.is_initialized
    handler.connect.assert_called_once()
    assert alloc.handler is handler
    assert alloc.single_token_size == 16  # 64 bytes / 4 tokens

    _, kwargs = maru_mod.MaruConfig.call_args
    assert kwargs["server_url"] == "tcp://localhost:9000"
    assert kwargs["pool_size"] == 1 << 30
    assert kwargs["chunk_size_bytes"] == 64
    assert kwargs["auto_connect"] is False


def test_init_layout_mismatch_raises(maru_mocks):
    alloc = MaruMemoryAllocator(_alloc_cfg())
    alloc.init_layout(*_ALLOC_LAYOUT)
    with pytest.raises(ValueError):
        alloc.init_layout(
            [torch.Size([8, 8])], [torch.float16], MemoryFormat.KV_2LTD, 4
        )


def test_connect_failure_raises(maru_mocks):
    handler, _, _ = maru_mocks
    handler.connect.return_value = False
    alloc = MaruMemoryAllocator(_alloc_cfg())
    with pytest.raises(RuntimeError):
        alloc.init_layout(*_ALLOC_LAYOUT)


def test_delegation_and_lifecycle(maru_mocks):
    _, adapter, _ = maru_mocks
    alloc = MaruMemoryAllocator(_alloc_cfg())
    alloc.init_layout(*_ALLOC_LAYOUT)
    obj = MagicMock()

    adapter.allocate.return_value = obj
    assert alloc.allocate([torch.Size([4, 8])], [torch.float16]) is obj

    alloc.get_by_location(1, 2, 64)
    adapter.get_by_location.assert_called_once_with(
        region_id=1, page_index=2, actual_size=64, single_token_size=16
    )

    alloc.create_store_handle(obj)
    adapter.create_store_handle.assert_called_once_with(obj)

    # abort_alloc returns the page via the adapter's real free
    alloc.abort_alloc(obj)
    adapter.free.assert_called_once_with(obj)

    # free/batched_free are no-ops (lifecycle owned by MaruServer)
    adapter.free.reset_mock()
    alloc.free(obj)
    alloc.batched_free([obj])
    adapter.free.assert_not_called()


# =========================================================================
# config / startup guards: flag parsing + rejected backend combinations
# (absorbed from the former test_l1_config_maru.py)
# =========================================================================


def _args(*extra: str, l1_size_gb: str = "1") -> list[str]:
    """Minimal required flags (eviction policy + L1 size) plus extras."""
    return [
        "--eviction-policy",
        "LRU",
        "--l1-size-gb",
        l1_size_gb,
        "--no-l1-use-lazy",
        *extra,
    ]


def _maru_memory(**overrides) -> L1MemoryManagerConfig:
    return L1MemoryManagerConfig(
        size_in_bytes=0,
        use_lazy=False,
        maru_config=MaruL1Config(
            server_url="maru://localhost:5555",
            pool_size_bytes=1 << 20,
            instance_id="t",
        ),
        **overrides,
    )


def _maru_sm_config(
    *, memory=None, store_policy="default", adapters=None
) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=memory if memory is not None else _maru_memory(),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapters or []),
        store_policy=store_policy,
    )


def test_maru_flags_build_maru_config():
    cfg = parse_args(
        _args(
            "--maru-server-url",
            "maru://localhost:9000",
            "--maru-pool-size-gb",
            "8",
            "--maru-instance-id",
            "node-a",
        )
    )
    maru = cfg.l1_manager_config.memory_config.maru_config
    assert isinstance(maru, MaruL1Config)
    assert maru.server_url == "maru://localhost:9000"
    assert maru.pool_size_bytes == 8 * (1 << 30)
    assert maru.instance_id == "node-a"


def test_maru_without_pool_size_raises():
    with pytest.raises(ValueError):
        parse_args(_args("--maru-server-url", "maru://localhost:9000"))


def test_maru_rejects_devdax_l1():
    with pytest.raises(ValueError, match="devdax"):
        _maru_sm_config(memory=_maru_memory(devdax_path="/dev/dax0.0", shm_name=""))


def test_maru_rejects_registered_l2(monkeypatch):
    # Simulate a registered/RDMA adapter via the shared region classifier.
    monkeypatch.setattr(
        "lmcache.v1.distributed.config._requires_single_l1_memory_region",
        lambda adapter_config: "nixl_store",
    )
    with pytest.raises(ValueError, match="registerable"):
        _maru_sm_config(
            adapters=[MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=1.0)]
        )


def test_maru_rejects_non_lmcache_driven_transfer():
    """maru requires the LMCache-driven transfer path (rejects engine/auto)."""
    # First Party
    from lmcache.v1.mp_observability.config import ObservabilityConfig
    from lmcache.v1.multiprocess.config import (
        CoordinatorConfig,
        HTTPFrontendConfig,
        MPServerConfig,
    )
    from lmcache.v1.multiprocess.http_server import run_http_server

    with pytest.raises(ValueError, match="lmcache_driven"):
        run_http_server(
            http_config=HTTPFrontendConfig(),
            mp_config=MPServerConfig(supported_transfer_mode="engine_driven"),
            storage_manager_config=_maru_sm_config(),
            obs_config=ObservabilityConfig(),
            coordinator_config=CoordinatorConfig(url=""),
        )


# =========================================================================
# reserve_read / finish_read: pin balance + page-reclaim safety under
# RPC failures, and the two staging regression guards
# =========================================================================


def test_reserve_read_pin_rpc_failure_is_a_miss():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.fail_pin = True
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)
    assert not handler.pins


def test_reserve_read_retrieve_rpc_failure_rolls_back_pins():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.fail_retrieve = True
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)
    assert handler.pins[object_key_to_string(k)] == 0


def test_reserve_read_pool_miss_rolls_back_pins():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    adapter.resolve_none = True
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)
    assert handler.pins[object_key_to_string(k)] == 0


def test_temporary_read_absorbing_overlapping_pins_releases_them():
    """A peer-registered key read while temporary-staged must not leak pins.

    A temporary promote stages ``k`` locally with no server pin. If a peer has
    registered ``k`` in the directory, a later ``reserve_read`` pins the shared
    copy and folds those pins into the temporary entry. Draining the reads must
    release every absorbed pin -- regression: the temporary release path used to
    reclaim the page without unpinning, leaking the pin permanently (the page
    became un-evictable on MaruServer).
    """
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)

    # Temporary promote: staged in _pending_read, no directory pin.
    manager.reserve_write([k], [True], _LAYOUT, mode="new")
    manager.finish_write_and_reserve_read([k])
    assert handler.pins.get(ks, 0) == 0  # temporary -> no server pin

    # A peer registers k; an overlapping reserve_read pins the shared copy and
    # absorbs that pin onto the temporary entry.
    _seed(handler, k)
    manager.reserve_read([k])
    assert handler.pins[ks] == 1

    # Drain both holds (temporary hold + absorbed read): every pin released.
    manager.finish_read([k])
    manager.finish_read([k])
    assert handler.pins[ks] == 0  # was 1 (leaked) before the fix
    assert ks not in manager._pending_read


def test_reserve_read_excludes_mid_write_key_no_double_staging():
    """A key mid-write on this instance stays KEY_NOT_READABLE even if a peer
    registered it -- it must not be pinned or read-staged.

    A local write reserves ``k`` (staged in _pending_write); before it finishes,
    a peer registers ``k`` in the shared directory. A reserve_read that slips in
    must refuse ``k``, not serve the peer copy: staging it would leave ``k`` in
    both _pending_write and _pending_read (double staging) and strand the write
    -- its promote would then return KEY_IN_WRONG_STATE and never pop
    _pending_write, orphaning the page until the write-TTL sweeper.
    """
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)

    # Local write in flight: k staged for write, not yet registered by us.
    manager.reserve_write([k], [True], _LAYOUT, mode="new")
    # A peer registers the same key in the shared directory mid-write.
    _seed(handler, k)

    # A read racing the in-flight write is refused, not served from the peer.
    res = manager.reserve_read([k])
    assert res[k] == (L1Error.KEY_NOT_READABLE, None)
    assert handler.pins.get(ks, 0) == 0  # no pin taken on the peer copy
    assert k not in manager._pending_read  # no double staging
    assert k in manager._pending_write  # write still in flight

    # The write still completes cleanly: promote pops _pending_write and stages
    # the read (no orphan, no KEY_IN_WRONG_STATE).
    fin = manager.finish_write_and_reserve_read([k])
    assert fin[k][0] == L1Error.SUCCESS
    assert k not in manager._pending_write


# =========================================================================
# reserve_write / finish_write: OOM, mode guard, and page-reclaim safety
# =========================================================================


def test_reserve_write_oom_marks_whole_batch():
    manager, _, adapter = make_maru_manager()
    adapter.oom = True
    res = manager.reserve_write([_key(1), _key(2)], [False] * 2, _LAYOUT, mode="new")
    assert all(v == (L1Error.OUT_OF_MEMORY, None) for v in res.values())


def test_reserve_write_rejects_non_new_mode():
    manager, _, _ = make_maru_manager()
    with pytest.raises(ValueError):
        manager.reserve_write([_key(1)], [False], _LAYOUT, mode="update")


def test_finish_write_handle_failure_reclaims_all_pages():
    manager, _, adapter = make_maru_manager()
    keys = [_key(1), _key(2)]
    adapter.fail_handle = True
    manager.reserve_write(keys, [False] * 2, _LAYOUT, mode="new")
    res = manager.finish_write(keys)
    assert all(err == L1Error.KEY_IN_WRONG_STATE for err in res.values())
    assert len(adapter.freed) == 2  # never reached the server: reclaimed


def test_finish_write_store_rpc_failure_never_recycles():
    """Unknown server state: pages must leak rather than be reused."""
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    handler.fail_store = True
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert manager.finish_write([k])[k] == L1Error.KEY_IN_WRONG_STATE
    assert not adapter.freed


def test_finish_write_dup_skip_is_success_without_abort():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    _seed(handler, k)  # another instance registered it meanwhile
    assert manager.finish_write([k])[k] == L1Error.SUCCESS  # dup-skip
    assert not adapter.freed  # no local double-free


# =========================================================================
# delete: cross-node pinned-key refusal
# =========================================================================


def test_delete_remotely_pinned_key_is_locked():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.pins[object_key_to_string(k)] = 1  # pinned by another instance

    assert manager.delete([k])[k] == L1Error.KEY_IS_LOCKED


# =========================================================================
# get_memory_usage (device-fill watermark) / is_key_evictable
# =========================================================================


def test_get_memory_usage_before_init_reports_pool_size():
    manager, _, _ = make_maru_manager()
    manager._allocator._cxl_adapter = None  # back to pre-init state
    used, total = manager.get_memory_usage()
    assert (used, total) == (0, 1 << 20)


def test_get_memory_usage_watermark_tracks_auto_expand():
    """total anchors to device fill when auto_expand is on, to the owned pool
    when off -- so a hard-capped pool evicts before it is exhausted instead of
    OOMing while the device still has (unusable) free space."""
    chunk = 64
    own_pool = 16 * chunk  # FakeMaruHandler.get_stats total_pool_size
    free = 100 * chunk

    # auto_expand on (default): total = owned pool + device free.
    mgr_on, handler_on, _ = make_maru_manager(chunk_size=chunk)
    handler_on.cxl_free = free
    _seed(handler_on, _key(1))  # one allocated page
    assert mgr_on.get_memory_usage() == (chunk, own_pool + free)

    # auto_expand off: device free is ignored; total is the owned pool alone.
    mgr_off, handler_off, _ = make_maru_manager(chunk_size=chunk, auto_expand=False)
    handler_off.cxl_free = free
    _seed(handler_off, _key(1))
    assert mgr_off.get_memory_usage() == (chunk, own_pool)


def test_is_key_evictable_tracks_local_staging():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    assert manager.is_key_evictable(k)
    manager.reserve_read([k])
    assert not manager.is_key_evictable(k)
    manager.finish_read([k])
    assert manager.is_key_evictable(k)


# =========================================================================
# finish_write_and_reserve_read (L2->L1 promote): retained vs temporary
# =========================================================================


def test_temporary_promote_page_reclaimed_after_read():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    page = manager.reserve_write([k], [True], _LAYOUT, mode="new")[k][1]
    manager.finish_write_and_reserve_read([k])

    assert manager.finish_read([k])[k] == L1Error.SUCCESS
    assert page.metadata.address in adapter.freed  # local page reclaimed
    assert not handler.unpin_log  # nothing was pinned
    assert manager.unsafe_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)


def test_retained_promote_extra_count_pins_n():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")

    manager.finish_write_and_reserve_read([k], extra_count=2)
    assert handler.pins[object_key_to_string(k)] == 3  # 1 + extra_count


def test_retained_promote_dup_skip_resolves_winner():
    """A peer registered the key first: re-resolve pins the winning page."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    _seed(handler, k, rid=9, pid=5)  # peer registered it meanwhile

    assert manager.finish_write_and_reserve_read([k])[k][0] == L1Error.SUCCESS
    assert handler.pins[object_key_to_string(k)] == 1


def test_retained_promote_store_failure_is_wrong_state():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    handler.fail_store_keys.add(object_key_to_string(k))
    manager.reserve_write([k], [False], _LAYOUT, mode="new")

    assert manager.finish_write_and_reserve_read([k])[k] == (
        L1Error.KEY_IN_WRONG_STATE,
        None,
    )
    assert k not in manager._pending_write  # write staging drained


# =========================================================================
# TTL sweeper: reclaim orphan write pages / read pins on expiry
# =========================================================================


def test_sweep_reclaims_expired_write():
    manager, _, adapter = make_maru_manager()
    k = _key(1)
    page = manager.reserve_write([k], [False], _LAYOUT, mode="new")[k][1]
    manager._pending_write[k].deadline = time.monotonic() - 1  # force expiry

    manager._sweep_once()
    assert k not in manager._pending_write
    assert page.metadata.address in adapter.freed  # returned to the owner


def test_sweep_reclaims_expired_read_unpins():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)
    _seed(handler, k)
    manager.reserve_read([k], extra_count=1)  # two pins
    manager._pending_read[k].deadline = time.monotonic() - 1

    manager._sweep_once()
    assert k not in manager._pending_read
    assert handler.pins[ks] == 0  # both pins released
    assert not adapter.freed  # registered page is not freed


def test_sweep_reclaims_expired_temporary_read():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    page = manager.reserve_write([k], [True], _LAYOUT, mode="new")[k][1]
    manager.finish_write_and_reserve_read([k])  # temporary read staged
    manager._pending_read[k].deadline = time.monotonic() - 1

    manager._sweep_once()
    assert k not in manager._pending_read
    assert page.metadata.address in adapter.freed  # private page reclaimed
    assert not handler.unpin_log  # temporary reads hold no pin


def test_sweep_leaves_live_staging():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])  # deadline ~ now + read_ttl (300s)

    manager._sweep_once()
    assert k in manager._pending_read  # not expired
    assert handler.pins[object_key_to_string(k)] == 1
    assert not adapter.freed


# =========================================================================
# notifications: maru-specific event_bus / listener nuances
# (the shared listener lifecycle is asserted in the conformance suite)
# =========================================================================


def _keys_for(bus, event_type):
    """Return the ``keys`` metadata of every publish of ``event_type``."""
    return [
        c.args[0].metadata["keys"]
        for c in bus.publish.call_args_list
        if c.args[0].event_type == event_type
    ]


def test_event_bus_write_read_delete_lifecycle():
    manager, handler, _ = make_maru_manager()
    manager._event_bus = MagicMock()
    k = _key(1)

    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    manager.finish_write([k])
    manager.reserve_read([k])
    manager.finish_read([k])
    manager.delete([k])

    assert _keys_for(manager._event_bus, EventType.L1_WRITE_RESERVED) == [[k]]
    assert _keys_for(manager._event_bus, EventType.L1_WRITE_FINISHED) == [[k]]
    assert _keys_for(manager._event_bus, EventType.L1_READ_RESERVED) == [[k]]
    assert _keys_for(manager._event_bus, EventType.L1_READ_FINISHED) == [[k]]
    assert [k] in _keys_for(manager._event_bus, EventType.L1_KEYS_EVICTED)  # delete


def test_finish_write_store_failure_is_not_fired():
    """A page that never registered must not notify write-finished."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    handler.fail_store_keys.add(object_key_to_string(k))
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    manager.finish_write([k])
    assert rec.kinds("write_finished") == [[]]  # fired once, empty


def test_finish_read_temporary_frees_page_and_fires_delete():
    """A temporary read (local staging) reclaims its page at refcount zero.

    The promote path that creates temporary reads lands in C5; here the entry
    is staged white-box to exercise C4's refcount-zero reclaim branch.
    """
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    page = adapter.batched_allocate(_LAYOUT.shapes, _LAYOUT.dtypes, 1)[0]
    manager._pending_read[k] = _PendingRead(mem_obj=page, refcount=1, is_temporary=True)
    rec = RecordingListener()
    manager.register_listener(rec)

    assert manager.finish_read([k])[k] == L1Error.SUCCESS
    # Temporary pages hold no server pin -- reclaimed, not unpinned.
    assert page.metadata.address in adapter.freed
    assert not handler.unpin_log
    assert rec.kinds("read_finished") == [[k]]
    assert rec.kinds("deleted_by_manager") == [[k]]
    assert k not in manager._pending_read


def test_delete_fires_only_for_removed_keys():
    """Locked / absent keys are not reported as manager deletes."""
    manager, handler, _ = make_maru_manager()
    removed, absent, locked = _key(1), _key(2), _key(3)
    _seed(handler, removed)
    _seed(handler, locked)
    handler.pins[object_key_to_string(locked)] = 1  # pinned elsewhere
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.delete([removed, absent, locked])
    assert rec.kinds("deleted_by_manager") == [[removed]]


# =========================================================================
# control integration: MaruL1Manager under the real tiering controllers
# (absorbed from the former test_maru_integration.py, C11b)
#
# A real StorageManager + StoreController/PrefetchController/EvictionController
# drive MaruL1Manager, whose CXL pool + MaruServer directory are the in-memory
# fakes. These assert maru's *control* integration (register / read-reserve /
# evict-delete under the real controllers); L1<->L2 byte movement is stock
# controller logic covered by the stock StorageManager tests.
# =========================================================================


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _maru_sm_with_fakes(chunk_size: int = 64, trigger_watermark: float = 0.8):
    """A real StorageManager whose maru L1 tier is backed by the fakes."""
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=0,
                use_lazy=False,
                maru_config=MaruL1Config(
                    server_url="maru://localhost:5555",
                    pool_size_bytes=1 << 20,
                    instance_id="t",
                ),
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(
            eviction_policy="LRU", trigger_watermark=trigger_watermark
        ),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=1.0)]
        ),
    )
    sm = StorageManager(config)
    assert isinstance(sm._l1_manager, MaruL1Manager)  # harness selected maru
    handler = FakeMaruHandler(chunk_size)
    adapter = FakeCxlAdapter(chunk_size)
    alloc = sm._l1_manager._allocator
    alloc._handler = handler
    alloc._cxl_adapter = adapter
    alloc._single_token_size = 16
    return sm, handler, adapter


def test_store_registers_in_maru_directory():
    """reserve_write -> finish_write through the full stack registers in maru."""
    sm, handler, _ = _maru_sm_with_fakes()
    try:
        k = _key(1)
        res = sm.reserve_write([k], _LAYOUT, mode="new")
        assert res[k] is not None
        sm.finish_write([k])
        assert object_key_to_string(k) in handler.store_map
    finally:
        sm.close()


def test_prefetch_hits_l1_resident_keys():
    """Prefetch of directory-resident keys is a full L1 hit (maru reserve_read)."""
    sm, _, _ = _maru_sm_with_fakes()
    try:
        keys = [_key(i) for i in range(3)]
        sm.reserve_write(keys, _LAYOUT, mode="new")
        sm.finish_write(keys)

        handle = sm.submit_prefetch_task(keys, _LAYOUT)
        assert _wait(lambda: sm.query_prefetch_status(handle) is not None)
        assert sm.query_prefetch_status(handle).count_leading_ones() == len(keys)
    finally:
        sm.close()


def test_eviction_deletes_from_maru_directory():
    """Watermark eviction drives MaruL1Manager.delete on the shared directory."""
    # Low watermark: the fake pool is 16 pages, so a handful of stored keys
    # crosses it and the eviction controller must reclaim some.
    sm, handler, _ = _maru_sm_with_fakes(trigger_watermark=0.1)
    try:
        keys = [_key(i) for i in range(6)]
        sm.reserve_write(keys, _LAYOUT, mode="new")
        sm.finish_write(keys)
        assert len(handler.store_map) == 6

        # The eviction controller runs on its own thread; it deletes evictable
        # keys from the maru directory until usage falls under the watermark.
        assert _wait(lambda: len(handler.store_map) < 6)
    finally:
        sm.close()
