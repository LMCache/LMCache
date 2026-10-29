# SPDX-License-Identifier: Apache-2.0
"""An expired lookup must not release a later reader's reservation."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event import Event as CacheEvent
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
import lmcache.v1.distributed.l1_manager as l1_manager_module


def finish_write_and_wait_for_store(sm: StorageManager, key: ObjectKey) -> None:
    """Commit a chunk and wait until its L2 store has released its read lock."""
    completed = Event()

    def on_event(event: CacheEvent) -> None:
        if event.event_type == EventType.L2_STORE_COMPLETED:
            completed.set()

    with patch.object(get_event_bus(), "publish", side_effect=on_event):
        sm.finish_write([key])
        if sm.l2_adapters():
            assert completed.wait(5), "L2 store did not complete"


@pytest.fixture(params=["L1", "L2"])
def lookup(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[LookupModule, StorageManager, IPCCacheServerKey, ObjectKey]]:
    """Real storage, prefetch and session lifecycle; use unpinned CPU allocation."""
    with (
        patch(
            "lmcache.v1.memory_management._allocate_cpu_memory",
            side_effect=lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
        ),
        patch("lmcache.v1.memory_management._free_cpu_memory"),
    ):
        sm = StorageManager(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=4096, use_lazy=False, align_bytes=64, shm_name=""
                    ),
                    read_ttl_seconds=1,
                ),
                eviction_config=EvictionConfig(eviction_policy="noop"),
                l2_adapter_config=L2AdaptersConfig(
                    [MockL2AdapterConfig(0.001, 1)] if request.param == "L2" else []
                ),
            )
        )
        ctx = MagicMock()
        ctx.chunk_size = ctx.token_hasher.chunk_size = 4
        ctx.token_hasher.compute_packed_chunk_hashes.return_value = [b"chunk"]
        ctx.token_hasher.hash_packed_chunk.return_value = b"chunk"
        ctx.session_manager = SessionManager(ctx.token_hasher, cleanup_interval=None)
        ctx.event_bus.has_subscribers.return_value = False
        layout = MemoryLayoutDesc([torch.Size([4])], [torch.float32])
        ctx.layout_desc_registry.find.return_value = layout
        ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc([-1])
        ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: layout}
        ctx.storage_manager = sm
        ctx.get_read_owners.return_value = None
        key = IPCCacheServerKey.from_token_ids(
            model_name="model",
            world_size=1,
            worker_id=None,
            token_ids=(1, 2, 3, 4),
            start=0,
            end=4,
            request_id="A",
            num_kv_readers=1,
        )
        obj_key = ObjectKey(b"chunk", "model", ObjectKey.ComputeKVRank(1, 0, 1, 0))
        assert obj_key in sm.reserve_write([obj_key], layout)
        finish_write_and_wait_for_store(sm, obj_key)
        if request.param == "L2":
            sm.clear()
            assert sm.get_l1_usage()[0] == 0  # Next lookup must reload from L2.
        try:
            yield LookupModule(ctx), sm, key, obj_key
        finally:
            ctx.session_manager.close()
            sm.close()


def start_lookup(module: LookupModule, key: IPCCacheServerKey) -> None:
    """Consume the real prefetch result so cleanup has the acquired lock set."""
    module.lookup(key, 1)
    assert module.wait_prefetch_status(key.request_id, timeout=5) == 1


@pytest.mark.parametrize("recreate", [False, True])
def test_expired_lookup_cannot_release_new_reader(lookup, recreate: bool) -> None:
    """The same public-API test fails on the original key-only cleanup."""
    module, sm, old, obj_key = lookup
    start_lookup(module, old)
    time.sleep(1.05)
    if recreate:
        sm.clear()
        layout = MemoryLayoutDesc([torch.Size([4])], [torch.float32])
        assert obj_key in sm.reserve_write([obj_key], layout)
        finish_write_and_wait_for_store(sm, obj_key)
    new = replace(old, request_id="B")
    start_lookup(module, new)
    with patch("lmcache.v1.distributed.storage_manager.publish_call_event") as trace:
        module.free_lookup_locks(old, 1)
    sm.clear()
    with sm.read_prefetched_results([obj_key]) as objects:
        assert objects is not None, "old cleanup released the new reader's lock"
    trace.assert_not_called()  # A rejected release must not be replayed.
    module.free_lookup_locks(new, 1)
    sm.clear()
    with sm.read_prefetched_results([obj_key]) as objects:
        assert objects is None, "normal cleanup leaked the new reader's lock"


def test_shared_readers_and_legacy_completion(lookup) -> None:
    """Live lookups share a count; individual worker completion stays key-only."""
    module, sm, old, obj_key = lookup
    start_lookup(module, old)
    new = replace(old, request_id="B", num_kv_readers=2)
    start_lookup(module, new)
    module.free_lookup_locks(old, 1)
    sm.finish_read_prefetched([obj_key])
    sm.clear()
    with sm.read_prefetched_results([obj_key]) as objects:
        assert objects is not None
    sm.finish_read_prefetched([obj_key])
    sm.clear()
    with sm.read_prefetched_results([obj_key]) as objects:
        assert objects is None


def test_removed_session_cannot_release_another_reader(lookup) -> None:
    """Missing session metadata is not permission to release by key alone."""
    module, sm, old, obj_key = lookup
    start_lookup(module, old)
    module.free_lookup_locks(old, 1)
    module.end_session(old.request_id)
    new = replace(old, request_id="B")
    start_lookup(module, new)
    module.free_lookup_locks(old, 1)
    sm.clear()
    with sm.read_prefetched_results([obj_key]) as objects:
        assert objects is not None
    module.free_lookup_locks(new, 1)


@pytest.fixture
def retriever(lookup, monkeypatch: pytest.MonkeyPatch):
    """Use real lookup storage with only the device stream/backend simulated."""
    lookup_module, sm, _key, obj_key = lookup
    ctx = lookup_module.context
    ctx.resolve_obj_keys.return_value = [[obj_key]]
    cache = MagicMock()
    cache.calculate_num_blocks.return_value = 1
    cache.kv_layer_groups_manager = SimpleNamespace(
        num_object_groups=1,
        num_kernel_groups=1,
        get_attn_desc=lambda: AttnWindowDesc([-1]),
    )
    backend = MagicMock()
    monkeypatch.setattr(transfer.DeviceHostFuncDispatcher, "start", lambda self: None)
    monkeypatch.setattr(device_ops, "drain_recorded_completions", lambda: [])
    module = LMCacheDrivenTransferModule(ctx)
    entry = SimpleNamespace(
        cache_context=cache, model_name="model", event_backend=backend
    )
    monkeypatch.setattr(
        module, "get_and_touch_context_entry", MagicMock(return_value=entry)
    )
    pending = []

    def submit(_stream, kind, payload):
        if kind == "finish_read_transfer":
            pending.append(payload)

    monkeypatch.setattr(transfer, "torch_dev", MagicMock())
    monkeypatch.setattr(
        transfer, "downsample_and_stage_block_ids", lambda _cache, ids: ids
    )
    monkeypatch.setattr(transfer, "submit_callback_to_stream", submit)
    monkeypatch.setattr("lmcache.v1.distributed.l1_manager.monotonic", lambda: 1.0)
    yield module, cache, backend, pending
    for receipt in pending:
        sm.finish_read_transfer(receipt)
    module.close()


@pytest.mark.parametrize("free_copy", [True, False])
def test_free_and_transfer_complete_only_their_own_same_generation_reads(
    lookup, retriever, free_copy: bool
) -> None:
    """FREE(A) plus completion(A) cannot remove B, and FREE(B) must not leak A."""
    lookup_module, sm, key, obj_key = lookup
    module, _cache, _backend, pending = retriever
    peer = replace(key, request_id="B")
    start_lookup(lookup_module, key)
    start_lookup(lookup_module, peer)
    assert module.retrieve(replace(key, worker_id=0), 11, [[0]], b"event")[1]
    lookup_module.free_lookup_locks(key if free_copy else peer, 1)
    lookup_module.free_lookup_locks(key if free_copy else peer, 1)
    assert sm.delete_l1_keys([obj_key]) == (0, 1)
    receipt = pending.pop()
    sm.finish_read_transfer(receipt)
    sm.finish_read_transfer(receipt)
    if free_copy:
        assert sm.delete_l1_keys([obj_key]) == (0, 1)
        lookup_module.free_lookup_locks(peer, 1)
    assert sm.unsafe_read([obj_key]) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0


def test_free_wins_before_retrieve_without_reading_peer(lookup, retriever) -> None:
    lookup_module, sm, key, obj_key = lookup
    module, cache, _backend, _pending = retriever
    peer = replace(key, request_id="B")
    start_lookup(lookup_module, key)
    start_lookup(lookup_module, peer)
    lookup_module.free_lookup_locks(key, 1)
    assert not module.retrieve(replace(key, worker_id=0), 11, [[0]], b"event")[1]
    cache.transfer_object_group.assert_not_called()
    assert sm.delete_l1_keys([obj_key]) == (0, 1)
    lookup_module.free_lookup_locks(peer, 1)
    assert sm.unsafe_read([obj_key]) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0


def test_mla_workers_failure_and_free_share_one_ownership_ledger(
    lookup, retriever
) -> None:
    """Three TP instances share one KV rank; a failed worker claims only once."""
    lookup_module, sm, original, obj_key = lookup
    module, _cache, _backend, pending = retriever
    key = replace(original, num_kv_readers=3)
    peer = replace(original, request_id="B")
    start_lookup(lookup_module, key)
    start_lookup(lookup_module, peer)
    worker = replace(key, worker_id=0)
    assert module.retrieve(worker, 11, [[0]], b"event")[1]
    assert not module.retrieve(worker, 11, [[0]], b"event")[1]
    entry = module.get_and_touch_context_entry.return_value
    module.get_and_touch_context_entry.return_value = None
    assert module.retrieve(worker, 12, [[0]], b"event") == (b"", False)
    assert module.retrieve(worker, 12, [[0]], b"event") == (b"", False)
    module.get_and_touch_context_entry.return_value = entry
    lookup_module.free_lookup_locks(key, 3)
    sm.finish_read_transfer(pending.pop())
    assert sm.delete_l1_keys([obj_key]) == (0, 1)
    lookup_module.free_lookup_locks(peer, 1)
    assert sm.unsafe_read([obj_key]) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0


@pytest.mark.parametrize("failure_stage", ["event", "callback"])
def test_failed_completion_publication_drains_before_unpin(
    lookup, retriever, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Even after FREE, the source stays resident until fallback sync completes."""
    lookup_module, sm, key, obj_key = lookup
    module, cache, backend, _pending = retriever
    start_lookup(lookup_module, key)

    def synchronize():
        monkeypatch.setattr(l1_manager_module, "monotonic", lambda: 1000.0)
        lookup_module.free_lookup_locks(key, 1)
        assert sm.delete_l1_keys([obj_key]) == (0, 1)

    cache.stream.synchronize.side_effect = synchronize
    if failure_stage == "event":
        backend.record_event.side_effect = RuntimeError("publication failed")
    else:
        original_submit = transfer.submit_callback_to_stream

        def submit(stream, kind, payload):
            if kind == "finish_read_transfer":
                raise RuntimeError("publication failed")
            original_submit(stream, kind, payload)

        monkeypatch.setattr(transfer, "submit_callback_to_stream", submit)
    with pytest.raises(RuntimeError, match="publication failed"):
        module.retrieve(replace(key, worker_id=0), 11, [[0]], b"event")
    cache.stream.synchronize.assert_called_once()
    assert sm.unsafe_read([obj_key]) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0


def test_failed_sync_preserves_source_until_copy_is_known_complete(
    lookup, retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_module, sm, key, obj_key = lookup
    module, cache, _backend, _pending = retriever
    start_lookup(lookup_module, key)
    attempted = []

    def submit(_stream, kind, payload):
        if kind == "finish_read_transfer":
            attempted.append(payload)
            raise RuntimeError("publication failed")

    monkeypatch.setattr(transfer, "submit_callback_to_stream", submit)
    cache.stream.synchronize.side_effect = RuntimeError("sync failed")
    with pytest.raises(RuntimeError, match="sync failed"):
        module.retrieve(replace(key, worker_id=0), 11, [[0]], b"event")
    monkeypatch.setattr(l1_manager_module, "monotonic", lambda: 1000.0)
    lookup_module.free_lookup_locks(key, 1)
    assert sm.delete_l1_keys([obj_key]) == (0, 1)
    # The fake copy is now quiescent; production must not assume this after failure.
    sm.finish_read_transfer(attempted.pop())
    assert sm.unsafe_read([obj_key]) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0


def test_failed_original_pin_retires_claim_without_releasing_new_reader(
    lookup, retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_module, sm, key, obj_key = lookup
    module, cache, _backend, pending = retriever
    start_lookup(lookup_module, key)
    monkeypatch.setattr(l1_manager_module, "monotonic", lambda: 1000.0)
    peer = replace(key, request_id="B")
    start_lookup(lookup_module, peer)
    assert not module.retrieve(replace(key, worker_id=0), 11, [[0]], b"event")[1]
    cache.transfer_object_group.assert_not_called()
    lookup_module.free_lookup_locks(key, 1)
    sm.finish_read_transfer(pending.pop())
    assert sm.delete_l1_keys([obj_key]) == (0, 1)
    lookup_module.free_lookup_locks(peer, 1)
    assert sm.unsafe_read([obj_key]) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0


def test_transfer_receipt_survives_session_removal(lookup, retriever) -> None:
    lookup_module, sm, key, obj_key = lookup
    module, _cache, _backend, pending = retriever
    start_lookup(lookup_module, key)
    assert module.retrieve(replace(key, worker_id=0), 11, [[0]], b"event")[1]
    lookup_module.end_session(key.request_id)
    assert sm.delete_l1_keys([obj_key]) == (0, 1)
    sm.finish_read_transfer(pending.pop())
    assert sm.unsafe_read([obj_key]) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0
