# SPDX-License-Identifier: Apache-2.0
"""An expired lookup must not release a later reader's reservation."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
from threading import Event
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest
import torch

# First Party
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
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

LookupFixture = tuple[LookupModule, StorageManager, IPCCacheServerKey, ObjectKey]


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
) -> Iterator[LookupFixture]:
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
        ctx.chunk_size = 4
        ctx.token_hasher = TokenHasher(chunk_size=4, hash_algorithm="blake3")
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
            token_ids=(0, 1, 2, 3),
            start=0,
            end=4,
            request_id="A",
            num_kv_readers=1,
        )
        chunk_hash = ctx.token_hasher.compute_packed_chunk_hashes(key.token_bytes)[0]
        obj_key = ObjectKey(chunk_hash, "model", ObjectKey.ComputeKVRank(1, 0, 1, 0))
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


def seed_prompt(
    lookup: LookupFixture,
) -> tuple[IPCCacheServerKey, list[ObjectKey]]:
    """Extend the fixture's one-chunk prompt to three real cached chunks."""
    module, sm, first, _ = lookup
    key = IPCCacheServerKey.from_token_ids(
        model_name=first.model_name,
        world_size=1,
        worker_id=None,
        token_ids=list(range(12)),
        start=0,
        end=12,
        request_id=first.request_id,
    )
    hashes = module.context.token_hasher.compute_packed_chunk_hashes(key.token_bytes)
    kv_rank = ObjectKey.ComputeKVRank(1, 0, 1, 0)
    keys = [ObjectKey(h, key.model_name, kv_rank) for h in hashes]
    layout = MemoryLayoutDesc([torch.Size([4])], [torch.float32])
    for obj_key in keys[1:]:
        assert obj_key in sm.reserve_write([obj_key], layout)
        finish_write_and_wait_for_store(sm, obj_key)
    return key, keys


def assert_all_reads_released(sm: StorageManager, keys: list[ObjectKey]) -> None:
    """Verify no read lease remains and clearing reclaims the entire test pool.

    L2-loaded temporary objects may disappear on their final release, before
    an explicit delete. Their deletion count is not a read-ownership contract.

    Args:
        sm: Fixture's real storage manager.
        keys: Every object stored by the test.
    """
    assert sm.unsafe_read(keys) == ([], [])
    sm.clear()
    assert sm.get_l1_usage()[0] == 0


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


@pytest.mark.parametrize("prefix_missing", [False, True])
def test_apc_retry_releases_the_original_tail(
    lookup: LookupFixture, prefix_missing: bool
) -> None:
    """A shorter full retry must not lose the earlier APC tail's reservation."""
    module, sm, _, _ = lookup
    key, keys = seed_prompt(lookup)
    module.lookup(replace(key, start=4), 1)
    assert module.wait_prefetch_status(key.request_id, timeout=5) == 3
    if prefix_missing:
        assert sm.delete_l1_keys(keys[:1])[1] == 0
        for _descriptor, adapter in sm.l2_adapters():
            adapter.delete(keys[:1])
    module.lookup(key, 1)
    hit = module.wait_prefetch_status(key.request_id, timeout=5)
    assert hit == (0 if prefix_missing else 3)
    module.free_lookup_locks(replace(key, end=hit * 4), 1, ((4, 12),))
    assert_all_reads_released(sm, keys)


def test_repeated_apc_cleanup_cannot_release_peer(lookup: LookupFixture) -> None:
    """The same old range is one reservation, even across repeated FREE calls."""
    module, sm, _, _ = lookup
    key, keys = seed_prompt(lookup)
    for request in (replace(key, start=4), key):
        module.lookup(request, 1)
        assert module.wait_prefetch_status(request.request_id, timeout=5) == 3
    peer = replace(key, request_id="peer")
    module.lookup(peer, 1)
    assert module.wait_prefetch_status(peer.request_id, timeout=5) == 3
    module.free_lookup_locks(key, 1, ((4, 12),))
    module.free_lookup_locks(replace(key, end=0), 1, ((4, 12),))
    assert sm.delete_l1_keys(keys) == (0, 3)
    module.free_lookup_locks(peer, 1)
    assert_all_reads_released(sm, keys)


def test_replay_tail_free_without_an_abandoned_attempt(lookup: LookupFixture) -> None:
    """A current full-prompt replay tail needs no old APC reservation."""
    module, sm, _, _ = lookup
    key, keys = seed_prompt(lookup)
    module.lookup(key, 1)
    assert module.wait_prefetch_status(key.request_id, timeout=5) == 3
    module.free_lookup_locks(replace(key, start=8), 1)
    assert sm.delete_l1_keys(keys[2:]) == (1, 0)
    module.free_lookup_locks(replace(key, end=8), 1)
    assert_all_reads_released(sm, keys)


def test_pending_apc_cleanup_declines_replacement_without_losing_locks(
    lookup: LookupFixture,
) -> None:
    """A bounded reservation slot declines work without returning an old hit."""
    module, sm, _, _ = lookup
    key, keys = seed_prompt(lookup)
    for request in (replace(key, start=4), key):
        module.lookup(request, 1)
        assert module.wait_prefetch_status(request.request_id, timeout=5) == 3
    module.lookup(key, 1)
    assert module.wait_prefetch_status(key.request_id, timeout=5) == 0
    assert module.query_prefetch_status(key.request_id) == 0
    module.free_lookup_locks(key, 1, ((4, 12),))
    assert_all_reads_released(sm, keys)


def test_apc_resolution_failure_does_not_consume_the_reservation(
    lookup: LookupFixture,
) -> None:
    """Resolving object keys must succeed before cleanup is claimed."""
    module, sm, _, _ = lookup
    key, keys = seed_prompt(lookup)
    for request in (replace(key, start=4), key):
        module.lookup(request, 1)
        assert module.wait_prefetch_status(request.request_id, timeout=5) == 3
    module.free_lookup_locks(key, 1)
    with patch.object(
        module.context.token_hasher,
        "compute_packed_chunk_hashes",
        side_effect=ValueError("resolution failed"),
    ):
        with pytest.raises(ValueError, match="resolution failed"):
            module.free_lookup_locks(replace(key, end=0), 1, ((4, 12),))
    module.free_lookup_locks(replace(key, end=0), 1, ((4, 12),))
    assert_all_reads_released(sm, keys)


def test_partial_apc_release_cannot_be_retried_against_a_peer(
    lookup: LookupFixture,
) -> None:
    """An unknown partial release must not grant a second decrement."""
    module, sm, _, _ = lookup
    key, keys = seed_prompt(lookup)
    for request in (replace(key, start=4), key):
        module.lookup(request, 1)
        assert module.wait_prefetch_status(request.request_id, timeout=5) == 3
    peer = replace(key, request_id="peer")
    module.lookup(peer, 1)
    assert module.wait_prefetch_status(peer.request_id, timeout=5) == 3
    module.free_lookup_locks(key, 1)
    release = sm.finish_read_prefetched

    def partially_release(
        objects: list[ObjectKey],
        *,
        read_locks: int = 1,
        l1_owners: dict[ObjectKey, int] | None = None,
        read_generations: dict[ObjectKey, int] | None = None,
    ) -> None:
        release(
            objects[:1],
            read_locks=read_locks,
            l1_owners=l1_owners,
            read_generations=read_generations,
        )
        raise RuntimeError("release outcome unknown")

    with patch.object(sm, "finish_read_prefetched", side_effect=partially_release):
        with pytest.raises(RuntimeError, match="release outcome unknown"):
            module.free_lookup_locks(replace(key, end=0), 1, ((4, 12),))
    module.free_lookup_locks(replace(key, end=0), 1, ((4, 12),))
    assert sm.delete_l1_keys(keys) == (0, 3)
    module.free_lookup_locks(peer, 1)
    assert sm.unsafe_read(keys[:2]) == ([], [])
    sm.clear()
    assert sm.delete_l1_keys(keys[2:]) == (0, 1)  # Unknown remainder awaits TTL.
