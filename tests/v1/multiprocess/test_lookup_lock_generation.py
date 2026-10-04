# SPDX-License-Identifier: Apache-2.0
"""An expired lookup must not release a later reader's reservation."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
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
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager


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
        ctx.token_hasher.compute_chunk_hashes.return_value = [b"chunk"]
        ctx.token_hasher.hash_tokens.return_value = b"chunk"
        ctx.session_manager = SessionManager(ctx.token_hasher, cleanup_interval=None)
        ctx.event_bus.has_subscribers.return_value = False
        layout = MemoryLayoutDesc([torch.Size([4])], [torch.float32])
        ctx.layout_desc_registry.find.return_value = layout
        ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc([-1])
        ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: layout}
        ctx.storage_manager = sm
        ctx.get_read_owners.return_value = None
        key = IPCCacheServerKey(
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
        sm.finish_write([obj_key])
        if request.param == "L2":
            adapter = sm.l2_adapters()[0][1]
            deadline = time.monotonic() + 5
            while (
                adapter.report_status()["stored_object_count"] != 1
                or sm.report_status()["store_controller"]["in_flight_task_count"]
            ):
                assert time.monotonic() < deadline, "L2 store did not complete"
                time.sleep(0.01)
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
        sm.finish_write([obj_key])
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
