# SPDX-License-Identifier: Apache-2.0
"""A shared Device-DAX L1 in StorageManager: placement, ownership, L2 affinity."""

# Standard
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.memory_orchestrator.api import RegionUsage
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from tests.v1.distributed.shared_l1.utils import key, shared_config
from tests.v1.distributed.utils import (
    SCRIPTED_WRITE_POLICY,
    ScriptedWritePolicy,
    single_row_spec,
)
from tests.v1.memory_orchestrator.harness import Orchestrator

pytestmark = pytest.mark.no_shared_allocator
LAYOUT = MemoryLayoutDesc([torch.Size([1000])], [torch.uint8])


def storage(*l1: L1ManagerConfig, l2: bool = False) -> StorageManager:
    """Build a StorageManager over ``l1`` in order, with one mock L2 if ``l2``."""
    adapters = L2AdaptersConfig([MockL2AdapterConfig(1, 1)] if l2 else [])
    noop = EvictionConfig("noop")
    config = StorageManagerConfig(l1[0], noop, adapters, l1_manager_configs=list(l1))
    return StorageManager(config)


def put(store: StorageManager, keys: list[ObjectKey], layout: MemoryLayoutDesc) -> None:
    """Write ``keys`` one by one; the i-th object (from 1) holds the value i."""
    for value, k in enumerate(keys, 1):
        reserved = store.reserve_write([k], layout)
        tensor = reserved[k].tensor
        assert tensor is not None
        tensor.fill_(value)
        store.finish_write(list(reserved), l1_owners=l1_owners_of(reserved))


def route(monkeypatch: pytest.MonkeyPatch, to_private: set[ObjectKey]) -> None:
    """Script writes: ``to_private`` keys go to the private L1, the rest to shared."""

    def answer(keys: list[ObjectKey], managers: list) -> dict[int, list[ObjectKey]]:
        shared, private = (manager.l1_manager_id for manager in managers)
        targets: dict[int, list[ObjectKey]] = {}
        for k in keys:
            targets.setdefault(private if k in to_private else shared, []).append(k)
        return targets

    monkeypatch.setattr(ScriptedWritePolicy, "answer", answer)


def prefetch(store: StorageManager, spec: PrefetchTaskSpec) -> PrefetchResult:
    """Submit a prefetch and wait for its result."""
    handle = store.submit_prefetch_task(spec)
    assert store.wait_prefetch_status(handle, timeout=10)
    result = store.query_prefetch_status(handle)
    assert result is not None
    return result


@pytest.mark.parametrize("shared_first", [False, True])
def test_shared_region_cannot_overlap_a_private_arena(
    orchestrator: Orchestrator, region: Path, shared_first: bool
) -> None:
    shared = shared_config(region, orchestrator.endpoint, "owner")
    private = replace(shared, tag="private", shared=None)
    with pytest.raises(ValueError, match="already owned by an L1"):
        storage(*([shared, private] if shared_first else [private, shared]))


@pytest.mark.parametrize("capacity", [2 * 4096])
def test_mixed_owners_complete_per_l1_and_a_full_region_reserves_nothing(
    make_store: Callable[..., StorageManager],
    usage: Callable[[], RegionUsage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, peer = make_store("writer", SCRIPTED_WRITE_POLICY), make_store("peer")
    route(monkeypatch, {key(2)})
    # An aborted store leaves the key writable; its extent stays consumed.
    aborted = writer.reserve_write([key(1)], LAYOUT)
    writer.abort_write(list(aborted), l1_owners_of(aborted))
    assert usage().writing == 0
    keys = [key(1), key(2)]
    put(writer, keys, LAYOUT)
    # The region is full: the policy's choice stands, nothing moves to DRAM.
    assert writer.reserve_write([key(3)], LAYOUT) == {}
    own = prefetch(writer, single_row_spec(keys + [key(3)], LAYOUT))
    remote = prefetch(peer, single_row_spec(keys + [key(3)], LAYOUT))
    assert own.hit_cells[0].get_indices_list() == [0, 1]
    assert remote.hit_cells[0].get_indices_list() == [0]
    assert own.l1_owners[key(1)] != own.l1_owners[key(2)]
    with writer.read_prefetched_results(keys, own.l1_owners) as objects:
        assert objects is not None
        assert all(torch.all(o.tensor == i) for i, o in enumerate(objects, 1))
    writer.finish_read_prefetched(keys, l1_owners=own.l1_owners)
    peer.finish_read_prefetched(keys[:1], l1_owners=remote.l1_owners)
    assert usage().read_leases == 0


@pytest.mark.parametrize("capacity", [2 * 4096])
def test_grouped_rows_split_between_private_and_shared_l1s(
    make_store: Callable[..., StorageManager],
    usage: Callable[[], RegionUsage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = MemoryLayoutDesc([torch.Size([500])], [torch.float16])
    rows = [
        GroupedObjectKeys([key(10 * group), key(10 * group + 1)], group, layout)
        for group, layout in enumerate((LAYOUT, other))
    ]
    keys = [k for row in rows for k in row.keys]
    writer = make_store("writer", SCRIPTED_WRITE_POLICY)
    route(monkeypatch, set(rows[0].keys))
    # Group 1 fills the region; group 0 goes to private DRAM.
    put(writer, rows[1].keys, other)
    put(writer, rows[0].keys, LAYOUT)
    result = prefetch(writer, PrefetchTaskSpec(rows))
    assert [b.get_indices_list() for b in result.hit_cells] == [[0, 1]] * 2
    assert len({result.l1_owners[k] for k in keys}) == 2
    writer.finish_read_prefetched(keys, l1_owners=result.l1_owners)
    # A wrong shared layout releases every lock taken, the private ones included.
    wrong = GroupedObjectKeys(rows[1].keys, 1, LAYOUT)
    with pytest.raises(ValueError, match="reader's layout"):
        writer.submit_prefetch_task(PrefetchTaskSpec([rows[0], wrong]))
    assert usage().read_leases == 0
    assert writer.report_status()["l1_managers"]["private"]["read_locked_count"] == 0


def test_l2_affinity_and_tag_filters_next_to_a_shared_l1(
    orchestrator: Orchestrator, region: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = shared_config(region, orchestrator.endpoint, "owner")
    private = L1ManagerConfig(
        L1MemoryManagerConfig(8192, use_lazy=False, shm_name=""), tag="_default"
    )
    remote = MemoryTags(MemorySegment.CXL, MemoryResidency.REMOTE)
    local = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
    # A default adapter wants DRAM/LOCAL first and binds to the private L1.
    with closing(storage(shared, private, l2=True)) as store:
        put(store, [key(1)], LAYOUT)
        for allowed in (None, frozenset({remote})):
            spec = single_row_spec([key(1)], LAYOUT, allowed_l1_memory_tags=allowed)
            result = prefetch(store, spec)
            assert result.l1_hit_cells[0].get_indices_list() == [0]
            store.finish_read_prefetched([key(1)], l1_owners=result.l1_owners)
        # Filters that exclude CXL/REMOTE never ask the orchestrator.
        lookups = Mock(side_effect=AssertionError("shared L1 was looked up"))
        monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", lookups)
        for allowed in (frozenset({local}), frozenset()):
            spec = single_row_spec([key(1)], LAYOUT, allowed_l1_memory_tags=allowed)
            assert prefetch(store, spec).l1_hit_cells[0].get_indices_list() == []
    monkeypatch.setattr(
        MockL2AdapterConfig, "get_wanted_l1_memory_tags", lambda self: (remote,)
    )
    with pytest.raises(ValueError, match="bind only to private L1s"):
        storage(shared, l2=True)
