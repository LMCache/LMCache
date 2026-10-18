# SPDX-License-Identifier: Apache-2.0
"""Native L1 allocation, locking and background byte-budget regressions."""

# Standard
from collections.abc import Iterator
from queue import Queue
from typing import Literal
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.memory_management import MemoryObj

pytestmark = pytest.mark.no_shared_allocator


def key(index: int) -> ObjectKey:
    """Return a distinct synthetic object group and cache identity."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "model", 0, object_group_id=index)


def layout(size: int) -> MemoryLayoutDesc:
    """Return a byte tensor layout of the requested logical payload size."""
    return MemoryLayoutDesc([torch.Size([size])], [torch.uint8])


def write(manager: L1Manager, index: int, size: int) -> MemoryObj:
    """Allocate, populate and finish one public L1 write."""
    result, obj = manager.reserve_write([key(index)], [False], layout(size))[key(index)]
    assert result == L1Error.SUCCESS and obj is not None
    assert obj.tensor is not None
    obj.tensor.fill_(index)
    assert manager.finish_write([key(index)])[key(index)] == L1Error.SUCCESS
    return obj


@pytest.fixture(params=[600])
def manager(request: pytest.FixtureRequest) -> Iterator[L1Manager]:
    """Yield a real aligned allocator, with the requested writer TTL."""
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=81920, use_lazy=False, align_bytes=4096, shm_name=""
            ),
            write_ttl_seconds=int(request.param),
        )
    )
    try:
        yield manager
    finally:
        manager.close()


@pytest.mark.parametrize("policy", ["LRU", "ARC"])
@pytest.mark.parametrize(
    "sizes",
    [[4096] * 7 + [36864], [8192] * 8, [1] * 7 + [32769]],
    ids=["mixed", "uniform", "padding"],
)
def test_background_cycle_reclaims_allocated_byte_fraction(
    manager: L1Manager,
    monkeypatch: pytest.MonkeyPatch,
    policy: Literal["LRU", "ARC"],
    sizes: list[int],
) -> None:
    """The first background eviction batch releases 16 KiB, including padding."""
    completed: Queue[dict[ObjectKey, L1Error]] = Queue()
    original_delete = manager.delete

    def observe_delete(
        keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        result = original_delete(keys, force=force)
        completed.put(result)
        return result

    # Observe the real public deletion without replacing global event state
    # or starting another consumer of the process-wide native event queue.
    monkeypatch.setattr(manager, "delete", observe_delete)
    controller = L1EvictionController(
        manager, EvictionConfig(policy, trigger_watermark=0.65, eviction_ratio=0.25)
    )
    physical_sizes = {
        key(index): write(manager, index, size).get_physical_size()
        for index, size in enumerate(sizes)
    }
    assert manager.get_memory_usage() == (65536, 81920)
    try:
        controller.start()
        result = completed.get(timeout=5)
        victims = [
            victim for victim, error in result.items() if error == L1Error.SUCCESS
        ]
        assert sum(physical_sizes[victim] for victim in victims) == 16384
        expected = 2 if sizes[0] == 8192 else 4
        assert victims == [key(index) for index in range(expected)]
    finally:
        controller.stop()


def test_evictable_size_counts_padding_and_rechecks_read_lock(
    manager: L1Manager,
) -> None:
    """Advisory sizing cannot authorize deleting a subsequently locked object."""
    write(manager, 0, 1)
    assert manager.get_evictable_size(key(0)) == 4096
    assert manager.reserve_read([key(0)])[key(0)][0] == L1Error.SUCCESS
    assert manager.get_evictable_size(key(0)) == 0
    assert manager.delete([key(0)])[key(0)] == L1Error.KEY_IS_LOCKED
    assert manager.finish_read([key(0)])[key(0)] == L1Error.SUCCESS
    assert manager.get_evictable_size(key(0)) == 4096
    assert manager.delete([key(0)])[key(0)] == L1Error.SUCCESS
    assert manager.get_evictable_size(key(0)) == 0
    assert manager.get_memory_usage()[0] == 0


def test_live_staging_protects_resident_bytes(manager: L1Manager) -> None:
    """An unlocked resident is not reclaimable while a new write is live."""
    admitted = manager.reserve_write([key(0)], [False], layout(1), tag="admit")
    assert admitted[key(0)][0] == L1Error.SUCCESS
    reserved = manager.reserve_write([key(0)], [False], layout(1), tag="writer")
    assert reserved[key(0)][0] == L1Error.SUCCESS
    assert manager.finish_write([key(0)], tag="admit")[key(0)] == L1Error.SUCCESS
    assert manager.get_evictable_size(key(0)) == 0
    assert manager.delete([key(0)])[key(0)] == L1Error.KEY_IS_LOCKED
    assert manager.get_memory_usage()[0] == 8192


@pytest.mark.parametrize("manager", [1], indirect=True)
def test_expired_staging_counts_only_bytes_delete_can_free(manager: L1Manager) -> None:
    """Expired buffers count even when another writer still protects a key."""
    old = manager.reserve_write([key(0)], [False], layout(1), tag="old")
    assert old[key(0)][0] == L1Error.SUCCESS
    time.sleep(1.1)
    live = manager.reserve_write([key(0)], [False], layout(1), tag="live")
    assert live[key(0)][0] == L1Error.SUCCESS
    admitted = manager.reserve_write([key(0)], [False], layout(1), tag="admit")
    assert admitted[key(0)][0] == L1Error.SUCCESS
    assert manager.finish_write([key(0)], tag="admit")[key(0)] == L1Error.SUCCESS
    assert manager.reserve_read([key(0)])[key(0)][0] == L1Error.SUCCESS
    before = manager.get_memory_usage()[0]
    reclaimable = manager.get_evictable_size(key(0))
    assert reclaimable == 4096
    assert manager.delete([key(0)])[key(0)] == L1Error.KEY_IS_LOCKED
    assert before - manager.get_memory_usage()[0] == reclaimable
    assert manager.get_evictable_size(key(0)) == 0


@pytest.mark.parametrize("manager", [1], indirect=True)
def test_expired_staging_and_unlocked_resident_sizes_add_up(manager: L1Manager) -> None:
    """Count both allocations for a key once all its staging locks expire."""
    for tag in ("old", "admit"):
        reserved = manager.reserve_write([key(0)], [False], layout(1), tag=tag)
        assert reserved[key(0)][0] == L1Error.SUCCESS
    assert manager.finish_write([key(0)], tag="admit")[key(0)] == L1Error.SUCCESS
    other = manager.reserve_write([key(1)], [False], layout(1), tag="old")
    assert other[key(1)][0] == L1Error.SUCCESS
    time.sleep(1.1)
    assert manager.get_evictable_size(key(0)) == 8192
    assert manager.get_evictable_size(key(1)) == 4096
    assert manager.delete([key(0), key(1)]) == {
        key(0): L1Error.SUCCESS,
        key(1): L1Error.SUCCESS,
    }
    assert manager.get_memory_usage()[0] == 0
