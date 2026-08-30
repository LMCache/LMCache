# SPDX-License-Identifier: Apache-2.0
"""Partial-prefix allocation behavior for ``L1Manager.reserve_write``."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.mp_observability.event_bus import EventType

POOL_BYTES = 64 * 1024 * 1024
OBJECT_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([2048, 1024])],
    dtypes=[torch.float32],
)
OVERSIZED_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([2048 * 16, 1024])],
    dtypes=[torch.float32],
)


class _RecordingListener:
    def __init__(self) -> None:
        self.reserved_write: list[ObjectKey] = []

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        self.reserved_write.extend(keys)

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass


class _RecordingEventBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def publish(self, event: Any) -> None:
        self.events.append(event)


@pytest.fixture
def manager() -> Iterator[L1Manager]:
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=POOL_BYTES,
            use_lazy=False,
            init_size_in_bytes=POOL_BYTES,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    instance = L1Manager(config)
    yield instance
    instance.close()


def _make_keys(count: int) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(index),
            model_name="partial-allocation-test",
            kv_rank=0,
        )
        for index in range(count)
    ]


def test_reserve_write_allocates_longest_prefix_on_oom(manager: L1Manager) -> None:
    """Only the unallocatable tail fails, and observers see the prefix."""
    keys = _make_keys(10)
    listener = _RecordingListener()
    event_bus = _RecordingEventBus()
    manager.register_listener(cast(L1ManagerListener, listener))
    manager._event_bus = cast(Any, event_bus)

    result = manager.reserve_write(keys, [False] * len(keys), OBJECT_LAYOUT)

    assert set(result) == set(keys)
    statuses = [result[key][0] for key in keys]
    num_success = statuses.count(L1Error.SUCCESS)
    assert 0 < num_success < len(keys)
    assert statuses == [L1Error.SUCCESS] * num_success + [L1Error.OUT_OF_MEMORY] * (
        len(keys) - num_success
    )

    successful_keys = keys[:num_success]
    assert all(result[key][1] is not None for key in successful_keys)
    assert all(result[key][1] is None for key in keys[num_success:])
    assert listener.reserved_write == successful_keys
    assert event_bus.events[-1].event_type == EventType.L1_WRITE_RESERVED
    assert event_bus.events[-1].metadata["keys"] == successful_keys


def test_partial_prefix_remains_committable(manager: L1Manager) -> None:
    keys = _make_keys(10)
    result = manager.reserve_write(keys, [False] * len(keys), OBJECT_LAYOUT)
    successful_keys = [key for key in keys if result[key][0] == L1Error.SUCCESS]

    assert successful_keys
    assert manager.finish_write(successful_keys) == {
        key: L1Error.SUCCESS for key in successful_keys
    }


@pytest.mark.parametrize("count", [1, 3])
def test_reserve_write_reports_all_oom_when_no_object_fits(
    manager: L1Manager, count: int
) -> None:
    keys = _make_keys(count)

    result = manager.reserve_write(keys, [False] * count, OVERSIZED_LAYOUT)

    assert result == {key: (L1Error.OUT_OF_MEMORY, None) for key in keys}


def test_largest_prefix_probe_finds_the_maximum_fit() -> None:
    """Probe allocation instead of relying on an inaccurate byte estimate."""

    class BoundedMemoryManager:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity
            self.allocate_counts: list[int] = []
            self.freed_counts: list[int] = []

        def allocate(
            self, _layout: MemoryLayoutDesc, count: int
        ) -> tuple[L1Error, list[Any]]:
            self.allocate_counts.append(count)
            if count > self.capacity:
                return L1Error.OUT_OF_MEMORY, []
            return L1Error.SUCCESS, [object() for _ in range(count)]

        def free(self, objects: list[Any]) -> L1Error:
            self.freed_counts.append(len(objects))
            return L1Error.SUCCESS

    memory_manager = BoundedMemoryManager(capacity=7)
    manager = SimpleNamespace(_memory_manager=memory_manager)

    err, objects = L1Manager._allocate_largest_prefix(
        manager,  # type: ignore[arg-type]
        OBJECT_LAYOUT,
        10,
    )

    assert err == L1Error.SUCCESS
    assert len(objects) == 7
    assert memory_manager.allocate_counts[-1] == 7
    assert 8 in memory_manager.allocate_counts
    assert memory_manager.freed_counts == [5, 7]
