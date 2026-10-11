# SPDX-License-Identifier: Apache-2.0
"""DAX task and StorageManager contracts using isolated temporary mmap files.

Run with RUN_DAX_L2_INTEGRATION=1; no DAX device or virtual machine is needed.
"""

# Standard
from pathlib import Path
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.dax_l2_adapter import (
    DaxDeviceConfig,
    DaxL2Adapter,
    DaxL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.reconfiguration import L2ReconfigureError
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj
from tests.v1.distributed.test_dax_l2_adapter import (
    bitmap_to_bools,
    create_memory_obj,
    create_object_key,
    load_and_wait,
    lookup_and_wait,
    store_and_wait,
    wait_for_condition,
    wait_for_event_fd,
)
from tests.v1.distributed.utils import single_row_spec

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        os.environ.get("RUN_DAX_L2_INTEGRATION") != "1",
        reason="set RUN_DAX_L2_INTEGRATION=1",
    ),
]

SLOT_BYTES = 2 << 20


def _device(workspace: Path, name: str, slots: int) -> str:
    """Create a file in the test workspace with capacity for ``slots`` payloads."""
    path = workspace / name
    with path.open("xb") as device:
        device.truncate(slots * SLOT_BYTES)
    return str(path)


def _config(path: str, slot: int, slots: int = 2) -> DaxL2AdapterConfig:
    """Configure a hotpluggable adapter mapping ``slots`` fixed-size payloads."""
    return DaxL2AdapterConfig(
        devices=[
            DaxDeviceConfig(device_path=path, max_dax_size_gb=slot * slots / 1024**3)
        ],
        slot_bytes=slot,
        hotplug_enabled=True,
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )


def _object(slot: int, seed: int) -> MemoryObj:
    """Allocate a two-dimensional byte payload with a deterministic pattern."""
    obj = create_memory_obj(shape=torch.Size([2, slot // 2]), dtype=torch.uint8)
    assert obj.raw_tensor is not None
    obj.raw_tensor.copy_(torch.arange(slot, dtype=torch.uint8) + seed)
    return obj


def _check(obj: MemoryObj, slot: int, seed: int) -> None:
    """Check the complete payload and layout against the original pattern."""
    assert obj.get_shapes() == [torch.Size([2, slot // 2])]
    assert obj.get_dtypes() == [torch.uint8]
    assert torch.equal(obj.raw_tensor, torch.arange(slot, dtype=torch.uint8) + seed)


def _storage_roundtrip(tmp_path: Path, dax_l1: bool) -> None:
    """Store through L2, evict the L1 copy, then verify the prefetched bytes."""
    slot = SLOT_BYTES
    path = _device(tmp_path, "l2", 2)
    memory = L1MemoryManagerConfig(
        size_in_bytes=2 * slot,
        use_lazy=False,
        shm_name="",
        align_bytes=4096,
        devdax_path=_device(tmp_path, "l1", 2) if dax_l1 else "",
    )
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=memory, write_ttl_seconds=60, read_ttl_seconds=60
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        l2_adapter_config=L2AdaptersConfig([_config(path, slot)]),
        store_policy="skip_l1",
    )
    manager = StorageManager(config)
    key = create_object_key(40)
    layout = MemoryLayoutDesc(shapes=[torch.Size([2, slot // 2])], dtypes=[torch.uint8])
    try:
        reserved = manager.reserve_write([key], layout)
        tensor = reserved[key].raw_tensor
        assert tensor is not None
        tensor.copy_(torch.arange(slot, dtype=torch.uint8) + 73)
        del tensor
        manager.finish_write([key])
        del reserved
        adapter = manager.l2_adapters()[0][1]
        assert wait_for_condition(
            lambda: (
                manager.report_status()["l1_manager"]["total_object_count"] == 0
                and adapter.get_usage().total_bytes_used == slot
            )
        )
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        assert manager.wait_prefetch_status(handle, timeout=5.0)
        result = manager.query_prefetch_status(handle)
        assert result is not None
        assert result.hit_cells[0].count_leading_ones() == 1
        assert result.l2_hit_cells[0].count_leading_ones() == 1
        with manager.read_prefetched_results([key]) as objects:
            assert objects is not None and len(objects) == 1
            _check(objects[0], slot, 73)
        del objects
        manager.finish_read_prefetched([key])
    finally:
        manager.close()


def test_async_capacity_locks_and_volatile_reopen(tmp_path: Path) -> None:
    """Task results are one-shot; locks protect bytes and full arenas reject writes."""
    slot = SLOT_BYTES
    path = _device(tmp_path, "dax", 2)
    config = _config(path, slot)
    adapter = DaxL2Adapter(config)
    objects = [_object(slot, seed) for seed in (3, 9, 21)]
    targets = [_object(slot, 0) for _ in range(3)]
    keys = [create_object_key(i) for i in range(3)]
    fds = [
        adapter.get_store_event_fd(),
        adapter.get_load_event_fd(),
        adapter.get_lookup_and_lock_event_fd(),
    ]
    try:
        task = adapter.submit_store_task(keys[:2], objects[:2])
        assert wait_for_event_fd(fds[0])
        assert adapter.pop_completed_store_tasks()[task].is_successful()
        assert adapter.pop_completed_store_tasks() == {}
        task = adapter.submit_store_task(keys[2:], objects[2:])
        assert wait_for_event_fd(fds[0])
        assert not adapter.pop_completed_store_tasks()[task].is_successful()
        assert adapter.get_usage().total_bytes_used == 2 * slot

        layout = MemoryLayoutDesc(
            shapes=objects[0].get_shapes(), dtypes=objects[0].get_dtypes()
        )
        task = adapter.submit_lookup_and_lock_task(keys, {0: layout})
        assert wait_for_event_fd(fds[2])
        result = adapter.query_lookup_and_lock_result(task)
        assert result is not None
        assert bitmap_to_bools(result, 3) == [True, True, False]
        assert adapter.query_lookup_and_lock_result(task) is None
        adapter.delete(keys)
        assert adapter.get_usage().total_bytes_used == 2 * slot
        task = adapter.submit_load_task(keys, targets)
        assert wait_for_event_fd(fds[1])
        result = adapter.query_load_result(task)
        assert result is not None
        assert bitmap_to_bools(result, 3) == [True, True, False]
        assert adapter.query_load_result(task) is None
        for obj, seed in zip(targets, (3, 9, 0), strict=True):
            _check(obj, slot, seed)
        adapter.submit_unlock(keys[:2])
        adapter.delete(keys[:1])
        assert adapter.get_usage().total_bytes_used == slot
        store_and_wait(adapter, keys[2], objects[2])
        # close must settle an outstanding copy before returning.
        adapter.submit_load_task(keys[1:], targets[1:])
        adapter.close()
        _check(targets[1], slot, 9)
        _check(targets[2], slot, 21)
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        adapter = DaxL2Adapter(config)
        assert lookup_and_wait(adapter, keys) == [False] * 3
        assert adapter.get_usage().total_bytes_used == 0
    finally:
        adapter.close()
        for obj in objects + targets:
            obj.ref_count_down()


def test_runtime_add_drain_migrate_and_blocked_remove(tmp_path: Path) -> None:
    """Adding capacity, draining and migration retain every byte and lock lifetime."""
    slot = SLOT_BYTES
    source = _device(tmp_path, "source", 1)
    destination = _device(tmp_path, "destination", 2)
    adapter = DaxL2Adapter(_config(source, slot, 1))
    obj, target = _object(slot, 37), _object(slot, 0)
    first, second = create_object_key(10), create_object_key(11)
    try:
        store_and_wait(adapter, first, obj)
        adapter.hotplug_add_device(destination, 2 * slot)
        assert lookup_and_wait(adapter, [first]) == [True]
        with pytest.raises(L2ReconfigureError) as error:
            adapter.hotplug_remove_device(source, "migrate")
        assert error.value.status_code == 409
        assert adapter.hotplug_status()["devices"][0]["state"] == "active"
        adapter.submit_unlock([first])
        assert adapter.hotplug_remove_device(source, "drain")["state"] == "draining"
        store_and_wait(adapter, second, obj)
        status = adapter.hotplug_status()["devices"]
        assert [d["live_slot_count"] for d in status] == [1, 1]
        assert load_and_wait(adapter, [first], [target]) == [True]
        _check(target, slot, 37)
        result = adapter.hotplug_remove_device(source, "migrate")
        assert result["state"] == "removed"
        assert result["moved_keys"] == 1
        for key in (first, second):
            assert lookup_and_wait(adapter, [key]) == [True]
            assert load_and_wait(adapter, [key], [target]) == [True]
            _check(target, slot, 37)
            adapter.submit_unlock([key])
        assert adapter.get_usage().total_bytes_used == 2 * slot
    finally:
        adapter.close()
        obj.ref_count_down()
        target.ref_count_down()


def test_mapping_resize(tmp_path: Path) -> None:
    """Grow/shrink the mapping within fixed capacity; blocked shrink retains data."""
    slot = SLOT_BYTES
    path = _device(tmp_path, "dax", 4)
    adapter = DaxL2Adapter(_config(path, slot))
    obj, target = _object(slot, 51), _object(slot, 0)
    keys = [create_object_key(20 + i) for i in range(3)]
    try:
        for key in keys[:2]:
            store_and_wait(adapter, key, obj)
        adapter.hotplug_resize_device(path, 4 * slot, "migrate")
        store_and_wait(adapter, keys[2], obj)
        assert lookup_and_wait(adapter, keys[2:]) == [True]
        with pytest.raises(L2ReconfigureError) as error:
            adapter.hotplug_resize_device(path, 2 * slot, "migrate")
        assert error.value.status_code == 409
        assert adapter.get_usage().total_capacity_bytes == 4 * slot
        adapter.submit_unlock(keys[2:])
        adapter.delete(keys[2:])
        adapter.hotplug_resize_device(path, 2 * slot, "migrate")
        assert adapter.get_usage().total_capacity_bytes == 2 * slot
        for key in keys[:2]:
            assert load_and_wait(adapter, [key], [target]) == [True]
            _check(target, slot, 51)
    finally:
        adapter.close()
        obj.ref_count_down()
        target.ref_count_down()


def test_storage_manager_dram_l1_roundtrip(tmp_path: Path) -> None:
    """Retrieve from DAX L2 after skip_l1 has removed the resident DRAM copy."""
    _storage_roundtrip(tmp_path, False)


def test_storage_manager_combined_dax_roundtrip(tmp_path: Path) -> None:
    """Round-trip through DAX L1 and L2 using distinct temporary mappings."""
    _storage_roundtrip(tmp_path, True)
