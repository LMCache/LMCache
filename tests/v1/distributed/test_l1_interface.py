# SPDX-License-Identifier: Apache-2.0
"""The L1ManagerInterface contract, run against each binding of the interface."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface, L1ManagerListener
from lmcache.v1.distributed.l1_manager import DevDaxL1Manager, DramL1Manager
import lmcache.v1.memory_management as memory_management

ALIGN = 4096
CAPACITY = 32 * ALIGN
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])  # 4 KiB


def key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "conformance", 0)


class RecordingListener(L1ManagerListener):
    def __init__(self) -> None:
        self.written: list[ObjectKey] = []
        self.read_reserved: list[ObjectKey] = []
        self.read_finished: list[ObjectKey] = []

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        self.read_reserved.extend(keys)

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        self.read_finished.extend(keys)

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        self.written.extend(keys)

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass


@pytest.fixture
def l1(monkeypatch: pytest.MonkeyPatch) -> Iterator[L1ManagerInterface]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    embedded = DramL1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=CAPACITY, use_lazy=False, shm_name="")
        )
    )
    yield embedded
    embedded.close()


def test_write_commit_read_release(l1: L1ManagerInterface) -> None:
    assert isinstance(l1, L1ManagerInterface)
    listener = RecordingListener()
    l1.register_listener(listener)

    error, obj = l1.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    assert obj.get_l1_manager() == l1.l1_manager_id
    assert l1.get_staging_memory_usage() > 0
    assert l1.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.KEY_NOT_WRITABLE
    )
    assert l1.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    obj.tensor.fill_(7.0)

    assert l1.finish_write([key(1)]) == {key(1): L1Error.SUCCESS}
    assert listener.written == [key(1)]
    assert l1.get_staging_memory_usage() == 0
    assert l1.finish_write([key(1)]) == {key(1): L1Error.KEY_NOT_EXIST}
    assert l1.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.KEY_NOT_WRITABLE
    )

    error, view = l1.reserve_read([key(1)], read_locks=2)[key(1)]
    assert error is L1Error.SUCCESS and view is not None and view.tensor is not None
    assert torch.all(view.tensor == 7.0)
    assert l1.unsafe_read([key(1)])[key(1)] == (L1Error.SUCCESS, view)
    assert l1.finish_read([key(1)], read_locks=2) == {key(1): L1Error.SUCCESS}
    assert l1.unsafe_read([key(1)])[key(1)][0] is not L1Error.SUCCESS
    assert listener.read_reserved == [key(1)]
    assert listener.read_finished == [key(1)]


def test_unregister_waits_for_active_read_callback(l1: L1ManagerInterface) -> None:
    obj_key = key(1)
    l1.reserve_write([obj_key], [False], LAYOUT)
    l1.finish_write_and_reserve_read([obj_key])
    callback_started = Event()
    release_callback = Event()
    removing = Event()

    class BlockingListener(RecordingListener):
        def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
            callback_started.set()
            assert release_callback.wait(timeout=5)
            super().on_l1_keys_read_finished(keys)

    listener = BlockingListener()
    l1.register_listener(listener)

    def unregister() -> None:
        removing.set()
        l1.unregister_listener(listener)
        assert listener.read_finished == [obj_key]

    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(l1.finish_read, [obj_key])
        try:
            assert callback_started.wait(timeout=5)
            removal = pool.submit(unregister)
            assert removing.wait(timeout=5)
            with pytest.raises(TimeoutError):
                removal.result(timeout=0.1)
        finally:
            release_callback.set()
        assert read.result(timeout=5) == {obj_key: L1Error.SUCCESS}
        removal.result(timeout=5)

    assert l1.reserve_read([obj_key])[obj_key][0] is L1Error.SUCCESS
    assert l1.finish_read([obj_key]) == {obj_key: L1Error.SUCCESS}
    assert listener.read_finished == [obj_key]


def test_aborted_write_is_never_readable(l1: L1ManagerInterface) -> None:
    assert l1.reserve_write([key(2)], [False], LAYOUT)[key(2)][0] is L1Error.SUCCESS
    assert l1.finish_write_and_delete([key(2)]) == {key(2): L1Error.SUCCESS}
    assert l1.reserve_read([key(2)])[key(2)] == (L1Error.KEY_NOT_EXIST, None)
    assert l1.finish_write([key(2)]) == {key(2): L1Error.KEY_NOT_EXIST}
    assert l1.reserve_write([key(2)], [False], LAYOUT)[key(2)][0] is L1Error.SUCCESS


def test_finish_write_and_reserve_read_hands_back_a_locked_object(
    l1: L1ManagerInterface,
) -> None:
    _, obj = l1.reserve_write([key(3)], [False], LAYOUT)[key(3)]
    assert obj is not None and obj.tensor is not None
    obj.tensor.fill_(3.0)
    error, view = l1.finish_write_and_reserve_read([key(3)], read_locks=1)[key(3)]
    assert error is L1Error.SUCCESS and view is not None and view.tensor is not None
    assert torch.all(view.tensor == 3.0)
    assert l1.finish_read([key(3)]) == {key(3): L1Error.SUCCESS}


def test_tags_are_independent_writers(l1: L1ManagerInterface) -> None:
    assert l1.reserve_write([key(4)], [False], LAYOUT, tag="a")[key(4)][0] is (
        L1Error.SUCCESS
    )
    # Only tag "a" can finish its reservation.
    assert l1.finish_write([key(4)], tag="b") == {key(4): L1Error.KEY_NOT_EXIST}
    assert l1.finish_write([key(4)], tag="a") == {key(4): L1Error.SUCCESS}


def test_medium_specific_calls_are_not_part_of_the_interface() -> None:
    # Device-DAX hot-plug and device ownership live on DevDaxL1Manager only.
    for name in (
        "owns_range",
        "add_devdax_device",
        "remove_devdax_device",
        "get_devdax_arena_statuses",
        "get_devdax_arena_status",
        "memory_region_count",
    ):
        assert not hasattr(L1ManagerInterface, name)
        assert hasattr(DevDaxL1Manager, name) or name == "memory_region_count"


def test_resident_key_needs_allow_resident_to_stage(l1: L1ManagerInterface) -> None:
    """A resident key is not writable by default. With ``allow_resident`` it
    is staged, and admission keeps the resident object, read-locks it, and
    frees the redundant staging buffer."""
    l1.reserve_write([key(5)], [False], LAYOUT, tag="a")
    assert l1.finish_write([key(5)], tag="a") == {key(5): L1Error.SUCCESS}
    resident = l1.reserve_read([key(5)])[key(5)][1]
    l1.finish_read([key(5)])

    assert l1.reserve_write([key(5)], [False], LAYOUT, tag="b")[key(5)][0] is (
        L1Error.KEY_NOT_WRITABLE
    )
    staged = l1.reserve_write([key(5)], [False], LAYOUT, tag="b", allow_resident=True)
    assert staged[key(5)][0] is L1Error.SUCCESS
    assert staged[key(5)][1] is not resident
    assert l1.get_staging_memory_usage() > 0

    admitted = l1.finish_write_and_reserve_read([key(5)], tag="b")
    assert admitted[key(5)] == (L1Error.SUCCESS, resident)
    assert l1.get_staging_memory_usage() == 0
    assert l1.is_key_evictable(key(5)) is False
    l1.finish_read([key(5)])
    assert l1.is_key_evictable(key(5)) is True


def test_prefetch_locked_bytes_follow_the_read_lock(l1: L1ManagerInterface) -> None:
    keys = [key(1), key(2), key(3)]
    staged = l1.reserve_write(keys, [False] * 3, LAYOUT, tag="pf")
    _error, staged_obj = staged[key(1)]
    assert staged_obj is not None
    size = staged_obj.get_size()
    l1.finish_write_and_reserve_read(keys[:2], tag="pf")
    l1.finish_write(keys[2:], tag="pf")  # resident but not read-locked
    l1.track_prefetch_locked(keys + [key(4)])  # key(4) is not resident
    l1.track_prefetch_locked([key(1)])  # charged once
    assert l1.get_prefetch_locked_bytes() == 2 * size

    l1.finish_read([key(1)])
    assert l1.get_prefetch_locked_bytes() == size
    l1.delete([key(2)], force=True)
    assert l1.get_prefetch_locked_bytes() == 0


def test_prefetch_locked_ends_with_the_prefetch_read(l1: L1ManagerInterface) -> None:
    """Once the prefetch's reader is done, a later lock on the same object by
    another reader (e.g. a store to L2) is not counted as prefetch-locked."""
    l1.reserve_write([key(5)], [False], LAYOUT, tag="pf")
    l1.finish_write_and_reserve_read([key(5)], tag="pf")
    l1.track_prefetch_locked([key(5)])
    l1.finish_read([key(5)])

    assert l1.reserve_read([key(5)])[key(5)][0] is L1Error.SUCCESS
    assert l1.get_prefetch_locked_bytes() == 0
    l1.finish_read([key(5)])


def test_prefetch_budget_uses_logical_bytes_for_mixed_objects(
    l1: L1ManagerInterface,
) -> None:
    keys = [key(6), key(7)]
    large = MemoryLayoutDesc([torch.Size([2 * ALIGN])], [torch.uint8])
    l1.reserve_write(keys[:1], [False], large)
    _, obj = l1.reserve_write(keys[1:], [False], LAYOUT)[keys[1]]
    assert isinstance(obj, memory_management.TensorMemoryObj)
    obj.set_used_size(17)
    l1.finish_write_and_reserve_read(keys)
    l1.track_prefetch_locked(keys)
    assert not l1.prefetch_locked_bytes_exceed(2 * ALIGN + 17)
    assert l1.prefetch_locked_bytes_exceed(2 * ALIGN + 16)
    l1.finish_read(keys)
