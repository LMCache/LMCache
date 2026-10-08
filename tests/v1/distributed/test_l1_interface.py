# SPDX-License-Identifier: Apache-2.0
"""The L1ManagerInterface contract, run against each binding of the interface."""

# Standard
from collections.abc import Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface, L1ManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

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
    embedded = L1Manager(
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
