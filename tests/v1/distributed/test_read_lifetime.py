# SPDX-License-Identifier: Apache-2.0
"""Public lifecycle contracts for named lookup leases and device acquisitions."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.memory_management import MemoryObj

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([128])], dtypes=[torch.float32])
KEY = ObjectKey(ObjectKey.IntHash2Bytes(1), "lifetime", 0)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Advance named lease deadlines without sleeping or changing system time."""
    now = [100.0]
    monkeypatch.setattr(
        "lmcache.v1.distributed.l1_manager.base.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    return now


@pytest.fixture
def manager(clock: list[float]) -> Iterator[DramL1Manager]:
    """Use the real L1 state machine and a small preallocated memory pool."""
    instance = DramL1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False),
            read_ttl_seconds=1,
        )
    )
    yield instance
    instance.close()


def admit(manager: DramL1Manager, temporary: bool = False) -> None:
    """Commit a known object through the public write API."""
    error, obj = manager.reserve_write([KEY], [temporary], LAYOUT)[KEY]
    assert error == L1Error.SUCCESS and obj is not None
    assert manager.finish_write([KEY])[KEY] == L1Error.SUCCESS


def acquire(
    manager: DramL1Manager, owner: str, worker: int = 0
) -> tuple[int, MemoryObj]:
    """Claim one share and assert the documented successful result."""
    error, acquired = manager.acquire_read([KEY], owner, worker)[KEY]
    assert error == L1Error.SUCCESS and acquired is not None
    return acquired


def test_expired_reader_cannot_borrow_or_release_new_lookup(
    manager: DramL1Manager,
    clock: list[float],
) -> None:
    """An old lookup neither acquires nor releases another lookup's shares."""
    admit(manager)
    manager.reserve_read([KEY], read_owner="old")
    clock[0] += 1.2
    manager.reserve_read([KEY], read_owner="new")
    assert manager.acquire_read([KEY], "old", 0)[KEY][0] == L1Error.KEY_NOT_READABLE
    manager.finish_read([KEY], read_owner="old")
    manager.finish_read([KEY], read_owner="old")
    handle, _ = acquire(manager, "new")
    manager.finish_active_reads([handle])
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS


def test_each_lookup_has_its_own_deadline(
    manager: DramL1Manager,
    clock: list[float],
) -> None:
    """A later reader does not extend the earlier reader's queue lease."""
    admit(manager)
    manager.reserve_read([KEY], read_owner="a")
    clock[0] += 0.6
    manager.reserve_read([KEY], read_owner="b")
    clock[0] += 0.6
    assert manager.acquire_read([KEY], "a", 0)[KEY][0] == L1Error.KEY_NOT_READABLE
    handle, _ = acquire(manager, "b")
    manager.finish_active_reads([handle])


def test_active_read_survives_expiry_cancel_clear_and_forced_delete(
    manager: DramL1Manager,
    clock: list[float],
) -> None:
    """Only completion, rather than a deadline or request cleanup, frees storage."""
    admit(manager)
    manager.reserve_read([KEY], read_owner="a")
    handle, obj = acquire(manager, "a")
    clock[0] += 10
    manager.finish_read([KEY], read_owner="a")
    manager.clear(force=True)
    assert manager.delete([KEY], force=True)[KEY] == L1Error.KEY_IS_LOCKED
    assert not manager.is_key_evictable(KEY)
    assert obj.is_valid()
    with pytest.raises(RuntimeError, match="active reads"):
        manager.close()
    manager.finish_active_reads([handle, handle])
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS


def test_tp_shares_and_duplicate_acquisition(manager: DramL1Manager) -> None:
    """A retry cannot consume another worker's share or double-release its hold."""
    admit(manager)
    manager.reserve_read([KEY], read_locks=2, read_owner="a")
    first, _ = acquire(manager, "a", 0)
    assert manager.acquire_read([KEY], "a", 0)[KEY][0] == L1Error.KEY_NOT_READABLE
    second, _ = acquire(manager, "a", 1)
    manager.finish_active_reads([first, first])
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_active_reads([second])
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS


def test_late_completion_does_not_touch_replacement(manager: DramL1Manager) -> None:
    """Recreated keys are independent from an earlier allocation's completion."""
    admit(manager)
    manager.reserve_read([KEY], read_owner="old")
    old, _ = acquire(manager, "old")
    manager.finish_active_reads([old])
    manager.delete([KEY])
    admit(manager)
    manager.reserve_read([KEY], read_owner="new")
    current, _ = acquire(manager, "new")
    manager.finish_active_reads([old])
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_active_reads([current])


def test_temporary_object_waits_for_every_reader(manager: DramL1Manager) -> None:
    """Cancelling the remaining queue shares cannot free an active temp object."""
    admit(manager, temporary=True)
    manager.reserve_read([KEY], read_locks=2, read_owner="a")
    handle, obj = acquire(manager, "a")
    manager.finish_read([KEY], read_owner="a")
    assert obj.is_valid()
    manager.finish_active_reads([handle])
    assert manager.acquire_read([KEY], "a", 1)[KEY][0] == L1Error.KEY_NOT_EXIST


def test_l2_admission_creates_named_reservation(manager: DramL1Manager) -> None:
    """The L2-to-L1 handoff has the same ownership semantics as an L1 hit."""
    manager.reserve_write([KEY], [False], LAYOUT, tag="load")
    manager.finish_write_and_reserve_read([KEY], tag="load", read_owner="a")
    handle, _ = acquire(manager, "a")
    assert manager.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_READABLE
    manager.finish_active_reads([handle])
