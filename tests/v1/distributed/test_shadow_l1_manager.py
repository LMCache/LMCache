# SPDX-License-Identifier: Apache-2.0
"""Shadow L1: indexing borrowed views without owning any payload."""

# Standard
from collections.abc import Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import (
    DramL1Manager,
    ShadowL1Manager,
)
from lmcache.v1.distributed.memory_manager import ShadowL1MemoryManager
from lmcache.v1.memory_management import (
    BorrowedTensorMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
)
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

PAGE = 4096
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([PAGE])], dtypes=[torch.uint8])
KEY = ObjectKey(chunk_hash=b"borrowed", model_name="shadow", kv_rank=0)


class Lease:
    """A controllable lease that counts how often it was released."""

    def __init__(self) -> None:
        self.valid = True
        self.releases = 0

    def is_valid(self) -> bool:
        return self.valid and self.releases == 0

    def release(self) -> None:
        self.releases += 1


def _view(lease: Lease) -> BorrowedTensorMemoryObj:
    return BorrowedTensorMemoryObj(
        torch.full((PAGE,), 7, dtype=torch.uint8),
        MemoryObjMetadata(
            shape=torch.Size([PAGE]),
            dtype=torch.uint8,
            address=0,
            phy_size=PAGE,
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
            shapes=[torch.Size([PAGE])],
            dtypes=[torch.uint8],
        ),
        lease_is_valid=lease.is_valid,
        on_release=lease.release,
    )


@pytest.fixture
def fake_pinned_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)


@pytest.fixture
def shadow(fake_pinned_memory: None) -> Iterator[ShadowL1Manager]:
    """A shadow L1 paired with a DRAM L1, as an RDMA borrower would use."""
    local = DramL1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=PAGE, use_lazy=False, shm_name=""),
            read_ttl_seconds=300,
        )
    )
    manager = ShadowL1Manager.next_to(local)
    yield manager
    manager.close()
    local.close()


def test_shadow_mirrors_its_local_medium_as_remote(shadow: ShadowL1Manager) -> None:
    """It reports the paired L1's backend and segment, as REMOTE, with no bytes."""
    assert shadow.backend_type is L1BackendType.DRAM
    assert shadow.get_memory_tags() == MemoryTags(
        MemorySegment.DRAM, MemoryResidency.REMOTE
    )
    assert shadow.config.tag == "_default.shadow"
    assert shadow.get_capacity_bytes() == 0
    assert shadow.get_memory_usage() == (0, 0)
    assert shadow.get_l1_memory_desc() is None


def test_admit_read_locks_without_allocating(shadow: ShadowL1Manager) -> None:
    lease = Lease()
    view = _view(lease)
    assert shadow.admit({KEY: view}, read_locks=2)[KEY] == (L1Error.SUCCESS, view)
    assert view.get_l1_manager() == shadow.l1_manager_id
    assert shadow.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, view)
    assert shadow.get_memory_usage() == (0, 0)
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 0
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 1
    assert shadow.reserve_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST


def test_duplicate_admission_releases_the_new_view(shadow: ShadowL1Manager) -> None:
    first_lease, second_lease = Lease(), Lease()
    first = _view(first_lease)
    assert shadow.admit({KEY: first})[KEY] == (L1Error.SUCCESS, first)
    assert shadow.admit({KEY: _view(second_lease)})[KEY] == (L1Error.SUCCESS, first)
    assert (first_lease.releases, second_lease.releases) == (0, 1)
    shadow.finish_read([KEY], read_locks=2)
    assert first_lease.releases == 1


def test_expired_view_is_not_admitted(shadow: ShadowL1Manager) -> None:
    lease = Lease()
    lease.valid = False
    assert shadow.admit({KEY: _view(lease)})[KEY] == (L1Error.KEY_IN_WRONG_STATE, None)
    assert lease.releases == 1
    assert shadow.reserve_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST


def test_reaper_keeps_held_views_and_releases_unheld_ones(
    shadow: ShadowL1Manager,
) -> None:
    lease = Lease()
    shadow.admit({KEY: _view(lease)})
    lease.valid = False
    shadow.reap_expired()
    assert lease.releases == 0
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_READABLE
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 1

    other = ObjectKey(chunk_hash=b"unheld", model_name="shadow", kv_rank=0)
    unheld = Lease()
    shadow.admit({other: _view(unheld)})
    shadow.finish_read([other])
    assert unheld.releases == 1


def test_deletion_releases_the_view(shadow: ShadowL1Manager) -> None:
    """Deleting a shadow -- as eviction or an admin delete does -- ends its lease."""
    lease = Lease()
    shadow.admit({KEY: _view(lease)})
    assert shadow.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    assert lease.releases == 0
    assert shadow.delete([KEY], force=True)[KEY] == L1Error.SUCCESS
    assert lease.releases == 1


def test_release_all_ends_every_lease(shadow: ShadowL1Manager) -> None:
    leases = [Lease(), Lease()]
    keys = [
        ObjectKey(chunk_hash=bytes([i]), model_name="shadow", kv_rank=0)
        for i in range(2)
    ]
    shadow.admit({k: _view(lease) for k, lease in zip(keys, leases, strict=True)})
    shadow.release_all()
    assert [lease.releases for lease in leases] == [1, 1]


def test_shadow_memory_manager_never_allocates() -> None:
    manager = ShadowL1MemoryManager()
    assert manager.allocate(LAYOUT, 1) == (L1Error.OUT_OF_MEMORY, [])
    lease = Lease()
    assert manager.free([_view(lease)]) == L1Error.SUCCESS
    assert lease.releases == 1
