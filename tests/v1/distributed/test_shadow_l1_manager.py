# SPDX-License-Identifier: Apache-2.0
"""Shadow L1: indexing borrowed views without owning any payload."""

# Standard
from collections.abc import Iterator
import time

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
from lmcache.v1.distributed.memory_manager import (
    BorrowedTensorMemoryObj,
    ShadowL1MemoryManager,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
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


def _view(lease: Lease, data: torch.Tensor | None = None) -> BorrowedTensorMemoryObj:
    return BorrowedTensorMemoryObj(
        torch.full((PAGE,), 7, dtype=torch.uint8) if data is None else data,
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


def _shadow(read_ttl_seconds: int = 300) -> tuple[ShadowL1Manager, DramL1Manager]:
    """A DRAM lender's shadow L1 and the local L1 whose config it inherits."""
    local = DramL1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=PAGE, use_lazy=False, shm_name=""),
            read_ttl_seconds=read_ttl_seconds,
        )
    )
    return ShadowL1Manager.for_lender(0, MemorySegment.DRAM, local.config), local


@pytest.fixture
def shadow(fake_pinned_memory: None) -> Iterator[ShadowL1Manager]:
    manager, local = _shadow()
    yield manager
    manager.close()
    local.close()


@pytest.fixture
def shadow_1s(fake_pinned_memory: None) -> Iterator[ShadowL1Manager]:
    """A shadow whose read-lock TTL lapses after one second."""
    manager, local = _shadow(read_ttl_seconds=1)
    yield manager
    manager.close()
    local.close()


def test_shadow_mirrors_its_local_medium_as_remote(shadow: ShadowL1Manager) -> None:
    """It reports the paired L1's backend and segment, as REMOTE, with no bytes."""
    assert shadow.backend_type is L1BackendType.DRAM
    assert shadow.get_memory_tags() == MemoryTags(
        MemorySegment.DRAM, MemoryResidency.REMOTE
    )
    assert shadow.config.tag == "shadow.dram.0"
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
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST


def test_resident_view_adopts_the_fresh_lease(shadow: ShadowL1Manager) -> None:
    """A second request's lookup renews a held view: the resident object
    stays, now under the newer lease, and the older reservation goes back."""
    first_lease, second_lease = Lease(), Lease()
    first = _view(first_lease)
    assert shadow.admit({KEY: first})[KEY] == (L1Error.SUCCESS, first)
    first_lease.valid = False  # the first reservation ran out meanwhile
    same_bytes = _view(second_lease, data=first.raw_data)
    assert shadow.admit({KEY: same_bytes})[KEY] == (L1Error.SUCCESS, first)
    assert (first_lease.releases, second_lease.releases) == (1, 0)
    assert first.is_valid() and not same_bytes.is_valid()
    assert shadow.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, first)
    shadow.finish_read([KEY], read_locks=2)
    assert second_lease.releases == 1


def test_a_view_of_other_bytes_is_a_miss_while_the_key_is_held(
    shadow: ShadowL1Manager,
) -> None:
    """The peer moved the key since the holders' lease: the held view keeps
    its bytes and lease, the fresh view is released, the request misses."""
    held_lease, fresh_lease = Lease(), Lease()
    held = _view(held_lease)
    assert shadow.admit({KEY: held})[KEY] == (L1Error.SUCCESS, held)
    assert shadow.admit({KEY: _view(fresh_lease)})[KEY] == (
        L1Error.KEY_NOT_READABLE,
        None,
    )
    assert (held_lease.releases, fresh_lease.releases) == (0, 1)
    assert shadow.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, held)
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert held_lease.releases == 1


def test_expired_view_is_not_admitted(shadow: ShadowL1Manager) -> None:
    lease = Lease()
    lease.valid = False
    assert shadow.admit({KEY: _view(lease)})[KEY] == (L1Error.KEY_IN_WRONG_STATE, None)
    assert lease.releases == 1
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST


def test_holder_keeps_an_expired_view_until_finish_read(
    shadow: ShadowL1Manager,
) -> None:
    """An ended lease refuses new readers but never tears the view out from
    under its holder; the lease is released by the holder's finish_read."""
    lease = Lease()
    shadow.admit({KEY: _view(lease)})
    lease.valid = False
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_READABLE
    assert lease.releases == 0
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 1


def test_deletion_releases_the_view(shadow: ShadowL1Manager) -> None:
    """Deleting a shadow -- as eviction or an admin delete does -- ends its lease."""
    lease = Lease()
    shadow.admit({KEY: _view(lease)})
    assert shadow.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    assert lease.releases == 0
    assert shadow.delete([KEY], force=True)[KEY] == L1Error.SUCCESS
    assert lease.releases == 1


def test_close_ends_every_lease_once(shadow: ShadowL1Manager) -> None:
    leases = [Lease(), Lease()]
    keys = [
        ObjectKey(chunk_hash=bytes([i]), model_name="shadow", kv_rank=0)
        for i in range(2)
    ]
    shadow.admit({k: _view(lease) for k, lease in zip(keys, leases, strict=True)})
    shadow.close()
    shadow.close()  # the lender and the storage manager may both close it
    assert [lease.releases for lease in leases] == [1, 1]


def test_shadow_memory_manager_never_allocates() -> None:
    manager = ShadowL1MemoryManager()
    assert manager.allocate(LAYOUT, 1) == (L1Error.OUT_OF_MEMORY, [])
    lease = Lease()
    assert manager.free([_view(lease)]) == L1Error.SUCCESS
    assert lease.releases == 1


def test_for_lender_rejects_a_segment_no_l1_backs(shadow: ShadowL1Manager) -> None:
    with pytest.raises(ValueError, match="vram"):
        ShadowL1Manager.for_lender(1, MemorySegment.VRAM, shadow.config)


def test_admit_sweeps_a_view_whose_read_ttl_lapsed(
    shadow_1s: ShadowL1Manager,
) -> None:
    """A reader past its TTL is gone and its late finish_read is refused, so
    the next admit must free the view whatever the lease says, or the
    lender's reservation stays pinned."""
    lease = Lease()
    shadow_1s.admit({KEY: _view(lease)})
    time.sleep(1.2)
    assert shadow_1s.finish_read([KEY])[KEY] == L1Error.KEY_IN_WRONG_STATE
    assert lease.releases == 0
    other = ObjectKey(chunk_hash=b"next", model_name="shadow", kv_rank=0)
    shadow_1s.admit({other: _view(Lease())})
    assert lease.releases == 1
    assert shadow_1s.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST
    time.sleep(1.2)
    shadow_1s.sweep_lapsed()  # the same sweep, without an admission
    assert shadow_1s.unsafe_read([other])[other][0] == L1Error.KEY_NOT_EXIST


def test_admit_replaces_a_stale_unheld_view(shadow_1s: ShadowL1Manager) -> None:
    """An expired view nobody holds must not refuse a fresh one for its key."""
    stale = Lease()
    shadow_1s.admit({KEY: _view(stale)})
    stale.valid = False
    time.sleep(1.2)  # the holder's TTL lapses: stale and unheld
    fresh = Lease()
    view = _view(fresh)
    assert shadow_1s.admit({KEY: view})[KEY] == (L1Error.SUCCESS, view)
    assert stale.releases == 1
    assert fresh.releases == 0
    assert shadow_1s.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, view)


def test_shadow_publishes_no_l1_events(fake_pinned_memory: None) -> None:
    """A shadow's keys are the peer's residency: none of admit, read, finish,
    delete or close publishes an L1 event, while the owned L1 built from the
    same config still does."""
    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=10_000))
    events: list[Event] = []
    for event_type in EventType:
        if event_type.name.startswith("L1_"):
            bus.subscribe(event_type, events.append)
    bus.start()
    try:
        manager, local = _shadow()
        try:
            lease = Lease()
            assert manager.admit({KEY: _view(lease)})[KEY][0] == L1Error.SUCCESS
            manager.unsafe_read([KEY])
            manager.finish_read([KEY])
            manager.admit({KEY: _view(Lease())})
            manager.delete([KEY], force=True)
            manager.close()
            local.reserve_write([KEY], [False], LAYOUT)
            local.finish_write([KEY])
        finally:
            manager.close()
            local.close()
        bus.stop()
        tags = {event.metadata["l1_tag"] for event in events}
        assert local.config.tag in tags
        assert manager.config.tag not in tags
    finally:
        init_event_bus(EventBusConfig(enabled=False))


def test_shadow_memory_manager_rejects_unleased_objects() -> None:
    with pytest.raises(TypeError):
        ShadowL1MemoryManager().free(
            [
                TensorMemoryObj(
                    torch.zeros(PAGE, dtype=torch.uint8), _view(Lease()).meta, None
                )
            ]
        )
