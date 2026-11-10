# SPDX-License-Identifier: Apache-2.0
"""Shadow L1: staging placeholders a borrower binds, without owning any payload."""

# Standard
from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock
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
    ShadowL1ManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import (
    DramL1Manager,
    ShadowL1Manager,
)
from lmcache.v1.distributed.memory_manager import (
    BorrowedMemoryObj,
    ShadowL1MemoryManager,
)
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

PAGE = 4096
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([PAGE])], dtypes=[torch.uint8])
KEY = ObjectKey(chunk_hash=b"borrowed", model_name="shadow", kv_rank=0)
TAG = "prefetch"


class Lease:
    """A controllable lease that counts how often it was released."""

    def __init__(self) -> None:
        self.valid = True
        self.releases = 0

    def is_valid(self) -> bool:
        return self.valid and self.releases == 0

    def release(self) -> None:
        self.releases += 1

    @property
    def pair(self):
        return self.is_valid, self.release


def _bytes() -> torch.Tensor:
    return torch.full((PAGE,), 7, dtype=torch.uint8)


class _Read:
    def __init__(self) -> None:
        self.started: list[tuple[int, int]] = []

    def __call__(self, region_id: int, offset: int) -> "_Read":
        self.started.append((region_id, offset))
        return self

    def is_finished(self) -> bool:
        return True

    def succeeded(self) -> bool:
        return True


@pytest.fixture
def fake_pinned_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)


def _shadow(ttl_seconds: int = 300) -> ShadowL1Manager:
    """A configured DRAM shadow L1 whose lock TTLs are ``ttl_seconds``."""
    return ShadowL1Manager(
        ShadowL1ManagerConfig(
            tag="shadow",
            segment=MemorySegment.DRAM,
            read_ttl_seconds=ttl_seconds,
            write_ttl_seconds=ttl_seconds,
        )
    )


@pytest.fixture
def shadow() -> Iterator[ShadowL1Manager]:
    manager = _shadow()
    yield manager
    manager.close()


@pytest.fixture
def shadow_1s() -> Iterator[ShadowL1Manager]:
    """A shadow whose lock TTLs lapse after one second."""
    manager = _shadow(ttl_seconds=1)
    yield manager
    manager.close()


def _stage(
    shadow: ShadowL1Manager, key: ObjectKey = KEY, tag: str = TAG
) -> BorrowedMemoryObj:
    """Reserve a placeholder for ``key`` as the prefetch controller does."""
    err, obj = shadow.reserve_write([key], [False], LAYOUT, tag=tag)[key]
    assert err is L1Error.SUCCESS and isinstance(obj, BorrowedMemoryObj)
    return obj


def _lend(
    shadow: ShadowL1Manager,
    lease: Lease,
    data: torch.Tensor | None = None,
    read_locks: int = 1,
    key: ObjectKey = KEY,
    tag: str = TAG,
) -> BorrowedMemoryObj:
    """Stage, bind and admit one view: the whole prefetch path of a borrower."""
    placeholder = _stage(shadow, key, tag)
    placeholder.bind(lease.pair, tensor=_bytes() if data is None else data)
    err, obj = shadow.finish_write_and_reserve_read([key], read_locks, tag=tag)[key]
    assert err is L1Error.SUCCESS and isinstance(obj, BorrowedMemoryObj)
    return obj


# -- the borrowed object -------------------------------------------------------


def test_placeholder_is_invalid_until_bound_once() -> None:
    _, (obj,) = ShadowL1MemoryManager().allocate(LAYOUT, 1)
    assert isinstance(obj, BorrowedMemoryObj)
    assert not obj.is_bound and not obj.is_valid() and obj.get_size() == PAGE
    with pytest.raises(RuntimeError, match="not bound"):
        _ = obj.lease
    with pytest.raises(ValueError, match="exactly one"):
        obj.bind(Lease().pair)
    with pytest.raises(ValueError, match="address"):
        obj.bind(Lease().pair, read_into=_Read(), address=0)
    with pytest.raises(ValueError, match="shorter"):
        obj.bind(Lease().pair, tensor=torch.zeros(PAGE - 1, dtype=torch.uint8))
    lease = Lease()
    obj.bind(lease.pair, tensor=_bytes())
    assert obj.is_bound and obj.is_valid() and not obj.is_remote
    assert obj.local_device() == torch.device("cpu")
    assert obj.raw_tensor is not None and obj.raw_tensor.numel() == PAGE
    with pytest.raises(RuntimeError, match="already bound"):
        obj.bind(Lease().pair, tensor=_bytes())
    obj.release()
    obj.release()
    assert lease.releases == 1 and not obj.is_valid()


def _placeholders(count: int) -> list[BorrowedMemoryObj]:
    """Fresh placeholders straight from the shadow's memory manager."""
    _, objs = ShadowL1MemoryManager().allocate(LAYOUT, count)
    assert all(isinstance(obj, BorrowedMemoryObj) for obj in objs)
    return cast(list[BorrowedMemoryObj], objs)


def test_remote_binding_reads_into_a_region_and_has_no_local_device() -> None:
    first, same, other, other_peer = _placeholders(4)
    read = _Read()
    first.bind(Lease().pair, read_into=read, address=0x1000, source="a")
    same.bind(Lease().pair, read_into=_Read(), address=0x1000, source="a")
    other.bind(Lease().pair, read_into=_Read(), address=0x2000, source="a")
    other_peer.bind(Lease().pair, read_into=_Read(), address=0x1000, source="b")
    assert first.is_remote and first.local_device() is None
    assert first.read_into(3, 64) is read and read.started == [(3, 64)]
    assert first.same_bytes(same) and not first.same_bytes(other)
    assert not first.same_bytes(other_peer)  # same offset, another peer
    (local,) = _placeholders(1)
    local.bind(Lease().pair, tensor=_bytes())
    assert not first.same_bytes(local) and not local.same_bytes(first)
    with pytest.raises(RuntimeError, match="local bytes"):
        local.read_into(0, 0)
    assert first.raw_tensor is None and local.raw_tensor is not None


def test_memory_manager_frees_by_ending_the_lease() -> None:
    manager = ShadowL1MemoryManager()
    err, objs = manager.allocate(LAYOUT, 2)
    assert err is L1Error.SUCCESS and len(objs) == 2
    lease = Lease()
    cast(BorrowedMemoryObj, objs[0]).bind(lease.pair, tensor=_bytes())
    assert manager.free(objs) == L1Error.SUCCESS  # the unbound one just goes
    assert lease.releases == 1
    assert manager.get_memory_usage() == (0, 0)
    with pytest.raises(TypeError):
        manager.free([TensorMemoryObj(_bytes(), objs[0].meta, None)])


# -- the shadow L1 -------------------------------------------------------------


def test_shadow_mirrors_its_local_medium_as_shadow(shadow: ShadowL1Manager) -> None:
    """It reports the medium backing its segment, as SHADOW, with no bytes."""
    assert shadow.backend_type is L1BackendType.DRAM
    assert shadow.get_memory_tags() == MemoryTags(
        MemorySegment.DRAM, MemoryResidency.SHADOW
    )
    assert shadow.config.tag == "shadow"
    assert shadow.get_capacity_bytes() == 0
    assert shadow.get_memory_usage() == (0, 0)
    assert shadow.get_l1_memory_desc() is None


def test_a_segment_no_l1_backs_is_refused() -> None:
    with pytest.raises(ValueError, match="vram"):
        ShadowL1Manager(ShadowL1ManagerConfig(tag="vram", segment=MemorySegment.VRAM))


def test_shadow_calls_no_listener(shadow: ShadowL1Manager) -> None:
    """Its store and eviction controllers register listeners like any L1's;
    a view is nothing to store or evict, so they hear nothing."""
    listener = MagicMock()
    shadow.register_listener(listener)
    _lend(shadow, Lease())
    shadow.unsafe_read([KEY])
    shadow.finish_read([KEY])
    shadow.delete([KEY], force=True)
    assert listener.method_calls == []


def test_borrow_read_locks_without_allocating(shadow: ShadowL1Manager) -> None:
    """reserve_write stages without a capacity check (and temporary, whatever
    the caller asked); the bound placeholder is admitted read-locked, and the
    last finish_read ends the lease."""
    lease = Lease()
    view = _lend(shadow, lease, read_locks=2)
    assert view.get_l1_manager() == shadow.l1_manager_id
    assert shadow.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, view)
    assert shadow.get_memory_usage() == (0, 0)
    assert shadow.get_staging_memory_usage() == 0
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 0
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 1
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST


def test_shadow_is_never_a_lookup_hit(shadow: ShadowL1Manager) -> None:
    """reserve_read misses even a resident view: every request borrows afresh."""
    _lend(shadow, Lease())
    assert shadow.reserve_read([KEY])[KEY] == (L1Error.KEY_NOT_EXIST, None)
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.SUCCESS


def test_a_lease_that_ended_by_admission_is_unreadable(shadow: ShadowL1Manager) -> None:
    """Admission refuses nothing: a lease that ended between the load and the
    admission is the same as one ending right after, so the view is resident
    but unreadable, and its holder's finish_read returns the lease."""
    lease = Lease()
    _stage(shadow).bind(lease.pair, tensor=_bytes())
    lease.valid = False
    assert shadow.finish_write_and_reserve_read([KEY], tag=TAG)[KEY][0] == (
        L1Error.SUCCESS
    )
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_READABLE
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 1


def test_resident_view_adopts_the_fresh_lease(shadow: ShadowL1Manager) -> None:
    """A second request's lookup renews a held view: it is staged again, and
    at admission the resident object stays, now under the newer lease, while
    the older reservation goes back with the discarded placeholder."""
    first_lease, second_lease = Lease(), Lease()
    first = _lend(shadow, first_lease)
    first_lease.valid = False  # the first reservation ran out meanwhile
    placeholder = _stage(shadow, tag="second")
    placeholder.bind(second_lease.pair, tensor=first.raw_data)
    assert shadow.finish_write_and_reserve_read([KEY], tag="second")[KEY] == (
        L1Error.SUCCESS,
        first,
    )
    assert (first_lease.releases, second_lease.releases) == (1, 0)
    assert first.is_valid() and not placeholder.is_valid()
    assert shadow.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, first)
    shadow.finish_read([KEY], read_locks=2)
    assert second_lease.releases == 1


def test_resident_view_of_other_bytes_serves_the_new_reader(
    shadow: ShadowL1Manager,
) -> None:
    """The peer moved the key since the holders' lease: the held view keeps
    its bytes and lease and serves the new reader too; the fresh reservation
    of the other bytes goes back."""
    held_lease, fresh_lease = Lease(), Lease()
    held = _lend(shadow, held_lease)
    placeholder = _stage(shadow, tag="second")
    placeholder.bind(fresh_lease.pair, tensor=_bytes())
    assert shadow.finish_write_and_reserve_read([KEY], tag="second")[KEY] == (
        L1Error.SUCCESS,
        held,
    )
    assert (held_lease.releases, fresh_lease.releases) == (0, 1)
    assert shadow.finish_read([KEY], read_locks=2)[KEY] == L1Error.SUCCESS
    assert held_lease.releases == 1


def test_holder_keeps_an_expired_view_until_finish_read(
    shadow: ShadowL1Manager,
) -> None:
    """An ended lease refuses new readers but never tears the view out from
    under its holder; the lease is released by the holder's finish_read."""
    lease = Lease()
    _lend(shadow, lease)
    lease.valid = False
    assert shadow.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_READABLE
    assert lease.releases == 0
    assert shadow.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert lease.releases == 1


def test_deletion_releases_the_view(shadow: ShadowL1Manager) -> None:
    """Deleting a shadow -- as an admin delete does -- ends its lease."""
    lease = Lease()
    _lend(shadow, lease)
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
    for key, lease in zip(keys, leases, strict=True):
        _lend(shadow, lease, key=key)
    shadow.close()
    shadow.close()  # the borrower and the storage manager may both close it
    assert [lease.releases for lease in leases] == [1, 1]


def test_reserve_write_sweeps_a_view_whose_read_ttl_lapsed(
    shadow_1s: ShadowL1Manager,
) -> None:
    """A reader past its TTL is gone and its late finish_read is refused, so
    the next reservation must free the view whatever the lease says, or the
    borrower's reservation stays pinned."""
    lease = Lease()
    _lend(shadow_1s, lease)
    time.sleep(1.2)
    assert shadow_1s.finish_read([KEY])[KEY] == L1Error.KEY_IN_WRONG_STATE
    assert lease.releases == 0
    other = ObjectKey(chunk_hash=b"next", model_name="shadow", kv_rank=0)
    _lend(shadow_1s, Lease(), key=other)
    assert lease.releases == 1
    assert shadow_1s.unsafe_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST
    time.sleep(1.2)
    third = ObjectKey(chunk_hash=b"third", model_name="shadow", kv_rank=0)
    _stage(shadow_1s, key=third)  # any reservation sweeps
    assert shadow_1s.unsafe_read([other])[other][0] == L1Error.KEY_NOT_EXIST


def test_expired_write_reservation_is_reclaimed(shadow_1s: ShadowL1Manager) -> None:
    """A bound placeholder whose write TTL lapsed before admission is refused
    by the base and, since a shadow has no eviction, swept at the next
    reservation: its lease goes back and the key can be staged afresh."""
    stale = Lease()
    _stage(shadow_1s).bind(stale.pair, tensor=_bytes())
    time.sleep(1.2)
    assert shadow_1s.finish_write_and_reserve_read([KEY], tag=TAG)[KEY][0] == (
        L1Error.KEY_IN_WRONG_STATE
    )
    assert stale.releases == 0
    fresh = Lease()
    view = _lend(shadow_1s, fresh)
    assert stale.releases == 1 and fresh.releases == 0
    assert shadow_1s.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, view)


def test_a_stale_unheld_view_does_not_refuse_a_fresh_one(
    shadow_1s: ShadowL1Manager,
) -> None:
    stale = Lease()
    _lend(shadow_1s, stale)
    stale.valid = False
    time.sleep(1.2)  # the holder's TTL lapses: stale and unheld
    fresh = Lease()
    view = _lend(shadow_1s, fresh)
    assert stale.releases == 1
    assert fresh.releases == 0
    assert shadow_1s.unsafe_read([KEY])[KEY] == (L1Error.SUCCESS, view)


def test_shadow_publishes_no_l1_events(fake_pinned_memory: None) -> None:
    """A shadow's keys are the peer's residency: none of reserve, admit, read,
    finish, delete or close publishes an L1 event, while an owned L1 does."""
    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=10_000))
    events: list[Event] = []
    for event_type in EventType:
        if event_type.name.startswith("L1_"):
            bus.subscribe(event_type, events.append)
    bus.start()
    try:
        manager = _shadow()
        local = DramL1Manager(
            L1ManagerConfig(
                L1MemoryManagerConfig(size_in_bytes=PAGE, use_lazy=False, shm_name="")
            )
        )
        try:
            _lend(manager, Lease())
            manager.unsafe_read([KEY])
            manager.finish_read([KEY])
            _lend(manager, Lease())
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
