# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the coordinator state machine."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey
from lmcache.v1.memory_coordinator.api import (
    InvalidReservationError,
    OutOfSpaceError,
    ReservationRef,
    StaleEpochError,
    WireLayout,
    WriteGrant,
    WriteReserveItem,
)
from lmcache.v1.memory_coordinator.pool import MemoryPool


def _key(seed: int) -> EncodedObjectKey:
    return EncodedObjectKey(f"{seed:08x}", "model", 0)


def _item(seed: int, elements: int = 64) -> WriteReserveItem:
    return WriteReserveItem(
        key=_key(seed),
        layout=WireLayout(shapes=((elements,),), dtypes=("float16",)),
    )


def _ref(grant: WriteGrant) -> ReservationRef:
    return ReservationRef(key=grant.key, token=grant.token)


def _pool(capacity: int = 64 * 1024) -> MemoryPool:
    return MemoryPool("region", capacity, 4096, "layout")


def test_write_lookup_is_atomic_immutable_and_one_copy() -> None:
    pool = _pool()
    grants = pool.reserve_writes([_item(1), _item(2)])
    assert all(grant is not None for grant in grants)
    writes = [grant for grant in grants if grant is not None]
    assert pool.lookup([_key(1), _key(9)]) == [None, None]

    pool.finish_writes([_ref(grant) for grant in writes])
    hits = pool.lookup([_key(1), _key(9), _key(2)])
    assert [hit.handle if hit else None for hit in hits] == [
        writes[0].handle,
        None,
        writes[1].handle,
    ]
    before = pool.status()
    assert pool.reserve_writes([_item(1)]) == [None]
    assert pool.status() == before
    assert before.object_count == 2


def test_abort_does_not_reuse_space_and_oom_batch_changes_nothing() -> None:
    pool = _pool(capacity=8192)
    first = pool.reserve_writes([_item(1)])[0]
    assert first is not None
    pool.abort_writes([_ref(first)])
    second = pool.reserve_writes([_item(2)])[0]
    assert second is not None
    assert second.handle.offset == 4096

    before = pool.status()
    with pytest.raises(OutOfSpaceError):
        pool.reserve_writes([_item(3), _item(4)])
    assert pool.status() == before


def test_tokens_and_epochs_fail_closed() -> None:
    pool = _pool()
    grant = pool.reserve_writes([_item(1)])[0]
    assert grant is not None
    with pytest.raises(InvalidReservationError):
        pool.finish_writes([ReservationRef(key=grant.key, token="wrong")])
    with pytest.raises(InvalidReservationError):
        pool.finish_writes([_ref(grant), _ref(grant)])
    with pytest.raises(StaleEpochError):
        pool.check_epoch("old-epoch")
    pool.finish_writes([_ref(grant)])
    with pytest.raises(InvalidReservationError):
        pool.finish_writes([_ref(grant)])
    with pytest.raises(InvalidReservationError):
        pool.abort_writes([_ref(grant)])


def test_key_identity_preserves_every_field() -> None:
    pool = _pool()
    key = _key(1)
    keys = [key] + [
        replace(key, **change)
        for change in (
            {"model_name": "other"},
            {"kv_rank": 1},
            {"object_group_id": 1},
            {"cache_salt": "tenant"},
        )
    ]
    items = [_item(1).model_copy(update={"key": item}) for item in keys]
    grants = pool.reserve_writes(items)
    assert all(grant is not None for grant in grants)
    assert pool.reserve_writes(items) == [None] * len(keys)
    pool.finish_writes([_ref(grant) for grant in grants if grant is not None])
    assert [hit.key for hit in pool.lookup(keys) if hit is not None] == keys


def test_duplicate_writer_has_one_winner() -> None:
    pool = _pool()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: pool.reserve_writes([_item(1)])[0], range(2))
        )
    assert sum(result is not None for result in results) == 1


def test_invalid_batch_is_rejected_before_allocation() -> None:
    pool = _pool()
    with pytest.raises(ValueError, match="duplicate"):
        pool.reserve_writes([_item(1), _item(1)])
    with pytest.raises(ValueError, match="positive"):
        pool.reserve_writes([_item(2, elements=0)])
    assert pool.status().used_bytes == 0
