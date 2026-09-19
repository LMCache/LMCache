# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the coordinator state machine."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.memory_coordinator.api import (
    InvalidReservationError,
    OutOfSpaceError,
    ReservationRef,
    StaleEpochError,
)
from lmcache.v1.memory_coordinator.pool import MemoryPool
import lmcache.v1.memory_coordinator.pool as pool_module

# Local
from .conftest import item as _item
from .conftest import key as _key
from .conftest import ref as _ref


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


@pytest.mark.parametrize("operation", ["finish_writes", "abort_writes"])
def test_pending_expiry_replaces_without_reusing_and_rejects_late_calls(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    now = 0.0
    monkeypatch.setattr(pool_module, "monotonic", lambda: now)
    pool = _pool()
    old = pool.reserve_writes([_item(1)])[0]
    assert old is not None
    now = 59.999
    assert pool.reserve_writes([_item(1)]) == [None]
    now = 60.0
    before = pool.status()
    late_call = getattr(pool, operation)
    with pytest.raises(InvalidReservationError):
        late_call([_ref(old)])
    assert pool.status() == before

    replacement = pool.reserve_writes([_item(1)])[0]
    assert replacement is not None
    assert replacement.handle.offset >= old.handle.offset + old.handle.length
    assert replacement.handle.generation == old.handle.generation + 1
    assert replacement.token != old.token
    assert pool.status().used_bytes > before.used_bytes
    pending = pool.status()
    with pytest.raises(InvalidReservationError):
        late_call([_ref(old)])
    assert pool.status() == pending
    assert pool.lookup([_key(1)]) == [None]
    assert pool.reserve_writes([_item(1)]) == [None]
    assert pool._objects[replacement.key].expires_at == 120.0

    pool.finish_writes([_ref(replacement)])
    hits = pool.lookup([_key(1)])
    assert hits[0] is not None and hits[0].handle == replacement.handle
    assert hits[0].layout == replacement.layout
    with pytest.raises(InvalidReservationError):
        late_call([_ref(old)])
    assert pool.lookup([_key(1)]) == hits
    assert pool.status() == pending
    assert pool._objects[replacement.key].expires_at == 120.0


def test_committed_object_outlives_pending_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(pool_module, "monotonic", lambda: now)
    pool = _pool()
    grant = pool.reserve_writes([_item(1)])[0]
    assert grant is not None
    # Discard the successful result to model a lost response, not an HTTP fault.
    pool.finish_writes([_ref(grant)])
    before = pool.status()
    now = 600.0
    hit = pool.lookup([_key(1)])[0]
    assert hit is not None and hit.handle == grant.handle
    assert pool.reserve_writes([_item(1)]) == [None]
    assert pool.status() == before


@pytest.mark.parametrize("failure", ["oom", "invalid"])
def test_failed_expiry_replacement_preserves_allocation_state(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    now = 0.0
    monkeypatch.setattr(pool_module, "monotonic", lambda: now)
    pool = _pool(capacity=8192)
    old = pool.reserve_writes([_item(1)])[0]
    assert old is not None
    now = 60.0
    before = pool.status()
    # Pending metadata is hidden by lookup, so inspect it for rollback equality.
    record = pool._objects[old.key]
    pending = (record.handle, record.layout, record.write_token, record.expires_at)
    bad_item = _item(2, elements=4096 if failure == "oom" else 0)
    with pytest.raises(OutOfSpaceError if failure == "oom" else ValueError):
        pool.reserve_writes([_item(1), bad_item])
    assert pool.status() == before
    assert pool._objects == {old.key: record}
    assert (
        record.handle,
        record.layout,
        record.write_token,
        record.expires_at,
    ) == pending
    replacement = pool.reserve_writes([_item(1)])[0]
    assert replacement is not None
    assert replacement.handle.offset == 4096
    assert replacement.handle.generation == old.handle.generation + 1


def test_expired_finish_batch_does_not_commit_live_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(pool_module, "monotonic", lambda: now)
    pool = _pool()
    expired = pool.reserve_writes([_item(1)])[0]
    now = 30.0
    live = pool.reserve_writes([_item(2)])[0]
    assert expired is not None and live is not None
    now = 60.0
    before = pool.status()
    with pytest.raises(InvalidReservationError):
        pool.finish_writes([_ref(live), _ref(expired)])
    assert pool.status() == before
    assert pool.lookup([_key(1), _key(2)]) == [None, None]
    pool.finish_writes([_ref(live)])
    hit = pool.lookup([_key(2)])[0]
    assert hit is not None and hit.handle == live.handle
