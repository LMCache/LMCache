# SPDX-License-Identifier: Apache-2.0
"""Unit tests for owner-scoped L1 read reservations."""

# First Party
from lmcache.v1.distributed.read_lock import ReadLock


def test_read_lock_keeps_owners_independent() -> None:
    now = [0.0]
    lock = ReadLock(ttl_seconds=1, clock=lambda: now[0])

    lock.lock("reader-a")
    lock.lock("reader-b")

    assert lock.is_locked("reader-a")
    assert lock.is_locked("reader-b")
    assert lock.unlock("reader-a")
    assert not lock.is_locked("reader-a")
    assert lock.is_locked("reader-b")


def test_read_lock_expiry_is_scoped_to_each_owner() -> None:
    now = [0.0]
    lock = ReadLock(ttl_seconds=1, clock=lambda: now[0])

    lock.lock("reader-a")
    now[0] = 0.5
    lock.lock("reader-b")
    now[0] = 1.1

    assert not lock.is_locked("reader-a")
    assert lock.is_locked("reader-b")
    assert not lock.unlock("reader-a")
    assert lock.unlock("reader-b")
    assert not lock.is_locked()


def test_legacy_owner_is_only_resolved_when_unambiguous() -> None:
    lock = ReadLock(ttl_seconds=10)

    lock.lock("reader-a")
    assert lock.legacy_owner() == "reader-a"
    lock.lock("reader-b")
    assert lock.legacy_owner() is None
