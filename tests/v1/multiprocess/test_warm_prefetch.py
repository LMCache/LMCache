# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the warm-prefetch job table (:mod:`warm_prefetch`).

Fakes the ``StorageManager``, so no real engine/CUDA/L2 is needed. Verifies the
no-lock contract: ``submit`` uses ``PrefetchLockMode.NO_LOCK`` with ``"full"``
fetching (the no-lock warm path), status is polled reactively, and completion
releases **nothing** (no ``finish_read`` — the warm holds no lock).
"""

# Standard
from dataclasses import dataclass
from typing import Optional

# First Party
from lmcache.v1.distributed.api import GroupedObjectKeys, ObjectKey, PrefetchLockMode
from lmcache.v1.multiprocess.warm_prefetch import (
    COMPLETED,
    PENDING,
    UNKNOWN,
    WarmPrefetchJobs,
)


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i),
        model_name="test_model",
        kv_rank=0,
    )


class _FakeHandle:
    def __init__(self, total: int) -> None:
        self.total_requested_keys = total


class _FakeBitmap:
    """Stands in for the found-key ``Bitmap``; ``popcount`` is the found count."""

    def __init__(self, n: int) -> None:
        self._n = n

    def popcount(self) -> int:
        return self._n


@dataclass
class _FakeStorageManager:
    found: int = 0
    delay_polls: int = 0

    submit_args: Optional[dict] = None
    finish_called: bool = False
    _polls: int = 0
    _total: int = 0

    def submit_prefetch_task(self, spec):
        keys = [key for row in spec.key_groups for key in row.keys]
        self._total = len(keys)
        self.submit_args = {
            "keys": keys,
            "lock_mode": spec.lock_mode,
            "fetching_policy": spec.fetching_policy,
        }
        return _FakeHandle(self._total)

    def query_prefetch_status(self, handle):
        if self._polls < self.delay_polls:
            self._polls += 1
            return None
        # First Party
        from lmcache.v1.distributed.api import PrefetchResult

        return PrefetchResult(
            hit_cells=[_FakeBitmap(self.found)],
            l1_hit_cells=[_FakeBitmap(0)],
            l2_hit_cells=[_FakeBitmap(self.found)],
        )

    def finish_read_prefetched(self, keys, read_locks: int = 1) -> None:
        # Must never be called: the warm holds no lock.
        self.finish_called = True


def test_submit_uses_retain_and_poll_completes_without_release():
    """submit goes through the WARM (no-lock) path; the caller polls
    (pending → completed); completion releases nothing and consumes the job."""
    keys = [_key(0), _key(1)]
    sm = _FakeStorageManager(found=2, delay_polls=2)
    jobs = WarmPrefetchJobs()

    request_id = jobs.submit(
        sm, [GroupedObjectKeys(keys=keys, object_group_id=0, layout_desc=object())]
    )
    assert sm.submit_args is not None
    assert sm.submit_args["keys"] == keys
    assert sm.submit_args["lock_mode"] is PrefetchLockMode.NO_LOCK
    assert sm.submit_args["fetching_policy"] == "full"

    # Pending while the load runs (reactive poll; no background loop).
    assert jobs.poll(sm, request_id).state == PENDING
    assert jobs.poll(sm, request_id).state == PENDING

    status = jobs.poll(sm, request_id)
    assert status.state == COMPLETED
    assert status.found_keys == 2
    assert status.total_keys == 2
    # No lock was held, so nothing is released.
    assert sm.finish_called is False

    # Exactly-once: the completing poll consumed the job.
    assert jobs.poll(sm, request_id).state == UNKNOWN


def test_poll_unknown_request_id():
    """Polling an id that was never submitted returns UNKNOWN."""
    sm = _FakeStorageManager()
    jobs = WarmPrefetchJobs()
    assert jobs.poll(sm, "does-not-exist").state == UNKNOWN
    assert sm.finish_called is False
