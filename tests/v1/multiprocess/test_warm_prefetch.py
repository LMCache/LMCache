# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the warm-prefetch job table (:mod:`warm_prefetch`).

Fakes the ``StorageManager``, so no real engine/CUDA/L2 is needed. Verifies the
no-lock contract: ``submit`` uses ``PrefetchMode.WARM`` (the no-lock warm
path), status is polled reactively, and completion releases **nothing** (no
``finish_read`` — the warm holds no lock).
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, cast
import threading

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    TrimPolicy,
)
from lmcache.v1.distributed.storage_manager import StorageManager
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
    completion_barrier: Optional[threading.Barrier] = None

    submit_args: Optional[dict] = None
    finish_called: bool = False
    _polls: int = 0
    _total: int = 0

    def submit_prefetch_task(self, spec):
        self._total = len(spec.keys)
        self.submit_args = {
            "keys": list(spec.keys),
            "mode": spec.mode,
            "policy": spec.policy,
        }
        return _FakeHandle(self._total)

    def query_prefetch_status(self, handle):
        if self._polls < self.delay_polls:
            self._polls += 1
            return None
        if self.completion_barrier is not None:
            self.completion_barrier.wait(timeout=5)
        return _FakeBitmap(self.found)

    def finish_read_prefetched(self, keys, read_locks: int = 1) -> None:
        # Must never be called: the warm holds no lock.
        self.finish_called = True


def test_submit_uses_retain_and_poll_completes_without_release():
    """submit goes through the WARM (no-lock) path; the caller polls
    (pending → completed); completion releases nothing and consumes the job."""
    keys = [_key(0), _key(1)]
    sm = _FakeStorageManager(found=2, delay_polls=2)
    jobs = WarmPrefetchJobs()

    request_id = jobs.submit(sm, keys, layout_desc=object())
    assert sm.submit_args is not None
    assert sm.submit_args["mode"] is PrefetchMode.WARM
    assert sm.submit_args["policy"] is TrimPolicy.SPARSE

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


def test_concurrent_completed_poll_is_consumed_exactly_once() -> None:
    """Only one concurrent poller may consume a completed job."""
    sm = _FakeStorageManager(
        found=1,
        completion_barrier=threading.Barrier(2),
    )
    storage_manager = cast(StorageManager, sm)
    jobs = WarmPrefetchJobs()
    request_id = jobs.submit(
        storage_manager,
        [_key(0)],
        layout_desc=cast(MemoryLayoutDesc, object()),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = [
            future.result(timeout=5)
            for future in (
                executor.submit(jobs.poll, storage_manager, request_id),
                executor.submit(jobs.poll, storage_manager, request_id),
            )
        ]

    assert sorted(status.state for status in statuses) == [COMPLETED, UNKNOWN]
