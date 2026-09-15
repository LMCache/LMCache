# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the warm-prefetch job table (:mod:`warm_prefetch`).

Fakes the ``StorageManager``, so no real engine/CUDA/L2 is needed. Verifies the
no-lock contract: ``submit`` uses ``PrefetchMode.WARM`` (the no-lock warm
path), status is polled reactively, and completion releases **nothing** (no
``finish_read`` — the warm holds no lock).
"""

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.api import ObjectKey, PrefetchMode, TrimPolicy
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
    """Stands in for the loaded-key ``Bitmap``: exactly ``loaded`` of ``size``
    positions are set."""

    def __init__(self, loaded: set[int], size: int) -> None:
        self._loaded = loaded
        self._size = size

    def popcount(self) -> int:
        return len(self._loaded)

    def __invert__(self) -> "_FakeBitmap":
        return _FakeBitmap(set(range(self._size)) - self._loaded, self._size)

    def get_indices_list(self) -> list[int]:
        return sorted(self._loaded)


@dataclass
class _FakeStorageManager:
    loaded: set[int] = field(default_factory=set)
    """Positions the load brings into L1."""
    delay_polls: int = 0
    """Status polls answered "still running" before the load completes."""

    submit_args: dict[str, object] = field(default_factory=dict)
    finish_called: bool = False
    _polls: int = 0

    def submit_prefetch_task(self, spec):
        self.submit_args = {
            "keys": list(spec.keys),
            "mode": spec.mode,
            "policy": spec.policy,
        }
        return _FakeHandle(len(spec.keys))

    def query_prefetch_status(self, handle):
        if self._polls < self.delay_polls:
            self._polls += 1
            return None
        return _FakeBitmap(self.loaded, handle.total_requested_keys)

    def finish_read_prefetched(self, keys, read_locks: int = 1) -> None:
        # Must never be called: the warm holds no lock.
        self.finish_called = True


def test_submit_uses_retain_and_poll_completes_without_release():
    """submit goes through the WARM (no-lock) path; the caller polls
    (pending → completed); completion releases nothing and consumes the job."""
    keys = [_key(0), _key(1)]
    sm = _FakeStorageManager(loaded={0, 1}, delay_polls=2)
    jobs = WarmPrefetchJobs()

    request_id = jobs.submit(sm, keys, layout_desc=object())
    assert sm.submit_args["mode"] is PrefetchMode.WARM
    assert sm.submit_args["policy"] is TrimPolicy.SPARSE

    # Pending while the load runs (reactive poll; no background loop).
    assert jobs.poll(sm, request_id).state == PENDING
    assert jobs.poll(sm, request_id).state == PENDING

    status = jobs.poll(sm, request_id)
    assert status.state == COMPLETED
    assert status.found_keys == 2
    assert status.total_keys == 2
    assert status.missing_key_indices == ()
    # No lock was held, so nothing is released.
    assert sm.finish_called is False

    # Exactly-once: the completing poll consumed the job.
    assert jobs.poll(sm, request_id).state == UNKNOWN


def test_completed_reports_missing_key_positions():
    """A partial load names the positions it did not bring in, in submitted
    key order, so a caller can act per key; the count agrees with found/total."""
    keys = [_key(0), _key(1), _key(2)]
    sm = _FakeStorageManager(loaded={0})
    jobs = WarmPrefetchJobs()

    request_id = jobs.submit(sm, keys, layout_desc=object())
    status = jobs.poll(sm, request_id)
    assert status.state == COMPLETED
    assert status.found_keys == 1
    assert status.total_keys == 3
    assert status.missing_key_indices == (1, 2)
    assert len(status.missing_key_indices) == status.total_keys - status.found_keys


def test_completed_reports_sparse_missing_key_positions():
    """Gaps in the loaded set are reported as the exact unloaded positions,
    not as a prefix count."""
    keys = [_key(i) for i in range(5)]
    sm = _FakeStorageManager(loaded={1, 3})
    jobs = WarmPrefetchJobs()

    request_id = jobs.submit(sm, keys, layout_desc=object())
    status = jobs.poll(sm, request_id)
    assert status.state == COMPLETED
    assert status.found_keys == 2
    assert status.total_keys == 5
    assert status.missing_key_indices == (0, 2, 4)


def test_poll_unknown_request_id():
    """Polling an id that was never submitted returns UNKNOWN."""
    sm = _FakeStorageManager()
    jobs = WarmPrefetchJobs()
    assert jobs.poll(sm, "does-not-exist").state == UNKNOWN
    assert sm.finish_called is False
