# SPDX-License-Identifier: Apache-2.0
"""Tests for AffinityThreadPool."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import affinity_pool as affinity_pool_mod
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool


def test_submit_returns_correct_result():
    pool = AffinityThreadPool(max_workers=2)
    future = pool.submit(lambda x: x * 2, 21, affinity_key=0)
    assert future.result(timeout=5) == 42
    pool.shutdown()


def test_affinity_routing():
    """Tasks with the same affinity_key always run on the same thread."""
    pool = AffinityThreadPool(max_workers=4, thread_name_prefix="test-affinity")
    results: dict[int, str] = {}

    def record_thread(key: int) -> str:
        name = threading.current_thread().name
        return name

    futures = []
    for key in [0, 1, 2, 3, 0, 1, 2, 3]:
        f = pool.submit(record_thread, key, affinity_key=key)
        futures.append((key, f))

    for key, f in futures:
        name = f.result(timeout=5)
        if key in results:
            assert results[key] == name, (
                f"Key {key} ran on {name} but previously on {results[key]}"
            )
        else:
            results[key] = name

    pool.shutdown()


def test_distinct_keys_get_distinct_workers():
    """Up to ``max_workers`` distinct keys each land on their own worker.

    Dynamic assignment binds each new key to the next free slot, so any set of
    keys (dense or not) within the worker count gets a 1:1 spread across
    threads -- no key collides onto a shared slot while another sits idle.
    """
    num_workers = 8
    pool = AffinityThreadPool(max_workers=num_workers, thread_name_prefix="dense")

    def record_thread() -> str:
        return threading.current_thread().name

    futures = [
        (rank, pool.submit(record_thread, affinity_key=rank))
        for rank in range(num_workers)
    ]
    threads = {rank: f.result(timeout=5) for rank, f in futures}

    # Every rank ran on a distinct worker thread -> no idle workers, no sharing.
    assert len(set(threads.values())) == num_workers
    # Keys are assigned slots in arrival order, so rank N lands on slot N here.
    for rank in range(num_workers):
        assert threads[rank] == f"dense-{rank}"

    pool.shutdown()


def test_modulo_colliding_keys_get_distinct_workers():
    """Keys that would collide under ``key % num_workers`` still get their own
    worker under dynamic assignment.

    With 4 workers the keys ``0, 4, 8, 12`` all map to slot 0 under a modulo
    scheme -- the bug this pool avoids. Dynamic assignment instead binds them to
    slots 0, 1, 2, 3, so each gets a dedicated thread.
    """
    pool = AffinityThreadPool(max_workers=4, thread_name_prefix="collide")

    def record_thread() -> str:
        return threading.current_thread().name

    keys = [0, 4, 8, 12]  # all == 0 (mod 4)
    threads = {
        k: pool.submit(record_thread, affinity_key=k).result(timeout=5) for k in keys
    }

    assert len(set(threads.values())) == 4, threads
    pool.shutdown()


def test_overflow_warning_on_shared_slot():
    """More distinct keys than workers wraps onto shared slots and warns once.

    LMCache loggers set ``propagate = False``, so assert on the module logger
    directly rather than via the ``caplog`` (root-logger) fixture.
    """
    # Standard
    from unittest.mock import patch

    # First Party
    from lmcache.v1.multiprocess import affinity_pool as affinity_pool_mod

    pool = AffinityThreadPool(max_workers=2, thread_name_prefix="overflow")
    with patch.object(affinity_pool_mod.logger, "warning") as mock_warning:
        # Keys 0 and 1 fill the two slots; key 2 is the first overflow (wraps
        # onto slot 0), and key 3 wraps too but must not warn a second time.
        for key in [0, 1, 2, 3]:
            pool.submit(lambda: None, affinity_key=key).result(timeout=5)

    assert mock_warning.call_count == 1
    assert "wrapped onto worker slot" in mock_warning.call_args.args[0]
    pool.shutdown()


def test_overflow_keys_share_a_slot():
    """When keys outnumber workers, overflow keys reuse earlier slots."""
    pool = AffinityThreadPool(max_workers=2, thread_name_prefix="share")

    def record_thread() -> str:
        return threading.current_thread().name

    threads = {
        key: pool.submit(record_thread, affinity_key=key).result(timeout=5)
        for key in [0, 1, 2, 3]
    }

    # Only two threads exist, so the four keys collapse onto two threads, and
    # the wrap is deterministic (key 2 shares key 0's slot, key 3 shares key 1's).
    assert len(set(threads.values())) == 2
    assert threads[0] == threads[2]
    assert threads[1] == threads[3]
    pool.shutdown()


def test_no_overflow_warning_within_worker_count():
    """Distinct keys within the worker count never warn, even when re-submitted."""
    # Standard
    from unittest.mock import patch

    # First Party
    from lmcache.v1.multiprocess import affinity_pool as affinity_pool_mod

    pool = AffinityThreadPool(max_workers=4, thread_name_prefix="nooverflow")
    with patch.object(affinity_pool_mod.logger, "warning") as mock_warning:
        for rank in range(4):
            pool.submit(lambda: None, affinity_key=rank).result(timeout=5)
        # Re-submitting the same keys reuses their slots -> still no overflow.
        for rank in range(4):
            pool.submit(lambda: None, affinity_key=rank).result(timeout=5)

    assert mock_warning.call_count == 0
    pool.shutdown()


def test_same_key_serialization():
    """Tasks with the same affinity key execute sequentially."""
    pool = AffinityThreadPool(max_workers=2)
    order: list[int] = []
    lock = threading.Lock()

    def append_value(val: int) -> None:
        with lock:
            order.append(val)
        # Small sleep to ensure ordering matters
        time.sleep(0.01)

    futures = [pool.submit(append_value, i, affinity_key=42) for i in range(5)]
    for f in futures:
        f.result(timeout=5)

    assert order == [0, 1, 2, 3, 4]
    pool.shutdown()


def test_different_keys_parallel():
    """Tasks with different affinity keys can run concurrently."""
    pool = AffinityThreadPool(max_workers=2)
    barrier = threading.Barrier(2, timeout=5)

    def wait_at_barrier() -> bool:
        barrier.wait()
        return True

    f1 = pool.submit(wait_at_barrier, affinity_key=0)
    f2 = pool.submit(wait_at_barrier, affinity_key=1)

    assert f1.result(timeout=5) is True
    assert f2.result(timeout=5) is True
    pool.shutdown()


def test_future_exception():
    pool = AffinityThreadPool(max_workers=1)

    def fail():
        raise ValueError("test error")

    future = pool.submit(fail, affinity_key=0)
    with pytest.raises(ValueError, match="test error"):
        future.result(timeout=5)
    pool.shutdown()


def test_future_done_callback():
    pool = AffinityThreadPool(max_workers=1)
    callback_results: list[int] = []
    event = threading.Event()

    def task():
        return 99

    future = pool.submit(task, affinity_key=0)

    def on_done(fut):
        callback_results.append(fut.result())
        event.set()

    future.add_done_callback(on_done)
    event.wait(timeout=5)
    assert callback_results == [99]
    pool.shutdown()


def test_shutdown_wait():
    pool = AffinityThreadPool(max_workers=1)
    completed = threading.Event()

    def slow_task():
        time.sleep(0.1)
        completed.set()

    pool.submit(slow_task, affinity_key=0)
    pool.shutdown(wait=True)
    assert completed.is_set()


def test_shutdown_no_wait():
    pool = AffinityThreadPool(max_workers=1)
    pool.submit(lambda: time.sleep(0.5), affinity_key=0)
    # Should return immediately without waiting
    pool.shutdown(wait=False)


@pytest.fixture
def two_worker_pool() -> Iterator[AffinityThreadPool]:
    pool = AffinityThreadPool(max_workers=2)
    try:
        yield pool
    finally:
        pool.shutdown()


def _thread_name() -> str:
    return threading.current_thread().name


@pytest.mark.parametrize("retired_key", [1001, 1002])
def test_retired_slots_are_reused(
    two_worker_pool: AffinityThreadPool, retired_key: int
) -> None:
    """Repeated restarts reclaim either slot without moving the surviving key."""
    pool = two_worker_pool
    threads = {
        key: pool.submit(_thread_name, affinity_key=key).result(timeout=5)
        for key in (1001, 1002)
    }
    survivor = 1002 if retired_key == 1001 else 1001
    retired_thread = threads[retired_key]
    assert retired_thread != threads[survivor]

    with (
        patch.object(affinity_pool_mod.logger, "warning") as warning,
        patch.object(affinity_pool_mod.logger, "info") as info,
    ):
        for replacement in range(1003, 1103):
            pool.release_key(retired_key)
            pool.release_key(retired_key)
            pool.release_key(-1)
            assert (
                pool.submit(_thread_name, affinity_key=replacement).result(timeout=5)
                == retired_thread
            )
            assert (
                pool.submit(_thread_name, affinity_key=survivor).result(timeout=5)
                == threads[survivor]
            )
            bound_key_count = info.call_args.args[-1]
            assert bound_key_count == 2
            retired_key = replacement
    warning.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_retirement_drains_pending_work_and_reactivated_key(
    two_worker_pool: AffinityThreadPool, outcome: str
) -> None:
    """Reaping cannot split running, queued, or reactivated work across threads."""
    pool = two_worker_pool
    survivor_thread = pool.submit(_thread_name, affinity_key=1).result(timeout=5)
    started = threading.Event()
    resume = threading.Event()
    order: list[str] = []

    def record(label: str) -> str:
        order.append(label)
        return _thread_name()

    def running() -> str:
        started.set()
        assert resume.wait(timeout=5)
        if outcome == "error":
            raise ValueError("task failed")
        return record("running")

    first = pool.submit(running, affinity_key=2)
    try:
        assert started.wait(timeout=5)
        queued = pool.submit(record, "queued", affinity_key=2)
        if outcome == "cancel":
            assert queued.cancel()

        with ThreadPoolExecutor(max_workers=1) as reaper:
            reaper.submit(pool.release_key, 2).result(timeout=5)
        assert (
            pool.submit(_thread_name, affinity_key=3).result(timeout=5)
            == survivor_thread
        )
        reactivated = pool.submit(record, "reactivated", affinity_key=2)
        assert not reactivated.done()
        resume.set()

        if outcome == "error":
            with pytest.raises(ValueError, match="task failed"):
                first.result(timeout=5)
        else:
            assert first.result(timeout=5) != survivor_thread
        if outcome != "cancel":
            assert queued.result(timeout=5) != survivor_thread
        retired_thread = reactivated.result(timeout=5)
        assert retired_thread != survivor_thread
        assert order == (
            ([] if outcome == "error" else ["running"])
            + ([] if outcome == "cancel" else ["queued"])
            + ["reactivated"]
        )
        assert (
            pool.submit(_thread_name, affinity_key=4).result(timeout=5)
            == retired_thread
        )
    finally:
        resume.set()


def test_retirement_from_a_worker_does_not_deadlock(
    two_worker_pool: AffinityThreadPool,
) -> None:
    pool = two_worker_pool

    def unregister() -> str:
        pool.release_key(1)
        return _thread_name()

    retired_thread = pool.submit(unregister, affinity_key=1).result(timeout=5)
    assert pool.submit(_thread_name, affinity_key=2).result(timeout=5) == retired_thread


def test_retiring_one_shared_key_does_not_free_its_live_peer(
    two_worker_pool: AffinityThreadPool,
) -> None:
    pool = two_worker_pool
    threads = {
        key: pool.submit(_thread_name, affinity_key=key).result(timeout=5)
        for key in (1, 2, 3)
    }
    assert threads[1] == threads[3] != threads[2]
    pool.release_key(1)
    pool.release_key(2)
    assert pool.submit(_thread_name, affinity_key=4).result(timeout=5) == threads[2]
    assert pool.submit(_thread_name, affinity_key=3).result(timeout=5) == threads[3]


def test_concurrent_submit_and_retire(two_worker_pool: AffinityThreadPool) -> None:
    pool = two_worker_pool
    barrier = threading.Barrier(4, timeout=5)

    def client(key: int) -> None:
        barrier.wait()
        for _ in range(50):
            first = pool.submit(_thread_name, affinity_key=key)
            second = pool.submit(_thread_name, affinity_key=key)
            assert first.result(timeout=5) == second.result(timeout=5)
            pool.release_key(key)

    with ThreadPoolExecutor(max_workers=4) as clients:
        futures = [clients.submit(client, key) for key in range(4)]
        for future in futures:
            future.result(timeout=10)
