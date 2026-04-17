# SPDX-License-Identifier: Apache-2.0
"""Tests for AffinityThreadPool."""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
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
