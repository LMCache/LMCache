# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `AsyncPQExecutor` shutdown."""

# Standard
import asyncio
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="test-pq-loop")
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


def run(loop, coro, timeout=10.0):
    """Run a coroutine on ``loop`` and wait for it, failing on a hang."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)


def test_shutdown_drains_queued_jobs(loop_in_thread):
    """``wait=True`` runs queued jobs and returns instead of blocking.

    Cancelling the workers before draining would stop one before it consumes
    its sentinel, leaving `task_done` uncalled and `queue.join()` waiting
    forever.
    """
    executor = AsyncPQExecutor(loop_in_thread)
    completed = []

    async def job(value: int) -> int:
        completed.append(value)
        return value

    async def scenario():
        submitted = [
            asyncio.ensure_future(executor.submit_job(job, i)) for i in range(8)
        ]
        # Let every submission reach the queue before shutting down.
        await asyncio.sleep(0.05)
        await executor.shutdown_async(wait=True)
        return await asyncio.gather(*submitted)

    results = run(loop_in_thread, scenario())

    assert sorted(results) == list(range(8))
    assert sorted(completed) == list(range(8))


def test_shutdown_without_wait_returns(loop_in_thread):
    """``wait=False`` returns promptly instead of draining the queue."""
    executor = AsyncPQExecutor(loop_in_thread)

    run(loop_in_thread, executor.shutdown_async(wait=False), timeout=5.0)


def test_shutdown_is_idempotent(loop_in_thread):
    """A second shutdown is a no-op rather than a second drain."""
    executor = AsyncPQExecutor(loop_in_thread)

    run(loop_in_thread, executor.shutdown_async(wait=True))
    run(loop_in_thread, executor.shutdown_async(wait=True), timeout=5.0)
