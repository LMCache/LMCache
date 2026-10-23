# SPDX-License-Identifier: Apache-2.0
"""A caller that gives up on a job (e.g. its asyncio.wait_for times out, as in
RemoteBackend.batched_async_contains) must not take an executor worker down.

Before the fix, the worker resolved the cancelled `done` future, hit
InvalidStateError and exited; after `max_workers` such timeouts every queued
job waited forever. CPU only: the executor runs on an event loop in a
background thread, as it does inside LMCache.
"""

# Standard
import asyncio
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQExecutor,
    AsyncPQThreadPoolExecutor,
)

WORKERS = 2


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def _run(loop, coro, timeout=10):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)


def _make(kind, loop):
    if kind == "async":

        async def slow(seconds):
            await asyncio.sleep(seconds)
            return seconds

        async def boom():
            raise ValueError("boom")

        return AsyncPQExecutor(loop, max_workers=WORKERS), slow, boom

    def slow_sync(seconds):
        time.sleep(seconds)
        return seconds

    def boom_sync():
        raise ValueError("boom")

    return AsyncPQThreadPoolExecutor(loop, max_workers=WORKERS), slow_sync, boom_sync


@pytest.mark.parametrize("kind", ["async", "thread"])
def test_cancelled_callers_do_not_kill_workers(loop, kind):
    executor, slow, _ = _make(kind, loop)
    try:

        async def give_up():
            # The job outlives the caller's timeout, as an L2 lookup behind a
            # slow server does.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(executor.submit_job(slow, 0.3), 0.05)

        # More abandoned jobs than workers: without the fix none survive.
        for _ in range(WORKERS * 2):
            _run(loop, give_up())
        time.sleep(0.5)  # let the abandoned jobs finish on the workers

        # Workers are still alive: a later job completes.
        assert _run(loop, executor.submit_job(slow, 0.01), timeout=5) == 0.01
    finally:
        executor.shutdown(wait=False)


@pytest.mark.parametrize("kind", ["async", "thread"])
def test_job_of_departed_caller_is_skipped(loop, kind):
    executor, slow, _ = _make(kind, loop)
    ran = []

    async def record_async():
        ran.append(True)

    def record_sync():
        ran.append(True)

    record = record_async if kind == "async" else record_sync
    try:

        async def scenario():
            # Occupy every worker, then queue a job whose caller gives up
            # while it is still waiting in the queue.
            blockers = [
                asyncio.ensure_future(executor.submit_job(slow, 0.3))
                for _ in range(WORKERS)
            ]
            await asyncio.sleep(0.05)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(executor.submit_job(record), 0.05)
            await asyncio.gather(*blockers)
            # A later job still runs.
            return await executor.submit_job(slow, 0.01)

        assert _run(loop, scenario()) == 0.01
        assert ran == []
    finally:
        executor.shutdown(wait=False)


@pytest.mark.parametrize("kind", ["async", "thread"])
def test_exceptions_still_reach_the_caller(loop, kind):
    executor, slow, boom = _make(kind, loop)
    try:
        with pytest.raises(ValueError, match="boom"):
            _run(loop, executor.submit_job(boom))
        assert _run(loop, executor.submit_job(slow, 0.01)) == 0.01
    finally:
        executor.shutdown(wait=False)
