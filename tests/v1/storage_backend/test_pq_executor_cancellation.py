# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import AsyncIterator
from typing import Any
import asyncio
import threading

# Third Party
import pytest
import pytest_asyncio

# First Party
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQExecutor,
    AsyncPQThreadPoolExecutor,
)

pytestmark = pytest.mark.no_shared_allocator


@pytest_asyncio.fixture(params=[AsyncPQExecutor, AsyncPQThreadPoolExecutor])
async def executor(request: pytest.FixtureRequest) -> AsyncIterator[AsyncPQExecutor]:
    """Use one worker so a lost worker is visible through the next submission."""
    instance = request.param(asyncio.get_running_loop(), max_workers=1)
    try:
        yield instance
    finally:
        # Queue draining during shutdown is a separate contract under review.
        await instance.shutdown_async(wait=False)
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True], ids=["return", "raise"])
async def test_cancelled_submitter_does_not_stop_worker(
    executor: AsyncPQExecutor, raises: bool
) -> None:
    """A cancelled caller must not prevent a subsequent job from completing."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    async_release = asyncio.Event()
    thread_release = threading.Event()

    def result() -> int:
        if raises:
            raise ValueError("job failed after caller cancellation")
        return 17

    async def async_job() -> int:
        started.set()
        await async_release.wait()
        return result()

    def thread_job() -> int:
        loop.call_soon_threadsafe(started.set)
        if not thread_release.wait(timeout=5):
            raise TimeoutError("test did not release the running job")
        return result()

    async def async_followup() -> int:
        return 29

    def thread_followup() -> int:
        return 29

    threaded = isinstance(executor, AsyncPQThreadPoolExecutor)
    job: Any = thread_job if threaded else async_job
    followup: Any = thread_followup if threaded else async_followup
    caller = asyncio.create_task(executor.submit_job(job))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        async_release.set()
        thread_release.set()
        assert await asyncio.wait_for(executor.submit_job(followup), timeout=2) == 29
    finally:
        async_release.set()
        thread_release.set()
        if not caller.done():
            caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True], ids=["return", "raise"])
async def test_executor_delivers_result_or_error_then_accepts_another_job(
    executor: AsyncPQExecutor, raises: bool
) -> None:
    """Normal success and failure preserve both delivery and worker availability."""

    def result() -> int:
        if raises:
            raise ValueError("ordinary job failure")
        return 17

    async def async_job() -> int:
        return result()

    def thread_job() -> int:
        return result()

    async def async_followup() -> int:
        return 29

    def thread_followup() -> int:
        return 29

    threaded = isinstance(executor, AsyncPQThreadPoolExecutor)
    job: Any = thread_job if threaded else async_job
    followup: Any = thread_followup if threaded else async_followup
    if raises:
        with pytest.raises(ValueError, match="ordinary job failure"):
            await asyncio.wait_for(executor.submit_job(job), timeout=2)
    else:
        assert await asyncio.wait_for(executor.submit_job(job), timeout=2) == 17
    assert await asyncio.wait_for(executor.submit_job(followup), timeout=2) == 29
