# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for benchmark request task tracking."""

# Standard
from functools import partial
from unittest.mock import MagicMock
import asyncio

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.workloads.base import DispatchTracker


def test_concurrency_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_inflight must be positive"):
        DispatchTracker(MagicMock(), max_inflight=0)


@pytest.mark.asyncio
async def test_bounded_dispatch_waits_for_a_free_slot() -> None:
    monitor = MagicMock()
    tracker = DispatchTracker(monitor, max_inflight=2)
    release = asyncio.Event()
    started: list[int] = []

    async def send(index: int) -> None:
        started.append(index)
        await release.wait()

    first = await tracker.start(lambda: send(0))
    second = await tracker.start(lambda: send(1))
    third_start = asyncio.create_task(tracker.start(lambda: send(2)))
    await asyncio.sleep(0)

    assert started == [0, 1]
    assert not third_start.done()
    assert tracker.has_pending

    release.set()
    third = await asyncio.wait_for(third_start, timeout=1)
    await asyncio.gather(first, second, third)
    await tracker.wait_one()
    assert not tracker.has_pending
    monitor.log_message.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_task_releases_slot_before_it_starts() -> None:
    monitor = MagicMock()
    tracker = DispatchTracker(monitor, max_inflight=1)
    sent = False

    async def send() -> None:
        nonlocal sent
        sent = True

    cancelled = await tracker.start(send)
    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)

    replacement = await asyncio.wait_for(tracker.start(send), timeout=1)
    await replacement
    await tracker.wait_one()
    assert sent
    assert not tracker.has_pending
    monitor.log_message.assert_not_called()


@pytest.mark.asyncio
async def test_cancelling_a_slot_wait_does_not_create_a_send() -> None:
    tracker = DispatchTracker(MagicMock(), max_inflight=1)
    release = asyncio.Event()
    created = False

    async def blocked_send() -> None:
        await release.wait()

    def cancelled_send() -> asyncio.Future[None]:
        nonlocal created
        created = True
        return asyncio.get_running_loop().create_future()

    first = await tracker.start(blocked_send)
    waiting = asyncio.create_task(tracker.start(cancelled_send))
    await asyncio.sleep(0)
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert not created

    release.set()
    await first
    replacement = await asyncio.wait_for(
        tracker.start(lambda: asyncio.sleep(0)), timeout=1
    )
    await replacement
    await tracker.wait_one()
    assert not tracker.has_pending


@pytest.mark.asyncio
async def test_unexpected_send_error_is_logged_and_releases_slot() -> None:
    monitor = MagicMock()
    tracker = DispatchTracker(monitor, max_inflight=1)

    async def fail() -> None:
        raise RuntimeError("send failed")

    failed = await tracker.start(fail)
    await asyncio.gather(failed, return_exceptions=True)
    replacement = await asyncio.wait_for(
        tracker.start(lambda: asyncio.sleep(0)), timeout=1
    )
    await replacement
    await tracker.wait_one()

    monitor.log_message.assert_called_once_with("Dispatch task failed: send failed")
    assert not tracker.has_pending


@pytest.mark.asyncio
async def test_unbounded_dispatch_starts_every_send_without_waiting() -> None:
    tracker = DispatchTracker(MagicMock())
    release = asyncio.Event()
    started: list[int] = []

    async def send(index: int) -> None:
        started.append(index)
        await release.wait()

    tasks = [await tracker.start(partial(send, i)) for i in range(3)]
    await asyncio.sleep(0)
    assert started == [0, 1, 2]

    release.set()
    await asyncio.gather(*tasks)
    await tracker.wait_one()
    assert not tracker.has_pending
