# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's background-loop runner and event-loop monitor."""

# Standard
import asyncio
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.background_loops import (
    BackgroundLoops,
    EventLoopMonitor,
)


@pytest.mark.asyncio
async def test_a_failed_pass_is_counted_and_the_loop_keeps_running():
    loops = BackgroundLoops()
    passes = 0

    async def flaky_pass() -> None:
        nonlocal passes
        passes += 1
        if passes == 1:
            raise RuntimeError("first pass fails")

    task = loops.start("flaky", 0.001, flaky_pass)
    while loops.stats()["flaky"].passes_ok < 2:
        await asyncio.sleep(0.005)
    task.cancel()

    stats = loops.stats()["flaky"]
    assert stats.passes_failed == 1
    assert stats.passes_ok >= 2
    assert stats.last_success_timestamp is not None
    assert stats.interval_seconds == 0.001


@pytest.mark.asyncio
async def test_no_success_is_recorded_while_every_pass_fails():
    loops = BackgroundLoops()

    async def failing_pass() -> None:
        raise RuntimeError("always fails")

    task = loops.start("broken", 0.001, failing_pass)
    while loops.stats()["broken"].passes_failed < 3:
        await asyncio.sleep(0.005)
    task.cancel()

    stats = loops.stats()["broken"]
    assert (stats.passes_ok, stats.last_success_timestamp) == (0, None)


@pytest.mark.asyncio
async def test_cancelling_the_task_stops_the_loop():
    loops = BackgroundLoops()

    async def quiet_pass() -> None:
        return None

    task = loops.start("quiet", 0.001, quiet_pass)
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_loop_name_can_be_started_once():
    loops = BackgroundLoops()

    async def quiet_pass() -> None:
        return None

    task = loops.start("health", 10.0, quiet_pass)
    with pytest.raises(ValueError, match="already started"):
        loops.start("health", 10.0, quiet_pass)
    task.cancel()


@pytest.mark.asyncio
async def test_the_monitor_adds_up_time_the_loop_was_blocked():
    monitor = EventLoopMonitor()
    task = asyncio.create_task(monitor.run())
    await asyncio.sleep(0)
    time.sleep(0.8)  # holds the event loop past the monitor's 0.5 s wake-up
    await asyncio.sleep(0.05)
    task.cancel()

    assert monitor.stats().blocked_seconds >= 0.2
