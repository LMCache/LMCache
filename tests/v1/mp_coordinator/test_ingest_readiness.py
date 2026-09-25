# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`IngestReadiness`."""

# Standard
import asyncio

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.ingest.event_source import (
    UNKNOWN_LAG,
    CacheEventSourceStatus,
    EventReplayCapability,
)
from lmcache.v1.mp_coordinator.ingest.readiness import IngestReadiness


class _StubSource:
    """A source reporting a fixed, settable lag."""

    def __init__(self, lag: int) -> None:
        self.lag = lag

    async def start(self) -> None:
        """Nothing to start."""

    async def stop(self) -> None:
        """Nothing to stop."""

    def status(self) -> CacheEventSourceStatus:
        """Return the currently set lag."""
        return CacheEventSourceStatus(
            source_name="stub",
            replay_capability=EventReplayCapability.SEEKABLE,
            lag=self.lag,
        )


def test_ready_when_lag_is_within_budget() -> None:
    readiness = IngestReadiness(_StubSource(lag=5), max_lag=10)

    status = readiness.status()

    assert status.ready is True
    assert status.lag == 5
    assert status.max_lag == 10


def test_ready_at_exactly_the_budget() -> None:
    readiness = IngestReadiness(_StubSource(lag=10), max_lag=10)

    assert readiness.status().ready is True


def test_not_ready_when_lag_exceeds_the_budget() -> None:
    readiness = IngestReadiness(_StubSource(lag=11), max_lag=10)

    status = readiness.status()

    assert status.ready is False
    assert status.lag == 11


def test_not_ready_when_lag_is_unknown() -> None:
    """Unknown is never evidence of being caught up -- the safe default."""
    readiness = IngestReadiness(_StubSource(lag=UNKNOWN_LAG), max_lag=10)

    status = readiness.status()

    assert status.ready is False
    assert status.lag == UNKNOWN_LAG


def test_always_ready_for_a_source_that_never_lags() -> None:
    """A push source (HTTP) reports lag 0 always; any non-negative budget
    -- including 0 -- leaves it ready."""
    readiness = IngestReadiness(_StubSource(lag=0), max_lag=0)

    assert readiness.status().ready is True


def test_reflects_a_lag_change_between_calls() -> None:
    """The verdict is read fresh each time, not cached at construction."""
    source = _StubSource(lag=0)
    readiness = IngestReadiness(source, max_lag=10)
    assert readiness.status().ready is True

    source.lag = 20

    assert readiness.status().ready is False


@pytest.mark.asyncio
async def test_wait_until_ready_returns_immediately_when_already_ready() -> None:
    readiness = IngestReadiness(_StubSource(lag=0), max_lag=10)

    await asyncio.wait_for(readiness.wait_until_ready(poll_interval=10.0), timeout=1.0)


@pytest.mark.asyncio
async def test_wait_until_ready_blocks_until_the_lag_clears() -> None:
    """A source that only catches up after a few checks makes the wait
    poll rather than return on its first look."""
    source = _StubSource(lag=100)

    async def _clear_the_lag_shortly() -> None:
        await asyncio.sleep(0.05)
        source.lag = 0

    readiness = IngestReadiness(source, max_lag=10)
    asyncio.create_task(_clear_the_lag_shortly())

    await asyncio.wait_for(readiness.wait_until_ready(poll_interval=0.01), timeout=1.0)
