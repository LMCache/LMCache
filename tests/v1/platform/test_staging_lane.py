# SPDX-License-Identifier: Apache-2.0
"""Staging lanes: bank rotation and the bank events. Streams and events
are faked, so these run without CUDA."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base import staging_lane
from lmcache.v1.platform.base.staging_lane import StagingLane

LOAD = "load"
SCATTER = "scatter"


def _lane() -> StagingLane:
    return StagingLane("gpu", LOAD, [MagicMock(), MagicMock()])


def test_lane_hands_out_its_banks_in_turn():
    lane = _lane()
    assert [lane.take_bank() for _ in range(5)] == [0, 1, 0, 1, 0]


@pytest.fixture
def events(monkeypatch):
    """Fake stream events; returns the log of ``("record", n, stream)``,
    ``("wait", n, stream)`` and ``("host_wait", n)`` entries, where ``n``
    numbers the events."""
    log: list[tuple] = []

    class _Event:
        def __init__(self, number: int) -> None:
            self._number = number

        def wait_on(self, stream: object) -> None:
            log.append(("wait", self._number, stream))

        def wait(self) -> None:
            log.append(("host_wait", self._number))

    def record_completion_event(
        device: object, stream: object, blocking: bool = False
    ) -> _Event:
        number = sum(1 for entry in log if entry[0] == "record")
        log.append(("record", number, stream))
        return _Event(number)

    monkeypatch.setattr(
        staging_lane,
        "platform_stream",
        SimpleNamespace(record_completion_event=record_completion_event),
    )
    return log


def test_free_bank_needs_no_wait_before_its_first_load(events):
    lane = _lane()
    lane.wait_free(0)
    assert events == []


def test_each_side_waits_for_the_other_sides_last_event(events):
    lane = _lane()
    for _ in range(2):
        lane.wait_free(0)
        lane.signal_staged(0, LOAD)
        lane.wait_staged(0, SCATTER)
        lane.signal_free(0, SCATTER)
    assert events == [
        ("record", 0, LOAD),
        ("wait", 0, SCATTER),
        ("record", 1, SCATTER),
        ("wait", 1, LOAD),
        ("record", 2, LOAD),
        ("wait", 2, SCATTER),
        ("record", 3, SCATTER),
    ]


def test_host_wait_blocks_on_the_last_free_event(events):
    lane = _lane()
    lane.wait_free_on_host(0)
    assert events == []
    lane.signal_free(0, SCATTER)
    lane.wait_free_on_host(0)
    assert events == [("record", 0, SCATTER), ("host_wait", 0)]


def test_bank_events_keep_the_banks_apart(events):
    lane = _lane()
    lane.signal_free(0, SCATTER)
    lane.wait_free(1)
    assert events == [("record", 0, SCATTER)]
