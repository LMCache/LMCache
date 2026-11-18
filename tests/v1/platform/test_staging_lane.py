# SPDX-License-Identifier: Apache-2.0
"""Staging lanes: bank rotation and the bank signals. Streams and events
are faked, so these run without CUDA."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base import staging_lane
from lmcache.v1.platform.base.staging_lane import BankSignals, StagingLane

LOAD = "load"
SCATTER = "scatter"


def test_lane_hands_out_its_banks_in_turn():
    lane = StagingLane(stream=LOAD, banks=["a", "b"], signals=None)
    assert [lane.take_bank() for _ in range(5)] == [0, 1, 0, 1, 0]


@pytest.fixture
def events(monkeypatch):
    """Fake stream events; returns the log of ``("record", n, stream)``,
    ``("wait", n, stream)`` and ``("poll", n)`` entries, where ``n`` numbers
    the events. An event is complete from its third poll on."""
    log: list[tuple] = []

    class _Event:
        def __init__(self, number: int) -> None:
            self._number = number

        def wait_on(self, stream: object) -> None:
            log.append(("wait", self._number, stream))

        def is_complete(self) -> bool:
            polls = sum(1 for entry in log if entry == ("poll", self._number))
            log.append(("poll", self._number))
            return polls == 2

    def record_completion_event(device: object, stream: object) -> _Event:
        number = sum(1 for entry in log if entry[0] == "record")
        log.append(("record", number, stream))
        return _Event(number)

    monkeypatch.setattr(
        staging_lane,
        "platform_stream",
        SimpleNamespace(record_completion_event=record_completion_event),
    )
    return log


def test_signals_free_bank_needs_no_wait_before_its_first_load(events):
    signals = BankSignals(device="gpu")
    signals.wait_free(0, LOAD)
    assert events == []


def test_signals_each_side_waits_for_the_other_sides_last_event(events):
    signals = BankSignals(device="gpu")
    for _ in range(2):
        signals.wait_free(0, LOAD)
        signals.signal_staged(0, LOAD)
        signals.wait_staged(0, SCATTER)
        signals.signal_free(0, SCATTER)
    assert events == [
        ("record", 0, LOAD),
        ("wait", 0, SCATTER),
        ("record", 1, SCATTER),
        ("wait", 1, LOAD),
        ("record", 2, LOAD),
        ("wait", 2, SCATTER),
        ("record", 3, SCATTER),
    ]


def test_signals_host_wait_polls_the_last_free_event_until_complete(events):
    signals = BankSignals(device="gpu")
    signals.wait_free_on_host(0)
    assert events == []
    signals.signal_free(0, SCATTER)
    signals.wait_free_on_host(0)
    assert events == [("record", 0, SCATTER)] + [("poll", 0)] * 3


def test_signals_keep_the_banks_apart(events):
    signals = BankSignals(device="gpu")
    signals.signal_free(0, SCATTER)
    signals.wait_free(1, LOAD)
    assert events == [("record", 0, SCATTER)]
