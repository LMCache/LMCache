# SPDX-License-Identifier: Apache-2.0
"""Staging lanes: bank rotation and the two bank-signal implementations.
Streams, events and the doorbell page are faked, so these run without
CUDA."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base import staging_lane
from lmcache.v1.platform.base.staging_lane import EventBankSignals, StagingLane
from lmcache.v1.platform.devices.cuda import stream_doorbell
from lmcache.v1.platform.devices.cuda.stream_doorbell import DoorbellBankSignals

LOAD = "load"
SCATTER = "scatter"


def test_lane_hands_out_its_banks_in_turn():
    lane = StagingLane(stream=LOAD, banks=["a", "b"], signals=None)
    assert [lane.take_bank() for _ in range(5)] == [0, 1, 0, 1, 0]


@pytest.fixture
def events(monkeypatch):
    """Fake stream events; returns the log of ``("record", n, stream)`` and
    ``("wait", n, stream)`` entries, where ``n`` numbers the events."""
    log: list[tuple] = []

    class _Event:
        def __init__(self, number: int) -> None:
            self._number = number

        def wait_on(self, stream: object) -> None:
            log.append(("wait", self._number, stream))

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


def test_event_signals_free_bank_needs_no_wait_before_its_first_load(events):
    signals = EventBankSignals(device="gpu")
    signals.wait_free(0, LOAD)
    assert events == []


def test_event_signals_each_side_waits_for_the_other_sides_last_event(events):
    signals = EventBankSignals(device="gpu")
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


def test_event_signals_keep_the_banks_apart(events):
    signals = EventBankSignals(device="gpu")
    signals.signal_free(0, SCATTER)
    signals.wait_free(1, LOAD)
    assert events == [("record", 0, SCATTER)]


class _FakeDoorbell:
    """Logs ``("write" | "wait", stream handle, word, value)``."""

    def __init__(self, first_word: int = 0) -> None:
        self.log: list[tuple] = []
        self.fail_writes = False
        self._first_word = first_word

    def take_words(self, count: int) -> int:
        return self._first_word

    def write(self, stream_handle: int, word: int, value: int) -> None:
        if self.fail_writes:
            raise RuntimeError("write rejected")
        self.log.append(("write", stream_handle, word, value))

    def wait_equal(self, stream_handle: int, word: int, value: int) -> None:
        self.log.append(("wait", stream_handle, word, value))


@pytest.fixture
def handles(monkeypatch):
    """Resolve the fake streams to the handles 1 (load) and 2 (scatter)."""
    monkeypatch.setattr(
        stream_doorbell,
        "platform_stream",
        SimpleNamespace(
            stream_handle=lambda device, stream: {LOAD: 1, SCATTER: 2}[stream]
        ),
    )


def _round(signals: DoorbellBankSignals, bank: int) -> None:
    signals.wait_free(bank, LOAD)
    signals.signal_staged(bank, LOAD)
    signals.wait_staged(bank, SCATTER)
    signals.signal_free(bank, SCATTER)


def test_doorbell_signals_pass_the_round_number_through_both_words(handles):
    """Bank 1 of a lane whose words start at 8: staged word 10, free word 11."""
    doorbell = _FakeDoorbell(first_word=8)
    signals = DoorbellBankSignals(doorbell, device="gpu")
    _round(signals, 1)
    _round(signals, 1)
    assert doorbell.log == [
        ("wait", 1, 11, 0),
        ("write", 1, 10, 1),
        ("wait", 2, 10, 1),
        ("write", 2, 11, 1),
        ("wait", 1, 11, 1),
        ("write", 1, 10, 2),
        ("wait", 2, 10, 2),
        ("write", 2, 11, 2),
    ]


def test_doorbell_signals_count_rounds_per_bank(handles):
    doorbell = _FakeDoorbell()
    signals = DoorbellBankSignals(doorbell, device="gpu")
    _round(signals, 0)
    _round(signals, 0)
    _round(signals, 1)
    assert doorbell.log[-4:] == [
        ("wait", 1, 3, 0),
        ("write", 1, 2, 1),
        ("wait", 2, 2, 1),
        ("write", 2, 3, 1),
    ]


def test_doorbell_round_does_not_advance_when_the_staged_write_fails(handles):
    """The next load must wait for the value the free word really holds."""
    doorbell = _FakeDoorbell()
    signals = DoorbellBankSignals(doorbell, device="gpu")
    _round(signals, 0)
    doorbell.fail_writes = True
    with pytest.raises(RuntimeError):
        signals.signal_staged(0, LOAD)
    doorbell.fail_writes = False
    doorbell.log.clear()
    signals.wait_free(0, LOAD)
    assert doorbell.log == [("wait", 1, 1, 1)]


def test_doorbell_round_number_wraps_at_32_bits(handles, monkeypatch):
    doorbell = _FakeDoorbell()
    signals = DoorbellBankSignals(doorbell, device="gpu")
    monkeypatch.setattr(signals, "_rounds", [0xFFFFFFFF, 0])
    _round(signals, 0)
    assert doorbell.log == [
        ("wait", 1, 1, 0xFFFFFFFF),
        ("write", 1, 0, 0),
        ("wait", 2, 0, 0),
        ("write", 2, 1, 0),
    ]


@pytest.mark.parametrize(
    "value, expected",
    [("1", True), ("true", True), ("0", False), ("", False)],
)
def test_doorbell_is_opt_in_through_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv(stream_doorbell.ENV_STAGING_DOORBELL, value)
    assert stream_doorbell.staging_doorbell_requested() is expected
