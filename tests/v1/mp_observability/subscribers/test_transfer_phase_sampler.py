# SPDX-License-Identifier: Apache-2.0
"""Tests for TransferPhaseSampler: pops finished samples from the native
phase-timing recorder onto the event bus when a transfer ends."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.subscribers import transfer_phase_sampler as mod
from lmcache.v1.mp_observability.subscribers.transfer_phase_sampler import (
    TransferPhaseSampler,
)


def _end_event() -> Event:
    return Event(event_type=EventType.MP_STORE_END, session_id="s")


def test_subscribes_to_both_end_events():
    subs = TransferPhaseSampler(MagicMock()).get_subscriptions()
    assert set(subs) == {EventType.MP_STORE_END, EventType.MP_RETRIEVE_END}


def test_publishes_one_event_carrying_all_samples(monkeypatch):
    """Popped samples are published verbatim as MP_TRANSFER_PHASE_SAMPLES."""
    samples = [
        (0, 1, 0, 2.5, 10**9, "s", 1.0, 1.0025),
        (1, 1, 0, 5.0, 10**9, "s", 1.0, 1.005),
    ]
    monkeypatch.setattr(mod, "_HAS_TRANSFER_PHASE_TIMING", True)
    monkeypatch.setattr(
        mod, "_device_ops", MagicMock(pop_completed_phase_timings=lambda: samples)
    )
    bus = MagicMock(name="event_bus")

    TransferPhaseSampler(bus).get_subscriptions()[EventType.MP_STORE_END](_end_event())

    bus.publish.assert_called_once()
    event = bus.publish.call_args.args[0]
    assert event.event_type == EventType.MP_TRANSFER_PHASE_SAMPLES
    assert event.metadata == {"samples": samples}


def test_no_event_when_nothing_finished(monkeypatch):
    monkeypatch.setattr(mod, "_HAS_TRANSFER_PHASE_TIMING", True)
    monkeypatch.setattr(
        mod, "_device_ops", MagicMock(pop_completed_phase_timings=lambda: [])
    )
    bus = MagicMock(name="event_bus")

    TransferPhaseSampler(bus).get_subscriptions()[EventType.MP_STORE_END](_end_event())

    bus.publish.assert_not_called()


def test_noop_without_native_op(monkeypatch):
    """Without the native op the sampler neither pops nor publishes."""
    monkeypatch.setattr(mod, "_HAS_TRANSFER_PHASE_TIMING", False)
    ops = MagicMock()
    monkeypatch.setattr(mod, "_device_ops", ops)
    bus = MagicMock(name="event_bus")

    TransferPhaseSampler(bus).get_subscriptions()[EventType.MP_STORE_END](_end_event())

    ops.pop_completed_phase_timings.assert_not_called()
    bus.publish.assert_not_called()
