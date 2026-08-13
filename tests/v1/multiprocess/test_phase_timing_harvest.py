# SPDX-License-Identifier: Apache-2.0
"""Tests for ``harvest_and_publish_phase_timings``: the drain-and-publish
bridge between the native phase-timing registry and the event bus."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod


def test_publishes_one_event_carrying_all_samples(monkeypatch):
    """Drained samples are published verbatim as MP_TRANSFER_PHASE_SAMPLES."""
    samples = [(0, 1, 0, 2.5, 10**9), (1, 1, 0, 5.0, 10**9)]
    monkeypatch.setattr(mod, "_HAS_PHASE_TIMING_HARVEST", True)
    monkeypatch.setattr(
        mod.lmc_ops,
        "harvest_transfer_phase_timings",
        lambda: samples,
        raising=False,
    )
    bus = MagicMock(name="event_bus")

    mod.harvest_and_publish_phase_timings(bus)

    bus.publish.assert_called_once()
    event = bus.publish.call_args.args[0]
    assert event.event_type == EventType.MP_TRANSFER_PHASE_SAMPLES
    assert event.metadata == {"samples": samples}


def test_no_event_when_nothing_finished(monkeypatch):
    """An empty harvest publishes nothing."""
    monkeypatch.setattr(mod, "_HAS_PHASE_TIMING_HARVEST", True)
    monkeypatch.setattr(
        mod.lmc_ops,
        "harvest_transfer_phase_timings",
        lambda: [],
        raising=False,
    )
    bus = MagicMock(name="event_bus")

    mod.harvest_and_publish_phase_timings(bus)

    bus.publish.assert_not_called()


def test_noop_without_native_op(monkeypatch):
    """Without the native op the function neither harvests nor publishes."""
    monkeypatch.setattr(mod, "_HAS_PHASE_TIMING_HARVEST", False)
    harvest = MagicMock(name="harvest")
    monkeypatch.setattr(
        mod.lmc_ops,
        "harvest_transfer_phase_timings",
        harvest,
        raising=False,
    )
    bus = MagicMock(name="event_bus")

    mod.harvest_and_publish_phase_timings(bus)

    harvest.assert_not_called()
    bus.publish.assert_not_called()
