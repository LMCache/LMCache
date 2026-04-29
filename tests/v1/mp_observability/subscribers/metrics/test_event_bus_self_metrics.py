# SPDX-License-Identifier: Apache-2.0

"""Tests for ``init_event_bus_self_metrics`` registration."""

# Standard
from unittest.mock import MagicMock
import sys

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventBusConfig,
    EventSubscriber,
)
from lmcache.v1.mp_observability.subscribers.metrics.event_bus import (
    init_event_bus_self_metrics,
)


def _install_otel_mock() -> tuple[MagicMock, MagicMock, dict]:
    """Replace ``opentelemetry`` and ``opentelemetry.metrics`` with a mock.

    Returns ``(mock_otel, mock_meter, saved_modules)``.  Restore via
    :func:`_restore_otel_mock`.
    """
    mock_meter = MagicMock()
    mock_otel = MagicMock()
    mock_otel.get_meter.return_value = mock_meter
    mock_otel.metrics = mock_otel
    mock_otel.Observation = lambda value, attrs=None: (value, attrs)

    saved = {k: sys.modules.get(k) for k in ("opentelemetry", "opentelemetry.metrics")}
    sys.modules["opentelemetry"] = mock_otel
    sys.modules["opentelemetry.metrics"] = mock_otel
    return mock_otel, mock_meter, saved


def _restore_otel_mock(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class TestRegistration:
    def test_registers_two_gauges(self):
        bus = EventBus(EventBusConfig(enabled=True))
        mock_otel, mock_meter, saved = _install_otel_mock()
        try:
            init_event_bus_self_metrics(bus, meter_name="test.meter")
            gauge_names = [
                call.args[0]
                for call in mock_meter.create_observable_gauge.call_args_list
            ]
            assert "lmcache_mp.event_bus.queue_depth" in gauge_names
            assert "lmcache_mp.event_bus.drain_lag_seconds" in gauge_names
        finally:
            _restore_otel_mock(saved)

    def test_registers_two_observable_counters(self):
        bus = EventBus(EventBusConfig(enabled=True))
        mock_otel, mock_meter, saved = _install_otel_mock()
        try:
            init_event_bus_self_metrics(bus, meter_name="test.meter")
            counter_names = [
                call.args[0]
                for call in mock_meter.create_observable_counter.call_args_list
            ]
            assert "lmcache_mp.event_bus.dropped_events_total" in counter_names
            assert "lmcache_mp.event_bus.subscriber_exceptions" in counter_names
        finally:
            _restore_otel_mock(saved)

    def test_queue_depth_callback_reads_bus(self):
        bus = EventBus(EventBusConfig(enabled=True))
        # Stage two events without starting the drain thread.
        bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s1"))
        bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s2"))

        mock_otel, mock_meter, saved = _install_otel_mock()
        try:
            init_event_bus_self_metrics(bus, meter_name="test.meter")
            gauge_calls = {
                call.args[0]: call.kwargs["callbacks"][0]
                for call in mock_meter.create_observable_gauge.call_args_list
            }
            cb = gauge_calls["lmcache_mp.event_bus.queue_depth"]
            result = cb(None)
            # register_gauge wraps a scalar return into a single Observation.
            assert result == [(2, None)]
        finally:
            _restore_otel_mock(saved)

    def test_dropped_counter_callback_reflects_drops(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=2))
        # Publish past capacity to force drops; nothing drains.
        for _ in range(5):
            bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s"))

        mock_otel, mock_meter, saved = _install_otel_mock()
        try:
            init_event_bus_self_metrics(bus, meter_name="test.meter")
            counter_calls = {
                call.args[0]: call.kwargs["callbacks"][0]
                for call in mock_meter.create_observable_counter.call_args_list
            }
            cb = counter_calls["lmcache_mp.event_bus.dropped_events_total"]
            result = cb(None)
            # Three events dropped (queue capacity 2, 5 published).
            assert result == [(3, None)]
        finally:
            _restore_otel_mock(saved)

    def test_exceptions_counter_callback_emits_per_subscriber(self):
        # Standard
        import time

        class _BadSub(EventSubscriber):
            def get_subscriptions(self):
                return {EventType.L1_READ_FINISHED: self._on_event}

            def _on_event(self, event):
                raise RuntimeError("boom")

        bus = EventBus(EventBusConfig(enabled=True))
        bus.register_subscriber(_BadSub())
        bus.start()
        bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s1"))
        bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s2"))
        time.sleep(0.15)
        bus.stop()

        mock_otel, mock_meter, saved = _install_otel_mock()
        try:
            init_event_bus_self_metrics(bus, meter_name="test.meter")
            counter_calls = {
                call.args[0]: call.kwargs["callbacks"][0]
                for call in mock_meter.create_observable_counter.call_args_list
            }
            cb = counter_calls["lmcache_mp.event_bus.subscriber_exceptions"]
            result = cb(None)
            # One Observation per subscriber name; both raised twice.
            assert (2, {"subscriber_name": "_BadSub"}) in result
        finally:
            _restore_otel_mock(saved)

    def test_exceptions_counter_empty_when_no_failures(self):
        bus = EventBus(EventBusConfig(enabled=True))
        mock_otel, mock_meter, saved = _install_otel_mock()
        try:
            init_event_bus_self_metrics(bus, meter_name="test.meter")
            counter_calls = {
                call.args[0]: call.kwargs["callbacks"][0]
                for call in mock_meter.create_observable_counter.call_args_list
            }
            cb = counter_calls["lmcache_mp.event_bus.subscriber_exceptions"]
            assert cb(None) == []
        finally:
            _restore_otel_mock(saved)
