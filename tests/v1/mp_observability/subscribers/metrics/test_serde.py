# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SerdeMetricsSubscriber."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.subscribers.metrics.serde import SerdeMetricsSubscriber
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    read_counters,
    reader as _reader,
)


def _histogram_count(name: str) -> int:
    data = _reader.get_metrics_data()
    if data is None:
        return 0
    total = 0
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != name:
                    continue
                for dp in metric.data.data_points:
                    total += int(getattr(dp, "count", 0))
    return total


def _event(
    event_type: EventType, session_id: str, timestamp: float, **metadata
) -> Event:
    return Event(
        event_type=event_type,
        session_id=session_id,
        timestamp=timestamp,
        metadata=metadata,
    )


@pytest.fixture
def subscriber() -> SerdeMetricsSubscriber:
    return SerdeMetricsSubscriber()


def test_subscribes_to_serde_start_and_end_events(subscriber: SerdeMetricsSubscriber):
    assert set(subscriber.get_subscriptions()) == {
        EventType.CB_SERDE_ENCODE_START,
        EventType.CB_SERDE_ENCODE_END,
        EventType.CB_SERDE_DECODE_START,
        EventType.CB_SERDE_DECODE_END,
    }


def test_encode_start_end_records_duration_and_byte_counters(
    subscriber: SerdeMetricsSubscriber,
):
    callbacks = subscriber.get_subscriptions()
    before = read_counters()
    before_hist = _histogram_count("lmcache_blend.serde_encode_duration_seconds")

    callbacks[EventType.CB_SERDE_ENCODE_START](
        _event(
            EventType.CB_SERDE_ENCODE_START,
            "encode-1",
            10.0,
            serde_type="fp8",
            num_objects=2,
        )
    )
    callbacks[EventType.CB_SERDE_ENCODE_END](
        _event(
            EventType.CB_SERDE_ENCODE_END,
            "encode-1",
            10.25,
            serde_type="fp8",
            bytes_in=4096,
            bytes_out=2048,
            success=True,
        )
    )

    delta = counter_delta(before, read_counters())
    assert (
        _histogram_count("lmcache_blend.serde_encode_duration_seconds") - before_hist
        == 1
    )
    assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 4096
    assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 2048


def test_decode_start_end_records_duration_and_byte_counters(
    subscriber: SerdeMetricsSubscriber,
):
    callbacks = subscriber.get_subscriptions()
    before = read_counters()
    before_hist = _histogram_count("lmcache_blend.serde_decode_duration_seconds")

    callbacks[EventType.CB_SERDE_DECODE_START](
        _event(
            EventType.CB_SERDE_DECODE_START,
            "decode-1",
            20.0,
            serde_type="cachegen",
            num_objects=1,
        )
    )
    callbacks[EventType.CB_SERDE_DECODE_END](
        _event(
            EventType.CB_SERDE_DECODE_END,
            "decode-1",
            20.1,
            serde_type="cachegen",
            bytes_in=1024,
            bytes_out=4096,
            success=True,
        )
    )

    delta = counter_delta(before, read_counters())
    assert (
        _histogram_count("lmcache_blend.serde_decode_duration_seconds") - before_hist
        == 1
    )
    assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 1024
    assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 4096


def test_encode_failure_increments_failure_counter(subscriber: SerdeMetricsSubscriber):
    before = read_counters()
    subscriber.get_subscriptions()[EventType.CB_SERDE_ENCODE_END](
        _event(
            EventType.CB_SERDE_ENCODE_END,
            "encode-fail",
            30.0,
            serde_type="fp8",
            success=False,
            failure_reason="ValueError",
        )
    )

    delta = counter_delta(before, read_counters())
    assert delta.get("lmcache_blend.serde_failures", 0) >= 1


def test_decode_failure_increments_failure_counter(subscriber: SerdeMetricsSubscriber):
    before = read_counters()
    subscriber.get_subscriptions()[EventType.CB_SERDE_DECODE_END](
        _event(
            EventType.CB_SERDE_DECODE_END,
            "decode-fail",
            40.0,
            serde_type="cachegen",
            success=False,
            failure_reason="RuntimeError",
        )
    )

    delta = counter_delta(before, read_counters())
    assert delta.get("lmcache_blend.serde_failures", 0) >= 1


def test_pending_ops_are_bounded(subscriber: SerdeMetricsSubscriber, monkeypatch):
    # Standard
    import lmcache.v1.mp_observability.subscribers.metrics.serde as serde_mod

    monkeypatch.setattr(serde_mod, "_MAX_PENDING_OPS", 2)
    callbacks = subscriber.get_subscriptions()

    for idx in range(4):
        callbacks[EventType.CB_SERDE_ENCODE_START](
            _event(
                EventType.CB_SERDE_ENCODE_START,
                f"encode-{idx}",
                float(idx),
                serde_type="fp8",
            )
        )

    assert len(subscriber._pending_ops) == 2
    assert "encode:encode-2" in subscriber._pending_ops
    assert "encode:encode-3" in subscriber._pending_ops
