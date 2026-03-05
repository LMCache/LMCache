# SPDX-License-Identifier: Apache-2.0

"""Tests for TelemetryEvent and EventType."""

# First Party
from lmcache.v1.mp_observability.telemetry.event import (
    EventType,
    TelemetryEvent,
)


class TestEventType:
    def test_has_start_and_end(self):
        assert EventType.START.value == "START"
        assert EventType.END.value == "END"

    def test_exactly_two_members(self):
        assert len(EventType) == 2


class TestTelemetryEvent:
    def test_default_metadata_is_empty_dict(self):
        event = TelemetryEvent(
            name="mp.store",
            event_type=EventType.START,
            session_id="abc",
        )
        assert event.metadata == {}

    def test_metadata_independence(self):
        e1 = TelemetryEvent(name="mp.store", event_type=EventType.START, session_id="a")
        e2 = TelemetryEvent(name="mp.store", event_type=EventType.START, session_id="b")
        e1.metadata["key"] = "value"
        assert "key" not in e2.metadata

    def test_custom_metadata(self):
        event = TelemetryEvent(
            name="mp.lookup",
            event_type=EventType.END,
            session_id="xyz",
            metadata={"tokens": 256, "hit": True, "ratio": 0.95, "model": "llama"},
        )
        assert event.metadata["tokens"] == 256
        assert event.metadata["hit"] is True
        assert event.metadata["ratio"] == 0.95
        assert event.metadata["model"] == "llama"

    def test_timestamp_defaults_to_zero(self):
        event = TelemetryEvent(
            name="mp.store", event_type=EventType.START, session_id="t"
        )
        assert event.timestamp == 0.0
