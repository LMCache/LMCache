# SPDX-License-Identifier: Apache-2.0
"""Tests for MUSA stream-ordered completion and event recording."""

# Third Party
import pytest

# First Party
from lmcache.v1.platform.musa import device_ops


def test_completion_is_enqueued_after_stream_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion publication follows synchronization of the MUSA stream."""
    calls: list[int] = []
    ops = device_ops.MusaDeviceOps()
    ops.drain_recorded_completions()
    monkeypatch.setattr(device_ops, "_synchronize_stream_pointer", calls.append)

    ops.record_completion_on_stream(17, "finish", b"payload")

    assert calls == [17]
    assert ops.drain_recorded_completions() == [("finish", b"payload")]


def test_event_is_enqueued_after_stream_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event publication follows synchronization of the MUSA stream."""
    calls: list[int] = []
    ops = device_ops.MusaDeviceOps()
    ops.drain_recorded_events()
    monkeypatch.setattr(device_ops, "_synchronize_stream_pointer", calls.append)

    ops.record_event_on_stream(
        23,
        "mp.store.end",
        "request-1",
        {"device": "musa:0"},
        {"stored_count": 1},
    )

    assert calls == [23]
    events = ops.drain_recorded_events()
    assert len(events) == 1
    event_type, session_id, _timestamp, string_metadata, int_metadata = events[0]
    assert event_type == "mp.store.end"
    assert session_id == "request-1"
    assert string_metadata == {"device": "musa:0"}
    assert int_metadata == {"stored_count": 1}
