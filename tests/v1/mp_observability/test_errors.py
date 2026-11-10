# SPDX-License-Identifier: Apache-2.0

"""Tests for LMCacheTimeoutError (observability-aware timeout error).

Verifies that the class is a drop-in ``TimeoutError`` subclass, is a no-op
when observability is disabled, and otherwise publishes a ``TIMEOUT_RAISED``
event to the global EventBus carrying the message, exception type, and a
captured stack trace.
"""

# Standard
from unittest.mock import Mock
import subprocess
import sys
import time

# Third Party
import pytest

# First Party
from lmcache.v1.common.timeouts import LMCacheTimeoutError
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBusConfig,
    init_event_bus,
)

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15


@pytest.fixture(autouse=True)
def _restore_global_bus():
    """Reset the global EventBus to the disabled default after each test."""
    yield
    init_event_bus(EventBusConfig(enabled=False))


class TestLMCacheTimeoutErrorType:
    def test_is_timeout_error_subclass(self) -> None:
        assert issubclass(LMCacheTimeoutError, TimeoutError)

    def test_caught_by_except_timeout_error(self) -> None:
        init_event_bus(EventBusConfig(enabled=False))
        with pytest.raises(TimeoutError):
            raise LMCacheTimeoutError("boom")

    def test_message_preserved(self) -> None:
        init_event_bus(EventBusConfig(enabled=False))
        err = LMCacheTimeoutError("the message")
        assert str(err) == "the message"


class TestLMCacheTimeoutErrorEmission:
    def test_no_event_when_observability_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bus = init_event_bus(EventBusConfig(enabled=False))
        captured: list[Event] = []
        bus.subscribe(EventType.TIMEOUT_RAISED, captured.append)
        format_stack = Mock(return_value=[])
        monkeypatch.setattr("traceback.format_stack", format_stack)
        bus.start()  # no-op when disabled
        LMCacheTimeoutError("nope")
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        assert captured == []
        format_stack.assert_not_called()

    def test_publishes_event_when_enabled(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        captured: list[Event] = []
        bus.subscribe(EventType.TIMEOUT_RAISED, captured.append)
        bus.start()
        LMCacheTimeoutError("timed out!", session_id="req-7")
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert len(captured) == 1
        event = captured[0]
        assert event.event_type == EventType.TIMEOUT_RAISED
        assert event.session_id == "req-7"
        assert event.metadata["message"] == "timed out!"
        assert event.metadata["exception_type"] == "LMCacheTimeoutError"
        # The construction stack should be captured and end at the caller.
        assert "test_publishes_event_when_enabled" in event.metadata["stacktrace"]
        assert "timeouts.py" not in event.metadata["stacktrace"]

    def test_subclass_reports_its_own_type(self) -> None:
        class MyTimeout(LMCacheTimeoutError):
            pass

        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        captured: list[Event] = []
        bus.subscribe(EventType.TIMEOUT_RAISED, captured.append)
        bus.start()
        MyTimeout("sub")
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert len(captured) == 1
        assert captured[0].metadata["exception_type"] == "MyTimeout"

    def test_publish_failure_still_raises_the_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))

        def broken_publish(event: Event) -> None:
            raise RuntimeError("bus is broken")

        monkeypatch.setattr(bus, "publish", broken_publish)
        with pytest.raises(LMCacheTimeoutError, match="still a timeout"):
            raise LMCacheTimeoutError("still a timeout")


def test_timeout_error_does_not_load_mp_observability() -> None:
    script = r"""
import sys

from lmcache.v1.common.timeouts import LMCacheTimeoutError

LMCacheTimeoutError("t")
loaded = [m for m in sys.modules if m.startswith("lmcache.v1.mp_observability")]
assert loaded == [], loaded
"""
    subprocess.run([sys.executable, "-c", script], check=True)
