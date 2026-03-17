# SPDX-License-Identifier: Apache-2.0

"""Tests for TelemetryController, singleton, and create_processors."""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.telemetry.config import TelemetryConfig
from lmcache.v1.mp_observability.telemetry.controller import (
    TelemetryController,
    create_processors,
    get_telemetry_controller,
    init_telemetry_controller,
    log_telemetry,
)
from lmcache.v1.mp_observability.telemetry.event import (
    EventType,
    TelemetryEvent,
)
from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessor,
)
from lmcache.v1.mp_observability.telemetry.processors.logging_processor import (
    LoggingProcessor,
    LoggingProcessorConfig,
)
import lmcache.v1.mp_observability.telemetry.controller as _ctrl_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    name: str = "test.op",
    event_type: EventType = EventType.START,
    session_id: str = "s1",
) -> TelemetryEvent:
    return TelemetryEvent(name=name, event_type=event_type, session_id=session_id)


class _RecordingProcessor(TelemetryProcessor):
    """Test processor that records events."""

    def __init__(self):
        self.events: list[TelemetryEvent] = []
        self.shutdown_called = False

    def on_new_event(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def shutdown(self) -> None:
        self.shutdown_called = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_global_controller():
    """Save and restore the global singleton so tests don't leak state."""
    saved = _ctrl_module._global_controller
    yield
    _ctrl_module._global_controller = saved


@pytest.fixture
def controller():
    """Enabled TelemetryController with small queue for fast tests."""
    return TelemetryController(TelemetryConfig(enabled=True, max_queue_size=100))


# ---------------------------------------------------------------------------
# Processor registration
# ---------------------------------------------------------------------------


class TestProcessorRegistration:
    def test_register_processor(self, controller):
        proc = _RecordingProcessor()
        controller.register_processor(proc)
        assert proc in controller._processors

    def test_register_multiple_processors(self, controller):
        procs = [_RecordingProcessor() for _ in range(3)]
        for p in procs:
            controller.register_processor(p)
        assert len(controller._processors) == 3

    def test_empty_initially(self, controller):
        assert len(controller._processors) == 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_launches_thread(self, controller):
        controller.start()
        assert controller._thread is not None
        assert controller._thread.is_alive()
        controller.stop()

    def test_disabled_noop(self):
        ctrl = TelemetryController(TelemetryConfig(enabled=False))
        ctrl.start()
        assert ctrl._thread is None

    def test_thread_is_daemon(self, controller):
        controller.start()
        assert controller._thread.daemon is True
        controller.stop()

    def test_thread_name(self, controller):
        controller.start()
        assert controller._thread.name == "TelemetryController"
        controller.stop()

    def test_double_start_is_idempotent(self, controller):
        controller.start()
        thread = controller._thread
        controller.start()
        assert controller._thread is thread
        controller.stop()

    def test_double_stop(self, controller):
        controller.start()
        controller.stop()
        controller.stop()

    def test_stop_without_start(self, controller):
        controller.stop()


# ---------------------------------------------------------------------------
# Event processing
# ---------------------------------------------------------------------------


class TestEventProcessing:
    def test_event_reaches_processor(self, controller):
        proc = _RecordingProcessor()
        controller.register_processor(proc)
        controller.start()

        controller.log(_make_event())
        time.sleep(0.15)
        controller.stop()

        assert len(proc.events) == 1
        assert proc.events[0].name == "test.op"

    def test_log_stamps_timestamp(self, controller):
        proc = _RecordingProcessor()
        controller.register_processor(proc)
        controller.start()

        before = time.time()
        controller.log(_make_event())
        after = time.time()
        time.sleep(0.15)
        controller.stop()

        assert len(proc.events) == 1
        assert before <= proc.events[0].timestamp <= after

    def test_multiple_events_in_order(self, controller):
        proc = _RecordingProcessor()
        controller.register_processor(proc)
        controller.start()

        for i in range(5):
            controller.log(_make_event(session_id=str(i)))
        time.sleep(0.15)
        controller.stop()

        assert len(proc.events) == 5
        for i, ev in enumerate(proc.events):
            assert ev.session_id == str(i)

    def test_late_registered_processor(self, controller):
        controller.start()
        time.sleep(0.05)

        proc = _RecordingProcessor()
        controller.register_processor(proc)

        controller.log(_make_event())
        time.sleep(0.15)
        controller.stop()

        assert len(proc.events) >= 1


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


class TestExceptionIsolation:
    def test_bad_processor_doesnt_block_good(self, controller):
        bad = MagicMock(spec=TelemetryProcessor)
        bad.on_new_event.side_effect = RuntimeError("boom")
        good = _RecordingProcessor()

        controller.register_processor(bad)
        controller.register_processor(good)
        controller.start()

        controller.log(_make_event())
        time.sleep(0.15)
        controller.stop()

        assert len(good.events) == 1

    def test_shutdown_exception_isolated(self, controller):
        bad = MagicMock(spec=TelemetryProcessor)
        bad.shutdown.side_effect = RuntimeError("shutdown boom")
        good = _RecordingProcessor()

        controller.register_processor(bad)
        controller.register_processor(good)
        controller.start()
        controller.stop()

        assert good.shutdown_called


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    def test_events_discarded_when_queue_full(self):
        ctrl = TelemetryController(TelemetryConfig(enabled=True, max_queue_size=5))
        # Don't start — nothing drains
        for _ in range(10):
            ctrl.log(_make_event())

        assert len(ctrl._queue) == 5
        assert ctrl._discard_count == 5


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalSingleton:
    def test_get_returns_instance(self):
        ctrl = get_telemetry_controller()
        assert isinstance(ctrl, TelemetryController)

    def test_init_replaces_global(self):
        old = get_telemetry_controller()
        new = init_telemetry_controller(TelemetryConfig(enabled=False))
        assert get_telemetry_controller() is new
        assert get_telemetry_controller() is not old

    def test_default_singleton_is_disabled(self):
        ctrl = get_telemetry_controller()
        assert ctrl._config.enabled is False

    def test_log_telemetry_convenience(self):
        config = TelemetryConfig(enabled=True, max_queue_size=100)
        init_telemetry_controller(config)
        log_telemetry(_make_event())
        assert len(get_telemetry_controller()._queue) == 1


# ---------------------------------------------------------------------------
# Disabled controller
# ---------------------------------------------------------------------------


class TestDisabledController:
    def test_log_is_noop_when_disabled(self):
        ctrl = TelemetryController(TelemetryConfig(enabled=False))
        ctrl.log(_make_event())
        assert len(ctrl._queue) == 0


# ---------------------------------------------------------------------------
# create_processors factory
# ---------------------------------------------------------------------------


class TestCreateProcessors:
    def test_creates_logging_processor(self):
        config = TelemetryConfig(
            processor_configs=[LoggingProcessorConfig.from_dict({})]
        )
        procs = create_processors(config)
        assert len(procs) == 1
        assert isinstance(procs[0], LoggingProcessor)

    def test_creates_multiple(self):
        config = TelemetryConfig(
            processor_configs=[
                LoggingProcessorConfig.from_dict({}),
                LoggingProcessorConfig.from_dict({}),
            ]
        )
        procs = create_processors(config)
        assert len(procs) == 2

    def test_empty_config(self):
        config = TelemetryConfig()
        procs = create_processors(config)
        assert procs == []
