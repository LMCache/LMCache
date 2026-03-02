# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PrometheusController (global-singleton design).

Tests cover:
- register_logger() adds loggers to all_loggers
- start() / stop() lifecycle (no deadlock, thread terminates)
- start() is a no-op when config.enabled is False
- stop() is safe when not started
- _run() calls log_prometheus() on every logger each interval
- _run() takes a snapshot of loggers under lock (thread-safe with late registration)
- Exceptions raised by a logger do not crash the run loop
- Global singleton: get_prometheus_controller() / init_prometheus_controller()
"""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.config import PrometheusConfig
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)
from lmcache.v1.mp_observability.prometheus_controller import (
    PrometheusController,
    get_prometheus_controller,
    init_prometheus_controller,
)
import lmcache.v1.mp_observability.prometheus_controller as _ctrl_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_prometheus_classes(monkeypatch):
    """Prevent real Prometheus metric registration during tests."""
    monkeypatch.setattr(PrometheusLogger, "_counter_cls", MagicMock)
    monkeypatch.setattr(PrometheusLogger, "_histogram_cls", MagicMock)


@pytest.fixture(autouse=True)
def restore_global_controller():
    """Save and restore the global _global_controller so tests don't leak state."""
    saved = _ctrl_module._global_controller
    yield
    _ctrl_module._global_controller = saved


@pytest.fixture
def controller():
    """PrometheusController with a very short log interval for fast tests."""
    return PrometheusController(
        PrometheusConfig(enabled=True, log_interval=0.02),
    )


# ---------------------------------------------------------------------------
# Logger registration
# ---------------------------------------------------------------------------


class TestLoggerRegistration:
    def test_register_logger_adds_to_all_loggers(self, controller):
        mock_logger = MagicMock(spec=PrometheusLogger)
        controller.register_logger(mock_logger)
        assert mock_logger in controller.all_loggers

    def test_register_multiple_loggers(self, controller):
        loggers = [MagicMock(spec=PrometheusLogger) for _ in range(3)]
        for lg in loggers:
            controller.register_logger(lg)
        assert len(controller.all_loggers) == 3
        for lg in loggers:
            assert lg in controller.all_loggers

    def test_all_loggers_empty_initially(self, controller):
        assert len(controller.all_loggers) == 0


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_launches_thread(self, controller):
        controller.start()
        assert controller._thread is not None
        assert controller._thread.is_alive()
        controller.stop()

    def test_stop_joins_thread(self, controller):
        controller.start()
        controller.stop()
        assert not controller._thread.is_alive()

    def test_stop_without_start_does_not_raise(self, controller):
        """Calling stop() before start() must not raise."""
        controller.stop()

    def test_start_noop_when_disabled(self):
        ctrl = PrometheusController(PrometheusConfig(enabled=False))
        ctrl.start()
        assert ctrl._thread is None

    def test_stop_safe_when_disabled(self):
        ctrl = PrometheusController(PrometheusConfig(enabled=False))
        ctrl.start()
        ctrl.stop()  # Should not raise

    def test_thread_is_daemon(self, controller):
        controller.start()
        assert controller._thread.daemon is True
        controller.stop()

    def test_thread_name(self, controller):
        controller.start()
        assert controller._thread.name == "PrometheusController"
        controller.stop()

    def test_double_stop_is_safe(self, controller):
        """Calling stop() twice must not raise."""
        controller.start()
        controller.stop()
        controller.stop()


# ---------------------------------------------------------------------------
# Periodic flushing
# ---------------------------------------------------------------------------


class TestPeriodicFlushing:
    def test_log_prometheus_called_multiple_times_during_run(self, controller):
        """With a 20 ms interval, running for ~150 ms should trigger >= 3 flushes."""
        mock_logger = MagicMock()
        controller.register_logger(mock_logger)

        controller.start()
        time.sleep(0.15)
        controller.stop()

        assert mock_logger.log_prometheus.call_count >= 3

    def test_log_prometheus_not_called_before_first_interval(self, controller):
        """log_prometheus() must NOT be called immediately on start."""
        mock_logger = MagicMock()
        controller.register_logger(mock_logger)

        controller.start()
        time.sleep(0.005)
        controller.stop()

        assert mock_logger.log_prometheus.call_count == 0

    def test_late_registered_logger_is_flushed(self, controller):
        """A logger registered after start() should still be flushed."""
        controller.start()
        time.sleep(0.01)

        late_logger = MagicMock()
        controller.register_logger(late_logger)
        time.sleep(0.10)
        controller.stop()

        assert late_logger.log_prometheus.call_count >= 1


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


class TestExceptionIsolation:
    def test_exception_in_logger_does_not_crash_loop(self, controller):
        """If a logger's log_prometheus() raises, the loop continues."""
        bad_logger = MagicMock()
        bad_logger.log_prometheus.side_effect = RuntimeError("boom")
        good_logger = MagicMock()

        controller.register_logger(bad_logger)
        controller.register_logger(good_logger)
        controller.start()
        time.sleep(0.12)
        controller.stop()

        assert good_logger.log_prometheus.call_count >= 3

    def test_bad_logger_does_not_prevent_repeated_calls(self, controller):
        """Even after an exception, the bad logger is retried each interval."""
        bad_logger = MagicMock()
        bad_logger.log_prometheus.side_effect = ValueError("always fails")
        controller.register_logger(bad_logger)

        controller.start()
        time.sleep(0.12)
        controller.stop()

        assert bad_logger.log_prometheus.call_count >= 3


# ---------------------------------------------------------------------------
# Stop flag terminates the loop
# ---------------------------------------------------------------------------


class TestStopFlag:
    def test_stop_terminates_loop_quickly(self, controller):
        """After stop(), the thread should join within a reasonable timeout."""
        controller.start()
        time.sleep(0.01)

        t0 = time.perf_counter()
        controller.stop()
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.5, f"stop() took too long: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalSingleton:
    def test_get_returns_valid_instance(self):
        ctrl = get_prometheus_controller()
        assert isinstance(ctrl, PrometheusController)

    def test_init_replaces_global(self):
        old = get_prometheus_controller()
        new = init_prometheus_controller(PrometheusConfig(enabled=False))
        assert get_prometheus_controller() is new
        assert get_prometheus_controller() is not old

    def test_default_singleton_is_disabled(self):
        """The module-level default should be disabled (safe no-op)."""
        # After restore_global_controller fixture restores the original:
        # just check we can register without error
        ctrl = get_prometheus_controller()
        mock_logger = MagicMock(spec=PrometheusLogger)
        ctrl.register_logger(mock_logger)
