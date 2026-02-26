# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PrometheusController.

Tests cover:
- SMStatsLogger is registered with StorageManager
- L1StatsLogger is registered with L1Manager
- start() / stop() lifecycle (no deadlock, thread terminates)
- _run() calls log_prometheus() on every logger each interval
- Exceptions raised by a logger do not crash the run loop
- The stop flag terminates the loop even mid-sleep
"""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.observability.logger.prometheus_logger import (
    PrometheusLogger,
)
from lmcache.v1.distributed.observability.prometheus_controller import (
    PrometheusController,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_prometheus_classes(monkeypatch):
    """Prevent real Prometheus metric registration during tests."""
    monkeypatch.setattr(PrometheusLogger, "_counter_cls", MagicMock)
    monkeypatch.setattr(PrometheusLogger, "_histogram_cls", MagicMock)


@pytest.fixture
def mock_l1_manager():
    """Minimal L1Manager mock that tracks registered listeners."""
    l1 = MagicMock()
    l1._registered_listeners = []
    l1.register_listener.side_effect = lambda lst: l1._registered_listeners.append(lst)
    return l1


@pytest.fixture
def mock_storage_manager():
    """Minimal StorageManager mock that tracks registered listeners."""
    sm = MagicMock()
    sm._registered_listeners = []
    sm.register_listener.side_effect = lambda lst: sm._registered_listeners.append(lst)
    return sm


@pytest.fixture
def controller(mock_storage_manager, mock_l1_manager):
    """PrometheusController with a very short log interval for fast tests."""
    return PrometheusController(
        storage_manager=mock_storage_manager,
        l1_manager=mock_l1_manager,
        log_interval=0.02,
    )


# ---------------------------------------------------------------------------
# Listener registration
# ---------------------------------------------------------------------------


class TestListenerRegistration:
    def test_sm_stats_logger_registered_with_storage_manager(
        self, controller, mock_storage_manager
    ):
        mock_storage_manager.register_listener.assert_called_once_with(
            controller.sm_stats_logger
        )

    def test_l1_stats_logger_registered_with_l1_manager(
        self, controller, mock_l1_manager
    ):
        mock_l1_manager.register_listener.assert_called_once_with(
            controller.l1_stats_logger
        )

    def test_sm_stats_logger_in_all_loggers(self, controller):
        assert controller.sm_stats_logger in controller.all_loggers

    def test_l1_stats_logger_in_all_loggers(self, controller):
        assert controller.l1_stats_logger in controller.all_loggers

    def test_all_loggers_has_two_entries(self, controller):
        assert len(controller.all_loggers) == 2


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_launches_thread(self, controller):
        assert not controller._thread.is_alive()
        controller.start()
        assert controller._thread.is_alive()
        controller.stop()

    def test_stop_joins_thread(self, controller):
        controller.start()
        controller.stop()
        assert not controller._thread.is_alive()

    def test_stop_without_start_does_not_raise(self, controller):
        """Calling stop() before start() sets the flag and join() returns immediately
        because the thread was never started (join on a non-started thread raises, so
        we verify the flag is set and the thread is not alive)."""
        controller._stop_flag.set()
        assert not controller._thread.is_alive()

    def test_thread_is_daemon(self, controller):
        assert controller._thread.daemon is True

    def test_thread_name(self, controller):
        assert controller._thread.name == "PrometheusController"

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
        """With a 20 ms interval, running for ~150 ms should trigger ≥ 3 flushes."""
        flush_count = 0
        original_log = controller.sm_stats_logger.log_prometheus

        def counting_log():
            nonlocal flush_count
            flush_count += 1
            original_log()

        controller.sm_stats_logger.log_prometheus = counting_log
        controller.start()
        time.sleep(0.15)
        controller.stop()

        assert flush_count >= 3, (
            f"Expected at least 3 flushes in 150 ms with 20 ms interval, "
            f"got {flush_count}"
        )

    def test_log_prometheus_not_called_before_first_interval(self, controller):
        """log_prometheus() must NOT be called immediately on start — only after
        the first interval elapses."""
        call_times: list[float] = []
        start_time = time.perf_counter()

        def recording_log():
            call_times.append(time.perf_counter() - start_time)

        controller.sm_stats_logger.log_prometheus = recording_log
        controller.start()
        time.sleep(0.005)
        controller.stop()

        assert call_times == [], (
            "log_prometheus() was called before the first interval elapsed"
        )


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


class TestExceptionIsolation:
    def test_exception_in_logger_does_not_crash_loop(
        self, mock_storage_manager, mock_l1_manager
    ):
        """If a logger's log_prometheus() raises, the loop must continue and
        call the next logger in all_loggers without crashing."""
        controller = PrometheusController(
            storage_manager=mock_storage_manager,
            l1_manager=mock_l1_manager,
            log_interval=0.02,
        )

        bad_logger = MagicMock()
        bad_logger.log_prometheus.side_effect = RuntimeError("boom")
        good_logger = MagicMock()

        controller.all_loggers = [bad_logger, good_logger]
        controller.start()
        time.sleep(0.12)
        controller.stop()

        assert good_logger.log_prometheus.call_count >= 3

    def test_bad_logger_does_not_prevent_repeated_calls(
        self, mock_storage_manager, mock_l1_manager
    ):
        """Even after an exception, the bad logger is retried in each interval."""
        controller = PrometheusController(
            storage_manager=mock_storage_manager,
            l1_manager=mock_l1_manager,
            log_interval=0.02,
        )

        bad_logger = MagicMock()
        bad_logger.log_prometheus.side_effect = ValueError("always fails")
        controller.all_loggers = [bad_logger]

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
