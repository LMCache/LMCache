# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for L1ManagerStatsLogger.

Tests cover:
- L1 counter accumulation for each callback
- reserved_read / reserved_write callbacks are no-ops
- deleted_by_manager increments eviction counter
- log_prometheus() atomically swaps and resets stats
- log_prometheus() forwards accumulated counts to Prometheus metrics
- Thread safety: concurrent callbacks from multiple threads
"""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.logger.l1_stats_logger import (
    L1ManagerStatsLogger,
)
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_key(n: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([n % 256]) * 32, model_name="m", kv_rank=0)


def make_keys(count: int, offset: int = 0) -> list[ObjectKey]:
    return [make_key(i + offset) for i in range(count)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_prometheus_classes(monkeypatch):
    """Replace real Prometheus metric classes with MagicMock to avoid
    duplicate-registration errors across test runs and to let us inspect calls.
    """
    monkeypatch.setattr(PrometheusLogger, "_counter_cls", MagicMock)
    monkeypatch.setattr(PrometheusLogger, "_histogram_cls", MagicMock)


@pytest.fixture
def logger() -> L1ManagerStatsLogger:
    return L1ManagerStatsLogger()


# ---------------------------------------------------------------------------
# L1-level counter callbacks
# ---------------------------------------------------------------------------


class TestL1CounterCallbacks:
    def test_reserved_read_is_noop(self, logger):
        """on_l1_keys_reserved_read is a no-op — does not increment counters."""
        logger.on_l1_keys_reserved_read(make_keys(5))
        assert logger.stats.interval_l1_read_keys == 0

    def test_read_finished_increments_key_counter(self, logger):
        logger.on_l1_keys_read_finished(make_keys(5))
        assert logger.stats.interval_l1_read_keys == 5

    def test_read_finished_multiple_batches_accumulate(self, logger):
        logger.on_l1_keys_read_finished(make_keys(3))
        logger.on_l1_keys_read_finished(make_keys(4, offset=10))
        assert logger.stats.interval_l1_read_keys == 7

    def test_reserved_write_is_noop(self, logger):
        """on_l1_keys_reserved_write is a no-op — does not increment counters."""
        logger.on_l1_keys_reserved_write(make_keys(6))
        assert logger.stats.interval_l1_write_keys == 0

    def test_write_finished_increments_key_counter(self, logger):
        logger.on_l1_keys_write_finished(make_keys(6))
        assert logger.stats.interval_l1_write_keys == 6

    def test_deleted_by_manager_increments_eviction_counter(self, logger):
        logger.on_l1_keys_deleted_by_manager(make_keys(2))
        assert logger.stats.interval_l1_evicted_keys == 2

    def test_multiple_delete_calls_accumulate(self, logger):
        logger.on_l1_keys_deleted_by_manager(make_keys(3))
        logger.on_l1_keys_deleted_by_manager(make_keys(4, offset=10))
        assert logger.stats.interval_l1_evicted_keys == 7

    def test_finish_write_and_reserve_read_increments_write_counter(self, logger):
        logger.on_l1_keys_finish_write_and_reserve_read(make_keys(4))
        assert logger.stats.interval_l1_write_keys == 4

    def test_finish_write_and_reserve_read_accumulates_with_write_finished(
        self, logger
    ):
        logger.on_l1_keys_write_finished(make_keys(3))
        logger.on_l1_keys_finish_write_and_reserve_read(make_keys(5, offset=10))
        assert logger.stats.interval_l1_write_keys == 8


# ---------------------------------------------------------------------------
# log_prometheus(): stats swap and Prometheus metric forwarding
# ---------------------------------------------------------------------------


class TestLogPrometheus:
    def test_stats_are_reset_after_log_prometheus(self, logger):
        logger.on_l1_keys_read_finished(make_keys(4))
        logger.on_l1_keys_deleted_by_manager(make_keys(1))

        logger.log_prometheus()

        s = logger.stats
        assert s.interval_l1_read_keys == 0
        assert s.interval_l1_write_keys == 0
        assert s.interval_l1_evicted_keys == 0

    def test_l1_counters_forwarded_to_prometheus(self, logger):
        logger.on_l1_keys_read_finished(make_keys(5))
        logger.on_l1_keys_write_finished(make_keys(3))
        logger.on_l1_keys_deleted_by_manager(make_keys(2))

        logger.log_prometheus()

        logger._l1_read_keys_counter.inc.assert_called_once_with(5)
        logger._l1_write_keys_counter.inc.assert_called_once_with(3)
        logger._l1_evicted_keys_counter.inc.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_l1_callbacks_accumulate_correctly(self, logger):
        n_threads = 8
        calls_per_thread = 50
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(calls_per_thread):
                    logger.on_l1_keys_read_finished(make_keys(1))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert logger.stats.interval_l1_read_keys == n_threads * calls_per_thread
