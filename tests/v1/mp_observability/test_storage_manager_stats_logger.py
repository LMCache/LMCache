# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for StorageManagerStatsLogger.

Tests cover:
- Stat accumulation for each SM callback
- No-op finish callbacks leave stats unchanged
- log_prometheus() atomically swaps and resets stats
- log_prometheus() forwards accumulated counts to Prometheus metrics
- Thread safety: concurrent callbacks from multiple threads
"""

# Standard
from unittest.mock import MagicMock, call
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)
from lmcache.v1.mp_observability.logger.storage_manager_stats_logger import (
    StorageManagerStatsLogger,
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
def logger() -> StorageManagerStatsLogger:
    return StorageManagerStatsLogger()


# ---------------------------------------------------------------------------
# SM-level callback tests
# ---------------------------------------------------------------------------


class TestSmReadCallbacks:
    def test_single_call_increments_request_count(self, logger):
        logger.on_sm_read_prefetched(succeeded_keys=make_keys(2), failed_keys=[])
        assert logger.stats.interval_sm_read_requests == 1

    def test_succeed_and_failed_keys_counted_separately(self, logger):
        logger.on_sm_read_prefetched(
            succeeded_keys=make_keys(3), failed_keys=make_keys(1, offset=100)
        )
        assert logger.stats.interval_sm_read_succeed_keys == 3
        assert logger.stats.interval_sm_read_failed_keys == 1

    def test_multiple_calls_accumulate(self, logger):
        for _ in range(5):
            logger.on_sm_read_prefetched(
                succeeded_keys=make_keys(2), failed_keys=make_keys(1, offset=50)
            )
        assert logger.stats.interval_sm_read_requests == 5
        assert logger.stats.interval_sm_read_succeed_keys == 10
        assert logger.stats.interval_sm_read_failed_keys == 5

    def test_all_misses(self, logger):
        logger.on_sm_read_prefetched(succeeded_keys=[], failed_keys=make_keys(4))
        assert logger.stats.interval_sm_read_succeed_keys == 0
        assert logger.stats.interval_sm_read_failed_keys == 4

    def test_finish_callback_is_noop(self, logger):
        logger.on_sm_read_prefetched(succeeded_keys=make_keys(2), failed_keys=[])
        before = logger.stats.interval_sm_read_requests

        logger.on_sm_read_prefetched_finished(
            succeeded_keys=make_keys(2), failed_keys=[]
        )
        assert logger.stats.interval_sm_read_requests == before
        assert logger.stats.interval_sm_read_succeed_keys == 2
        assert logger.stats.interval_sm_read_failed_keys == 0


class TestSmWriteCallbacks:
    def test_single_call_increments_request_count(self, logger):
        logger.on_sm_reserved_write(succeeded_keys=make_keys(3), failed_keys=[])
        assert logger.stats.interval_sm_write_requests == 1

    def test_success_and_failed_keys_counted_separately(self, logger):
        logger.on_sm_reserved_write(
            succeeded_keys=make_keys(4), failed_keys=make_keys(2, offset=100)
        )
        assert logger.stats.interval_sm_write_succeed_keys == 4
        assert logger.stats.interval_sm_write_failed_keys == 2

    def test_multiple_calls_accumulate(self, logger):
        for _ in range(3):
            logger.on_sm_reserved_write(
                succeeded_keys=make_keys(2), failed_keys=make_keys(1, offset=50)
            )
        assert logger.stats.interval_sm_write_requests == 3
        assert logger.stats.interval_sm_write_succeed_keys == 6
        assert logger.stats.interval_sm_write_failed_keys == 3

    def test_finish_callback_is_noop(self, logger):
        logger.on_sm_reserved_write(succeeded_keys=make_keys(2), failed_keys=[])
        before_requests = logger.stats.interval_sm_write_requests
        before_succeed = logger.stats.interval_sm_write_succeed_keys

        logger.on_sm_write_finished(succeeded_keys=make_keys(2), failed_keys=[])

        assert logger.stats.interval_sm_write_requests == before_requests
        assert logger.stats.interval_sm_write_succeed_keys == before_succeed


# ---------------------------------------------------------------------------
# log_prometheus(): stats swap and Prometheus metric forwarding
# ---------------------------------------------------------------------------


class TestLogPrometheus:
    def test_stats_are_reset_after_log_prometheus(self, logger):
        logger.on_sm_read_prefetched(succeeded_keys=make_keys(3), failed_keys=[])
        logger.on_sm_reserved_write(succeeded_keys=make_keys(2), failed_keys=[])

        logger.log_prometheus()

        s = logger.stats
        assert s.interval_sm_read_requests == 0
        assert s.interval_sm_read_succeed_keys == 0
        assert s.interval_sm_read_failed_keys == 0
        assert s.interval_sm_write_requests == 0
        assert s.interval_sm_write_succeed_keys == 0
        assert s.interval_sm_write_failed_keys == 0

    def test_sm_counters_forwarded_to_prometheus(self, logger):
        logger.on_sm_read_prefetched(
            succeeded_keys=make_keys(3), failed_keys=make_keys(1, offset=50)
        )
        logger.on_sm_reserved_write(
            succeeded_keys=make_keys(2), failed_keys=make_keys(1, offset=100)
        )

        logger.log_prometheus()

        logger._sm_read_requests_counter.inc.assert_called_once_with(1)
        logger._sm_read_succeed_keys_counter.inc.assert_called_once_with(3)
        logger._sm_read_failed_keys_counter.inc.assert_called_once_with(1)
        logger._sm_write_requests_counter.inc.assert_called_once_with(1)
        logger._sm_write_succeed_keys_counter.inc.assert_called_once_with(2)
        logger._sm_write_failed_keys_counter.inc.assert_called_once_with(1)

    def test_log_prometheus_with_zero_stats_still_calls_inc(self, logger):
        logger.log_prometheus()
        logger._sm_read_requests_counter.inc.assert_called_once_with(0)

    def test_multiple_log_prometheus_calls_are_independent(self, logger):
        logger.on_sm_read_prefetched(succeeded_keys=make_keys(2), failed_keys=[])
        logger.log_prometheus()

        logger.log_prometheus()

        calls = logger._sm_read_requests_counter.inc.call_args_list
        assert calls == [call(1), call(0)]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_sm_callbacks_accumulate_correctly(self, logger):
        n_threads = 8
        calls_per_thread = 50
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(calls_per_thread):
                    logger.on_sm_read_prefetched(
                        succeeded_keys=make_keys(1), failed_keys=[]
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert logger.stats.interval_sm_read_requests == n_threads * calls_per_thread
        assert (
            logger.stats.interval_sm_read_succeed_keys == n_threads * calls_per_thread
        )

    def test_concurrent_log_prometheus_and_callbacks(self, logger):
        n_callbacks = 200
        errors: list[Exception] = []

        def fire_callbacks():
            try:
                for _ in range(n_callbacks):
                    logger.on_sm_read_prefetched(
                        succeeded_keys=make_keys(1), failed_keys=[]
                    )
            except Exception as e:
                errors.append(e)

        def flush_periodically():
            try:
                for _ in range(10):
                    time.sleep(0.001)
                    logger.log_prometheus()
            except Exception as e:
                errors.append(e)

        t_cb = threading.Thread(target=fire_callbacks)
        t_flush = threading.Thread(target=flush_periodically)
        t_cb.start()
        t_flush.start()
        t_cb.join()
        t_flush.join()

        assert not errors
