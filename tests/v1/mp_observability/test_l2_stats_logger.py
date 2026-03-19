# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for L2StatsLogger.

Tests cover:
- on_l2_keys_stored, on_l2_keys_accessed, on_l2_keys_deleted
  are no-ops that do not raise
- log_prometheus is a no-op that does not raise
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.logger.l2_stats_logger import (
    L2ManagerStatsLogger,
)
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)


@pytest.fixture(autouse=True)
def mock_prometheus_classes(monkeypatch):
    """Replace real Prometheus metric classes with MagicMock to avoid
    duplicate-registration errors across test runs."""
    monkeypatch.setattr(PrometheusLogger, "_counter_cls", MagicMock)
    monkeypatch.setattr(PrometheusLogger, "_histogram_cls", MagicMock)


@pytest.fixture
def logger() -> L2ManagerStatsLogger:
    return L2ManagerStatsLogger()


class TestL2Callbacks:
    def test_l2_keys_stored_is_noop(self, logger):
        """on_l2_keys_stored must not raise."""
        logger.on_l2_keys_stored([])

    def test_l2_keys_accessed_is_noop(self, logger):
        """on_l2_keys_accessed must not raise."""
        logger.on_l2_keys_accessed([])

    def test_l2_keys_deleted_is_noop(self, logger):
        """on_l2_keys_deleted must not raise."""
        logger.on_l2_keys_deleted([])

    def test_log_prometheus_is_noop(self, logger):
        """log_prometheus must not raise."""
        logger.log_prometheus()
