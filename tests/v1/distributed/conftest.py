# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for distributed module tests.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)


@pytest.fixture(autouse=True)
def mock_prometheus_classes(monkeypatch):
    """Prevent real Prometheus metric registration during distributed tests.

    L1Manager and StorageManager now self-register observability loggers on
    construction. Without this fixture, creating multiple instances across
    tests would collide in the global Prometheus CollectorRegistry.
    """
    monkeypatch.setattr(PrometheusLogger, "_counter_cls", MagicMock)
    monkeypatch.setattr(PrometheusLogger, "_histogram_cls", MagicMock)
    monkeypatch.setattr(PrometheusLogger, "_gauge_cls", MagicMock)
