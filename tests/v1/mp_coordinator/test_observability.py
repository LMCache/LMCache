# SPDX-License-Identifier: Apache-2.0
"""Tests for MP coordinator metrics initialization."""

# Standard
from unittest.mock import patch

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.observability import init_coordinator_metrics


def test_disabled_metrics_are_not_initialized() -> None:
    config = MPCoordinatorConfig(metrics_enabled=False)

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_not_called()


def test_prometheus_metrics_reuse_coordinator_http_server() -> None:
    config = MPCoordinatorConfig(metrics_enabled=True)

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_called_once_with(
        otlp_endpoint=None,
        resource_attributes={"service.name": "lmcache-mp-coordinator"},
        start_http_server=False,
    )


def test_otlp_metrics_reuse_shared_initializer() -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=True,
        otlp_endpoint="http://collector:4317",
    )

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_called_once_with(
        otlp_endpoint="http://collector:4317",
        resource_attributes={"service.name": "lmcache-mp-coordinator"},
        start_http_server=False,
    )
