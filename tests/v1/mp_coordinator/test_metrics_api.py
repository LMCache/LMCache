# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP coordinator Prometheus endpoint."""

# Standard
from unittest.mock import patch

# Third Party
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
import pytest

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.observability import register_coordinator_metrics

# Local
from .otel_reader import private_meter, read_values


def _client(config: MPCoordinatorConfig) -> TestClient:
    return TestClient(create_app(config))


def test_prometheus_pull_returns_registry_exposition() -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=True,
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )
    exposition = b"# HELP coordinator_test Coordinator test metric\n"

    with (
        patch(
            "lmcache.v1.mp_coordinator.http_apis.metrics_api.generate_latest",
            return_value=exposition,
        ) as mock_generate,
        _client(config) as client,
    ):
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.content == exposition
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    mock_generate.assert_called_once()


def test_disabled_metrics_return_404() -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=False,
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )

    with (
        patch(
            "lmcache.v1.mp_coordinator.http_apis.metrics_api.generate_latest"
        ) as mock_generate,
        _client(config) as client,
    ):
        response = client.get("/metrics")

    assert response.status_code == 404
    mock_generate.assert_not_called()


def test_otlp_push_returns_404() -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=True,
        otlp_endpoint="http://collector:4317",
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )

    with (
        patch(
            "lmcache.v1.mp_coordinator.http_apis.metrics_api.generate_latest"
        ) as mock_generate,
        _client(config) as client,
    ):
        response = client.get("/metrics")

    assert response.status_code == 404
    mock_generate.assert_not_called()


@pytest.mark.parametrize("otlp_endpoint", [None, "http://collector:4317"])
def test_enabled_app_registers_metrics_over_its_own_components(
    otlp_endpoint: str | None,
) -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=True,
        otlp_endpoint=otlp_endpoint,
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )
    meter, reader = private_meter()

    with patch(
        "lmcache.v1.mp_coordinator.app.register_coordinator_metrics",
        lambda: register_coordinator_metrics(meter),
    ):
        create_app(config)

    names = set(read_values(reader))
    prefix = "lmcache_coordinator."
    # One metric from each component the app builds, found without the app
    # naming any of them.
    assert {
        prefix + "key_directory.placements",
        prefix + "ingest.event_batches_received",
        prefix + "ingest.batches_delivered",
        prefix + "event_loop.blocked_seconds",
        prefix + "controller.up",
    } <= names
    # An HTTP-source coordinator exports no Kafka metrics.
    assert not any(name.startswith(prefix + "kafka.") for name in names)


def test_disabled_app_does_not_register_metrics() -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=False,
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )

    with patch(
        "lmcache.v1.mp_coordinator.app.register_coordinator_metrics"
    ) as mock_register:
        create_app(config)

    mock_register.assert_not_called()
