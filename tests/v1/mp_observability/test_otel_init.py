# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``start_http_server`` flag in OTel metrics init."""

# Standard
from types import ModuleType
from unittest.mock import MagicMock, patch
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability import otel_init
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    init_observability,
)
from lmcache.v1.mp_observability.otel_init import (
    init_grpc_otel_metrics,
    init_otel_metrics,
)


@pytest.fixture(autouse=True)
def _mock_otel_provider(monkeypatch):
    """Avoid mutating the process-global OTel MeterProvider."""
    # Third Party
    from opentelemetry import metrics as otel_metrics

    monkeypatch.setattr(otel_metrics, "set_meter_provider", MagicMock())


def test_init_otel_metrics_starts_http_server_by_default():
    with (
        patch("prometheus_client.start_http_server") as mock_start,
        patch(
            "lmcache.v1.mp_observability.otel_init.init_grpc_otel_metrics"
        ) as mock_grpc,
    ):
        init_otel_metrics(prometheus_port=19090)
    mock_start.assert_called_once_with(19090)
    mock_grpc.assert_not_called()


def test_init_otel_metrics_skips_http_server_when_disabled():
    with (
        patch("prometheus_client.start_http_server") as mock_start,
        patch(
            "lmcache.v1.mp_observability.otel_init.init_grpc_otel_metrics"
        ) as mock_grpc,
    ):
        init_otel_metrics(prometheus_port=19091, start_http_server=False)
    mock_start.assert_not_called()
    mock_grpc.assert_not_called()


def test_init_otel_metrics_registers_grpc_metrics_when_enabled():
    with (
        patch("prometheus_client.start_http_server"),
        patch(
            "lmcache.v1.mp_observability.otel_init.init_grpc_otel_metrics"
        ) as mock_grpc,
    ):
        init_otel_metrics(enable_grpc_metrics=True)
    mock_grpc.assert_called_once()


def test_init_otel_metrics_can_disable_grpc_metrics():
    with (
        patch("prometheus_client.start_http_server"),
        patch(
            "lmcache.v1.mp_observability.otel_init.init_grpc_otel_metrics"
        ) as mock_grpc,
    ):
        init_otel_metrics(enable_grpc_metrics=False)
    mock_grpc.assert_not_called()


def test_init_otel_metrics_otlp_mode_never_starts_prom_server():
    """OTLP push mode must not spawn a Prometheus HTTP server,
    regardless of ``start_http_server``."""
    with (
        patch(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter"
        ),
        patch("prometheus_client.start_http_server") as mock_start,
        patch(
            "lmcache.v1.mp_observability.otel_init.init_grpc_otel_metrics"
        ) as mock_grpc,
    ):
        init_otel_metrics(
            otlp_endpoint="http://localhost:4317",
            start_http_server=True,
        )
    mock_start.assert_not_called()
    mock_grpc.assert_not_called()


def test_init_grpc_otel_metrics_registers_official_plugin(monkeypatch):
    registered_plugins = []

    class FakeOpenTelemetryPlugin:
        def __init__(self, *, meter_provider):
            self.meter_provider = meter_provider

        def register_global(self):
            registered_plugins.append(self)

    module = ModuleType("grpc_observability")
    module.OpenTelemetryPlugin = FakeOpenTelemetryPlugin
    monkeypatch.setitem(sys.modules, "grpc_observability", module)
    monkeypatch.setattr(otel_init.sys, "platform", "linux")
    monkeypatch.setattr(otel_init, "_GRPC_OTEL_PLUGIN", None)

    provider = object()
    init_grpc_otel_metrics(provider)

    assert len(registered_plugins) == 1
    assert registered_plugins[0].meter_provider is provider


def test_init_grpc_otel_metrics_is_idempotent(monkeypatch):
    class FakeOpenTelemetryPlugin:
        def __init__(self, *, meter_provider):
            self.meter_provider = meter_provider

        def register_global(self):
            raise AssertionError("should not register again")

    module = ModuleType("grpc_observability")
    module.OpenTelemetryPlugin = FakeOpenTelemetryPlugin
    monkeypatch.setitem(sys.modules, "grpc_observability", module)
    monkeypatch.setattr(otel_init.sys, "platform", "linux")
    monkeypatch.setattr(otel_init, "_GRPC_OTEL_PLUGIN", object())

    init_grpc_otel_metrics(object())


def test_init_grpc_otel_metrics_skips_non_linux_without_warning(monkeypatch):
    monkeypatch.setattr(otel_init.sys, "platform", "darwin")
    monkeypatch.delitem(sys.modules, "grpc_observability", raising=False)
    monkeypatch.setattr(otel_init, "_GRPC_OTEL_PLUGIN", None)

    with (
        patch("lmcache.v1.mp_observability.otel_init.logger.info") as mock_info,
        patch("lmcache.v1.mp_observability.otel_init.logger.warning") as mock_warning,
    ):
        init_grpc_otel_metrics(object())

    mock_info.assert_called_once()
    mock_warning.assert_not_called()


def _stub_metrics_subscriber_modules(monkeypatch):
    class _Subscriber:
        def __init__(self, *args, **kwargs):
            pass

        def register(self, bus):
            pass

    sampler_module = ModuleType(
        "lmcache.v1.mp_observability.subscribers.transfer_phase_sampler"
    )
    sampler_module.TransferPhaseSampler = _Subscriber
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.mp_observability.subscribers.transfer_phase_sampler",
        sampler_module,
    )

    metrics_module = ModuleType("lmcache.v1.mp_observability.subscribers.metrics")
    for name in [
        "BlendMetricsSubscriber",
        "EngineMetricsSubscriber",
        "EventBusSelfMetricsSubscriber",
        "L0L1ThroughputSubscriber",
        "L0LifecycleSubscriber",
        "L1EvictionLoopSubscriber",
        "L1FailureMetricsSubscriber",
        "L1LifecycleSubscriber",
        "L1MetricsSubscriber",
        "L2FailureMetricsSubscriber",
        "L2MetricsSubscriber",
        "L2ThroughputSubscriber",
        "LookupMetricsSubscriber",
        "MPTransferCountersSubscriber",
        "SMLifecycleSubscriber",
        "TimeoutMetricsSubscriber",
        "TransferPhaseMetricsSubscriber",
    ]:
        setattr(metrics_module, name, _Subscriber)
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.mp_observability.subscribers.metrics",
        metrics_module,
    )


def test_init_observability_propagates_flag_false(monkeypatch):
    """``init_observability`` must forward ``False`` to
    ``init_otel_metrics``."""
    _stub_metrics_subscriber_modules(monkeypatch)
    cfg = ObservabilityConfig(enabled=True, metrics_enabled=True, logging_enabled=False)
    with patch("lmcache.v1.mp_observability.otel_init.init_otel_metrics") as mock_init:
        init_observability(cfg, start_prometheus_http_server=False)
    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["start_http_server"] is False
    assert mock_init.call_args.kwargs["enable_grpc_metrics"] is False


def test_init_observability_propagates_flag_true_by_default(monkeypatch):
    _stub_metrics_subscriber_modules(monkeypatch)
    cfg = ObservabilityConfig(enabled=True, metrics_enabled=True, logging_enabled=False)
    with patch("lmcache.v1.mp_observability.otel_init.init_otel_metrics") as mock_init:
        init_observability(cfg)
    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["start_http_server"] is True
    assert mock_init.call_args.kwargs["enable_grpc_metrics"] is False


def test_init_observability_propagates_grpc_metrics_flag(monkeypatch):
    _stub_metrics_subscriber_modules(monkeypatch)
    cfg = ObservabilityConfig(
        enabled=True,
        metrics_enabled=True,
        grpc_metrics_enabled=False,
        logging_enabled=False,
    )
    with patch("lmcache.v1.mp_observability.otel_init.init_otel_metrics") as mock_init:
        init_observability(cfg)
    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["enable_grpc_metrics"] is False


def test_init_observability_propagates_grpc_metrics_enabled(monkeypatch):
    _stub_metrics_subscriber_modules(monkeypatch)
    cfg = ObservabilityConfig(
        enabled=True,
        metrics_enabled=True,
        grpc_metrics_enabled=True,
        logging_enabled=False,
    )
    with patch("lmcache.v1.mp_observability.otel_init.init_otel_metrics") as mock_init:
        init_observability(cfg)
    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["enable_grpc_metrics"] is True
