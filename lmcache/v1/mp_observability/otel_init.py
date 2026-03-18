# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry SDK initialization for the MP observability system."""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.config import ObservabilityConfig

logger = init_logger(__name__)


def init_otel_metrics(config: ObservabilityConfig) -> None:
    """Set up the OpenTelemetry MeterProvider with a Prometheus exporter.

    After this call, any ``opentelemetry.metrics.get_meter()`` in subscriber
    code will produce metrics that are scraped via the ``/metrics`` HTTP
    endpoint on ``config.prometheus_port``.

    This is a no-op when ``config.metrics_enabled`` is False.
    """
    if not config.metrics_enabled:
        return

    # Third Party
    from opentelemetry import metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider

    reader = PrometheusMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    logger.info(
        "OTel MeterProvider initialised with PrometheusMetricReader (port=%d)",
        config.prometheus_port,
    )


def init_otel_tracing(config: ObservabilityConfig) -> None:
    """Set up the OpenTelemetry TracerProvider.

    Currently a placeholder — uncomment and configure an exporter
    (e.g. OTLP, Jaeger) when tracing is needed.

    This is a no-op when ``config.tracing_enabled`` is False.
    """
    if not config.tracing_enabled:
        return

    # Third Party
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    provider = TracerProvider()
    # Default to console export; swap for OTLPSpanExporter in production.
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OTel TracerProvider initialised")
