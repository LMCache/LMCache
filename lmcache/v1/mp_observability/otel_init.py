# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry SDK initialization for the MP observability system.

Supports two modes:
- **OTLP push** (production): metrics are pushed to an OTel collector.
- **Prometheus pull** (dev/debug): metrics are served on a local ``/metrics``
  endpoint via ``prometheus_client``, no collector needed.

The mode is selected by the *otlp_endpoint* argument to ``init_otel_metrics()``:
- Pass an endpoint string → OTLP push mode
- Pass ``None`` (default) and set ``OTEL_EXPORTER_OTLP_ENDPOINT`` → OTLP push
- Pass ``None`` with no env var set → Prometheus pull fallback
"""

# Future
from __future__ import annotations

# Standard
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def init_otel_metrics(
    otlp_endpoint: str | None = None,
    prometheus_port: int | None = None,
) -> None:
    """Set up the OpenTelemetry MeterProvider.

    When an OTLP endpoint is available (via *otlp_endpoint* arg or
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var), metrics are pushed to an
    OTel collector.  Otherwise, falls back to an in-process Prometheus
    endpoint on *prometheus_port* (default 9090) so metrics can be
    queried directly without a collector.

    Args:
        otlp_endpoint: OTLP gRPC endpoint (e.g. ``http://localhost:4317``).
            When ``None``, reads from ``OTEL_EXPORTER_OTLP_ENDPOINT``.
            If that is also unset, falls back to Prometheus mode.
        prometheus_port: Port for the fallback Prometheus ``/metrics``
            endpoint.  Only used when OTLP is not configured.  Defaults
            to 9090.
    """
    # Third Party
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider

    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    if otlp_endpoint is not None:
        # OTLP push mode
        # Third Party
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics.export import (
            PeriodicExportingMetricReader,
        )

        exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=10000)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
        logger.info(
            "OTel MeterProvider initialised with OTLP exporter (%s)",
            otlp_endpoint,
        )
    else:
        # Prometheus pull fallback — no collector needed
        # Third Party
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        import prometheus_client

        if prometheus_port is None:
            prometheus_port = 9090

        reader = PrometheusMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
        prometheus_client.start_http_server(prometheus_port)
        logger.info(
            "OTel MeterProvider initialised with Prometheus fallback "
            "(http://0.0.0.0:%d/metrics)",
            prometheus_port,
        )


def init_otel_tracing(otlp_endpoint: str | None = None) -> None:
    """Set up the OpenTelemetry TracerProvider with an OTLP exporter.

    Args:
        otlp_endpoint: OTLP gRPC endpoint.  Defaults to
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.  If unset, tracing
            init is skipped (no-op).
    """
    # Third Party
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    if otlp_endpoint is None:
        logger.debug("No OTLP endpoint configured, skipping tracing init")
        return

    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info(
        "OTel TracerProvider initialised with OTLP exporter (%s)",
        otlp_endpoint,
    )
