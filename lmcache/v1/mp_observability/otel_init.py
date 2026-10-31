# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry SDK initialization for the MP observability system.

Supports two modes, controlled by the ``otlp_endpoint`` field in
``ObservabilityConfig``:

- **OTLP push** (production): metrics/traces are pushed to an OTel collector.
- **Prometheus pull** (dev/debug): metrics are served on a local ``/metrics``
  endpoint via ``prometheus_client``, no collector needed.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING
import os
import sys

if TYPE_CHECKING:
    # Third Party
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import Metric, MetricReader, MetricsData
    from opentelemetry.sdk.resources import Resource

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_GRPC_OTEL_PLUGIN: object | None = None
_GRPC_METRICS_ENABLED = False
_METRICS_RESOURCE: "Resource | None" = None
_METRICS_SOURCES: list[Callable[[], Iterable["Metric"]]] = []

# gRFC A66's latency buckets, in seconds. grpcio-observability leaves the
# SDK's default buckets (0 to 10000 s) in place, which put every sub-second
# call in the first one.
_GRPC_LATENCY_BOUNDS = (
    0.0, 0.00001, 0.00005, 0.0001, 0.0003, 0.0006, 0.0008, 0.001, 0.002,
    0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.013, 0.016, 0.02, 0.025,
    0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4,
    0.5, 0.65, 0.8, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
)  # fmt: skip

# grpcio-observability flushes its stats from a Python thread every 0.5s by
# default and holds the GIL for the whole batch. Under load that stalls requests
# for several ms, so flush smaller batches more often.
_GRPC_EXPORT_INTERVAL_ENV = "GRPC_PYTHON_CENSUS_EXPORT_BATCH_INTERVAL_SECS"
_GRPC_EXPORT_INTERVAL_SECS = "0.05"


def _build_resource(resource_attributes: dict[str, str] | None) -> "Resource":
    """Build an OTel ``Resource`` from the given attribute dict.

    Returns an empty ``Resource`` when *resource_attributes* is empty or
    ``None`` so that telemetry carries no stale process-level tags.
    """
    # Third Party
    from opentelemetry.sdk.resources import Resource

    if not resource_attributes:
        return Resource.create({})
    return Resource.create(dict(resource_attributes))


def init_otel_metrics(
    otlp_endpoint: str | None = None,
    prometheus_port: int | None = None,
    resource_attributes: dict[str, str] | None = None,
    start_http_server: bool = True,
    enable_grpc_metrics: bool = False,
) -> None:
    """Set up the OpenTelemetry MeterProvider.

    Args:
        otlp_endpoint: OTLP gRPC endpoint (e.g. ``http://localhost:4317``).
            When set, metrics are pushed to an OTel collector.
            When ``None``, falls back to Prometheus pull mode.
        prometheus_port: Port for the fallback Prometheus ``/metrics``
            endpoint.  Only used when *otlp_endpoint* is ``None``.
            Defaults to 9090.
        resource_attributes: Optional ``{attr_name: value}`` map attached
            to the ``MeterProvider`` ``Resource``.  Every metric emitted
            through the provider carries these attributes.  Intended for
            process-level identity (e.g. ``service.instance.id``) — never
            for per-request tags.
        start_http_server: Whether to start a standalone Prometheus
            HTTP server.  Set to ``False`` when metrics are already
            served by an external HTTP framework (e.g. FastAPI).
        enable_grpc_metrics: Whether to register gRPC Python's
            OpenTelemetry metrics plugin against the same MeterProvider.
    """
    # Third Party
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider

    global _GRPC_METRICS_ENABLED, _METRICS_RESOURCE  # pylint: disable=global-statement # noqa: PLW0603

    resource = _build_resource(resource_attributes)
    _GRPC_METRICS_ENABLED = enable_grpc_metrics
    _METRICS_RESOURCE = resource
    views = _grpc_metrics_views() if enable_grpc_metrics else []

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
        reader = _with_metrics_sources(PeriodicExportingMetricReader)(
            exporter, export_interval_millis=10000
        )
        provider = MeterProvider(
            metric_readers=[reader], resource=resource, views=views
        )
        metrics.set_meter_provider(provider)
        if enable_grpc_metrics:
            init_grpc_otel_metrics(provider)
        logger.info(
            "OTel MeterProvider initialised with OTLP exporter (%s), resource=%s",
            otlp_endpoint,
            dict(resource.attributes),
        )
    else:
        # Prometheus pull fallback — no collector needed
        # Third Party
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        import prometheus_client

        if prometheus_port is None:
            prometheus_port = 9090

        reader = _with_metrics_sources(PrometheusMetricReader)()
        provider = MeterProvider(
            metric_readers=[reader], resource=resource, views=views
        )
        metrics.set_meter_provider(provider)
        if enable_grpc_metrics:
            init_grpc_otel_metrics(provider)
        if start_http_server:
            prometheus_client.start_http_server(prometheus_port)
            logger.info(
                "OTel MeterProvider initialised with Prometheus fallback "
                "(http://0.0.0.0:%d/metrics), resource=%s",
                prometheus_port,
                dict(resource.attributes),
            )
        else:
            logger.info(
                "OTel MeterProvider initialised with Prometheus fallback "
                "(standalone metrics HTTP server disabled; "
                "/metrics must be exposed by the caller), resource=%s",
                dict(resource.attributes),
            )


def _grpc_metrics_views() -> list:
    # Third Party
    from opentelemetry.sdk.metrics.view import (
        ExplicitBucketHistogramAggregation,
        View,
    )

    return [
        View(
            instrument_name="grpc.server.call.duration",
            aggregation=ExplicitBucketHistogramAggregation(_GRPC_LATENCY_BOUNDS),
        )
    ]


def grpc_metrics_enabled() -> bool:
    """Whether metrics were initialized with gRPC server metrics turned on."""
    return _GRPC_METRICS_ENABLED


def register_metrics_source(
    source: Callable[[], Iterable["Metric"]],
) -> Callable[[], None]:
    """Export metrics that are collected outside the OTel SDK.

    For components that keep their own cumulative totals instead of recording
    through a meter, such as a native extension. ``source`` is called on every
    collection (each OTLP export or Prometheus scrape) and returns OTel
    ``Metric`` objects with cumulative data points. If it raises, it is
    skipped for that collection and the other metrics are still exported.

    Args:
        source: Returns the metrics to export.

    Returns:
        A function that unregisters the source.
    """
    _METRICS_SOURCES.append(source)

    def unregister() -> None:
        if source in _METRICS_SOURCES:
            _METRICS_SOURCES.remove(source)

    return unregister


def _with_metrics_sources(reader_class: type["MetricReader"]) -> type["MetricReader"]:
    """Subclass a reader so each collection also carries the sources' metrics."""

    # OTel Python has no MetricProducer hook yet, so add them where a reader
    # receives each collection.
    class _Reader(reader_class):  # type: ignore[valid-type, misc]
        def _receive_metrics(
            self,
            metrics_data: "MetricsData",
            timeout_millis: float = 10_000,
            **kwargs,
        ) -> None:
            super()._receive_metrics(
                _add_source_metrics(metrics_data), timeout_millis, **kwargs
            )

    return _Reader


def _add_source_metrics(metrics_data: "MetricsData | None") -> "MetricsData | None":
    """Return ``metrics_data`` with the sources' metrics in a scope of their own.

    Returns ``metrics_data`` unchanged when there are no sources, so the
    collected metrics stay exactly as the SDK produced them.
    """
    if not _METRICS_SOURCES:
        return metrics_data
    extra: list["Metric"] = []
    for source in list(_METRICS_SOURCES):
        try:
            extra.extend(source())
        except Exception:
            logger.warning("Skipping a metrics source that failed", exc_info=True)
    if not extra:
        return metrics_data

    try:
        return _with_scope(metrics_data, extra)
    except Exception:
        # Never let the sources break the metrics the SDK collected.
        logger.warning("Could not add source metrics", exc_info=True)
        return metrics_data


def _with_scope(
    metrics_data: "MetricsData | None", extra: list["Metric"]
) -> "MetricsData":
    # Third Party
    from opentelemetry.sdk.metrics.export import (
        MetricsData,
        ResourceMetrics,
        ScopeMetrics,
    )
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    scope = ScopeMetrics(
        scope=InstrumentationScope("lmcache.metrics_sources"),
        metrics=extra,
        schema_url="",
    )
    resource_metrics = list(metrics_data.resource_metrics) if metrics_data else []
    if resource_metrics:
        first = resource_metrics[0]
        resource_metrics[0] = ResourceMetrics(
            resource=first.resource,
            scope_metrics=[*first.scope_metrics, scope],
            schema_url=first.schema_url,
        )
    elif _METRICS_RESOURCE is not None:
        resource_metrics = [
            ResourceMetrics(
                resource=_METRICS_RESOURCE, scope_metrics=[scope], schema_url=""
            )
        ]
    return MetricsData(resource_metrics=resource_metrics)


def init_grpc_otel_metrics(meter_provider: "MeterProvider") -> None:
    """Register gRPC Python runtime metrics with the active OTel provider.

    The official ``grpcio-observability`` package exports gRPC Core metrics
    into OpenTelemetry. It is currently distributed for Linux, so LMCache
    treats it as an optional runtime integration and keeps metrics startup
    working when the package is unavailable.

    Sets ``GRPC_PYTHON_CENSUS_EXPORT_BATCH_INTERVAL_SECS`` to 0.05 unless it
    is already set, so the plugin flushes its stats in small batches.

    Args:
        meter_provider: The OTel SDK provider used by the LMCache metrics
            pipeline.
    """
    global _GRPC_OTEL_PLUGIN  # pylint: disable=global-statement # noqa: PLW0603

    if _GRPC_OTEL_PLUGIN is not None:
        logger.debug("gRPC OpenTelemetry metrics plugin is already registered")
        return

    if sys.platform != "linux":
        logger.info(
            "gRPC runtime metrics are not available on %s; "
            "grpcio-observability currently supports Linux.",
            sys.platform,
        )
        return

    # Only read when grpc_observability is first imported.
    os.environ.setdefault(_GRPC_EXPORT_INTERVAL_ENV, _GRPC_EXPORT_INTERVAL_SECS)
    try:
        # Third Party
        from grpc_observability import OpenTelemetryPlugin
    except ImportError:
        logger.warning(
            "grpcio-observability is not installed; gRPC runtime metrics "
            "will not be exported. Install grpcio-observability on Linux "
            "or pass --disable-grpc-metrics to suppress this warning."
        )
        return

    plugin = OpenTelemetryPlugin(meter_provider=meter_provider)
    try:
        plugin.register_global()
    except RuntimeError as exc:
        logger.warning(
            "gRPC OpenTelemetry metrics plugin could not be registered: %s",
            exc,
        )
        return

    _GRPC_OTEL_PLUGIN = plugin
    logger.info("gRPC OpenTelemetry runtime metrics are enabled")


def init_otel_tracing(
    otlp_endpoint: str | None = None,
    resource_attributes: dict[str, str] | None = None,
) -> None:
    """Set up the OpenTelemetry TracerProvider with an OTLP exporter.

    Tracing requires an OTLP endpoint — there is no local fallback.
    When *otlp_endpoint* is ``None``, tracing init is skipped.

    Args:
        otlp_endpoint: OTLP gRPC endpoint.  When ``None``, tracing
            init is skipped (no-op).
        resource_attributes: Optional ``{attr_name: value}`` map attached
            to the ``TracerProvider`` ``Resource``.  Every span emitted
            through the provider carries these attributes.
    """
    if otlp_endpoint is None:
        logger.debug("No OTLP endpoint configured, skipping tracing init")
        return

    # Third Party
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = _build_resource(resource_attributes)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info(
        "OTel TracerProvider initialised with OTLP exporter (%s), resource=%s",
        otlp_endpoint,
        dict(resource.attributes),
    )


def register_gauge(
    meter_name: str,
    gauge_name: str,
    description: str,
    func: Callable[[], int | float]
    | Callable[[], list[tuple[int | float, dict[str, object]]]],
) -> None:
    """Register an OTel observable gauge with a callback.

    This is a convenience wrapper that hides the OTel boilerplate.
    If OTel is not available, the call is silently ignored.

    Two callback shapes are accepted:

    - **Single value (no attributes).** ``func`` returns ``int`` / ``float``
      and the gauge emits one datapoint per scrape with no attributes.
      Use this for whole-process metrics like
      ``lmcache_mp.active_prefetch_jobs``.
    - **Per-attribute-set values.** ``func`` returns a list of
      ``(value, attrs)`` tuples; the gauge emits one datapoint per tuple
      with the given attributes.  Use this for per-adapter or per-tier
      metrics that share a name but vary by attribute.  An empty list
      reports no datapoints (the metric simply does not appear in the
      next scrape, which is the correct shape when there is nothing to
      observe).

    Args:
        meter_name: OTel meter name (e.g. ``lmcache.mp_server``).
        gauge_name: Metric name (e.g.
            ``lmcache_mp.active_prefetch_jobs``).
        description: Human-readable description of the gauge.
        func: Zero-arg callable.  Either a function returning the current
            scalar value, or a function returning a list of
            ``(value, attrs)`` tuples for tagged observations.
    """
    try:
        # Third Party
        from opentelemetry import metrics as otel_metrics

        def _callback(_options):
            result = func()
            if isinstance(result, list):
                return [
                    otel_metrics.Observation(value, attrs) for value, attrs in result
                ]
            return [otel_metrics.Observation(result)]

        meter = otel_metrics.get_meter(meter_name)
        meter.create_observable_gauge(
            gauge_name,
            callbacks=[_callback],
            description=description,
        )
    except ImportError:
        logger.debug(
            "opentelemetry package not found, skipping gauge %s",
            gauge_name,
        )
