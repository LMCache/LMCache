# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the MP-mode observability stack.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING
import argparse

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_observability.event_bus import EventBus


@dataclass
class ObservabilityConfig:
    """Unified configuration for the EventBus-based observability system.

    Controls the EventBus, OTel metrics/tracing pipelines, and subscriber
    registration.
    """

    enabled: bool = True
    """Master switch for the EventBus."""

    max_queue_size: int = 10_000
    """Maximum events in the EventBus queue before tail-drop."""

    metrics_enabled: bool = True
    """Register metrics subscribers (OTel counters / histograms)."""

    logging_enabled: bool = True
    """Register logging subscribers."""

    tracing_enabled: bool = False
    """Register span subscribers (OTel traces)."""

    otlp_endpoint: str | None = None
    """OTLP gRPC endpoint (e.g. ``http://localhost:4317``).  When set,
    metrics and traces are pushed to an OTel collector.  When ``None``,
    metrics fall back to an in-process Prometheus ``/metrics`` endpoint."""

    prometheus_port: int = 9090
    """Port for the Prometheus /metrics endpoint.  Only used when
    ``otlp_endpoint`` is ``None`` (Prometheus pull fallback)."""


DEFAULT_OBSERVABILITY_CONFIG = ObservabilityConfig(enabled=False)


def add_observability_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add observability configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with observability arguments added.
    """
    group = parser.add_argument_group(
        "Observability", "Configuration for metrics, logging, and tracing"
    )
    group.add_argument(
        "--disable-observability",
        action="store_true",
        default=False,
        help="Disable the observability EventBus entirely.",
    )
    group.add_argument(
        "--disable-metrics",
        action="store_true",
        default=False,
        help="Disable metrics subscribers (OTel counters).",
    )
    group.add_argument(
        "--disable-logging",
        action="store_true",
        default=False,
        help="Disable logging subscribers.",
    )
    group.add_argument(
        "--enable-tracing",
        action="store_true",
        default=False,
        help="Enable span subscribers (OTel traces). Disabled by default.",
    )
    group.add_argument(
        "--otlp-endpoint",
        type=str,
        default=None,
        help=(
            "OTLP gRPC endpoint (e.g. http://localhost:4317). "
            "When set, metrics/traces are pushed to an OTel collector. "
            "When unset, falls back to Prometheus pull mode."
        ),
    )
    group.add_argument(
        "--event-bus-queue-size",
        type=int,
        default=10_000,
        help=(
            "Maximum number of events in the EventBus queue before "
            "tail-drop. Default is 10000."
        ),
    )
    group.add_argument(
        "--prometheus-port",
        type=int,
        default=9090,
        help=(
            "Port for the Prometheus /metrics endpoint. "
            "Only used when --otlp-endpoint is not set. Default is 9090."
        ),
    )
    return parser


def parse_args_to_observability_config(
    args: argparse.Namespace,
) -> ObservabilityConfig:
    """Convert parsed command line arguments to an ObservabilityConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        The configuration object.
    """
    config = ObservabilityConfig(
        enabled=not args.disable_observability,
        max_queue_size=args.event_bus_queue_size,
        metrics_enabled=not args.disable_metrics,
        logging_enabled=not args.disable_logging,
        tracing_enabled=args.enable_tracing,
        otlp_endpoint=args.otlp_endpoint,
        prometheus_port=args.prometheus_port,
    )

    if config.tracing_enabled and config.otlp_endpoint is None:
        raise ValueError(
            "--enable-tracing requires --otlp-endpoint to be set. "
            "Tracing needs an OTLP gRPC endpoint to export spans."
        )

    return config


def init_observability(obs_config: ObservabilityConfig) -> EventBus:
    """Initialize OTel providers, EventBus, and register subscribers.

    This is the single entry-point that every MP server calls at startup.
    Returns a **started** EventBus.
    """
    # First Party
    from lmcache.v1.mp_observability.event_bus import (
        EventBusConfig,
        init_event_bus,
    )

    # Set up OTel providers BEFORE creating subscribers so that
    # module-level get_meter()/get_tracer() calls bind to the real provider
    if obs_config.enabled and obs_config.metrics_enabled:
        # First Party
        from lmcache.v1.mp_observability.otel_init import init_otel_metrics

        init_otel_metrics(
            otlp_endpoint=obs_config.otlp_endpoint,
            prometheus_port=obs_config.prometheus_port,
        )

    if obs_config.enabled and obs_config.tracing_enabled:
        # First Party
        from lmcache.v1.mp_observability.otel_init import init_otel_tracing

        init_otel_tracing(otlp_endpoint=obs_config.otlp_endpoint)

    bus = init_event_bus(
        EventBusConfig(
            enabled=obs_config.enabled,
            max_queue_size=obs_config.max_queue_size,
        )
    )

    if obs_config.metrics_enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.metrics import (
            L1MetricsSubscriber,
            L2MetricsSubscriber,
            SMMetricsSubscriber,
        )

        bus.register_subscriber(L1MetricsSubscriber())
        bus.register_subscriber(L2MetricsSubscriber())
        bus.register_subscriber(SMMetricsSubscriber())

    if obs_config.logging_enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.logging import (
            L1LoggingSubscriber,
            L2LoggingSubscriber,
            MPServerLoggingSubscriber,
            SMLoggingSubscriber,
        )

        bus.register_subscriber(MPServerLoggingSubscriber())
        bus.register_subscriber(L1LoggingSubscriber())
        bus.register_subscriber(L2LoggingSubscriber())
        bus.register_subscriber(SMLoggingSubscriber())

    if obs_config.tracing_enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.tracing import (
            MPServerTracingSubscriber,
        )

        bus.register_subscriber(MPServerTracingSubscriber())

    bus.start()
    return bus
