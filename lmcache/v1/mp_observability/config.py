# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the MP-mode observability stack.
"""

# Standard
from dataclasses import dataclass
import argparse


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
    return ObservabilityConfig(
        enabled=not args.disable_observability,
        metrics_enabled=not args.disable_metrics,
        logging_enabled=not args.disable_logging,
        tracing_enabled=args.enable_tracing,
        otlp_endpoint=args.otlp_endpoint,
        prometheus_port=args.prometheus_port,
    )
