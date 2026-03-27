# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the MP-mode observability stack.
"""

# Standard
from dataclasses import dataclass
import argparse


@dataclass
class PrometheusConfig:
    """
    The configuration for the Prometheus observability stack.
    """

    enabled: bool = True
    """ Whether to enable Prometheus metrics collection and HTTP server. """

    port: int = 9090
    """ Port to expose the Prometheus /metrics endpoint on. """

    log_interval: float = 10.0
    """ How often (in seconds) to flush accumulated stats to Prometheus. """


DEFAULT_PROMETHEUS_CONFIG = PrometheusConfig(enabled=False)


def add_prometheus_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add Prometheus configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The same parser with Prometheus arguments added.
    """
    prometheus_group = parser.add_argument_group(
        "Prometheus Observability", "Configuration for Prometheus metrics"
    )
    prometheus_group.add_argument(
        "--disable-prometheus",
        action="store_true",
        default=False,
        help="Disable Prometheus metrics collection and HTTP server.",
    )
    prometheus_group.add_argument(
        "--prometheus-port",
        type=int,
        default=9090,
        help="Port to expose the Prometheus /metrics endpoint on. Default is 9090.",
    )
    prometheus_group.add_argument(
        "--prometheus-log-interval",
        type=float,
        default=10.0,
        help="How often (in seconds) to flush stats to Prometheus. Default is 10.0.",
    )
    return parser


def parse_args_to_prometheus_config(
    args: argparse.Namespace,
) -> PrometheusConfig:
    """
    Convert parsed command line arguments to a PrometheusConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        PrometheusConfig: The configuration object.
    """
    return PrometheusConfig(
        enabled=not args.disable_prometheus,
        port=args.prometheus_port,
        log_interval=args.prometheus_log_interval,
    )


# ---------------------------------------------------------------------------
# Unified observability config (new EventBus-based system)
# ---------------------------------------------------------------------------


@dataclass
class ObservabilityConfig:
    """Unified configuration for the EventBus-based observability system.

    This config drives the new EventBus + OTel pipeline.  During the
    migration period it coexists with ``PrometheusConfig`` and
    ``TelemetryConfig``.
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

    prometheus_port: int = 9090
    """Port for the Prometheus /metrics endpoint (via OTel exporter)."""


DEFAULT_OBSERVABILITY_CONFIG = ObservabilityConfig(enabled=False)
