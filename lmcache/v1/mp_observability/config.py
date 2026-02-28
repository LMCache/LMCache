# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the MP-mode Prometheus observability stack.
"""

# Standard
from dataclasses import dataclass


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
