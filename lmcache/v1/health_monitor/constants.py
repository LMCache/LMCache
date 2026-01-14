# SPDX-License-Identifier: Apache-2.0
"""
Constants for health monitoring.
"""

# Standard
from enum import Enum


class FallbackPolicy(str, Enum):
    """Fallback policy when health check fails."""

    RECOMPUTE = "recompute"  # Skip all cache operations, fall back to recomputation
    LOCAL_CPU = "local_cpu"  # Fall back to local CPU backend only


# Ping error codes
PING_TIMEOUT_ERROR_CODE = -1
PING_GENERIC_ERROR_CODE = -2

# Configuration keys
PING_TIMEOUT_CONFIG_KEY = "ping_timeout"
PING_INTERVAL_CONFIG_KEY = "ping_interval"
FALLBACK_POLICY_CONFIG_KEY = "fallback_policy"

# Default values
DEFAULT_PING_TIMEOUT = 5.0
DEFAULT_PING_INTERVAL = 30.0
DEFAULT_FALLBACK_POLICY = FallbackPolicy.RECOMPUTE

# Memory thresholds
DEFAULT_MEMORY_THRESHOLD_PERCENT = 95.0  # Unhealthy if memory usage > 95%
