# SPDX-License-Identifier: Apache-2.0
"""
Constants for health monitoring.
"""

# Ping error codes
PING_TIMEOUT_ERROR_CODE = -1
PING_GENERIC_ERROR_CODE = -2

# Configuration keys
PING_TIMEOUT_CONFIG_KEY = "ping_timeout"
PING_INTERVAL_CONFIG_KEY = "ping_interval"

# Default values
DEFAULT_PING_TIMEOUT = 5.0
DEFAULT_PING_INTERVAL = 30.0

# Memory thresholds
DEFAULT_MEMORY_THRESHOLD_PERCENT = 95.0  # Unhealthy if memory usage > 95%
