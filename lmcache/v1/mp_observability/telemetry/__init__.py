# SPDX-License-Identifier: Apache-2.0

"""Telemetry event system for MP observability.

Public API — intentionally narrow so call sites only depend on what they need.
"""

# First Party
from lmcache.v1.mp_observability.telemetry.config import (
    TelemetryConfig,
    add_telemetry_args,
    parse_args_to_telemetry_config,
)
from lmcache.v1.mp_observability.telemetry.controller import (
    get_telemetry_controller,
    init_telemetry_controller,
    log_telemetry,
    make_end_event,
    make_start_event,
)
from lmcache.v1.mp_observability.telemetry.event import (
    EventType,
    TelemetryEvent,
)
from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessor,
    TelemetryProcessorConfig,
)
from lmcache.v1.mp_observability.telemetry.processors.logging_processor import (
    LoggingProcessorConfig,
)

__all__ = [
    "EventType",
    "TelemetryEvent",
    "TelemetryProcessor",
    "TelemetryProcessorConfig",
    "TelemetryConfig",
    "LoggingProcessorConfig",
    "init_telemetry_controller",
    "get_telemetry_controller",
    "log_telemetry",
    "make_start_event",
    "make_end_event",
    "add_telemetry_args",
    "parse_args_to_telemetry_config",
]
