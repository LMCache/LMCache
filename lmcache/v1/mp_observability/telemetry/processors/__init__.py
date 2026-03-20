# SPDX-License-Identifier: Apache-2.0

"""Telemetry processors package.

Imports submodules to trigger registration of built-in processor types.
Re-exports base classes for convenience.
"""

# First Party
# Import submodules to trigger registration
from lmcache.v1.mp_observability.telemetry.processors import (  # noqa: F401
    logging_processor,
)
from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessor,
    TelemetryProcessorConfig,
    get_registered_telemetry_processor_types,
    register_telemetry_processor_type,
)

__all__ = [
    "TelemetryProcessor",
    "TelemetryProcessorConfig",
    "get_registered_telemetry_processor_types",
    "register_telemetry_processor_type",
]
