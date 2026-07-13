# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim; the implementation lives in
:mod:`lmcache.usage_telemetry` (single-process reporters in
:mod:`lmcache.usage_telemetry.non_mp`)."""

# First Party
from lmcache.usage_telemetry import (
    ContinuousContextMessage,
    EngineMessage,
    EnvMessage,
    MetadataMessage,
)
from lmcache.usage_telemetry.non_mp import (
    ContinuousUsageContext,
    InitializeUsageContext,
    UsageContext,
)

__all__ = [
    "ContinuousContextMessage",
    "ContinuousUsageContext",
    "EngineMessage",
    "EnvMessage",
    "InitializeUsageContext",
    "MetadataMessage",
    "UsageContext",
]
