# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim; the implementation lives in
:mod:`lmcache.usage_telemetry`."""

# First Party
from lmcache.usage_telemetry import (
    ContinuousContextMessage,
    EnvMessage,
)

__all__ = [
    "ContinuousContextMessage",
    "EnvMessage",
]
