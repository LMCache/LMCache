# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives (fallbacks used when no
accelerator-backed implementation is available).

Importing this package self-registers a :class:`CpuPlatform` instance
with :mod:`lmcache.v1.platform._registry`, so the stream dispatcher
can pick it up either as the active backend (on CPU-only hosts) or
as the default fallback when no concrete accelerator backend matches.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform._registry import (
    DEFAULT_BACKEND,
    Platform,
    register_platform,
)
from lmcache.v1.platform.cpu.stream import MockExternalStream

logger = init_logger(__name__)


class CpuPlatform(Platform):
    """CPU fallback platform — always available, never declines."""

    device_type = DEFAULT_BACKEND  # "cpu"

    def make_external_stream(self, raw_ptr: int, _device_index: int) -> Optional[Any]:
        logger.info("make_external_stream: MockExternalStream %s", raw_ptr)
        return MockExternalStream(raw_ptr)


register_platform(CpuPlatform())
