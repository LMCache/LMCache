# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for TensorRT-LLM's built-in ``lmcache`` preset."""

# First Party
from lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter import (
    LMCacheKvConnectorScheduler,
    LMCacheKvConnectorWorker,
)

__all__ = ["LMCacheKvConnectorScheduler", "LMCacheKvConnectorWorker"]
