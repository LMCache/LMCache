# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for TensorRT-LLM's in-process LMCache connector."""

# First Party
from lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter import (
    LMCacheConnectorMetadata,
    LMCacheKvConnectorScheduler,
    LMCacheKvConnectorWorker,
    destroy_engine,
)

__all__ = [
    "LMCacheConnectorMetadata",
    "LMCacheKvConnectorScheduler",
    "LMCacheKvConnectorWorker",
    "destroy_engine",
]
