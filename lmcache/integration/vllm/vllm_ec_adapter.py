# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the in-process vLLM encoder-cache adapter."""

# First Party
from lmcache.inprocess.integration.vllm.vllm_ec_adapter import (
    LMCacheECConnectorImpl,
    LMCacheECConnectorMetadata,
    MMMeta,
)

__all__ = ["LMCacheECConnectorImpl", "LMCacheECConnectorMetadata", "MMMeta"]
