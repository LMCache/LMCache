# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the in-process vLLM connector."""

# First Party
from lmcache.inprocess.integration.vllm.lmcache_connector_v1 import (
    LMCacheConnectorV1Dynamic,
)

__all__ = ["LMCacheConnectorV1Dynamic"]
