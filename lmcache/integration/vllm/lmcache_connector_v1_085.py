# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the vLLM 0.8.5 in-process connector."""

# First Party
from lmcache.inprocess.integration.vllm.lmcache_connector_v1_085 import (
    LMCacheConnectorV1Dynamic,
)

__all__ = ["LMCacheConnectorV1Dynamic"]
