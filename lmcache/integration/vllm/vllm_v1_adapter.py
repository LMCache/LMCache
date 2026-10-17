# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the in-process vLLM adapter."""

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    DisaggSpec,
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
    extract_request_configs,
)

__all__ = [
    "DisaggSpec",
    "LMCacheConnectorMetadata",
    "LMCacheConnectorV1Impl",
    "LoadSpec",
    "ReqMeta",
    "RequestTracker",
    "SaveSpec",
    "extract_request_configs",
]
