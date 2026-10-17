# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the in-process vLLM service factory."""

# First Party
from lmcache.inprocess.integration.vllm.vllm_service_factory import (
    VllmServiceFactory,
)

__all__ = ["VllmServiceFactory"]
