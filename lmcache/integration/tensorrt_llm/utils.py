# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for in-process TensorRT-LLM helpers."""

# First Party
from lmcache.inprocess.integration.tensorrt_llm.utils import (
    ENGINE_NAME,
    create_trtllm_metadata,
    lmcache_get_config,
)

__all__ = ["ENGINE_NAME", "create_trtllm_metadata", "lmcache_get_config"]
