# SPDX-License-Identifier: Apache-2.0
"""
LMCache Multi-process Mode

This module provides the multi-process mode for LMCache, allowing vLLM instances
to connect to a separate LMCache server process for KV cache management.
"""

# First Party
from lmcache.v1.multiprocess.mp_metadata import (
    create_mp_server_metadata,
    create_mp_server_metadata_from_gpu_context,
)
from lmcache.v1.multiprocess.mp_storage_manager import MPStorageManager

__all__ = [
    "create_mp_server_metadata",
    "create_mp_server_metadata_from_gpu_context",
    "MPStorageManager",
]
