# SPDX-License-Identifier: Apache-2.0
"""
Standalone Metadata Builder for LMCache.

This module contains the logic to construct LMCacheEngineMetadata
for standalone mode (no serving engine dependency).
"""

# Standard
from typing import Any, Dict, Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


class StandaloneMetadataBuilder:
    """Builder for creating LMCacheEngineMetadata for standalone mode."""

    @staticmethod
    def build(
        lmcache_config: LMCacheEngineConfig,
        model_name: str,
        world_size: int,
        worker_id: int,
        kv_dtype: torch.dtype,
        kv_shape: tuple,
        use_mla: bool = False,
        layer_groups: Optional[Dict[str, Any]] = None,
    ) -> LMCacheEngineMetadata:
        """
        Build LMCacheEngineMetadata for standalone mode.

        Args:
            lmcache_config: LMCache engine configuration
            model_name: Name of the model
            world_size: Total number of distributed workers
            worker_id: ID of this worker
            kv_dtype: Data type for KV cache
            kv_shape: Shape of KV cache tensors
                (num_layer, 2, chunk_size, num_kv_head, head_size)
            use_mla: Whether to use Multi-Level Attention
            layer_groups: Optional layer group specifications

        Returns:
            LMCacheEngineMetadata instance for standalone mode
        """
        logger.info(
            "Building standalone metadata: model=%s, world_size=%d, worker_id=%d, "
            "kv_dtype=%s, kv_shape=%s, use_mla=%s",
            model_name,
            world_size,
            worker_id,
            kv_dtype,
            kv_shape,
            use_mla,
        )

        # Create metadata
        metadata = LMCacheEngineMetadata(
            use_case="standalone",
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role="worker",  # Standalone is always worker role
        )

        # Build layer groups if provided
        if layer_groups:
            metadata.kv_layer_groups_manager.build_kv_layer_groups(layer_groups)
            logger.info(
                "Built %d layer groups for standalone mode",
                metadata.kv_layer_groups_manager.num_groups,
            )

        return metadata
