# SPDX-License-Identifier: Apache-2.0
"""
SGLang Metadata Builder for LMCache.

This module contains the logic to construct LMCacheEngineMetadata
from SGLang configuration, isolating SGLang-specific dependencies.
"""

# Standard
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # Third Party
    from sglang.srt.configs.model_config import ModelConfig

logger = init_logger(__name__)


class SGLangMetadataBuilder:
    """Builder for creating LMCacheEngineMetadata from SGLang configuration."""

    @staticmethod
    def build(
        model_config: "ModelConfig",
        lmcache_config: LMCacheEngineConfig,
        tp_size: int,
        local_rank: int,
        global_rank: int,
        kv_dtype: torch.dtype,
    ) -> LMCacheEngineMetadata:
        """
        Build LMCacheEngineMetadata from SGLang configuration.

        Args:
            model_config: SGLang model configuration
            lmcache_config: LMCache engine configuration
            tp_size: Tensor parallel size
            local_rank: Local GPU device index (for device selection)
            global_rank: Global tensor parallel rank (for metadata)
            kv_dtype: Data type for KV cache tensors

        Returns:
            LMCacheEngineMetadata instance with SGLang-specific configuration
        """
        # Construct kv shape (for mem pool)
        num_layer = model_config.num_hidden_layers
        chunk_size = lmcache_config.chunk_size
        num_kv_head = model_config.get_num_kv_heads(tp_size)
        head_dim = model_config.head_dim

        kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_dim)

        logger.info(
            "Building SGLang metadata: num_layer=%d, chunk_size=%d, "
            "num_kv_head=%d, head_dim=%d, hidden_dim=%d, kv_shape=%s",
            num_layer,
            chunk_size,
            num_kv_head,
            head_dim,
            num_kv_head * head_dim,
            kv_shape,
        )

        # Set current device using local GPU index
        torch.cuda.device(local_rank)
        _ = torch.device(f"cuda:{local_rank}")

        # Create metadata (use global rank for metadata - tensor parallel rank)
        metadata = LMCacheEngineMetadata(
            use_case="sglang",
            model_name=model_config.model_path,
            world_size=tp_size,
            worker_id=global_rank,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
        )

        return metadata
