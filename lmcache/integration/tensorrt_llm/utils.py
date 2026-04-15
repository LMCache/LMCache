# SPDX-License-Identifier: Apache-2.0
"""Utility helpers for the TensorRT-LLM integration."""

# Standard
from typing import TYPE_CHECKING
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # Third Party
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

logger = init_logger(__name__)
ENGINE_NAME = "trtllm-instance"


def lmcache_get_config() -> LMCacheEngineConfig:
    """Get LMCache configuration from LMCACHE_CONFIG_FILE env var or defaults."""
    if "LMCACHE_CONFIG_FILE" not in os.environ:
        logger.warning(
            "No LMCache configuration file is set. Trying to read"
            " configurations from the environment variables."
        )
        logger.warning(
            "You can set the configuration file through "
            "the environment variable: LMCACHE_CONFIG_FILE"
        )
        config = LMCacheEngineConfig.from_env()
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)

    return config


def create_trtllm_metadata(
    llm_args: "TorchLlmArgs",
    kv_cache_tensor: torch.Tensor,
    config: LMCacheEngineConfig,
    num_kv_heads: int,
    head_dim: int,
) -> LMCacheMetadata:
    """Construct LMCacheMetadata from TRT-LLM args and KV cache pool tensor.

    Args:
        llm_args: TorchLlmArgs instance from TRT-LLM.
        kv_cache_tensor: TRT-LLM KV pool tensor of shape
            ``[num_blocks, num_layers, kv_factor, block_size_flat]``.
        config: LMCacheEngineConfig for chunk_size.
        num_kv_heads: Number of KV attention heads (per TP rank).
        head_dim: Dimension of each attention head.

    Returns:
        LMCacheMetadata populated from the TRT-LLM runtime state.
    """
    # Third Party
    import tensorrt_llm

    num_blocks, num_layers, kv_factor, block_size_flat = kv_cache_tensor.shape
    kv_shape = (num_layers, kv_factor, config.chunk_size, num_kv_heads, head_dim)

    rank = tensorrt_llm.mpi_rank()
    tp_size = llm_args.tensor_parallel_size
    pp_size = llm_args.pipeline_parallel_size
    world_size = tp_size * pp_size
    local_world_size = tp_size
    local_worker_id = rank % local_world_size
    model_name = str(getattr(llm_args, "model", "unknown_model"))

    metadata = LMCacheMetadata(
        model_name=model_name,
        world_size=world_size,
        local_world_size=local_world_size,
        worker_id=rank,
        local_worker_id=local_worker_id,
        kv_dtype=kv_cache_tensor.dtype,
        kv_shape=kv_shape,
        chunk_size=config.chunk_size,
    )

    return metadata
