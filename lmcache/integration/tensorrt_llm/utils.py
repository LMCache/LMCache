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
    """Return an LMCacheEngineConfig from ``LMCACHE_CONFIG_FILE`` or env.

    Mirrors the in-process pattern used by the vLLM adapter: prefer a
    config file if the env var is set; otherwise pull from individual
    ``LMCACHE_*`` environment variables.
    """
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
        logger.info("Loading LMCache config file %s", config_file)
        config = LMCacheEngineConfig.from_file(config_file)

    return config


def create_trtllm_metadata(
    llm_args: "TorchLlmArgs",
    kv_cache_tensor: torch.Tensor,
    config: LMCacheEngineConfig,
    num_kv_heads: int,
    head_dim: int,
) -> LMCacheMetadata:
    """Construct LMCacheMetadata from TRT-LLM args and the KV pool tensor.

    Args:
        llm_args: TRT-LLM ``TorchLlmArgs``.
        kv_cache_tensor: TRT-LLM KV pool tensor of shape
            ``[num_blocks, num_layers, kv_factor, flat]`` where
            ``flat = num_kv_heads * tokens_per_block * head_dim``.
        config: The LMCache engine config (used for ``chunk_size``).
        num_kv_heads: Per-rank number of KV attention heads.
        head_dim: Per-head dimension.

    Returns:
        Populated :class:`LMCacheMetadata`.
    """
    # Third Party
    import tensorrt_llm

    _, num_layers, kv_factor, _ = kv_cache_tensor.shape
    kv_shape = (num_layers, kv_factor, config.chunk_size, num_kv_heads, head_dim)

    rank = tensorrt_llm.mpi_rank()
    tp_size = llm_args.tensor_parallel_size
    pp_size = llm_args.pipeline_parallel_size
    world_size = tp_size * pp_size
    local_world_size = tp_size
    local_worker_id = rank % local_world_size
    model_name = str(getattr(llm_args, "model", "unknown_model"))

    # Detect MLA via env var (TRT-LLM doesn't expose an MLA flag on llm_args).
    # If LMCACHE_USE_MLA is set, respect it; otherwise try to detect from
    # kv_factor (MLA models have kv_factor=1, meaning a single latent vector
    # instead of separate K/V).
    import os
    use_mla = os.environ.get("LMCACHE_USE_MLA", "").strip().lower() in (
        "1", "true", "yes",
    )
    if not use_mla and kv_factor == 1:
        # Heuristic: kv_factor=1 likely means MLA (single latent, no separate K/V)
        logger.info(
            "TRT-LLM: detected kv_factor=1, likely MLA model. "
            "Set LMCACHE_USE_MLA=1 to enable MLA cache locking."
        )

    return LMCacheMetadata(
        model_name=model_name,
        world_size=world_size,
        local_world_size=local_world_size,
        worker_id=rank,
        local_worker_id=local_worker_id,
        kv_dtype=kv_cache_tensor.dtype,
        kv_shape=kv_shape,
        chunk_size=config.chunk_size,
        use_mla=use_mla,
    )
