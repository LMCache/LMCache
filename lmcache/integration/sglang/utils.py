# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

# First Party
from lmcache.config import LMCacheEngineMetadata

logger = init_logger(__name__)
ENGINE_NAME = "sglang-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config() -> LMCacheEngineConfig:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.
    """
    logger.info(f"LMCACHE_CONFIG_FILE: {os.getenv('LMCACHE_CONFIG_FILE')}")
    if "LMCACHE_CONFIG_FILE" not in os.environ:
        logger.warn(
            "No LMCache configuration file is set. Trying to read"
            " configurations from the environment variables."
        )
        logger.warn(
            "You can set the configuration file through "
            "the environment variable: LMCACHE_CONFIG_FILE"
        )
        config = LMCacheEngineConfig.from_env()
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)

    return config


class SGLangMetadataBuilder:
    """Builder for creating LMCacheEngineMetadata from SGLang configuration."""

    @staticmethod
    def from_sglang_config(
        model_config: "ModelConfig",
        tp_size: int,
        global_rank: int,
        kv_dtype: torch.dtype,
        lmcache_config: LMCacheEngineConfig,
    ) -> LMCacheEngineMetadata:
        """
        Create LMCacheEngineMetadata from SGLang configuration.

        Args:
            model_config: SGLang model configuration
            tp_size: Tensor parallel size
            global_rank: Global tensor parallel rank
            kv_dtype: Data type for KV cache tensors
            lmcache_config: LMCache engine configuration

        Returns:
            LMCacheEngineMetadata
        """
        # First Party
        from lmcache.config import LMCacheEngineMetadata

        num_layer = model_config.num_hidden_layers
        chunk_size = lmcache_config.chunk_size
        num_kv_head = model_config.get_num_kv_heads(tp_size)
        head_dim = model_config.head_dim

        kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_dim)

        return LMCacheEngineMetadata(
            model_config.model_path,
            tp_size,
            global_rank,
            "sgl",
            kv_dtype,
            kv_shape,
        )
