# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Union
import os

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.multimodal.inputs import PlaceholderRange

# Third Party
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

# First Party
from lmcache.config import LMCacheEngineConfig as Config  # type: ignore[assignment]
from lmcache.logging import init_logger
from lmcache.v1.config import (
    LMCacheEngineConfig as V1Config,  # type: ignore[assignment]
)



logger = init_logger(__name__)
ENGINE_NAME = "vllm-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config() -> Union[Config, V1Config]:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.
    """

    if is_false(os.getenv("LMCACHE_USE_EXPERIMENTAL", "True")):
        logger.warning(
            "Detected LMCACHE_USE_EXPERIMENTAL is set to False. "
            "Using legacy configuration is deprecated and will "
            "be remove soon! Please set LMCACHE_USE_EXPERIMENTAL "
            "to True."
        )
        LMCacheEngineConfig = Config  # type: ignore[assignment]
    else:
        LMCacheEngineConfig = V1Config  # type: ignore[assignment]

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


def hex_hash_to_int16(s: str) -> int:
    """
    Convert a hex hash string to a 16-bit integer.
    """
    return int(s, 16) & 0xFFFF


def apply_mm_hashes_to_token_ids(
    token_ids: torch.Tensor,
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
) -> torch.Tensor:
    """
    Overwrite token_ids in-place for multimodal placeholders using
    efficient slice assignments.
    """
    n = token_ids.size(0)
    for hash_str, placeholder in zip(mm_hashes, mm_positions, strict=False):
        start, length = placeholder.offset, placeholder.length
        if start >= n:
            continue
        end = min(start + length, n)
        token_ids[start:end] = hex_hash_to_int16(hash_str)
    return token_ids


def mla_enabled(model_config: "ModelConfig") -> bool:
    return (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    )


def create_lmcache_metadata(
    vllm_config=None, model_config=None, parallel_config=None, cache_config=None
):
    """
    Create LMCacheEngineMetadata from vLLM configuration.

    This function extracts common metadata creation logic that was duplicated
    across multiple files.

    Args:
        vllm_config: vLLM configuration object containing model, parallel, and
                    cache configs (alternative to individual config parameters)
        model_config: Model configuration (alternative to vllm_config)
        parallel_config: Parallel configuration (alternative to vllm_config)
        cache_config: Cache configuration (alternative to vllm_config)

    Returns:
        tuple: (LMCacheEngineMetadata, LMCacheEngineConfig)
    """
    # Third Party
    from vllm.utils import get_kv_cache_torch_dtype

    # First Party
    from lmcache.config import LMCacheEngineMetadata

    config = lmcache_get_config()
    # Support both vllm_config object and individual config parameters
    if vllm_config is not None:
        model_cfg = vllm_config.model_config
        parallel_cfg = vllm_config.parallel_config
        cache_cfg = vllm_config.cache_config
    else:
        model_cfg = model_config
        parallel_cfg = parallel_config
        cache_cfg = cache_config

    # Get KV cache dtype
    kv_dtype = get_kv_cache_torch_dtype(cache_cfg.cache_dtype, model_cfg.dtype)

    # Check if MLA is enabled
    use_mla = mla_enabled(model_cfg)

    # Construct KV shape (for memory pool)
    num_layer = model_cfg.get_num_layers(parallel_cfg)
    chunk_size = config.chunk_size
    num_kv_head = model_cfg.get_num_kv_heads(parallel_cfg)
    head_size = model_cfg.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)

    # Create metadata
    metadata = LMCacheEngineMetadata(
        model_cfg.model,
        parallel_cfg.world_size,
        parallel_cfg.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
    )

    return metadata, config


def get_layer_to_kv_cache_group_id_mapping(
        kv_cache_config: KVCacheConfig
) -> tuple[dict[str, int], dict[int, int]]:
    """
    Create mappings from layer names/IDs to KV cache group IDs for hybrid 
    memory allocation.
    
    This function constructs two mapping dictionaries that allow efficient 
    lookup of which KV cache group a given layer belongs to. This is 
    essential for the hybrid memory allocator to properly manage memory 
    allocation across different layers.
    
    The function processes each KV cache group and creates mappings for both 
    the full layer names and the extracted numeric layer IDs. This enables 
    fast lookups using either identifier format.
    
    Args:
        kv_cache_config (KVCacheConfig): Configuration object 
            containing KV cache groups and their associated layer names.
    
    Returns:
        tuple[dict[str, int], dict[int, int]]: A tuple containing:
            - layer_name_to_kv_cache_group_id: Maps full layer names (e.g., 
              'language_model.model.layers.18.self_attn.attn') to their
              corresponding KV cache group ID.
            - layer_id_to_kv_cache_group_id: Maps numeric layer IDs (e.g., 
                18) to their corresponding KV cache group ID.
    
    Raises:
        ValueError: If a layer name cannot be parsed to extract 
            the layer ID.
    
    Example:
        >>> config = KVCacheConfig(kv_cache_groups=[...])
        >>> name_map, id_map = get_layer_to_kv_cache_group_id_mapping(config)
        >>> name_map['language_model.model.layers.18.self_attn.attn']
        0
        >>> id_map[5]
        1
    """
    # NOTE(Kuntai): for hybrid memory allocator, we need to map from layer id
    # or layer name to kv cache group id.
    layer_name_to_kv_cache_group_id = {}
    layer_id_to_kv_cache_group_id = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups

    # Third Party
    from vllm.model_executor.models.utils import extract_layer_index

    # construct the mapping from layer name/id to kv cache group id
    for kv_cache_group_id, group in enumerate(kv_cache_groups):
        for layer_name in group.layer_names:
            layer_name_to_kv_cache_group_id[layer_name] = kv_cache_group_id
            layer_id = extract_layer_index(layer_name)
            layer_id_to_kv_cache_group_id[layer_id] = kv_cache_group_id

    return layer_name_to_kv_cache_group_id, layer_id_to_kv_cache_group_id
