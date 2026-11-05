# SPDX-License-Identifier: Apache-2.0
"""vLLM helpers that construct the legacy in-process engine."""

# Standard
from typing import Optional
import os
import threading

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig, load_ec_engine_config
from lmcache.inprocess.v1.config_base import apply_remote_configs, fetch_remote_config
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    calculate_draft_layers,
    calculate_local_rank_and_world_size,
    extract_mm_features,
    extract_request_configs_from_sampling_params,
    get_size_bytes,
    is_dp_rank0,
    mla_enabled,
    vllm_layout_hints,
)
from lmcache.logging import init_logger

__all__ = [
    "ENGINE_NAME",
    "apply_mm_hashes_to_token_ids",
    "calculate_draft_layers",
    "calculate_local_rank_and_world_size",
    "extract_mm_features",
    "extract_request_configs_from_sampling_params",
    "get_size_bytes",
    "is_dp_rank0",
    "mla_enabled",
    "vllm_layout_hints",
]

logger = init_logger(__name__)

_config_instance: Optional[LMCacheEngineConfig] = None
_config_lock = threading.Lock()


def lmcache_get_or_create_config() -> LMCacheEngineConfig:
    """Load and cache the configuration for the in-process engine."""
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                if "LMCACHE_CONFIG_FILE" not in os.environ:
                    logger.warning(
                        "No LMCache configuration file is set. Trying to read "
                        "configurations from environment variables."
                    )
                    _config_instance = LMCacheEngineConfig.from_env()
                    _config_instance.validate()
                else:
                    config_file = os.environ["LMCACHE_CONFIG_FILE"]
                    logger.info("Loading LMCache config file %s", config_file)
                    _config_instance = LMCacheEngineConfig.from_file(config_file)
                    _config_instance.update_config_from_env()

                remote_config_url = _config_instance.remote_config_url
                if remote_config_url:
                    remote_response = fetch_remote_config(
                        remote_config_url,
                        _config_instance.app_id,
                        _config_instance,
                    )
                    if remote_response:
                        _config_instance = apply_remote_configs(
                            _config_instance, remote_response
                        )
                    else:
                        logger.warning(
                            "Failed to fetch remote configuration from %s. "
                            "Using local configuration only.",
                            remote_config_url,
                        )
    return _config_instance


def create_lmcache_ec_config() -> LMCacheEngineConfig:
    """Create EC config from LMCache config plus EC-specific overrides."""
    return load_ec_engine_config(base_config=lmcache_get_or_create_config())


def create_lmcache_metadata(
    vllm_config=None,
    model_config=None,
    parallel_config=None,
    cache_config=None,
    role=None,
):
    """Create in-process engine metadata from vLLM configuration."""
    try:
        # Third Party
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    except ImportError:
        # Third Party
        from vllm.utils import get_kv_cache_torch_dtype

    config = lmcache_get_or_create_config()
    if vllm_config is not None:
        model_cfg = vllm_config.model_config
        parallel_cfg = vllm_config.parallel_config
        cache_cfg = vllm_config.cache_config
    else:
        model_cfg = model_config
        parallel_cfg = parallel_config
        cache_cfg = cache_config

    kv_dtype = get_kv_cache_torch_dtype(cache_cfg.cache_dtype, model_cfg.dtype)
    use_mla = mla_enabled(model_cfg)
    kv_shape = (
        model_cfg.get_num_layers(parallel_cfg),
        1 if use_mla else 2,
        config.chunk_size,
        model_cfg.get_num_kv_heads(parallel_cfg),
        model_cfg.get_head_size(),
    )

    engine_id = None
    kv_connector_extra_config = None
    if vllm_config is not None and hasattr(vllm_config, "kv_transfer_config"):
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            engine_id = getattr(kv_transfer_config, "engine_id", None)
            kv_connector_extra_config = getattr(
                kv_transfer_config, "kv_connector_extra_config", None
            )

    metadata = LMCacheMetadata(
        model_name=model_cfg.model,
        world_size=parallel_cfg.world_size,
        local_world_size=parallel_cfg.world_size,
        worker_id=parallel_cfg.rank,
        local_worker_id=parallel_cfg.rank,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
        role=role,
        served_model_name=model_cfg.served_model_name,
        engine_id=engine_id,
        kv_connector_extra_config=kv_connector_extra_config,
    )
    return metadata, config


def validate_mla_config(config: LMCacheEngineConfig, use_mla: bool) -> None:
    """Validate MLA-related in-process configuration."""
    if use_mla and config.remote_serde not in ("naive", None):
        raise ValueError("MLA only works with naive serde mode..")
    if use_mla and config.use_layerwise and config.enable_blending:
        raise ValueError(
            "We haven't supported MLA with Cacheblend yet. Please disable blending."
        )
