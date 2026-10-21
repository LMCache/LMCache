# SPDX-License-Identifier: Apache-2.0
"""Cache identity for vLLM KV encodings."""

# Third Party
from vllm.config import VllmConfig

# First Party
from lmcache.v1.cache_format import KVCacheFormat


def get_kv_cache_format(vllm_config: VllmConfig) -> str:
    """Project raw vLLM KV precision settings into LMCache's format schema.

    Args:
        vllm_config: Initialized configuration shared by scheduler and worker.

    Returns:
        Engine-neutral cache-format identifier for the configured precision.
    """
    cache = vllm_config.cache_config
    return KVCacheFormat(
        model_dtype=str(vllm_config.model_config.dtype),
        cache_dtype=cache.cache_dtype,
        skip_layers=tuple(cache.kv_cache_dtype_skip_layers),
        mamba_cache_dtype=cache.mamba_cache_dtype,
        mamba_ssm_cache_dtype=cache.mamba_ssm_cache_dtype,
    ).to_cache_format()
