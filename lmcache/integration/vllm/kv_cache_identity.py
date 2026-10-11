# SPDX-License-Identifier: Apache-2.0
"""Cache identity for vLLM KV encodings."""

# Third Party
from vllm.config import VllmConfig
from vllm.config.utils import hash_factors

_KV_DTYPE_NAMESPACE = "kv-dtype-v1-"


def get_kv_cache_format(vllm_config: VllmConfig) -> str:
    """Hash raw vLLM KV precision settings into a cache-format ID."""
    cache = vllm_config.cache_config
    factors: dict[str, object] = {
        "model_dtype": str(vllm_config.model_config.dtype),
        "cache_dtype": cache.cache_dtype,
        "skip_layers": sorted(set(cache.kv_cache_dtype_skip_layers)),
        "mamba_cache_dtype": cache.mamba_cache_dtype,
        "mamba_ssm_cache_dtype": cache.mamba_ssm_cache_dtype,
    }
    return f"{_KV_DTYPE_NAMESPACE}{hash_factors(factors)[:32]}"
