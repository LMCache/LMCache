# SPDX-License-Identifier: Apache-2.0
"""Cache identity for vLLM KV encodings copied without dtype conversion."""

# Third Party
from vllm.config import VllmConfig
from vllm.config.utils import hash_factors

_KV_DTYPE_NAMESPACE = "##lmcache-kv-dtype-v1-"


def get_kv_dtype_decorated_model_name(vllm_config: VllmConfig, model_name: str) -> str:
    """Namespace the cache model name by raw KV precision settings.

    Args:
        vllm_config: Initialized vLLM configuration shared by scheduler/worker.
        model_name: Cache model name, including any DCP layout decoration.

    Returns:
        Cache model name with a dtype identifier. New clients miss legacy
        objects; model loading and the public served name are unaffected.

    Note:
        Raw settings remain distinct even when they resolve to equivalent
        encodings (e.g. auto and bfloat16). This isolates configured dtypes,
        not every layout, weight revision or scale difference.
    """
    cache = vllm_config.cache_config
    factors: dict[str, object] = {
        "model_dtype": str(vllm_config.model_config.dtype),
        "cache_dtype": cache.cache_dtype,
        "skip_layers": sorted(set(cache.kv_cache_dtype_skip_layers)),
        "mamba_cache_dtype": cache.mamba_cache_dtype,
        "mamba_ssm_cache_dtype": cache.mamba_ssm_cache_dtype,
    }
    return f"{model_name}{_KV_DTYPE_NAMESPACE}{hash_factors(factors)[:32]}"
