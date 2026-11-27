# SPDX-License-Identifier: Apache-2.0
"""Cache identity for vLLM KV encodings."""

# Standard
import json

# Third Party
from vllm.config import VllmConfig
import blake3


def get_kv_cache_format(vllm_config: VllmConfig) -> str:
    """Fingerprint configured KV precision, ignoring skip-layer order and duplicates.

    Args:
        vllm_config: Initialized configuration shared by scheduler and worker.

    Returns:
        Versioned cache-format identifier with a 128-bit BLAKE3 digest.

    Note:
        This fingerprints configured precision settings, not resolved physical
        encodings, model weights, quantization scales, or every KV layout.
    """
    cache = vllm_config.cache_config
    factors = {
        "model_dtype": str(vllm_config.model_config.dtype),
        "cache_dtype": cache.cache_dtype,
        "skip_layers": sorted(set(cache.kv_cache_dtype_skip_layers)),
        "mamba_cache_dtype": cache.mamba_cache_dtype,
        "mamba_ssm_cache_dtype": cache.mamba_ssm_cache_dtype,
    }
    payload = json.dumps(factors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"kv-dtype-v1-{blake3.blake3(payload).hexdigest(length=16)}"
