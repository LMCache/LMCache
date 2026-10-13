# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral identity for opaque KV precision settings."""

# Standard
from dataclasses import asdict, dataclass
import json

# Third Party
import blake3


@dataclass(frozen=True)
class KVCacheFormat:
    """KV precision settings used to isolate incompatible caches.

    Attributes:
        model_dtype: Model computation dtype name.
        cache_dtype: Configured attention KV dtype name.
        skip_layers: Layer identifiers excluded from KV dtype conversion;
            order and duplicates do not affect the identity.
        mamba_cache_dtype: Configured recurrent convolution-state dtype name.
        mamba_ssm_cache_dtype: Configured recurrent SSM-state dtype name.
    """

    model_dtype: str
    cache_dtype: str
    skip_layers: tuple[str, ...]
    mamba_cache_dtype: str
    mamba_ssm_cache_dtype: str

    def to_cache_format(self) -> str:
        """Return a deterministic identifier using LMCache's BLAKE3 algorithm.

        Returns:
            Versioned namespace plus a 128-bit digest of canonical JSON.
            Serialization and hashing are independent of the serving engine
            and Python's process-specific hash seed.
        """
        factors = asdict(self)
        factors["skip_layers"] = sorted(set(self.skip_layers))
        payload = json.dumps(factors, sort_keys=True, separators=(",", ":"))
        return f"kv-dtype-v1-{blake3.blake3(payload.encode('utf-8')).hexdigest()[:32]}"
