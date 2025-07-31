# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple

# First Party
from lmcache.config import (
    GlobalConfig,
    LMCacheEngineConfig,
    LMCacheEngineMetadata,
)
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.fast_serde import (
    FastDeserializer,
    FastSerializer,
)
from lmcache.storage_backend.serde.safe_serde import (
    SafeDeserializer,
    SafeSerializer,
)
from lmcache.storage_backend.serde.serde import (
    Deserializer,
    DeserializerDebugWrapper,
    Serializer,
    SerializerDebugWrapper,
)
from lmcache.storage_backend.serde.torch_serde import (
    TorchDeserializer,
    TorchSerializer,
)


def CreateSerde(
    serde_type: str,
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
) -> Tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None

    if serde_type == "torch":
        s, d = TorchSerializer(), TorchDeserializer(metadata.kv_dtype)
    elif serde_type == "safetensor":
        s, d = SafeSerializer(), SafeDeserializer(metadata.kv_dtype)
    elif serde_type == "cachegen":
        s, d = (
            CacheGenSerializer(config, metadata),
            CacheGenDeserializer(config, metadata, metadata.kv_dtype),
        )
    elif serde_type == "fast":
        s, d = FastSerializer(), FastDeserializer(metadata.kv_dtype)
    else:
        raise ValueError(f"Invalid serde type: {serde_type}")

    if GlobalConfig.is_debug():
        return SerializerDebugWrapper(s), DeserializerDebugWrapper(d)
    else:
        return s, d


__all__ = [
    "Serializer",
    "Deserializer",
    "TorchSerializer",
    "TorchDeserializer",
    "CacheGenDeserializer",
    "CacheGenSerializer",
    "CreateSerde",
]
