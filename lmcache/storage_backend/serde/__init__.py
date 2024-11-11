from typing import Optional, Tuple

import torch

from lmcache.config import (GlobalConfig, LMCacheEngineConfig,
                            LMCacheEngineMetadata)
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.fast_serde import (FastDeserializer,
                                                      FastSerializer)
from lmcache.storage_backend.serde.safe_serde import (SafeDeserializer,
                                                      SafeSerializer)
from lmcache.storage_backend.serde.serde import (Deserializer,
                                                 DeserializerDebugWrapper,
                                                 Serializer,
                                                 SerializerDebugWrapper)
from lmcache.storage_backend.serde.torch_serde import (TorchDeserializer,
                                                       TorchSerializer)

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "double": torch.float64,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def CreateSerde(
    serde_type: str,
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
) -> Tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None
    print(STR_DTYPE_TO_TORCH_DTYPE[metadata.dtype])
    if serde_type == "torch":
        s, d = TorchSerializer(), TorchDeserializer(
            STR_DTYPE_TO_TORCH_DTYPE[metadata.dtype])
    elif serde_type == "safetensor":
        s, d = SafeSerializer(), SafeDeserializer(
            STR_DTYPE_TO_TORCH_DTYPE[metadata.dtype])
    elif serde_type == "cachegen":
        s, d = CacheGenSerializer(config, metadata), CacheGenDeserializer(
            config, metadata, STR_DTYPE_TO_TORCH_DTYPE[metadata.dtype])
    elif serde_type == "fast":
        s, d = FastSerializer(), FastDeserializer(
            STR_DTYPE_TO_TORCH_DTYPE[metadata.dtype])
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
