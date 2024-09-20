from lmcache.storage_backend.serde.serde import Serializer, Deserializer, SerializerDebugWrapper, DeserializerDebugWrapper
from lmcache.storage_backend.serde.torch_serde import TorchSerializer, TorchDeserializer
from lmcache.storage_backend.serde.safe_serde import SafeSerializer, SafeDeserializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.fast_serde import FastSerializer, FastDeserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata, GlobalConfig

def CreateSerde(serde_type: str, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
    if serde_type == "torch":
        s, d = TorchSerializer(), TorchDeserializer()
    elif serde_type == "safetensor":
        s, d = SafeSerializer(), SafeDeserializer()
    elif serde_type == "cachegen":
        s, d = CacheGenSerializer(config, metadata), CacheGenDeserializer(config, metadata)
    elif serde_type == "fast":
        s, d = FastSerializer(), FastDeserializer()
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
