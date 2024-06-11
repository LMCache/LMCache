from lmcache.storage_backend.serde.serde import Serializer, Deserializer, SerializerDebugWrapper, DeserializerDebugWrapper
from lmcache.storage_backend.serde.torch_serde import TorchSerializer, TorchDeserializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata, GlobalConfig

def CreateSerde(serde_type: str, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
    if serde_type == "torch":
        s, d = TorchSerializer(), TorchDeserializer()
    elif serde_type == "cachegen":
        s, d = CacheGenSerializer(config, metadata), CacheGenDeserializer(config, metadata)
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
