# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
