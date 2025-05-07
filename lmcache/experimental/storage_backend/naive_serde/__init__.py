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

from typing import Optional, Tuple

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import MemoryAllocatorInterface
from lmcache.experimental.storage_backend.naive_serde.cachegen_decoder import \
    CacheGenDeserializer
from lmcache.experimental.storage_backend.naive_serde.cachegen_encoder import \
    CacheGenSerializer
from lmcache.experimental.storage_backend.naive_serde.kivi_serde import (
    KIVIDeserializer, KIVISerializer)
from lmcache.experimental.storage_backend.naive_serde.naive_serde import (
    NaiveDeserializer, NaiveSerializer)
from lmcache.experimental.storage_backend.naive_serde.serde import (
    Deserializer, Serializer)


def CreateSerde(
    serde_type: str,
    memory_allocator: MemoryAllocatorInterface,
    metadata: LMCacheEngineMetadata,
    config: LMCacheEngineConfig,
) -> Tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None

    if serde_type == "naive":
        s, d = NaiveSerializer(memory_allocator), \
            NaiveDeserializer()
    elif serde_type == "kivi":
        s, d = KIVISerializer(memory_allocator), \
            KIVIDeserializer(memory_allocator)
    elif serde_type == "cachegen":
        s, d = CacheGenSerializer(
                config, metadata, memory_allocator), \
            CacheGenDeserializer(
                config, metadata, memory_allocator)
    else:
        raise ValueError(f"Invalid type: {serde_type}")

    return s, d


__all__ = [
    "Serializer",
    "Deserializer",
    "KIVISerializer",
    "KIVIDeserializer",
    "CreateSerde",
]
