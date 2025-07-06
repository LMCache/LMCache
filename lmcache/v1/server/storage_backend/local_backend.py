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
from collections import OrderedDict
from typing import List, Optional
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import LMSMemoryObj
from lmcache.v1.protocol import ClientMetaMessage
from lmcache.v1.server.storage_backend.abstract_backend import LMSBackendInterface
from lmcache.v1.storage_backend.evictor.base_evictor import PutStatus
from lmcache.v1.storage_backend.evictor.lru_evictor import LRUEvictor

logger = init_logger(__name__)


class LMSLocalBackend(LMSBackendInterface):
    def __init__(
        self,
        max_cache_size: float = 10.0,
    ):
        self.dict: OrderedDict[CacheEngineKey, LMSMemoryObj] = OrderedDict()

        self.lock = threading.Lock()

        # Initialize LRU evictor with max cache size in GB
        self.evictor = LRUEvictor(max_cache_size=max_cache_size)
        logger.info(
            f"Initialized LMSLocalBackend with max_cache_size={max_cache_size}GB"
        )

    # TODO
    def list_keys(self) -> List[CacheEngineKey]:
        with self.lock:
            return list(self.dict.keys())

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.lock:
            return key in self.dict

    # TODO
    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        with self.lock:
            if key in self.dict:
                self.dict.pop(key)
                logger.debug(f"Removed cache item: {key}")
            else:
                logger.debug(f"Attempted to remove non-existent key: {key}")

    def put(
        self,
        client_meta: ClientMetaMessage,
        kv_chunk_bytes: bytearray,
    ) -> None:
        with self.lock:
            # Create memory object
            memory_obj = LMSMemoryObj(
                kv_chunk_bytes,
                client_meta.length,
                client_meta.fmt,
                client_meta.dtype,
                client_meta.shape,
            )

            # Check if eviction is needed
            cache_size = memory_obj.get_size()
            evict_keys, put_status = self.evictor.update_on_put(self.dict, cache_size)

            # If cache is too large, abort
            if put_status == PutStatus.ILLEGAL:
                logger.warning(
                    f"Cannot store cache item {client_meta.key}: "
                    f"exceeds cache size limit"
                )
                return

            # Evict old items
            for evict_key in evict_keys:
                if evict_key in self.dict:
                    self.dict.pop(evict_key)
                    logger.debug(f"Evicted cache item: {evict_key}")

            # Store new item
            self.dict[client_meta.key] = memory_obj

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[LMSMemoryObj]:
        with self.lock:
            memory_obj = self.dict.get(key, None)

            # Update cache recency for LRU
            if memory_obj is not None:
                self.evictor.update_on_hit(key, self.dict)

            return memory_obj

    def close(self):
        pass


# TODO(Jiayi): please implement the remote disk backend
# class LMSLocalDiskBackend(LMSBackendInterface):
#    pass
