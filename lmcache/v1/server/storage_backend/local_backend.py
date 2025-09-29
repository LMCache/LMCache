# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import List, Optional
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import LMSMemoryObj
from lmcache.v1.protocol import ClientMetaMessage
from lmcache.v1.server.storage_backend.abstract_backend import LMSBackendInterface

logger = init_logger(__name__)


class AsyncLRU:
    def __init__(self, capacity: int):
        self.dict: OrderedDict[CacheEngineKey, LMSMemoryObj] = OrderedDict()
        self.capacity = capacity * 1024**3
        self.size = 0
        self.lock = asyncio.Lock()

    async def exists(self, key: CacheEngineKey, pin: bool = False) -> bool:
        async with self.lock:
            if key in self.dict:
                if pin:
                    self.dict[key].pin_count += 1
                self.dict.move_to_end(key)
                return True
            return False

    async def get(self, key: CacheEngineKey) -> Optional[LMSMemoryObj]:
        async with self.lock:
            if key not in self.dict:
                return None
            self.dict.move_to_end(key)
            return self.dict[key]

    async def put(self, key: CacheEngineKey, value: LMSMemoryObj):
        async with self.lock:
            alloc_size = value.length
            if alloc_size > self.capacity:
                raise ValueError(
                    f"Allocation size {alloc_size} is",
                    f"larger than the capacity {self.capacity}",
                )
            if key in self.dict:
                self.dict.move_to_end(key)
                return None
            self.dict[key] = value
            self.size += alloc_size
            # TODO: currently, there is a possibility that the LMServer accepts lots
            # of requests and we actually hold more than the capacity in the LRU
            # if we cannot evict due to pins. This should be ok unless the CPU space
            # is extremely tight.
            while self.size > self.capacity:
                for key, value in reversed(self.dict.items()):
                    if value.pin_count == 0:
                        self.dict.pop(key)
                        self.size -= value.length
                        if self.size <= self.capacity:
                            break

    async def remove(self, key: CacheEngineKey):
        async with self.lock:
            value = self.dict.pop(key)
            self.size -= value.length

    async def batched_unpin(self, keys: List[CacheEngineKey]):
        async with self.lock:
            for key in keys:
                if key in self.dict:
                    self.dict[key].pin_count -= 1

    async def list(self) -> List[CacheEngineKey]:
        async with self.lock:
            return list(self.dict.keys())

    async def close(self):
        async with self.lock:
            self.dict.clear()


class LMSLocalBackend(LMSBackendInterface):
    def __init__(
        self,
        capacity: int,
    ) -> None:
        self.lru = AsyncLRU(capacity)

    async def list_keys(self) -> List[CacheEngineKey]:
        return await self.lru.list()

    async def contains(
        self,
        key: CacheEngineKey,
        pin: bool = False,
    ) -> bool:
        return await self.lru.exists(key, pin)

    async def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        await self.lru.remove(key)

    async def put(
        self,
        client_meta: ClientMetaMessage,
        kv_chunk_bytes: bytearray,
    ) -> None:
        await self.lru.put(
            client_meta.key,
            LMSMemoryObj(
                kv_chunk_bytes,
                client_meta.length,
                client_meta.fmt,
                client_meta.dtype,
                client_meta.shape,
            ),
        )

    @_lmcache_nvtx_annotate
    async def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[LMSMemoryObj]:
        return await self.lru.get(key)

    async def batched_unpin(self, keys: List[CacheEngineKey]) -> None:
        await self.lru.batched_unpin(keys)

    async def close(self):
        await self.lru.close()


# TODO(Jiayi): please implement the remote disk backend
# class LMSLocalDiskBackend(LMSBackendInterface):
#    pass
