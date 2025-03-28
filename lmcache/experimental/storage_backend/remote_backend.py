import asyncio
import threading
import time
from concurrent.futures import Future
from typing import List, Optional

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.connector import CreateConnector
from lmcache.experimental.storage_backend.naive_serde import CreateSerde
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

logger = init_logger(__name__)


class RemoteBackend(StorageBackendInterface):

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
        lookup_server: Optional[LookupServerInterface] = None,
    ):

        self.put_tasks: List[CacheEngineKey] = []
        self.put_tasks_lock = threading.Lock()

        assert config.remote_url is not None
        # Initialize connection
        self.connection = CreateConnector(config.remote_url, loop,
                                          memory_allocator)

        self.remote_url = config.remote_url

        self.memory_allocator = memory_allocator

        self.loop = loop

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, memory_allocator, metadata, config)

        logger.info(f"Connected to remote storage at {config.remote_url}")

        # TODO(Jiayi): If we want to have cache admission policies,
        # we must make decision (whether to send or not) at the local side

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.connection.exists(key),
                                                  self.loop)
        return future.result()

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_tasks_lock:
            return key in self.put_tasks

    def put_callback(self, future: Future, key: CacheEngineKey):
        """
        Callback function for put tasks.
        """
        self.put_tasks_lock.acquire()
        self.put_tasks.remove(key)
        self.put_tasks_lock.release()

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:

        self.memory_allocator.ref_count_up(memory_obj)

        self.put_tasks_lock.acquire()
        self.put_tasks.append(key)
        self.put_tasks_lock.release()

        compressed_memory_obj = self.serializer.serialize(memory_obj)

        future = asyncio.run_coroutine_threadsafe(
            self.connection.put(key, compressed_memory_obj), self.loop)

        self.memory_allocator.ref_count_down(memory_obj)

        lambda_callback = lambda f: \
                self.put_callback(f, key)
        future.add_done_callback(lambda_callback)

        return future

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        pass

    @_lmcache_nvtx_annotate
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self.connection.get(key),
                                                  self.loop)
        memory_obj = future.result()

        t2 = time.perf_counter()
        if memory_obj is None:
            return None
        obj_size = memory_obj.get_size()
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(f"Get takes {(t2 - t1) * 1000:.6f} msec, "
                     f"Bytes loaded: {obj_size / 1e6:.4f} MBytes, "
                     f"deserialization takes {(t3 - t2) * 1000:.6f} msec")
        return decompressed_memory_obj

    def close(self):
        future = asyncio.run_coroutine_threadsafe(self.connection.close(),
                                                  self.loop)
        future.result()
        logger.info("Remote backend closed.")
