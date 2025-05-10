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

import asyncio
import threading
import time
from concurrent.futures import Future
from functools import wraps
from typing import List, Optional

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.connector import CreateConnector
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.experimental.storage_backend.naive_serde import CreateSerde
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
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
        self.lock = threading.Lock()

        assert config.remote_url is not None

        self.remote_url = config.remote_url

        self.memory_allocator = memory_allocator

        self.loop = loop
        self.config = config

        # Re-establish connection only when the connection
        # has been lost for 10 secs
        self.connection: Optional[RemoteConnector] = None
        self.min_reconnect_interval = 10
        self.failure_time = -1000000.0
        self._init_connection()

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, memory_allocator, metadata, config)

        logger.info(f"Connected to remote storage at {config.remote_url}")

        # TODO(Jiayi): If we want to have cache admission policies,
        # we must make decision (whether to send or not) at the local side

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    def __str__(self):
        return self.__class__.__name__

    def _init_connection(self):
        # Initialize connection
        if self.connection is not None:
            return
        if (time.time() - self.failure_time) < self.min_reconnect_interval:
            logger.warning("Connection will not be re-established yet "
                           "since it has not been long enough since "
                           "the last failure")
            return
        try:
            assert self.config.remote_url is not None
            self.connection = CreateConnector(self.config.remote_url,
                                              self.loop, self.memory_allocator,
                                              self.config)
            logger.info("Connection initialized/re-established "
                        f"at {self.config.remote_url}")
        except Exception as e:
            with self.lock:
                self.failure_time = time.time()
            logger.warning(
                f"Failed to initialize/re-establish remote connection: {e}")
            self.connection = None

    @staticmethod
    def _init_connection_wrapper(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._init_connection()
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    @_init_connection_wrapper
    def contains(self, key: CacheEngineKey) -> bool:
        if self.connection is None:
            logger.warning("Connection is None in contains, returning False")
            return False

        future = asyncio.run_coroutine_threadsafe(self.connection.exists(key),
                                                  self.loop)
        try:
            res = future.result()
            return res
        except Exception as e:
            with self.lock:
                self.connection = None
                self.failure_time = time.time()
            logger.warning(f"Remote connection failed in contains: {e}")
            logger.warning("Returning False")
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def put_callback(self, future: Future, key: CacheEngineKey):
        """
        Callback function for put tasks.
        """
        self.lock.acquire()
        self.put_tasks.remove(key)
        self.lock.release()

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:

        if self.connection is None:
            logger.warning(
                "Connection is None in submit_put_task, returning None")
            return None

        self.memory_allocator.ref_count_up(memory_obj)

        self.lock.acquire()
        self.put_tasks.append(key)
        self.lock.release()

        compressed_memory_obj = self.serializer.serialize(memory_obj)
        self.memory_allocator.ref_count_down(memory_obj)

        # NOTE: No need to do error handling here
        # since the `future` is never waited
        future = asyncio.run_coroutine_threadsafe(
            self.connection_put_wrapper(key, compressed_memory_obj), self.loop)
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

        if self.connection is None:
            logger.warning(
                "Connection is None in get_blocking, returning None")
            return None
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(
            self.connection_get_wrapper(key), self.loop)

        try:
            memory_obj = future.result()
        except Exception as e:
            with self.lock:
                self.connection = None
                self.failure_time = time.time()
            logger.warning(f"Error occurred in get_blocking: {e}")
            logger.warning("Returning None")
            return None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get_sync(
            (t2 - t1) * 1000)
        if memory_obj is None:
            return None
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(f"Get takes {(t2 - t1) * 1000:.6f} msec, "
                     f"deserialization takes {(t3 - t2) * 1000:.6f} msec")
        return decompressed_memory_obj

    async def connection_put_wrapper(self, key: CacheEngineKey,
                                     memory_obj: MemoryObj):
        obj_size = memory_obj.get_size()
        begin = time.perf_counter()
        assert self.connection is not None
        await self.connection.put(key, memory_obj)
        end = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_put(
            (end - begin) * 1000)
        self.stats_monitor.update_interval_remote_write_metrics(obj_size)
        logger.debug(f"Bytes offloaded: {obj_size / 1e6:.4f} MBytes, ")

    async def connection_get_wrapper(self, key: CacheEngineKey):
        begin = time.perf_counter()
        assert self.connection is not None
        memory_obj = await self.connection.get(key)
        end = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get(
            (end - begin) * 1000)
        if memory_obj is not None:
            obj_size = memory_obj.get_size()
            self.stats_monitor.update_interval_remote_read_metrics(obj_size)
            logger.debug(f"Bytes loaded: {obj_size / 1e6:.4f} MBytes, ")
        return memory_obj

    def close(self):
        try:
            assert self.connection is not None
            future = asyncio.run_coroutine_threadsafe(self.connection.close(),
                                                      self.loop)
            future.result()
            logger.info("Remote backend closed.")
        except Exception as e:
            logger.warning(
                f"Error occurred when closing remote connection: {e}")
