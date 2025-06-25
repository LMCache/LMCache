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
from concurrent.futures import Future, TimeoutError
from functools import wraps
from typing import List, Optional
import asyncio
import threading
import time

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.naive_serde import CreateSerde

logger = init_logger(__name__)


class RemoteBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.put_tasks: List[CacheEngineKey] = []
        self.lock = threading.Lock()

        assert config.remote_url is not None

        self.remote_url = config.remote_url
        self.blocking_timeout_secs = config.blocking_timeout_secs

        self.local_cpu_backend = local_cpu_backend

        self.loop = loop
        self.config = config
        self.metadata = metadata

        # Re-establish connection only when the connection
        # has been lost for 10 secs
        self.connection: Optional[RemoteConnector] = None
        self.min_reconnect_interval = 10
        self.failure_time = -1000000.0
        self._init_connection()

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, metadata, config
        )

        # Precompute MLA mode status
        self._mla_worker_id_as0_mode = (
            config.extra_config is not None
            and config.extra_config.get("remote_enable_mla_worker_id_as0", False)
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )
        logger.info(f"metadata={metadata}")
        logger.info(
            f"Connected to remote storage at {config.remote_url}, "
            f"remote_mla_worker_id_as_0 mode: {self._mla_worker_id_as0_mode}"
        )

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
            logger.warning(
                "Connection will not be re-established yet "
                "since it has not been long enough since "
                "the last failure"
            )
            return
        try:
            assert self.config.remote_url is not None
            self.connection = CreateConnector(
                self.config.remote_url,
                self.loop,
                self.local_cpu_backend,
                self.config,
            )
            logger.info(
                f"Connection initialized/re-established at {self.config.remote_url}"
            )
        except Exception as e:
            with self.lock:
                self.failure_time = time.time()
            logger.warning(f"Failed to initialize/re-establish remote connection: {e}")
            self.connection = None

    @staticmethod
    def _init_connection_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._init_connection()
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    # TODO(Jiayi): handle `pin` semantics
    @_init_connection_wrapper
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.connection is None:
            logger.warning("Connection is None in contains, returning False")
            return False

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = CacheEngineKey(
                key.fmt, key.model_name, key.world_size, 0, key.chunk_hash
            )

        future = asyncio.run_coroutine_threadsafe(
            self.connection.exists(key), self.loop
        )
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
            logger.warning("Connection is None in submit_put_task, returning None")
            return None

        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return None

        memory_obj.ref_count_up()

        self.lock.acquire()
        self.put_tasks.append(key)
        self.lock.release()

        compressed_memory_obj = self.serializer.serialize(memory_obj)
        memory_obj.ref_count_down()

        # NOTE: No need to do error handling here
        # since the `future` is never waited
        future = asyncio.run_coroutine_threadsafe(
            self.connection.put(key, compressed_memory_obj), self.loop
        )
        lambda_callback = lambda f: self.put_callback(f, key)
        future.add_done_callback(lambda_callback)
        return future

    def batched_submit_put_task(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> Optional[List[Future]]:
        return [
            self.submit_put_task(key, memory_obj)
            for key, memory_obj in zip(keys, memory_objs, strict=False)
        ]

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """

        if self.connection is None:
            logger.warning("Connection is None in get_blocking, returning None")
            return None
        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = CacheEngineKey(
                key.fmt, key.model_name, key.world_size, 0, key.chunk_hash
            )
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)

        try:
            memory_obj = future.result(self.blocking_timeout_secs)
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning("get blocking timeout, trigger cancel the future task")
                future.cancel()
            with self.lock:
                self.connection = None
                self.failure_time = time.time()
            logger.warning(f"Error occurred in get_blocking: {e}")
            logger.warning("Returning None")
            return None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get_sync((t2 - t1) * 1000)
        if memory_obj is None:
            return None
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(
            f"Get takes {(t2 - t1) * 1000:.6f} msec, "
            f"deserialization takes {(t3 - t2) * 1000:.6f} msec"
        )
        return decompressed_memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        raise NotImplementedError

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support pin. "
            "This method is a no-op and will return True."
        )
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support unpin. "
            "This method is a no-op and will return True."
        )
        return True

    def close(self):
        try:
            assert self.connection is not None
            future = asyncio.run_coroutine_threadsafe(
                self.connection.close(), self.loop
            )
            future.result()
            logger.info("Remote backend closed.")
        except Exception as e:
            logger.warning(f"Error occurred when closing remote connection: {e}")
