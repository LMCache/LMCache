# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class InstrumentedRemoteConnector(RemoteConnector):
    """
    A connector that instruments the underlying connector with
    metrics collection and logging capabilities.
    """

    def __init__(self, connector: RemoteConnector):
        self._connector = connector
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.name = self.__repr__()

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        obj_size = memory_obj.get_size()
        begin = time.perf_counter()
        try:
            await self._connector.put(key, memory_obj)
        finally:
            # Ensure reference count is decreased even if exception occurs
            memory_obj.ref_count_down()

        end = time.perf_counter()
        self._stats_monitor.update_interval_remote_time_to_put((end - begin) * 1000)
        self._stats_monitor.update_interval_remote_write_metrics(obj_size)
        logger.debug(
            f"[{self.name}]Bytes offloaded: {obj_size / 1e6:.3f} MBytes "
            f"in {(end - begin) * 1000:.3f}ms"
        )

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        begin = time.perf_counter()
        memory_obj = await self._connector.get(key)
        end = time.perf_counter()
        self._stats_monitor.update_interval_remote_time_to_get((end - begin) * 1000)
        if memory_obj is not None:
            obj_size = memory_obj.get_size()
            self._stats_monitor.update_interval_remote_read_metrics(obj_size)
            logger.debug(
                f"[{self.name}]Bytes loaded: {obj_size / 1e6:.3f} MBytes "
                f"in {(end - begin) * 1000:.3f}ms"
            )
        return memory_obj

    @property
    def has_batched_get(self) -> bool:
        return hasattr(self._connector, "batched_get")

    @property
    def has_batched_put(self) -> bool:
        return hasattr(self._connector, "batched_put")

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        begin = time.perf_counter()
        memory_objs = await self._connector.batched_get(keys)
        end = time.perf_counter()
        self._stats_monitor.update_interval_remote_time_to_get((end - begin) * 1000)
        total_obj_size = 0
        for memory_obj in memory_objs:
            if memory_obj is not None:
                total_obj_size += memory_obj.get_size()
        self._stats_monitor.update_interval_remote_read_metrics(total_obj_size)
        logger.debug(
            f"[{self.name}]Bytes loaded: {total_obj_size / 1e6:.3f} MBytes "
            f"in {(end - begin) * 1000:.3f}ms"
        )
        return memory_objs

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        begin = time.perf_counter()
        total_obj_size = 0
        try:
            await self._connector.batched_put(keys, memory_objs)
        finally:
            for memory_obj in memory_objs:
                total_obj_size += memory_obj.get_size()
                memory_obj.ref_count_down()
        end = time.perf_counter()
        self._stats_monitor.update_interval_remote_time_to_put((end - begin) * 1000)
        self._stats_monitor.update_interval_remote_write_metrics(total_obj_size)
        logger.debug(
            f"[{self.name}]Bytes offloaded: {total_obj_size / 1e6:.3f} MBytes "
            f"in {(end - begin) * 1000:.3f}ms"
        )

    # Delegate all other methods to the underlying connector
    async def exists(self, key: CacheEngineKey) -> bool:
        return await self._connector.exists(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self._connector.exists_sync(key)

    async def list(self) -> List[str]:
        return await self._connector.list()

    async def close(self) -> None:
        await self._connector.close()

    def getWrappedConnector(self) -> RemoteConnector:
        return self._connector

    def support_ping(self) -> bool:
        return self._connector.support_ping()

    async def ping(self) -> int:
        return await self._connector.ping()

    def support_batched_get(self) -> bool:
        return self._connector.support_batched_get()

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        return await self._connector.batched_get(keys)

    def __repr__(self) -> str:
        return f"InstrumentedRemoteConnector({self._connector})"
