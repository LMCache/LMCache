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
            f"Bytes offloaded: {obj_size / 1e6:.3f} MBytes "
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
                f"Bytes loaded: {obj_size / 1e6:.3f} MBytes "
                f"in {(end - begin) * 1000:.3f}ms"
            )
        return memory_obj

    # Delegate all other methods to the underlying connector
    async def exists(self, key: CacheEngineKey) -> bool:
        return await self._connector.exists(key)

    async def list(self) -> List[str]:
        return await self._connector.list()

    async def close(self) -> None:
        await self._connector.close()

    def getWrappedConnector(self) -> RemoteConnector:
        return self._connector
