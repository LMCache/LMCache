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
import inspect
import os
from typing import List, Optional, Tuple, Union, no_type_check

import valkey
from valkey import Valkey

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.protocol import METADATA_BYTES_LEN, RedisMetadata
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class BaseValkeyConnector(RemoteConnector):
    """Base Valkey connector with common operations"""

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator

    @property
    def read_client(self) -> Valkey:
        """Client for read operations (to be implemented by subclasses)"""
        raise NotImplementedError

    @property
    def write_client(self) -> Valkey:
        """Client for write operations (to be implemented by subclasses)"""
        raise NotImplementedError

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.read_client.exists(key.to_string()))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        combined_bytes = self.read_client.get(key.to_string())
        if combined_bytes is None:
            return None

        assert not inspect.isawaitable(combined_bytes)

        valkey_metadata = RedisMetadata.deserialize(
            memoryview(combined_bytes[:METADATA_BYTES_LEN]))
        kv_bytes = combined_bytes[METADATA_BYTES_LEN:METADATA_BYTES_LEN +
                                  valkey_metadata.length]

        memory_obj = self.memory_allocator.allocate(
            valkey_metadata.shape,
            valkey_metadata.dtype,
            valkey_metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        view = memoryview(memory_obj.byte_array)
        view[:valkey_metadata.length] = kv_bytes

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        kv_bytes = memory_obj.byte_array
        valkey_metadata = RedisMetadata(len(kv_bytes), memory_obj.get_shape(),
                                        memory_obj.get_dtype(),
                                        memory_obj.get_memory_format())

        combined_bytes = valkey_metadata.serialize() + kv_bytes
        try:
            self.write_client.set(key_str, combined_bytes)
        except Exception as e:
            logger.error(f"Failed to put key {key_str},"
                         f"meta type: {type(valkey_metadata)},"
                         f"data: {type(kv_bytes)}: {e}")

        self.memory_allocator.ref_count_down(memory_obj)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass


class ValkeyConnector(BaseValkeyConnector):
    """
    The remote url should start with "valkey://" and only have one
    host-port pair
    """

    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        super().__init__(memory_allocator)
        self._client = Valkey(host=host, port=port, decode_responses=False)
        self.loop = loop

    @property
    def read_client(self) -> Valkey:
        return self._client

    @property
    def write_client(self) -> Valkey:
        return self._client

    async def close(self):
        self._client.close()


class ValkeySentinelConnector(BaseValkeyConnector):
    """
    Uses valkey.Sentinel to connect to a Valkey cluster.
    The hosts are specified in the config file, started with "valkey-sentinel://"
    and separated by commas.

    Example:
        remote_url: "valkey-sentinel://localhost:26379,localhost:26380,localhost:26381"

    Extra environment variables:
    - VALKEY_SERVICE_NAME (required) -- service name for valkey.
    - VALKEY_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    ENV_VALKEY_TIMEOUT = "VALKEY_TIMEOUT"
    ENV_VALKEY_SERVICE_NAME = "VALKEY_SERVICE_NAME"

    def __init__(self, hosts_and_ports: List[Tuple[str, Union[str, int]]],
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        super().__init__(memory_allocator)
        # Get service name
        match os.environ.get(self.ENV_VALKEY_SERVICE_NAME):
            case None:
                logger.warning(
                    f"Environment variable {self.ENV_VALKEY_SERVICE_NAME} is"
                    f" not found, using default value 'valkeymaster'")
                service_name = "valkeymaster"
            case value:
                service_name = value

        timeout: float = -1000.0

        # Get timeout
        match os.environ.get(self.ENV_VALKEY_TIMEOUT):
            case None:
                timeout = 1
            case value:
                timeout = float(value)

        logger.info(f"Host and ports: {hosts_and_ports}")
        self.sentinel = valkey.Sentinel(hosts_and_ports, timeout)
        # FIXME(maobaolong):  _master would be changed to _slave
        self._master = self.sentinel.master_for(service_name,
                                                socket_timeout=timeout)
        self._slave = self.sentinel.slave_for(service_name,
                                              socket_timeout=timeout)

    @property
    def read_client(self) -> Valkey:
        return self._slave

    @property
    def write_client(self) -> Valkey:
        return self._master

    async def close(self):
        self._master.close()
        self._slave.close()
