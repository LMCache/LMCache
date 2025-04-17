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

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.protocol import RedisMetadata
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class ValkeyConnector(RemoteConnector):
    """
    The remote url should start with "valkey://" and only have one
    host-port pair
    """

    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        self.connection = valkey.Valkey(host=host,
                                        port=port,
                                        decode_responses=False)

        self.memory_allocator = memory_allocator
        self.loop = loop

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.connection.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        valkey_metadata_bytes = self.connection.get(key_str + "metadata")

        if valkey_metadata_bytes is None:
            return None

        assert not inspect.isawaitable(valkey_metadata_bytes)

        valkey_metadata = RedisMetadata.deserialize(
            memoryview(valkey_metadata_bytes))

        memory_obj = self.memory_allocator.allocate(
            valkey_metadata.shape,
            valkey_metadata.dtype,
            valkey_metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        kv_bytes = self.connection.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # consistency issues.
            # and kv cache in one key.
            logger.warning("Key exists but KV cache does not exist."
                           "Might happen when the cache is evicted by valkey.")
            self.connection.delete(key_str + "metadata")
            return None

        view = memoryview(memory_obj.byte_array)
        view[:valkey_metadata.length] = kv_bytes

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        valkey_metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape,
                                              kv_dtype,
                                              memory_format).serialize()

        key_str = key.to_string()
        self.connection.set(key_str + "metadata", valkey_metadata_bytes)
        self.connection.set(key_str + "kv_bytes", kv_bytes)

        self.memory_allocator.ref_count_down(memory_obj)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.connection.close()
        logger.info("Closed the valkey connection")


class ValkeySentinelConnector(RemoteConnector):
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
        # Get service name
        match os.environ.get(self.ENV_VALKEY_SERVICE_NAME):
            case None:
                logger.warning(
                    f"Environment variable {self.ENV_VALKEY_SERVICE_NAME} is "
                    f"not found, using default value 'valkeymaster'")
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
        self.master = self.sentinel.master_for(service_name,
                                               socket_timeout=timeout)
        self.slave = self.sentinel.slave_for(service_name,
                                             socket_timeout=timeout)

        self.memory_allocator = memory_allocator

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.slave.exists(key.to_string() + "metadata")

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        valkey_metadata_bytes = self.slave.get(key_str + "metadata")

        if valkey_metadata_bytes is None:
            return None

        assert not inspect.isawaitable(valkey_metadata_bytes)

        valkey_metadata = RedisMetadata.deserialize(valkey_metadata_bytes)

        memory_obj = self.memory_allocator.allocate(
            valkey_metadata.shape,
            valkey_metadata.dtype,
            valkey_metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        kv_bytes = self.slave.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # consistency issues.
            # for the sake of performance.
            logger.warning("Key exists but KV cache does not exist."
                           "Might happen when the cache is evicted by valkey.")
            self.master.delete(key_str + "metadata")
            return None

        view = memoryview(memory_obj.byte_array)
        view[0:valkey_metadata.length] = kv_bytes

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        valkey_metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape,
                                              kv_dtype,
                                              memory_format).serialize()

        key_str = key.to_string()
        self.master.set(key_str + "metadata", valkey_metadata_bytes)
        self.master.set(key_str + "kv_bytes", kv_bytes)

        self.memory_allocator.ref_count_down(memory_obj)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.master.close()
        self.slave.close()
