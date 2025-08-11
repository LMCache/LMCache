# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, no_type_check
import asyncio
import inspect
import os

# Third Party
import redis

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# TODO(Jiayi): Use `redis.asyncio`
# NOTE(Jiayi): `redis-py` supports async operations, but data copy
# cannot be avoided. `hiredis` is more lower-level but asyncio is
# not supported.


class RedisConnector(RemoteConnector):
    """
    The remote url should start with "redis://" and only have one host-port pair
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        self.connection = redis.from_url(url=url, decode_responses=False)
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.connection.exists(key.to_string() + "metadata"))

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return bool(self.connection.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        metadata_bytes = self.connection.get(key_str + "metadata")

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # TODO(Jiayi): Find a way to do `get` inplace
        kv_bytes = self.connection.get(key_str + "kv_bytes")
        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A better way is to aggregate metadata
            # and kv cache in one key.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            self.connection.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[: metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[: metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[: metadata.length] = converted

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shape, kv_dtype, memory_format
        ).serialize()

        key_str = key.to_string()
        self.connection.set(key_str + "metadata", metadata_bytes)
        self.connection.set(key_str + "kv_bytes", kv_bytes)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.connection.close()
        logger.info("Closed the redis connection")


class RedisSentinelConnector(RemoteConnector):
    """
    Uses redis.Sentinel to connect to a Redis cluster.
    The hosts are specified in the config file, started with "redis-sentinel://"
    and separated by commas.

    Example:
        remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"

    Extra environment variables:
    - REDIS_SERVICE_NAME (required) -- service name for redis.
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    ENV_REDIS_TIMEOUT = "REDIS_TIMEOUT"
    ENV_REDIS_SERVICE_NAME = "REDIS_SERVICE_NAME"

    def __init__(
        self,
        hosts_and_ports: List[Tuple[str, int]],
        username: str,
        password: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # Get service name
        match os.environ.get(self.ENV_REDIS_SERVICE_NAME):
            case None:
                logger.warning(
                    f"Environment variable {self.ENV_REDIS_SERVICE_NAME} is "
                    f"not found, using default value 'redismaster'"
                )
                service_name = "redismaster"
            case value:
                service_name = value

        timeout: float = -1000.0

        # Get timeout
        match os.environ.get(self.ENV_REDIS_TIMEOUT):
            case None:
                timeout = 1
            case value:
                timeout = float(value)

        logger.info(f"Host and ports: {hosts_and_ports}")
        self.sentinel = redis.Sentinel(hosts_and_ports, socket_timeout=timeout)
        self.master = self.sentinel.master_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )
        self.slave = self.sentinel.slave_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )

        self.local_cpu_backend = local_cpu_backend

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        metadata_bytes = self.slave.get(key_str + "metadata")

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # TODO(Jiayi): Find a way to do `get` inplace
        kv_bytes = self.slave.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A background sweeper might be better
            # for the sake of performance.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            self.master.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[0 : metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[0 : metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[0 : metadata.length] = converted

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shape, kv_dtype, memory_format
        ).serialize()

        key_str = key.to_string()
        self.master.set(key_str + "metadata", metadata_bytes)
        self.master.set(key_str + "kv_bytes", kv_bytes)

        memory_obj.ref_count_down()

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.master.close()
        self.slave.close()
