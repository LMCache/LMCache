# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, no_type_check
import asyncio
import os
import struct

# Third Party
from redis.asyncio import from_url as redis_from_url
from redis.asyncio.sentinel import Sentinel
import redis

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# Constants for the combined format
METADATA_HEADER_SIZE = 28  # 7 integers * 4 bytes each from RemoteMetadata.serialize()


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
        self.connection = redis_from_url(url=url, decode_responses=False)
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        # pip install hiredis automatically enables hiredis (asyncio compatible)
        logger.info(
            f"HIREDIS_AVAILABLE: {getattr(redis.connection, 'HIREDIS_AVAILABLE', None)}"
        )

    @staticmethod
    def _pack_metadata_and_data(memory_obj: MemoryObj) -> bytes:
        """Pack metadata and KV bytes into a single byte array with header format."""
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata = RemoteMetadata(len(kv_bytes), kv_shape, kv_dtype, memory_format)
        metadata_bytes = metadata.serialize()

        # Combine: [metadata_header][kv_data]
        return metadata_bytes + kv_bytes

    @staticmethod
    def _unpack_metadata_and_data(
        combined_bytes: bytes,
    ) -> Tuple[RemoteMetadata, bytes]:
        """Unpack metadata and KV bytes from combined format."""
        if len(combined_bytes) < METADATA_HEADER_SIZE:
            raise ValueError(
                f"Combined bytes too small: "
                f"{len(combined_bytes)} < {METADATA_HEADER_SIZE}"
            )

        metadata_bytes = combined_bytes[:METADATA_HEADER_SIZE]
        kv_bytes = combined_bytes[METADATA_HEADER_SIZE:]

        metadata = RemoteMetadata.deserialize(metadata_bytes)
        return metadata, kv_bytes

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.connection.exists(key.to_string())

    def exists_sync(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError("exists_sync is not supported for RedisConnector yet")

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        # Get combined metadata + KV bytes
        combined_bytes = await self.connection.get(key_str)
        if combined_bytes is None:
            logger.warning(f"Key {key_str} does not exist in the cache")
            return None

        try:
            metadata, kv_bytes = self._unpack_metadata_and_data(combined_bytes)
        except (ValueError, struct.error) as e:
            logger.error(f"Failed to unpack data for key {key_str}: {e}")
            return None

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
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

    def support_batched_get(self) -> bool:
        return True

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        # TODO: currently batching degrades
        # performance with async redis client
        # need to figure out why batching
        # degrades async performance

        # temporary pipeline
        async with self.connection.pipeline(transaction=False) as pipeline:
            key_strings = [key.to_string() for key in keys]

            for key_str in key_strings:
                pipeline.get(key_str)

            combined_bytes_list = await pipeline.execute()
            final_memory_objs = []

            for i, combined_bytes in enumerate(combined_bytes_list):
                if combined_bytes is None:
                    logger.warning(f"Key {keys[i]} does not exist in the cache")
                    final_memory_objs.append(None)
                    continue

                try:
                    metadata, kv_bytes = self._unpack_metadata_and_data(combined_bytes)
                except (ValueError, struct.error) as e:
                    logger.error(f"Failed to unpack data for key {keys[i]}: {e}")
                    final_memory_objs.append(None)
                    continue

                memory_obj = self.local_cpu_backend.allocate(
                    metadata.shape,
                    metadata.dtype,
                    metadata.fmt,
                )
                if memory_obj is None:
                    logger.warning("Failed to allocate memory during remote receive")
                    final_memory_objs.append(None)
                    continue

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

                final_memory_objs.append(memory_obj)

            return final_memory_objs

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        combined_bytes = self._pack_metadata_and_data(memory_obj)
        await self.connection.set(key_str, combined_bytes)

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        async with self.connection.pipeline(transaction=False) as pipeline:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                key_str = key.to_string()
                combined_bytes = self._pack_metadata_and_data(memory_obj)
                pipeline.set(key_str, combined_bytes)
            await pipeline.execute()

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.connection.aclose()
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
        self.sentinel = Sentinel(hosts_and_ports, socket_timeout=timeout)
        self.master = self.sentinel.master_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )
        self.slave = self.sentinel.slave_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )

        self.local_cpu_backend = local_cpu_backend

    @staticmethod
    def _pack_metadata_and_data(memory_obj: MemoryObj) -> bytes:
        """Pack metadata and KV bytes into a single byte array with header format."""
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata = RemoteMetadata(len(kv_bytes), kv_shape, kv_dtype, memory_format)
        metadata_bytes = metadata.serialize()

        # Combine: [metadata_header][kv_data]
        return metadata_bytes + kv_bytes

    @staticmethod
    def _unpack_metadata_and_data(
        combined_bytes: bytes,
    ) -> Tuple[RemoteMetadata, bytes]:
        """Unpack metadata and KV bytes from combined format."""
        if len(combined_bytes) < METADATA_HEADER_SIZE:
            raise ValueError(
                f"Combined bytes too small: "
                f"{len(combined_bytes)} < {METADATA_HEADER_SIZE}"
            )

        metadata_bytes = combined_bytes[:METADATA_HEADER_SIZE]
        kv_bytes = combined_bytes[METADATA_HEADER_SIZE:]

        metadata = RemoteMetadata.deserialize(metadata_bytes)
        return metadata, kv_bytes

    async def exists(self, key: CacheEngineKey) -> bool:
        # Simple Redis key existence check
        return await self.slave.exists(key.to_string())

    def exists_sync(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError(
            "exists_sync is not supported for RedisSentinelConnector yet"
        )

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        # Get combined metadata + KV bytes
        combined_bytes = await self.slave.get(key_str)
        if combined_bytes is None:
            logger.warning(f"Key {key_str} does not exist in the cache")
            return None

        try:
            metadata, kv_bytes = self._unpack_metadata_and_data(combined_bytes)
        except (ValueError, struct.error) as e:
            logger.error(f"Failed to unpack data for key {key_str}: {e}")
            return None

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
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

    def support_batched_get(self) -> bool:
        return True

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Batched get operation for Redis Sentinel."""
        async with self.slave.pipeline(transaction=False) as pipeline:
            key_strings = [key.to_string() for key in keys]

            for key_str in key_strings:
                pipeline.get(key_str)

            combined_bytes_list = await pipeline.execute()
            final_memory_objs = []

            for i, combined_bytes in enumerate(combined_bytes_list):
                if combined_bytes is None:
                    logger.warning(f"Key {keys[i]} does not exist in the cache")
                    final_memory_objs.append(None)
                    continue

                try:
                    metadata, kv_bytes = self._unpack_metadata_and_data(combined_bytes)
                except (ValueError, struct.error) as e:
                    logger.error(f"Failed to unpack data for key {keys[i]}: {e}")
                    final_memory_objs.append(None)
                    continue

                memory_obj = self.local_cpu_backend.allocate(
                    metadata.shape,
                    metadata.dtype,
                    metadata.fmt,
                )
                if memory_obj is None:
                    logger.warning("Failed to allocate memory during remote receive")
                    final_memory_objs.append(None)
                    continue

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

                final_memory_objs.append(memory_obj)

            return final_memory_objs

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        combined_bytes = self._pack_metadata_and_data(memory_obj)
        await self.master.set(key_str, combined_bytes)

        memory_obj.ref_count_down()

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        """Batched put operation for Redis Sentinel."""
        async with self.master.pipeline(transaction=False) as pipeline:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                key_str = key.to_string()
                combined_bytes = self._pack_metadata_and_data(memory_obj)
                pipeline.set(key_str, combined_bytes)
            await pipeline.execute()

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.master.aclose()
        await self.slave.aclose()
        logger.info("Closed Redis Sentinel clients")
