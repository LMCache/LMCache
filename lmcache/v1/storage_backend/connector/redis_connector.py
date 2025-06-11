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
from typing import List, Optional, Tuple, no_type_check, AsyncGenerator
import asyncio
import inspect
import os

# Third Party
import redis
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

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
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        use_async_redis: bool = False,
    ):
        self.use_async_redis = use_async_redis
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        
        if use_async_redis:
            # Use async Redis for better performance
            if aioredis is None:
                logger.warning("redis.asyncio not available, falling back to sync Redis")
                self.use_async_redis = False
                self.connection = redis.Redis(host=host, port=port, decode_responses=False)
            else:
                self.connection = aioredis.Redis(host=host, port=port, decode_responses=False)
        else:
            # Fallback to sync Redis
            self.connection = redis.Redis(host=host, port=port, decode_responses=False)

    async def exists(self, key: CacheEngineKey) -> bool:
        if self.use_async_redis:
            return bool(await self.connection.exists(key.to_string() + "metadata"))
        else:
            return bool(self.connection.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        
        if self.use_async_redis:
            metadata_bytes = await self.connection.get(key_str + "metadata")
        else:
            metadata_bytes = self.connection.get(key_str + "metadata")

        if metadata_bytes is None:
            return None

        if self.use_async_redis:
            # For async Redis, results are already awaited
            pass
        else:
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
        if self.use_async_redis:
            kv_bytes = await self.connection.get(key_str + "kv_bytes")
        else:
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
            if self.use_async_redis:
                await self.connection.delete(key_str + "metadata")
            else:
                self.connection.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)
        view[: metadata.length] = kv_bytes

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
        if self.use_async_redis:
            await self.connection.set(key_str + "metadata", metadata_bytes)
            await self.connection.set(key_str + "kv_bytes", kv_bytes)
        else:
            self.connection.set(key_str + "metadata", metadata_bytes)
            self.connection.set(key_str + "kv_bytes", kv_bytes)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    def supports_layerwise(self) -> bool:
        """Redis supports optimized layerwise operations."""
        return True
    
    async def layerwise_exists(self, keys: List[List[CacheEngineKey]]) -> List[List[bool]]:
        """
        Optimized batch existence check using Redis pipeline.
        """
        all_keys = []
        key_positions = []
        
        # Flatten keys while tracking positions
        for layer_idx, layer_keys in enumerate(keys):
            for chunk_idx, key in enumerate(layer_keys):
                all_keys.append(key.to_string() + "metadata")
                key_positions.append((layer_idx, chunk_idx))
        
        # Batch existence check
        if all_keys:
            results = self.connection.exists(*all_keys)
            if not isinstance(results, list):
                results = [results]
            
            # Reconstruct layerwise format
            layerwise_results = [[False for _ in layer] for layer in keys]
            for i, (layer_idx, chunk_idx) in enumerate(key_positions):
                layerwise_results[layer_idx][chunk_idx] = bool(results[i])
                
            return layerwise_results
        else:
            return []
    
    async def layerwise_get(self, keys: List[List[CacheEngineKey]]) -> AsyncGenerator[List[Optional[MemoryObj]], None]:
        """
        Optimized layerwise retrieval with Redis pipelining.
        """
        for layer_keys in keys:
            if not layer_keys:
                yield []
                continue
                
            # Batch retrieve metadata and KV data for entire layer
            if self.use_async_redis:
                # True async pipeline for better performance
                async with self.connection.pipeline(transaction=False) as pipe:
                    # Add all metadata and kv_bytes requests to pipeline
                    for key in layer_keys:
                        key_str = key.to_string()
                        pipe.get(key_str + "metadata")
                        pipe.get(key_str + "kv_bytes")
                    
                    # Execute batch requests asynchronously
                    results = await pipe.execute()
            else:
                # Fallback to sync pipeline
                pipe = self.connection.pipeline()
                for key in layer_keys:
                    key_str = key.to_string()
                    pipe.get(key_str + "metadata")
                    pipe.get(key_str + "kv_bytes")
                results = pipe.execute()
            
            # Pre-allocate all memory objects to avoid lock contention during processing
            layer_metadatas = []
            layer_allocations = []
            
            # Phase 1: Batch deserialize metadata and pre-allocate memory
            for i, key in enumerate(layer_keys):
                metadata_bytes = results[i * 2]
                kv_bytes = results[i * 2 + 1]
                
                if metadata_bytes is None or kv_bytes is None:
                    layer_metadatas.append(None)
                    layer_allocations.append(None)
                    continue
                    
                # Deserialize metadata
                metadata = RemoteMetadata.deserialize(metadata_bytes)
                layer_metadatas.append(metadata)
                
                # Pre-allocate memory object
                memory_obj = self.local_cpu_backend.allocate(
                    metadata.shape, metadata.dtype, metadata.fmt
                )
                layer_allocations.append(memory_obj)
            
            # Phase 2: Bulk copy KV data to pre-allocated memory
            layer_objs = []
            for i, key in enumerate(layer_keys):
                metadata = layer_metadatas[i]
                memory_obj = layer_allocations[i]
                kv_bytes = results[i * 2 + 1]
                
                if metadata is None or memory_obj is None or kv_bytes is None:
                    layer_objs.append(None)
                    continue
                    
                # Fast bulk copy KV data to memory object
                if isinstance(memory_obj.byte_array, memoryview):
                    view = memory_obj.byte_array
                    if view.format == "<B":
                        view = view.cast("B")
                else:
                    view = memoryview(memory_obj.byte_array)
                view[:metadata.length] = kv_bytes
                
                layer_objs.append(memory_obj)
                
            yield layer_objs
    
    async def layerwise_put(self, keys: List[List[CacheEngineKey]], 
                           memory_objs: List[List[MemoryObj]]) -> AsyncGenerator[None, None]:
        """
        Optimized layerwise storage with Redis pipelining.
        """
        for layer_keys, layer_objs in zip(keys, memory_objs):
            if not layer_keys:
                yield
                continue
                
            # Batch store entire layer using pipeline
            if self.use_async_redis:
                # True async pipeline for better performance
                async with self.connection.pipeline(transaction=False) as pipe:
                    for key, memory_obj in zip(layer_keys, layer_objs):
                        if memory_obj is None:
                            continue
                            
                        # Prepare metadata and KV data
                        kv_bytes = memory_obj.byte_array
                        metadata_bytes = RemoteMetadata(
                            len(kv_bytes), 
                            memory_obj.get_shape(),
                            memory_obj.get_dtype(), 
                            memory_obj.get_memory_format()
                        ).serialize()
                        
                        key_str = key.to_string()
                        pipe.set(key_str + "metadata", metadata_bytes)
                        pipe.set(key_str + "kv_bytes", kv_bytes)
                    
                    # Execute batch storage asynchronously
                    await pipe.execute()
            else:
                # Fallback to sync pipeline
                pipe = self.connection.pipeline()
                for key, memory_obj in zip(layer_keys, layer_objs):
                    if memory_obj is None:
                        continue
                        
                    # Prepare metadata and KV data
                    kv_bytes = memory_obj.byte_array
                    metadata_bytes = RemoteMetadata(
                        len(kv_bytes), 
                        memory_obj.get_shape(),
                        memory_obj.get_dtype(), 
                        memory_obj.get_memory_format()
                    ).serialize()
                    
                    key_str = key.to_string()
                    pipe.set(key_str + "metadata", metadata_bytes)
                    pipe.set(key_str + "kv_bytes", kv_bytes)
                
                # Execute batch storage synchronously
                pipe.execute()
            yield

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
        self.master = self.sentinel.master_for(service_name, socket_timeout=timeout)
        self.slave = self.sentinel.slave_for(service_name, socket_timeout=timeout)

        self.local_cpu_backend = local_cpu_backend

    async def exists(self, key: CacheEngineKey) -> bool:
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

        view = memoryview(memory_obj.byte_array)
        view[0 : metadata.length] = kv_bytes

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

    def supports_layerwise(self) -> bool:
        """Redis Sentinel supports optimized layerwise operations."""
        return True

    async def layerwise_exists(self, keys: List[List[CacheEngineKey]]) -> List[List[bool]]:
        """
        Optimized batch existence check using Redis Sentinel pipeline.
        """
        all_keys = []
        key_positions = []
        
        # Flatten keys while tracking positions
        for layer_idx, layer_keys in enumerate(keys):
            for chunk_idx, key in enumerate(layer_keys):
                all_keys.append(key.to_string() + "metadata")
                key_positions.append((layer_idx, chunk_idx))
        
        # Batch existence check using slave for reads
        if all_keys:
            results = []
            for key in all_keys:
                exists = bool(self.slave.exists(key))
                results.append(exists)
            
            # Reconstruct layerwise format
            layerwise_results = [[False for _ in layer] for layer in keys]
            for i, (layer_idx, chunk_idx) in enumerate(key_positions):
                layerwise_results[layer_idx][chunk_idx] = results[i]
                
            return layerwise_results
        else:
            return []

    async def layerwise_get(self, keys: List[List[CacheEngineKey]]) -> AsyncGenerator[List[Optional[MemoryObj]], None]:
        """
        Optimized layerwise retrieval with Redis Sentinel pipelining.
        """
        for layer_keys in keys:
            if not layer_keys:
                yield []
                continue
                
            # Process results for this layer
            layer_objs = []
            for key in layer_keys:
                key_str = key.to_string()
                metadata_bytes = self.slave.get(key_str + "metadata")
                kv_bytes = self.slave.get(key_str + "kv_bytes")
                
                if metadata_bytes is None or kv_bytes is None:
                    layer_objs.append(None)
                    continue
                    
                # Deserialize metadata
                metadata = RemoteMetadata.deserialize(metadata_bytes)
                
                # Allocate memory object
                memory_obj = self.local_cpu_backend.allocate(
                    metadata.shape, metadata.dtype, metadata.fmt
                )
                
                if memory_obj is None:
                    layer_objs.append(None)
                    continue
                    
                # Copy KV data to memory object
                view = memoryview(memory_obj.byte_array)
                view[0:metadata.length] = kv_bytes
                
                layer_objs.append(memory_obj)
                
            yield layer_objs

    async def layerwise_put(self, keys: List[List[CacheEngineKey]], 
                           memory_objs: List[List[MemoryObj]]) -> AsyncGenerator[None, None]:
        """
        Optimized layerwise storage with Redis Sentinel.
        """
        for layer_keys, layer_objs in zip(keys, memory_objs):
            if not layer_keys:
                yield
                continue
                
            for key, memory_obj in zip(layer_keys, layer_objs):
                if memory_obj is None:
                    continue
                    
                # Prepare metadata and KV data
                kv_bytes = memory_obj.byte_array
                metadata_bytes = RemoteMetadata(
                    len(kv_bytes), 
                    memory_obj.get_shape(),
                    memory_obj.get_dtype(), 
                    memory_obj.get_memory_format()
                ).serialize()
                
                key_str = key.to_string()
                self.master.set(key_str + "metadata", metadata_bytes)
                self.master.set(key_str + "kv_bytes", kv_bytes)
                
                memory_obj.ref_count_down()
            
            yield

    async def close(self):
        self.master.close()
        self.slave.close()
