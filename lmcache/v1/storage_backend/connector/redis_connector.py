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
from typing import List, Optional, Tuple, no_type_check
import asyncio
import inspect
import os

# Third Party
import redis
import torch

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
    ):
        self.connection = redis.Redis(host=host, port=port, decode_responses=False)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.connection.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        metadata_bytes = self.connection.get(key_str + "metadata")

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))
        
        # Reconstruct original shape by removing trailing zeros (reverse of padding)
        original_shape = []
        for dim in metadata.shape:
            if dim > 0:
                original_shape.append(dim)
            else:
                break  # Stop at first zero (trailing zeros were padding)
        original_shape = torch.Size(original_shape)
        logger.debug(f"Shape reconstructed from {metadata.shape} to {original_shape}")

        memory_obj = self.local_cpu_backend.allocate(
            original_shape,  # Use original shape, not padded shape
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
        view[: metadata.length] = kv_bytes

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        # Pad shape to 4 dimensions as required by RemoteMetadata protocol
        # "Pass in shape [x, 0, 0, 0] if it is a bytes memory object"
        padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
        padded_shape = torch.Size(padded_shape[:4])  # Ensure exactly 4 dimensions
        
        metadata_bytes = RemoteMetadata(
            len(kv_bytes), padded_shape, kv_dtype, memory_format
        ).serialize()

        key_str = key.to_string()
        self.connection.set(key_str + "metadata", metadata_bytes)
        self.connection.set(key_str + "kv_bytes", kv_bytes)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def batched_get(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        """
        Optimized batch get using Redis pipeline.
        
        Reduces network round trips from 2*N (metadata + kv_bytes for each key) 
        to 1 pipeline execution.
        """
        if not keys:
            return []
        
        # Use Redis pipeline for batch operations
        pipe = self.connection.pipeline()
        
        # Add all metadata and kv_bytes gets to pipeline
        key_strs = [key.to_string() for key in keys]
        for key_str in key_strs:
            pipe.get(key_str + "metadata")
            pipe.get(key_str + "kv_bytes")
        
        # Execute pipeline - gets all results in one network round trip
        try:
            results = pipe.execute()
        except Exception as e:
            logger.warning(f"Redis pipeline execution failed: {e}")
            return [None] * len(keys)
        
        # Process results - results are [meta1, kv1, meta2, kv2, ...]
        memory_objs = []
        for i, key in enumerate(keys):
            metadata_bytes = results[i * 2]
            kv_bytes = results[i * 2 + 1]
            
            if metadata_bytes is None or kv_bytes is None:
                if metadata_bytes is not None and kv_bytes is None:
                    # Cleanup orphaned metadata
                    logger.warning(
                        "Key exists but KV cache does not exist. "
                        "Might happen when the cache is evicted by redis."
                    )
                    self.connection.delete(key_strs[i] + "metadata")
                memory_objs.append(None)
                continue
            
            try:
                # Deserialize metadata
                metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))
                
                # Reconstruct original shape by removing trailing zeros (reverse of padding)
                original_shape = []
                for dim in metadata.shape:
                    if dim > 0:
                        original_shape.append(dim)
                    else:
                        break  # Stop at first zero (trailing zeros were padding)
                original_shape = torch.Size(original_shape)
                
                # Allocate memory object
                memory_obj = self.local_cpu_backend.allocate(
                    original_shape,  # Use original shape, not padded shape
                    metadata.dtype, 
                    metadata.fmt,
                )
                if memory_obj is None:
                    logger.warning("Failed to allocate memory during batch remote receive")
                    memory_objs.append(None)
                    continue
                
                # Copy data into memory object
                if isinstance(memory_obj.byte_array, memoryview):
                    view = memory_obj.byte_array
                    if view.format == "<B":
                        view = view.cast("B")
                else:
                    view = memoryview(memory_obj.byte_array)
                view[:metadata.length] = kv_bytes
                
                memory_objs.append(memory_obj)
                
            except Exception as e:
                logger.warning(f"Failed to process batched result for key {keys[i]}: {e}")
                memory_objs.append(None)
        
        return memory_objs

    async def batched_put(self, keys_and_objs: List[Tuple[CacheEngineKey, MemoryObj]]):
        """
        Optimized batch put using Redis pipeline.
        
        Reduces network round trips from 2*N (metadata + kv_bytes for each key)
        to 1 pipeline execution.
        """
        if not keys_and_objs:
            return
        
        # Use Redis pipeline for batch operations
        pipe = self.connection.pipeline()
        
        # Add all sets to pipeline
        for key, memory_obj in keys_and_objs:
            # Prepare data (same as individual put)
            kv_bytes = memory_obj.byte_array
            kv_shape = memory_obj.get_shape()
            kv_dtype = memory_obj.get_dtype()
            memory_format = memory_obj.get_memory_format()
            
            # Pad shape to 4 dimensions as required by RemoteMetadata protocol
            padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
            padded_shape = torch.Size(padded_shape[:4])  # Ensure exactly 4 dimensions
            
            metadata_bytes = RemoteMetadata(
                len(kv_bytes), padded_shape, kv_dtype, memory_format
            ).serialize()
            
            key_str = key.to_string()
            
            # Add to pipeline (instead of executing immediately)
            pipe.set(key_str + "metadata", metadata_bytes)
            pipe.set(key_str + "kv_bytes", kv_bytes)
        
        # Execute all operations in one network round trip
        try:
            pipe.execute()
        except Exception as e:
            logger.warning(f"Redis pipeline execution failed in batched_put: {e}")

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
        
        # Reconstruct original shape by removing trailing zeros (reverse of padding)
        original_shape = []
        for dim in metadata.shape:
            if dim > 0:
                original_shape.append(dim)
            else:
                break  # Stop at first zero (trailing zeros were padding)
        original_shape = torch.Size(original_shape)

        memory_obj = self.local_cpu_backend.allocate(
            original_shape,  # Use original shape, not padded shape
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

        # Pad shape to 4 dimensions as required by RemoteMetadata protocol
        padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
        padded_shape = torch.Size(padded_shape[:4])  # Ensure exactly 4 dimensions

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), padded_shape, kv_dtype, memory_format
        ).serialize()

        key_str = key.to_string()
        self.master.set(key_str + "metadata", metadata_bytes)
        self.master.set(key_str + "kv_bytes", kv_bytes)

        memory_obj.ref_count_down()

    async def batched_get(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        """
        Optimized batch get using Redis Sentinel pipeline on slave.
        
        Uses slave for read operations to distribute load.
        """
        if not keys:
            return []
        
        # Use slave pipeline for batch read operations
        pipe = self.slave.pipeline()
        
        # Add all metadata and kv_bytes gets to pipeline
        key_strs = [key.to_string() for key in keys]
        for key_str in key_strs:
            pipe.get(key_str + "metadata")
            pipe.get(key_str + "kv_bytes")
        
        # Execute pipeline
        try:
            results = pipe.execute()
        except Exception as e:
            logger.warning(f"Redis Sentinel pipeline execution failed: {e}")
            return [None] * len(keys)
        
        # Process results (same logic as RedisConnector)
        memory_objs = []
        for i, key in enumerate(keys):
            metadata_bytes = results[i * 2]
            kv_bytes = results[i * 2 + 1]
            
            if metadata_bytes is None or kv_bytes is None:
                if metadata_bytes is not None and kv_bytes is None:
                    # Cleanup orphaned metadata (use master for write)
                    logger.warning(
                        "Key exists but KV cache does not exist. "
                        "Might happen when the cache is evicted by redis."
                    )
                    self.master.delete(key_strs[i] + "metadata")
                memory_objs.append(None)
                continue
            
            try:
                # Deserialize metadata
                metadata = RemoteMetadata.deserialize(metadata_bytes)
                
                # Reconstruct original shape by removing trailing zeros (reverse of padding)
                original_shape = []
                for dim in metadata.shape:
                    if dim > 0:
                        original_shape.append(dim)
                    else:
                        break  # Stop at first zero (trailing zeros were padding)
                original_shape = torch.Size(original_shape)
                
                # Allocate memory object
                memory_obj = self.local_cpu_backend.allocate(
                    original_shape,  # Use original shape, not padded shape
                    metadata.dtype,
                    metadata.fmt,
                )
                if memory_obj is None:
                    logger.warning("Failed to allocate memory during batch remote receive")
                    memory_objs.append(None)
                    continue
                
                # Copy data into memory object
                view = memoryview(memory_obj.byte_array)
                view[0:metadata.length] = kv_bytes
                
                memory_objs.append(memory_obj)
                
            except Exception as e:
                logger.warning(f"Failed to process batched result for key {keys[i]}: {e}")
                memory_objs.append(None)
        
        return memory_objs

    async def batched_put(self, keys_and_objs: List[Tuple[CacheEngineKey, MemoryObj]]):
        """
        Optimized batch put using Redis Sentinel pipeline on master.
        
        Uses master for write operations to ensure consistency.
        """
        if not keys_and_objs:
            return
        
        # Use master pipeline for batch write operations
        pipe = self.master.pipeline()
        
        # Add all sets to pipeline
        for key, memory_obj in keys_and_objs:
            # Prepare data (same as individual put)
            kv_bytes = memory_obj.byte_array
            kv_shape = memory_obj.get_shape()
            kv_dtype = memory_obj.get_dtype()
            memory_format = memory_obj.get_memory_format()
            
            # Pad shape to 4 dimensions as required by RemoteMetadata protocol
            padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
            padded_shape = torch.Size(padded_shape[:4])  # Ensure exactly 4 dimensions
            
            metadata_bytes = RemoteMetadata(
                len(kv_bytes), padded_shape, kv_dtype, memory_format
            ).serialize()
            
            key_str = key.to_string()
            
            # Add to pipeline
            pipe.set(key_str + "metadata", metadata_bytes)
            pipe.set(key_str + "kv_bytes", kv_bytes)
            
            # Handle ref count (same as individual put)
            memory_obj.ref_count_down()
        
        # Execute all operations in one network round trip
        try:
            pipe.execute()
        except Exception as e:
            logger.warning(f"Redis Sentinel pipeline execution failed in batched_put: {e}")

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.master.close()
        self.slave.close()
