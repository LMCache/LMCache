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
from typing import List, Optional, AsyncGenerator
import abc

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class RemoteConnector(metaclass=abc.ABCMeta):
    """
    Interface for remote connector
    """

    @abc.abstractmethod
    async def exists(self, key: CacheEngineKey) -> bool:
        """
        Check if the remote server contains the key

        Input:
            key: a string

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Get the memory_obj of the corresponding key

        Input:
            key: the key of the corresponding object

        Returns:
            The memory_obj of the corresponding key
            Return None if the key does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Send the memory_obj with the corresponding key directly
        to the remote server. Will decrease the ref count after
        send finishes.

        Input:
            key: the CacheEngine key
            memory_obj: the memory_obj of the corresponding key
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def list(self) -> List[str]:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        """
        Close remote server

        """
        raise NotImplementedError

    # New layerwise methods with default implementations for backwards compatibility
    async def layerwise_exists(self, keys: List[List[CacheEngineKey]]) -> List[List[bool]]:
        """
        Check existence of keys in layerwise format.
        
        Args:
            keys: List[List[CacheEngineKey]] - [layer][chunk] format
            
        Returns:
            List[List[bool]] - existence status in same format
            
        Default implementation: calls exists() for each key individually
        """
        results = []
        for layer_keys in keys:
            layer_results = []
            for key in layer_keys:
                exists = await self.exists(key)
                layer_results.append(exists)
            results.append(layer_results)
        return results
    
    async def layerwise_get(self, keys: List[List[CacheEngineKey]]) -> AsyncGenerator[List[Optional[MemoryObj]], None]:
        """
        Generator-based layerwise retrieval for streaming.
        
        Args:
            keys: List[List[CacheEngineKey]] - [layer][chunk] format
            
        Yields:
            List[Optional[MemoryObj]] - memory objects for each layer
            
        Default implementation: processes layer by layer using regular get()
        """
        for layer_keys in keys:
            layer_objs = []
            for key in layer_keys:
                obj = await self.get(key)
                layer_objs.append(obj)
            yield layer_objs
    
    async def layerwise_put(self, keys: List[List[CacheEngineKey]], 
                           memory_objs: List[List[MemoryObj]]) -> AsyncGenerator[None, None]:
        """
        Generator-based layerwise storage for streaming.
        
        Args:
            keys: List[List[CacheEngineKey]] - [layer][chunk] format
            memory_objs: List[List[MemoryObj]] - corresponding memory objects
            
        Yields:
            None - yields after each layer completion
            
        Default implementation: processes layer by layer using regular put()
        """
        for layer_keys, layer_objs in zip(keys, memory_objs):
            for key, obj in zip(layer_keys, layer_objs):
                await self.put(key, obj)
            yield
            
    def supports_layerwise(self) -> bool:
        """
        Check if this connector supports optimized layerwise operations.
        
        Returns:
            bool - True if layerwise methods are optimized, False for default fallback
        """
        return False  # Default implementation uses fallback
