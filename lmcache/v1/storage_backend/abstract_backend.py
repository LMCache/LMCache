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

import abc
from concurrent.futures import Future
from typing import Optional

import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj


class StorageBackendInterface(metaclass=abc.ABCMeta):

    def __init__(
        self,
        dst_device: str = "cuda",
    ):
        """
        Initialize the storage backend. 

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.

        :raise: RuntimeError if the device is not valid
        """
        try:
            torch.device(dst_device)
        except RuntimeError:
            raise

        self.dst_device = dst_device

    @abc.abstractmethod
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend. 
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :param bool pin: Whether to pin the key. 
            If True, the corresponding KV cache will be
            pinned in the storage backend.
        
        :return: True if the key exists, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing put tasks. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        """
        An async function to put the MemoryObj into the storage backend.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param MemoryObj obj: The MemoryObj to be stored.
        
        :return: a future object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        An async function to get the MemoryObj from the storage backend.

        :param CacheEngineKey key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        A blcocking function to get the kv cache from the storage backend.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :return: MemoryObj. None if the key does not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        A non-blcocking function to get the kv cache from the storage backend.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Pin a memory object so it will not be evicted.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :return: a bool indicates whether pin is successful.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Unpin a memory object so it can be evicted.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :return: a bool indicates whether unpin is successful.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self, ) -> None:
        """
        Close the storage backend.
        """
        raise NotImplementedError
