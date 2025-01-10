import abc
from concurrent.futures import Future
from typing import Optional

import torch

from lmcache.experimental.memory_management import BufferMemoryObj, MemoryObj
from lmcache.utils import CacheEngineKey


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
    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the storage backend. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """
        Insert the key after data is put to storage backend.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Future:
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
    ) -> Optional[BufferMemoryObj]:
        """
        A blcocking function to get the kv cache from the storage backend.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :return: BufferMemoryObj. None if the key does not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self, ) -> None:
        """
        Close the storage backend.
        """
        raise NotImplementedError
