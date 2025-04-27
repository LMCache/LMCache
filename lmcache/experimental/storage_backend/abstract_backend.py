import abc
from concurrent.futures import Future
from typing import Optional

import torch

from lmcache.experimental.memory_management import MemoryObj
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
        
        :param key: The key to check (CacheEngineKey).
        :return: True if the key exists, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing put tasks. 
        
        :param key: The key to check (CacheEngineKey).
        :return: True if a put task for this key is currently active, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        """
        An async function to put the MemoryObj into the storage backend.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param MemoryObj obj: The MemoryObj to be stored. The underlying tensor
                              is expected to have shape [2, num_layers, ...].
        
        :return: A future object representing the asynchronous put operation, 
                 or None if the operation could not be submitted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        An async function to prefetch the MemoryObj from the storage backend.

        The future, when completed, will yield the MemoryObj containing the KV cache
        tensor with shape [2, num_layers, ...], or None if the key was not found.

        :param CacheEngineKey key: The key of the MemoryObj to prefetch.

        :return: A future object representing the asynchronous prefetch operation. 
                 Returns None if the key does not exist or prefetching cannot be initiated.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        A blocking function to get the MemoryObj from the storage backend.

        The returned MemoryObj contains the KV cache tensor with shape 
        [2, num_layers, ...].
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :return: MemoryObj containing the KV cache tensor if the key exists, 
                 otherwise None. The tensor will be on the device specified by `self.dst_device`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self, ) -> None:
        """
        Close the storage backend and release resources.
        """
        raise NotImplementedError
