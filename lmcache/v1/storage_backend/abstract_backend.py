# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import List, Optional
import abc

# Third Party
import torch

# First Party
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

    # NOTE (Jiayi): Using batched interface allows the underlying implementation
    # have more flexibility to do optimizations.
    @abc.abstractmethod
    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec=None,
    ) -> Optional[List[Future]]:
        """
        An async function to put the MemoryObj into the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :param List[MemoryObj] objs: The MemoryObjs to be stored.

        :return: a list of future objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> bool:
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
        A blocking function to get the kv cache from the storage backend.

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

    # NOTE(Jiayi): Please re-implement this method if the storage backend
    # can benefit from batched get.
    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.

        :return: a list of memory objects.
        """
        mem_objs = []
        for key in keys:
            mem_objs.append(self.get_blocking(key))
        return mem_objs

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
    def remove(self, key: CacheEngineKey, free_obj: bool = True) -> bool:
        """
        remove a memory object.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param bool free_obj: Whether to free the MemoryObj after removing it.

        :return: a bool indicates whether remove is successful.
        """
        raise NotImplementedError

    # TODO(Jiayi): Optimize batched remove
    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        free_obj: bool = True,
    ) -> int:
        """
        Remove a list of memory objects.

        :param list[CacheEngineKey] keys: The keys of the MemoryObjs.
        :param bool free_obj: Whether to free the MemoryObjs after removing them.

        :return: a int indicates the number of removed memory objects.
        """
        num_removed = 0
        for key in keys:
            num_removed += self.remove(key, free_obj=free_obj)
        return num_removed

    @abc.abstractmethod
    def close(
        self,
    ) -> None:
        """
        Close the storage backend.
        """
        raise NotImplementedError
