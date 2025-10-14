# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union
import abc
import asyncio

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend import LocalCPUBackend


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
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> Union[List[Future], None]:
        """
        An async function to put the MemoryObj into the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :param List[MemoryObj] objs: The MemoryObjs to be stored.

        :return:  Union[List[Future], None]: A list of `Future` objects if the
        storage persistence operation is asynchronous and is successful.
        `None` if the operation is synchronous, or the asynchronous fails
        or is skipped.

        :note: This function will have the side effect that modifies the
            underlying key-value mappings in the storage backend. The side
            effect may change the result of lookup and get.
        """
        raise NotImplementedError

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """
        An async version of batched_submit_put_task.
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

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        A non-blocking function to get the kv cache from the storage backend.
        """
        raise NotImplementedError

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check whether keys are in the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.

        :param bool pin: Whether to pin the keys.
            If True, the corresponding KV caches will be
            pinned in the storage backend.

        :return: The number of keys that exist in the storage backend.
        """
        raise NotImplementedError

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        A non-blcocking function to get the kv cache from the storage backend.

        :param list[CacheEngineKey] keys: The keys of the list of MemoryObjs.

        :return: a list of Memoryobjs.
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
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        remove a memory object.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param bool force: Whether to it is a forced remove from the external.

        :return: a bool indicates whether remove is successful.
        """
        raise NotImplementedError

    # TODO(Jiayi): Optimize batched remove
    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """
        Remove a list of memory objects.

        :param list[CacheEngineKey] keys: The keys of the MemoryObjs.
        :param bool force: Whether to force remove the memory objects.

        :return: a int indicates the number of removed memory objects.
        """
        num_removed = 0
        for key in keys:
            num_removed += self.remove(key, force=force)
        return num_removed

    @abc.abstractmethod
    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        """
        Get the allocator backend that is used by the current storage backend
        to allocate memory objects during `get` operations.

        :return: an instance of AllocateBackendInterface
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(
        self,
    ) -> None:
        """
        Close the storage backend.
        """
        raise NotImplementedError


class AllocatorBackendInterface(StorageBackendInterface):
    """
    AllocatorBackendInterface extends the StorageBackendInterface with
    the ability to actively allocate the memory objects.
    """

    @abc.abstractmethod
    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> MemoryAllocatorInterface:
        """
        Create the correct memory allocator for the current storage backend

        Args:
            config: The cache engine config
            metadata: the cache engine metadata

        Returns:
            The memory allocator for this storage backend
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        """
        Returns:
            The underlying memory allocator
        """
        raise NotImplementedError

    @abc.abstractmethod
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocates memory in the backend to hold a tensor of the given shape.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.
        :param bool eviction: whether to enable eviction when allocating.
        :param bool busy_loop: whether to enable a busy loop to wait
            for in-progress store operations to finish and release the
            memory space for retrieve.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """
        Allocates memory in the backend to hold a tensor of the given shape
        in a batched manner. The allocated memory objects will have the same
        shape, dtype, and format.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
        :param int batch_size: The number of memory objects to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.
        :param bool eviction: whether to enable eviction when allocating.
        :param bool busy_loop: whether to enable a busy loop to wait
            for in-progress store operations to finish and release the
            memory space for retrieve.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    def calculate_chunk_budget(self) -> int:
        """
        Calculate the chunk budget for the allocator backend.
        """
        raise NotImplementedError


class ConfigurableStorageBackendInterface(StorageBackendInterface):
    """The Configurable Storage Backend Interface needs to be implemented
    when you want to add a storage backend in a configurable or plug and play
    fashion."""

    def __init__(
        self,
        dst_device: str = "cuda",
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initialize a configurable storage backend. This constructor will be called
        when loading the configurable storage backends from the configuration file.

        :param str dst_device: The target device for tensor operations
            (e.g., "cuda" or "cpu").
        :param LMCacheEngineConfig config: Optional configuration object for the
            cache engine.
        :param LMCacheEngineMetadata metadata: Optional metadata describing the cache
            engine state or version.
        :param LocalCPUBackend local_cpu_backend: Optional backend for local CPU-based
            inference or caching.
        :param asyncio.AbstractEventLoop loop: Optional asyncio event loop for
            asynchronous operations.
        """
        super().__init__(dst_device=dst_device)
        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop
