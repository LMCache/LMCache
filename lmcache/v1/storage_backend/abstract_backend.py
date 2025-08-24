# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import List, Optional, Sequence
import abc
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)


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
        self.closed = False

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

    """ nixl_backend_v3.py uses `batched_submit_put_task()`
    @abc.abstractmethod
    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        transfer_spec=None,
    ) -> Optional[Future]:
        ###
        An async function to put a single MemoryObj into the storage backend.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param MemoryObj memory_obj: The MemoryObj to be stored.
        :param transfer_spec: Transfer specification for the operation.

        :return: a future object
        ###
        raise NotImplementedError
    """

    # NOTE (Jiayi): Using batched interface allows the underlying implementation
    # have more flexibility to do optimizations.
    @abc.abstractmethod
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        """
        An async function to put the MemoryObj into the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :param List[MemoryObj] objs: The MemoryObjs to be stored.

        :return: Nothing

        :note: This function will have the side effect that modifies the
            underlying key-value mappings in the storage backend. The side
            effect may change the result of lookup and get.
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
    def close(
        self,
    ) -> None:
        """
        Close the storage backend.
        """
        raise NotImplementedError


class AllocatorBackendInterface(StorageBackendInterface):
    """
    return self.allocator_backend.allocate(
        shape, dtype, fmt, eviction=eviction, busy_loop=busy_loop
    )
    """

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
    ) -> Optional[MemoryObj]:
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

    def _log_operation_complete(
        self,
        operation: str,
        key: CacheEngineKey,
        start_time: float,
        result=None,
    ) -> None:
        size_bytes = None
        throughput_info = ""
        elapsed_time = time.perf_counter() - start_time
        if result is not None:
            if hasattr(result, "get_size"):
                size_bytes = result.get_size()
            elif isinstance(result, dict) and "size_bytes" in result:
                size_bytes = result["size_bytes"]
        if size_bytes is not None and elapsed_time > 0:
            throughput_size = size_bytes / (1024**3)
            throughput_gbps = throughput_size / elapsed_time
            throughput_info = (
                f",(size={throughput_size:.2f} GB),"
                f"throughput={throughput_gbps:.2f} GB/s"
            )

        logger.debug(
            f"{operation} completed for key {key.to_string()} "
            f"in {elapsed_time * 1000:.2f}ms"
            f"{throughput_info}"
        )

    def _get_memory_obj_size(self, memory_obj: MemoryObj) -> int:
        return memory_obj.get_size()
