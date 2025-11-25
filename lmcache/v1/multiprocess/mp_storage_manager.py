# SPDX-License-Identifier: Apache-2.0
# NOTE: this file will be moved and the class implementation
# will be largely refactored in the future.
# Standard
from itertools import compress
from typing import Union
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MixedMemoryAllocator
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

logger = init_logger(__name__)

ReserveHandle = int
ReserveResult = tuple[ReserveHandle, dict[IPCCacheEngineKey, MemoryObj]]


class MemoryExhaustedError(Exception):
    """Raised when the memory allocation cannot be completed due to
    insufficient memory.
    """

    pass


class MPStorageManager:
    def __init__(self, cpu_buffer_size: float):
        """
        Args:
            cpu_buffer_size: the total size (in GB) of CPU memory buffer
                to be used for storage
        """
        # Allocator for CPU memory (note: this will be moved to storage backend
        # implementation in the future)
        size_in_bytes = int(cpu_buffer_size * (1 << 30))  # Convert GB to bytes
        self._memory_allocator = MixedMemoryAllocator(size_in_bytes)
        self._allocator_lock = threading.Lock()

        # Reserved memory objects
        self._reserved_memory_object_pools: dict[
            ReserveHandle, dict[IPCCacheEngineKey, MemoryObj]
        ] = {}
        self._reserved_keys: set[IPCCacheEngineKey] = set()
        self._reserve_handle = 0
        self._reserve_handle_lock = threading.Lock()

        # Committed memory objects
        self._commited_memory_objects: dict[IPCCacheEngineKey, MemoryObj] = {}

        # The lock for reserved buffer and committed buffer
        self._buffer_lock = threading.Lock()

        # self.lock for debug
        self.lock = threading.Lock()

    def _allocate_new_reserve_handle(self) -> ReserveHandle:
        """Allocate a new reserve handle in a thread-safe manner."""
        with self._reserve_handle_lock:
            handle = self._reserve_handle
            self._reserve_handle += 1
        return handle

    def _has_key(self, key: IPCCacheEngineKey) -> bool:
        """Check whether the given key already exists in the storage manager.
        Both reserved and committed keys will be considered.

        Not thread-safe, should be protected by the buffer lock
        """
        if key in self._reserved_keys:
            return True
        if key in self._commited_memory_objects:
            return True
        return False

    def reserve(
        self,
        keys: list[IPCCacheEngineKey],
        shape: Union[torch.Size, tuple[int, ...]],
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> ReserveResult:
        """Allocate the memory objects to store the given keys in the storage
        manager. If some keys already exist in the storage manager (no matter
        whether it is reserved or committed), the allocation will be skipped.
        It will return a list of boolean to identify which key is "skipped".

        Args:
            keys: the list of keys corresponding to the storage

        Returns:
            ReserveHandle: a special handle to represent this reservation.
                Will be used in "commit".
            dict[IPCCacheEngineKey, MemoryObj]: a dictionary mapping from
                reserved keys to the allocated memory objects.

        Raises:
            MemoryExhaustedError: if the allocation cannot be completed

        Note:
            This function should be thread-safe
        """
        # Compute number of keys to allocate
        handle = self._allocate_new_reserve_handle()

        num_objects_to_allocate = 0
        ret_mask: list[bool] = []
        for key in keys:
            # NOTE: we do fine-grained locking here since we want to
            # make sure multiple threads can reserve a part of the keys
            # if they have the identical keys submitted at the same time
            with self._buffer_lock:
                if self._has_key(key):
                    ret_mask.append(False)
                else:
                    ret_mask.append(True)
                    num_objects_to_allocate += 1
                    self._reserved_keys.add(key)

        if num_objects_to_allocate == 0:
            # No allocation needed
            with self._buffer_lock:
                self._reserved_memory_object_pools[handle] = {}
            return handle, {}

        # Allocate memory objects
        objects = self._memory_allocator.batched_allocate(
            shape, dtype, num_objects_to_allocate, fmt
        )
        if objects is None:
            # TODO: trigger eviction logic here
            raise MemoryExhaustedError(
                "Memory allocation failed due to insufficient memory."
            )

        # Record the reserved memory objects
        reserved_dict = {
            k: v for k, v in zip(compress(keys, ret_mask), objects, strict=False)
        }
        with self._buffer_lock:
            self._reserved_memory_object_pools[handle] = reserved_dict

        return handle, reserved_dict

    def commit(
        self,
        reserve_handle: ReserveHandle,
    ) -> None:
        """Mark the reserved memory objects as "ready to be used/retrieved".

        Args:
            reserve_handle: the handle returned from the "reserve" function.

        Raises:
            RuntimeError: if the reserve handle is invalid.
        """
        with self._buffer_lock:
            reserved_dict = self._reserved_memory_object_pools.pop(reserve_handle, None)
            if reserved_dict is None:
                raise RuntimeError(f"Invalid reserve handle: {reserve_handle}")
            self._commited_memory_objects.update(reserved_dict)
            for key in reserved_dict:
                self._reserved_keys.remove(key)

    def lookup(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> int:
        """Lookup the and lock memory objects for the given keys.

        Args:
            keys: the list of keys to lookup

        Returns:
            int: the total number of found keys (prefix matching)
        """
        # TODO: implement LOCK mechanism
        found_count = 0
        with self._buffer_lock:
            for key in keys:
                if key in self._commited_memory_objects:
                    found_count += 1
                else:
                    break
        return found_count

    def retrieve(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> list[MemoryObj]:
        """Retrieve the memory objects for the given keys.
        The memory objects should be locked before retrieval.
        It will unlock the memory objects after retrieval.

        Args:
            keys: the list of keys to retrieve

        Returns:
            list[MemoryObj]: the list of memory objects corresponding to
                the input keys. It requires all keys to be found.

        Raises:
            RuntimeError if there are one or more memory objects that are
                not found.
        """
        # TODO: implement UNLOCK mechanism
        # TODO: make it a context manager. i.e. caller will do the following
        # ```
        # with storage_manager.retrieve(keys) as objs:
        # ```
        # This will help unlock and free the objects automatically.
        with self._buffer_lock:
            try:
                return [self._commited_memory_objects[key] for key in keys]
            except KeyError as e:
                raise RuntimeError(f"Key not found: {e.args[0]}") from e

    def prefetch(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> None:
        """Prefetch the memory objects for the given keys into L1 memory.

        Args:
            keys: the list of keys to prefetch
        """
        raise NotImplementedError

    def close(self):
        """
        Release the resources held by the storage manager.
        """
        self._memory_allocator.close()

    def memcheck(self):
        """
        Check the memory usage of the storage manager.
        """
        with self._allocator_lock:
            return self._memory_allocator.memcheck()

    def clear(self):
        """
        Clear all the memory objects in the storage manager.
        """
        # TODO: Need to think about how to safely clear the reserved
        # but not committed memory objects
        with self._buffer_lock:
            for key, obj in self._commited_memory_objects.items():
                obj.ref_count_down()
            logger.info(
                "Cleared %d committed memory objects.",
                len(self._commited_memory_objects),
            )
            self._commited_memory_objects.clear()

            for handle, reserved_list in self._reserved_memory_object_pools.items():
                for key, obj in reserved_list.items():
                    obj.ref_count_down()
            logger.info(
                "Cleared %d reserved memory objects pools.",
                len(self._reserved_memory_object_pools),
            )
            self._reserved_memory_object_pools.clear()
            self._reserved_keys.clear()
