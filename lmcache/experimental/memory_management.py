import abc
import ctypes
import threading
from ctypes import Array, c_ubyte
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import sortedcontainers
import torch

from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor

logger = init_logger(__name__)


class MemoryFormat(Enum):
    UNDEFINED = 0
    """[2, num_layers, num_tokens, hidden_dim]
    """
    KV_BLOB = 1
    """Compressed binary array format
    """
    BINARY = 2

    def token_dim(self) -> int:
        if self == MemoryFormat.KV_BLOB:
            return 2
        elif self == MemoryFormat.BINARY:
            return 0
        return 0


@dataclass
class FreeBlock:
    """Metadata class used by the memory allocators
    """
    start: int
    size: int

    def can_be_coalesced(self, succ: "FreeBlock") -> bool:
        return self.start + self.size == succ.start


@dataclass
class MemoryObjMetadata:
    # The 'logical' shape of the tensor
    shape: torch.Size

    # The 'logical' dtype of the tensor
    dtype: torch.dtype

    # The 'physical address' of the tensor
    address: int

    # The 'physical size' in bytes of the allocated memory
    phy_size: int

    # Reference count
    ref_count: int

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED


class MemoryObj:
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self, raw_data: torch.Tensor, metadata: MemoryObjMetadata):
        self.raw_data = raw_data
        self.metadata = metadata
        self.valid = True

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        num_elements = self.raw_data.numel()
        element_size = self.raw_data.element_size()
        size_in_bytes = num_elements * element_size
        return size_in_bytes

    def get_shape(self) -> torch.Size:
        return self.metadata.shape

    def get_dtype(self) -> torch.dtype:
        return self.metadata.dtype

    def get_memory_format(self) -> MemoryFormat:
        return self.metadata.fmt

    def get_physical_size(self) -> int:
        return self.metadata.phy_size

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return self.raw_data.view(self.metadata.dtype)\
                            .view(self.metadata.shape)

    @property
    def byte_array(self) -> Array[c_ubyte]:
        kv_chunk = self.tensor
        assert kv_chunk is not None
        num_bytes = kv_chunk.numel() * kv_chunk.element_size()
        ptr = kv_chunk.data_ptr()
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = (ctypes.c_ubyte * num_bytes).from_address(
            ctypes.addressof(ubyte_ptr.contents))
        return byte_array


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
    ) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.
        
        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(self, memory_obj: MemoryObj):
        """
        Frees the memory allocated for the given MemoryObj.
        Note that this function shouldn't be explicitly called.
        Instead, use `ref_count_down` to decrease ref count.

        :param MemoryObj memory_obj: The MemoryObj to free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_up(self, memory_obj: MemoryObj):
        """
        Increase ref count for the given MemoryObj.

        :param MemoryObj memory_obj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_down(self, memory_obj: MemoryObj):
        """
        Decrease ref count for the given MemoryObj.

        :param MemoryObj memory_obj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ref_count(self, memory_obj: MemoryObj):
        """
        Get ref count for the given MemoryObj.

        :param MemoryObj memory_obj.
        """
        raise NotImplementedError


class TensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a "explicit list" memory allocator.
    """
    ALIGN_BYTES = 512

    def __init__(self, tensor: torch.Tensor):
        self.buffer = tensor.view(torch.uint8).flatten()

        self.explicit_list = sortedcontainers.SortedList(key=lambda x: x.start)

        self.explicit_list.add(FreeBlock(start=0, size=self.buffer.numel()))

        # For debugging purposes
        self.num_active_allocations = 0
        self.total_allocated_size = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @staticmethod
    def _Compute_raw_size(shape: torch.Size, dtype: torch.dtype) -> int:
        return shape.numel() * dtype.itemsize

    @staticmethod
    def _Compute_aligned_size(raw_size: int) -> int:
        align = TensorMemoryAllocator.ALIGN_BYTES
        return (raw_size + align - 1) & ~(align - 1)

    def _coalesce(self, curr_block: FreeBlock, prev_block: Optional[FreeBlock],
                  succ_block: Optional[FreeBlock]):
        """
        Coalesces the current block with the previous and/or successor block.
        This assumes the curr_block is NOT in self.explicit_list

        Returns True if the current block was coalesced, otherwise False.
        """
        if prev_block is not None and \
                prev_block.can_be_coalesced(curr_block):
            merge_prev = True
        else:
            merge_prev = False

        if succ_block is not None and \
                curr_block.can_be_coalesced(succ_block):
            merge_succ = True
        else:
            merge_succ = False

        if merge_prev and merge_succ:
            prev_block.size += curr_block.size + succ_block.size  # type: ignore
            self.explicit_list.remove(succ_block)
        elif merge_prev:
            prev_block.size += curr_block.size  # type: ignore
        elif merge_succ:
            # NOTE: logically, this won't change the order of the succ_block,
            #       so we don't need to do a "remove" and "reinsert" here
            self.explicit_list.remove(succ_block)
            succ_block.start -= curr_block.size  # type: ignore
            succ_block.size += curr_block.size  # type: ignore
            self.explicit_list.add(succ_block)

        return merge_prev or merge_succ

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> Optional[MemoryObj]:
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        # Calculate the size of the tensor
        raw_size = TensorMemoryAllocator._Compute_raw_size(shape, dtype)
        aligned_size = TensorMemoryAllocator._Compute_aligned_size(raw_size)

        # Find the first block that fits the shape
        for block in self.explicit_list:
            if block.size >= aligned_size:
                break
        else:
            logger.warning("Failed to allocate memory for "
                           "tensor({shape}, {dtype}) because "
                           "no memory is available")
            return None

        # Do not add the block back if `block.size == aligned_size`
        self.explicit_list.remove(block)
        # Update the explicit list
        if block.size > aligned_size:
            self.explicit_list.add(
                FreeBlock(start=block.start + aligned_size,
                          size=block.size - aligned_size))

        # Update debug status
        self.total_allocated_size += aligned_size
        self.num_active_allocations += 1
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

        # Allocate the block
        return MemoryObj(
            raw_data=self.buffer[block.start:block.start + raw_size],
            metadata=MemoryObjMetadata(shape, dtype, block.start, aligned_size,
                                       1, fmt))

    def free(self, memory_obj: MemoryObj):
        if not memory_obj.is_valid():
            return

        new_free_block = FreeBlock(start=memory_obj.metadata.address,
                                   size=memory_obj.metadata.phy_size)
        index = self.explicit_list.bisect_right(new_free_block)
        prev_block = self.explicit_list[index - 1] if index > 0 else None
        succ_block = self.explicit_list[index] \
                if index < len(self.explicit_list) else None

        coalesced = self._coalesce(new_free_block, prev_block, succ_block)

        if not coalesced:
            self.explicit_list.add(new_free_block)
        memory_obj.invalidate()

        # Update debug status
        self.total_allocated_size -= memory_obj.metadata.phy_size
        self.num_active_allocations = max(0, self.num_active_allocations - 1)
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

    def ref_count_up(self, memory_obj: MemoryObj):
        memory_obj.metadata.ref_count += 1

    def ref_count_down(self, memory_obj: MemoryObj):
        memory_obj.metadata.ref_count -= 1
        if memory_obj.metadata.ref_count == 0:
            self.free(memory_obj)

    def get_ref_count(self, memory_obj: MemoryObj):
        return memory_obj.metadata.ref_count

    def memcheck(self):
        """For debug purposes.
        Returns True is everything is fine, otherwise False.
        """
        clear = True
        logger.info("Checking memory allocator consistency")
        logger.info(
            f" - Total active allocations: {self.num_active_allocations}")
        logger.info(f" - Total allocated size: "
                    f"{self.total_allocated_size / 1048576} MB")

        # Check the real total free size
        total_free_size = sum([block.size for block in self.explicit_list])
        logger.info(f" - Total free size: {total_free_size / 1048576} MB")

        # Check if the numbers are consistent
        if total_free_size + self.total_allocated_size != self.buffer.numel():
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        # Check if the blocks are coalesced
        for prev, succ in zip(self.explicit_list[:-1], self.explicit_list[1:]):
            if prev.can_be_coalesced(succ):
                logger.error("Memory allocator has non-coalesced blocks")
                logger.error("This implies a bug in the memory allocator")
                clear = False
        return clear

    def __del__(self):
        del self.buffer


class HostMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated Host memory.
    """

    def __init__(self, size: int):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, device="cpu")
        self.allocator = TensorMemoryAllocator(buffer)

        self.host_mem_lock = threading.Lock()

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt)

    def free(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()

    def ref_count_up(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.ref_count_up(memory_obj)

    def ref_count_down(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.ref_count_down(memory_obj)

    def get_ref_count(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            return self.allocator.get_ref_count(memory_obj)


class PinMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory.
    """

    def __init__(self, size: int):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        self.allocator = TensorMemoryAllocator(buffer)

        self.host_mem_lock = threading.Lock()

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt)

    def free(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    def ref_count_up(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.ref_count_up(memory_obj)

    def ref_count_down(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.ref_count_down(memory_obj)

    def get_ref_count(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            return self.allocator.get_ref_count(memory_obj)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()


class GPUMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated Host memory.
    """

    def __init__(self, size: int, device="cuda"):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, device=device)
        self.allocator = TensorMemoryAllocator(buffer)

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> Optional[MemoryObj]:
        return self.allocator.allocate(shape, dtype, fmt)

    def free(self, memory_obj: MemoryObj):
        self.allocator.free(memory_obj)

    def ref_count_up(self, memory_obj: MemoryObj):
        self.allocator.ref_count_up(memory_obj)

    def ref_count_down(self, memory_obj: MemoryObj):
        self.allocator.ref_count_down(memory_obj)

    def get_ref_count(self, memory_obj: MemoryObj):
        return self.allocator.get_ref_count(memory_obj)

    def memcheck(self):
        return self.allocator.memcheck()
