import abc
import ctypes
import threading
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

    BINARY_BUFFER = 3

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
    dtype: Optional[torch.dtype]

    # The 'physical address' of the tensor
    address: int

    # The 'physical size' in bytes of the allocated memory
    phy_size: int

    # Reference count
    ref_count: int

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED

    def get_size(self):
        """
        Calculate the size of the memory object in bytes 
        """
        if self.shape.numel() == 0:
            return 0
        if self.dtype is None:
            return 0
        num_elements = self.shape.numel()
        element_size = self.dtype.itemsize
        size_in_bytes = num_elements * element_size
        return size_in_bytes


class MemoryObj(metaclass=abc.ABCMeta):
    """
    MemoryObj interface.
    """

    @abc.abstractmethod
    def invalidate(self):
        """
        Invalidate the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid(self):
        """
        Check if the MemoryObj is valid.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Get the size of the MemoryObj in bytes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self) -> torch.Size:
        """
        Get the shape of the MemoryObj.
        """
        raise NotImplementedError

    def get_dtype(self) -> Optional[torch.dtype]:
        """
        Get the dtype of the MemoryObj.
        """
        return None

    @abc.abstractmethod
    def get_memory_format(self) -> MemoryFormat:
        """
        Get the memory format of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_physical_size(self) -> int:
        """
        Get the physical size of the MemoryObj in bytes.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def metadata(self) -> MemoryObjMetadata:
        """
        Get the metada of the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tensor(self) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def byte_array(self) -> bytes:
        """
        Get the byte array from the MemoryObj.
        """
        raise NotImplementedError


class TensorMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self, raw_data: torch.Tensor, metadata: MemoryObjMetadata):
        self.raw_data = raw_data
        self.meta = metadata
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
        assert self.metadata.dtype is not None
        return self.metadata.dtype

    def get_memory_format(self) -> MemoryFormat:
        return self.metadata.fmt

    def get_physical_size(self) -> int:
        return self.metadata.phy_size

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.metadata.dtype is not None
        return self.raw_data.view(self.metadata.dtype)\
                            .view(self.metadata.shape)

    @property
    def byte_array(self) -> bytes:
        kv_chunk = self.tensor
        assert kv_chunk is not None
        num_bytes = kv_chunk.numel() * kv_chunk.element_size()
        ptr = kv_chunk.data_ptr()
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = (ctypes.c_ubyte * num_bytes).from_address(
            ctypes.addressof(ubyte_ptr.contents))
        return memoryview(byte_array)


class CopyLessMemoryObj(TensorMemoryObj):

    def __init__(self, raw_data, metadata, callback):
        super().__init__(raw_data, metadata)
        self.callback = callback

    def __del__(self):
        self.callback()


class BytesBufferMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self,
                 raw_bytes: bytes,
                 metadata: Optional[MemoryObjMetadata] = None):
        self.raw_data = raw_bytes
        if metadata is None:
            bytes_shape = torch.Size([len(self.raw_data), 0, 0, 0])
            self.meta = MemoryObjMetadata(shape=bytes_shape,
                                          dtype=None,
                                          address=0,
                                          phy_size=0,
                                          ref_count=1,
                                          fmt=MemoryFormat.BINARY_BUFFER)
        else:
            self.meta = metadata
        self.valid = True

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        return len(self.raw_data)

    def get_shape(self) -> torch.Size:
        return torch.Size([len(self.raw_data), 0, 0, 0])

    def get_dtype(self) -> Optional[torch.dtype]:
        return None

    def get_memory_format(self) -> MemoryFormat:
        return self.metadata.fmt

    def get_physical_size(self) -> int:
        return self.metadata.phy_size

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return None

    @property
    def byte_array(self) -> bytes:
        return self.raw_data


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
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
    def dry_allocate(self, shape: torch.Size,
                     dtype: Optional[torch.dtype]) -> MemoryObjMetadata:
        """
        A 'dry run' allocation that returns the metadata of the
        allocated memory without actually allocating it.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
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
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> Optional[TensorMemoryObj]:
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"

        # Calculate the size of the tensor
        raw_size = TensorMemoryAllocator._Compute_raw_size(shape, dtype)
        aligned_size = TensorMemoryAllocator._Compute_aligned_size(raw_size)

        # Find the first block that fits the shape
        for block in self.explicit_list:
            if block.size >= aligned_size:
                break
        else:
            logger.warning(f"Failed to allocate memory for "
                           f"tensor({shape}, {dtype}) because "
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
        return TensorMemoryObj(
            raw_data=self.buffer[block.start:block.start + raw_size],
            metadata=MemoryObjMetadata(shape, dtype, block.start, aligned_size,
                                       1, fmt))

    def dry_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> MemoryObjMetadata:
        """
        A 'dry run' allocation that returns the metadata of the
        allocated memory without actually allocating it.
        """
        raise NotImplementedError

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


class BufferAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory.
    """

    def __init__(self, device="cpu"):
        """
        :param str device: The device of the buffer memory.
        """
        self.device = device

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
    ) -> BytesBufferMemoryObj:
        n = shape[0]
        byte_array = bytearray(n)
        return BytesBufferMemoryObj(byte_array)

    def dry_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
    ) -> MemoryObjMetadata:
        n = shape[0]
        return MemoryObjMetadata(shape=torch.Size([n, 0, 0, 0]),
                                 dtype=None,
                                 address=0,
                                 phy_size=0,
                                 ref_count=1,
                                 fmt=MemoryFormat.BINARY_BUFFER)

    def free(self, memory_obj: MemoryObj):
        return

    def ref_count_up(self, memory_obj: MemoryObj):
        pass

    def ref_count_down(self, memory_obj: MemoryObj):
        pass

    def get_ref_count(self, memory_obj: MemoryObj):
        pass

    def memcheck(self):
        return True


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
        dtype: Optional[torch.dtype],
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

    def dry_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> MemoryObjMetadata:
        with self.host_mem_lock:
            return self.allocator.dry_allocate(shape, dtype, fmt)


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
        dtype: Optional[torch.dtype],
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

    def dry_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> MemoryObjMetadata:
        with self.host_mem_lock:
            return self.allocator.dry_allocate(shape, dtype, fmt)


class MixedMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates (1) memory in the pre-allocated pinned memory.
              (2) byte_array buffer memory.
    """

    def __init__(self, size: int):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        self.pin_allocator = TensorMemoryAllocator(buffer)
        self.buffer_allocator = BufferAllocator("cpu")

        self.host_mem_lock = threading.Lock()

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> Optional[MemoryObj]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shape, dtype, fmt)
        elif fmt == MemoryFormat.KV_BLOB:
            with self.host_mem_lock:
                return self.pin_allocator.allocate(shape, dtype, fmt)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def dry_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> MemoryObjMetadata:
        raise NotImplementedError

    def free(self, memory_obj: MemoryObj):
        fmt = memory_obj.get_memory_format()
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.free(memory_obj)
        elif fmt == MemoryFormat.KV_BLOB:
            with self.host_mem_lock:
                self.pin_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def ref_count_up(self, memory_obj: MemoryObj):
        fmt = memory_obj.get_memory_format()
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.ref_count_up(memory_obj)
        elif fmt == MemoryFormat.KV_BLOB:
            with self.host_mem_lock:
                self.pin_allocator.ref_count_up(memory_obj)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def ref_count_down(self, memory_obj: MemoryObj):
        fmt = memory_obj.get_memory_format()
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.ref_count_down(memory_obj)
        elif fmt == MemoryFormat.KV_BLOB:
            with self.host_mem_lock:
                self.pin_allocator.ref_count_down(memory_obj)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def get_ref_count(self, memory_obj: MemoryObj):
        fmt = memory_obj.get_memory_format()
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.get_ref_count(memory_obj)
        elif fmt == MemoryFormat.KV_BLOB:
            with self.host_mem_lock:
                return self.pin_allocator.get_ref_count(memory_obj)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def memcheck(self):
        with self.host_mem_lock:
            return self.pin_allocator.memcheck()


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
        dtype: Optional[torch.dtype],
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

    def dry_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> MemoryObjMetadata:
        return self.allocator.dry_allocate(shape, dtype, fmt)


class AdHocMemoryAllocator(MemoryAllocatorInterface):
    """
    AdHocMemoryAllocator is a simple allocator that does not actually 
    allocate memory. It is used for testing purposes only.
    """

    def __init__(self, device: str = "cpu"):
        """
        :param str device: The device of the ad hoc memory allocator.
        """
        self.device = device

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> Optional[MemoryObj]:
        """
        Returns a dummy MemoryObj for testing purposes.
        """
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"

        # Return a dummy object with no actual memory allocation
        return TensorMemoryObj(raw_data=torch.empty(shape,
                                                    dtype=dtype,
                                                    device=self.device),
                               metadata=MemoryObjMetadata(shape=shape,
                                                          dtype=dtype,
                                                          address=0,
                                                          phy_size=0,
                                                          ref_count=1,
                                                          fmt=fmt))

    def dry_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_BLOB,
    ) -> MemoryObjMetadata:
        """
        Returns a dummy MemoryObjMetadata for testing purposes.
        """
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"

        return MemoryObjMetadata(shape=shape,
                                 dtype=dtype,
                                 address=0,
                                 phy_size=0,
                                 ref_count=1,
                                 fmt=fmt)

    def free(self, memory_obj: MemoryObj):
        pass

    def ref_count_up(self, memory_obj: MemoryObj):
        pass

    def ref_count_down(self, memory_obj: MemoryObj):
        pass

    def get_ref_count(self, memory_obj: MemoryObj):
        return 0

    def memcheck(self):
        return True
