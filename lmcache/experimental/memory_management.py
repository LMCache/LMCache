import torch
from typing import Optional, Union, Tuple
import abc
from dataclasses import dataclass
import sortedcontainers

from lmcache.logging import init_logger
logger = init_logger(__name__)

@dataclass
class FreeBlock:
    """Metadata class used by the memory allocators
    """
    start: int
    size: int

    def can_be_coalesced(self, 
                         succ: "FreeBlock") -> bool:
        return self.start + self.size == succ.start

@dataclass
class MemoryObjMetadata:
    shape: torch.Size  # The 'logical' shape of the tensor
    dtype: torch.dtype # The 'logical' dtype of the tensor
    address: int       # The 'physical address' of the tensor
    phy_size: int      # The 'physical size' in bytes of the allocated memory

class MemoryObj:
    """
    Wraps a raw flat tensor with some metadata
    """
    def __init__(
            self, 
            raw_data: torch.Tensor, 
            metadata: MemoryObjMetadata):
        self.raw_data = raw_data
        self.metadata = metadata
        self.valid = True

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warn("Trying to access an invalidated MemoryObj")
            return None

        return self.raw_data.view(self.metadata.dtype)\
                            .view(self.metadata.shape)

class MemoryAllocatorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate(self, 
                 shape: torch.Size, 
                 dtype: torch.dtype) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.

        :param torch.Size shape: The shape of the tensor to allocate.
        
        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(self, memory_obj: MemoryObj):
        """
        Frees the memory allocated for the given MemoryObj.

        :param MemoryObj memory_obj: The MemoryObj to free.
        """
        raise NotImplementedError

class TensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a "explicit list" memory allocator.
    """
    ALIGN_BYTES = 512


    def __init__(self, tensor: torch.Tensor):
        self.buffer = tensor.view(torch.uint8).flatten()

        self.explicit_list = sortedcontainers.SortedList(
            key=lambda x: x.start
        )

        self.explicit_list.add(
            FreeBlock(
                start=0,
                size=self.buffer.numel()
            )
        )

        # For debugging purposes
        self.num_active_allocations = 0
        self.total_allocated_size = 0

    @staticmethod
    def _Compute_raw_size(shape: torch.Size, dtype: torch.dtype) -> int:
        return shape.numel() * dtype.itemsize

    @staticmethod
    def _Compute_aligned_size(raw_size: int) -> int:
        align = TensorMemoryAllocator.ALIGN_BYTES
        return (raw_size + align - 1) & ~(align - 1)

    def _coalesce(
            self, 
            curr_block: FreeBlock, 
            prev_block: Optional[FreeBlock],
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
            prev_block.size += curr_block.size + succ_block.size # type: ignore
            self.explicit_list.remove(succ_block)
        elif merge_prev:
            prev_block.size += curr_block.size # type: ignore
        elif merge_succ:
            # NOTE: logically, this won't change the order of the succ_block,
            #       so we don't need to do a "remove" and "reinsert" here
            succ_block.start -= curr_block.size # type: ignore

        return merge_prev or merge_succ

    def allocate(self, 
                 shape: Union[torch.Size, Tuple[int, ...]],
                 dtype: torch.dtype) -> Optional[MemoryObj]:
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

        # Update the explicit list
        if block.size > aligned_size:
            self.explicit_list.remove(block)
            self.explicit_list.add(
                FreeBlock(
                    start=block.start + aligned_size,
                    size=block.size - aligned_size
                )
            )

        # Update debug status
        self.total_allocated_size += aligned_size
        self.num_active_allocations += 1 

        # Allocate the block
        return MemoryObj(
            raw_data=self.buffer[block.start:block.start + raw_size],
            metadata=MemoryObjMetadata(shape, dtype, block.start, aligned_size)
        )

    def free(self, memory_obj: MemoryObj):
        if not memory_obj.is_valid():
            return 

        new_free_block = FreeBlock(
            start=memory_obj.metadata.address,
            size=memory_obj.metadata.phy_size
        )
        index = self.explicit_list.bisect_right(new_free_block)
        prev_block = self.explicit_list[index - 1] if index > 0 else None
        succ_block = self.explicit_list[index] \
                if index < len(self.explicit_list) else None

        coalesced = self._coalesce(
            new_free_block, prev_block, succ_block
        )

        if not coalesced:
            self.explicit_list.add(new_free_block)

        memory_obj.invalidate()

        # Update debug status
        self.total_allocated_size -= memory_obj.metadata.phy_size
        self.num_active_allocations = max(0, self.num_active_allocations - 1)


    def memcheck(self):
        """For debug purposes.
        Returns True is everything is fine, otherwise False.
        """
        clear = True
        logger.info("Checking memory allocator consistency")
        logger.info(
                f" - Total active allocations: {self.num_active_allocations}")
        logger.info(
                f" - Total allocated size: "
                f"{self.total_allocated_size / 1048576} MB")

        # Check the real total free size 
        total_free_size = sum([block.size for block in self.explicit_list])
        logger.info(f" - Total free size: {total_free_size / 1048576} MB")

        # Check if the numbers are consistent
        if total_free_size + self.total_allocated_size != self.buffer.numel():
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        # Check if the blocks are coallesced
        for prev, succ in zip(self.explicit_list[:-1], self.explicit_list[1:]):
            if prev.can_be_coalesced(succ):
                logger.error("Memory allocator has non-coalesced blocks")
                logger.error("This implies a bug in the memory allocator")
                clear = False
        return clear

