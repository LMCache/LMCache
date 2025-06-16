# Copyright 2024-2025 LMCache Authors.
# Copyright 2025 Ilya Yanok, Serapheim Dimitropoulos.
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

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
import abc
import ctypes
import threading

# Third Party
import sortedcontainers
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class MemoryFormat(Enum):
    UNDEFINED = 0
    """[2, num_layers, num_tokens, hidden_dim]
    """
    # KV_BLOB = 1
    KV_2LTD = 1
    """[num_tokens, 2, hidden_dim]
    """
    # LAYER_KV_BLOB = 2
    KV_T2D = 2
    """[2, num_tokens, hidden_dim]
    """

    KV_2TD = 3
    """Compressed binary array format
    """
    BINARY = 4

    BINARY_BUFFER = 5

    KV_MLA_FMT = 6
    """[1, num_layers, num_tokens, aligned_head_size]
    """

    def token_dim(self) -> int:
        if self == MemoryFormat.KV_2LTD:
            return 2
        elif self == MemoryFormat.BINARY:
            return 0
        return 0


@dataclass
class FreeBlock:
    """Metadata class used by the memory allocators"""

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

    # TODO(Jiayi): Need to differentiate between temporary pin
    # and persistent pin. Or maybe it's better to use only
    # `ref_count` to manage these semantics.
    # Whether the object is pinned and cannot be evicted
    is_pin: bool = False

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED

    # Positions when the cache is stored
    cached_positions: Optional[torch.Tensor] = None

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

    def to_dict(self):
        # Note(Kuntai): this is used for serializing MemoryObjMetadata via
        # msgpack.
        return {
            "__type__": "MemoryObjMetadata",
            "shape": list(self.shape),  # torch.Size -> list
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "address": self.address,
            "phy_size": self.phy_size,
            "ref_count": self.ref_count,
            "fmt": self.fmt.value,
        }

    @staticmethod
    def from_dict(d):
        dtype_str = d["dtype"]
        dtype = getattr(torch, dtype_str.replace("torch.", "")) if dtype_str else None
        return MemoryObjMetadata(
            shape=torch.Size(d["shape"]),
            dtype=dtype,
            address=d["address"],
            phy_size=d["phy_size"],
            ref_count=d["ref_count"],
            fmt=MemoryFormat(d["fmt"]),
        )


class MemoryObj(metaclass=abc.ABCMeta):
    """
    MemoryObj interface.
    """

    def __init__(self, metadata: MemoryObjMetadata):
        self.meta = metadata

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

    @abc.abstractmethod
    def pin(self) -> bool:
        """
        Pin the memory obj so that it will not be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_up(self):
        """
        Increase ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unpin(self) -> bool:
        """
        Unpin the memory obj so that it can be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_down(self):
        """
        Decrease ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ref_count(self) -> int:
        """
        Get ref count for the given MemoryObj.
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

    @property
    @abc.abstractmethod
    def is_pinned(self) -> bool:
        """
        Check whether the memory obj is pinned.
        """
        raise NotImplementedError


class TensorMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"] = None,
    ):
        self.raw_data = raw_data
        self.meta = metadata
        self.valid = True
        self.lock = threading.Lock()
        self.parent_allocator = parent_allocator

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
        return self.meta.shape

    def get_dtype(self) -> torch.dtype:
        return self.meta.dtype

    def get_memory_format(self) -> MemoryFormat:
        with self.lock:
            return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self):
        with self.lock:
            self.meta.ref_count += 1

    def ref_count_down(self):
        with self.lock:
            self.meta.ref_count -= 1
            if (
                self.meta.ref_count == 0
                and self.parent_allocator is not None
                and self.meta.is_pin is False
            ):
                self.parent_allocator.free(self)

    def get_ref_count(self) -> int:
        with self.lock:
            return self.meta.ref_count

    def pin(self) -> bool:
        self.metadata.is_pin = True
        return True

    def unpin(self) -> bool:
        self.metadata.is_pin = False
        return True

    @property
    def metadata(self) -> MemoryObjMetadata:
        with self.lock:
            return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.dtype is not None
        return self.raw_data.view(self.meta.dtype).view(self.meta.shape)

    @property
    def byte_array(self) -> bytes:
        kv_chunk = self.tensor
        assert kv_chunk is not None
        num_bytes = kv_chunk.numel() * kv_chunk.element_size()
        ptr = kv_chunk.data_ptr()
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = (ctypes.c_ubyte * num_bytes).from_address(
            ctypes.addressof(ubyte_ptr.contents)
        )
        return memoryview(byte_array)

    @property
    def is_pinned(self) -> bool:
        return self.metadata.is_pin


class BytesBufferMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self, raw_bytes: bytes, metadata: Optional[MemoryObjMetadata] = None):
        self.raw_data = raw_bytes
        if metadata is None:
            bytes_shape = torch.Size([len(self.raw_data), 0, 0, 0])
            self.meta = MemoryObjMetadata(
                shape=bytes_shape,
                dtype=None,
                address=0,
                phy_size=0,
                ref_count=1,
                is_pin=False,
                fmt=MemoryFormat.BINARY_BUFFER,
            )
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

    def pin(self) -> bool:
        self.metadata.is_pin = True
        return True

    def unpin(self) -> bool:
        self.metadata.is_pin = False
        return True

    def ref_count_up(self):
        pass

    def ref_count_down(self):
        pass

    def get_ref_count(self) -> int:
        return 1

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

    @property
    def is_pinned(self) -> bool:
        return self.metadata.is_pin


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
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
    ) -> Optional[MemoryObj]:
        """
        Batched allocate the memory to hold a tensor of the given shape.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
        :param int batch_size: The number of tensors to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A lisf of MemoryObjs wrapping the allocated memory.
            Returns None if the allocation failed.

        :rtype: Optional[List[MemoryObj]]
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
    def batched_free(self, memory_objs: List[MemoryObj]):
        """
        Frees the memory allocated for the given list of MemoryObjs.

        :param List[MemoryObj] memory_objs: The list of MemoryObjs
            to free.
        """
        raise NotImplementedError


class TensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a "explicit list" memory allocator.
    """

    ALIGN_BYTES = 512

    def __init__(self, tensor: torch.Tensor, align_bytes: int = ALIGN_BYTES):
        self.buffer = tensor.view(torch.uint8).flatten()
        self.align_bytes = align_bytes

        self.explicit_list = sortedcontainers.SortedList(key=lambda x: x.start)

        self.explicit_list.add(FreeBlock(start=0, size=self.buffer.numel()))

        # For debugging purposes
        self.num_active_allocations = 0
        self.total_allocated_size = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @staticmethod
    @_lmcache_nvtx_annotate
    def _Compute_raw_size(shape: torch.Size, dtype: torch.dtype) -> int:
        return shape.numel() * dtype.itemsize

    @staticmethod
    @_lmcache_nvtx_annotate
    def _Compute_aligned_size(raw_size: int, align: int) -> int:
        return (raw_size + align - 1) & ~(align - 1)

    @_lmcache_nvtx_annotate
    def _coalesce(
        self,
        curr_block: FreeBlock,
        prev_block: Optional[FreeBlock],
        succ_block: Optional[FreeBlock],
    ):
        """
        Coalesces the current block with the previous and/or successor block.
        This assumes the curr_block is NOT in self.explicit_list

        Returns True if the current block was coalesced, otherwise False.
        """
        if prev_block is not None and prev_block.can_be_coalesced(curr_block):
            merge_prev = True
        else:
            merge_prev = False

        if succ_block is not None and curr_block.can_be_coalesced(succ_block):
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

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        parent_allocator: Optional["MemoryAllocatorInterface"] = None,
    ) -> Optional[TensorMemoryObj]:
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"
        # Calculate the size of the tensor
        raw_size = TensorMemoryAllocator._Compute_raw_size(shape, dtype)
        if raw_size % self.align_bytes != 0:
            aligned_size = TensorMemoryAllocator._Compute_aligned_size(
                raw_size, self.align_bytes
            )
        else:
            aligned_size = raw_size

        # Find the first block that fits the shape
        for block in self.explicit_list:
            if block.size >= aligned_size:
                break
        else:
            logger.debug(
                f"Failed to allocate memory for "
                f"tensor({shape}, {dtype}) because "
                "no memory is available"
            )
            return None

        # Do not add the block back if `block.size == aligned_size`
        self.explicit_list.remove(block)
        # Update the explicit list
        if block.size > aligned_size:
            self.explicit_list.add(
                FreeBlock(
                    start=block.start + aligned_size,
                    size=block.size - aligned_size,
                )
            )

        # TODO (Jiayi): need a flag to drop these debug ops
        # Update debug status
        self.total_allocated_size += aligned_size
        self.num_active_allocations += 1
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

        # Allocate the block
        return TensorMemoryObj(
            raw_data=self.buffer[block.start : block.start + raw_size],
            metadata=MemoryObjMetadata(
                shape, dtype, block.start, aligned_size, 1, False, fmt
            ),
            parent_allocator=parent_allocator,
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        parent_allocator: Optional["MemoryAllocatorInterface"] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        """
        Batched allocate tensor memory objs with equal sizes.
        """
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"

        # Calculate the size of the tensor
        unit_raw_size = TensorMemoryAllocator._Compute_raw_size(shape, dtype)

        if unit_raw_size % self.align_bytes != 0:
            unit_aligned_size = TensorMemoryAllocator._Compute_aligned_size(
                unit_raw_size, self.align_bytes
            )
        else:
            unit_aligned_size = unit_raw_size

        total_aligned_size = unit_aligned_size * batch_size

        # Find the first block that fits the shape
        for block in self.explicit_list:
            if block.size >= total_aligned_size:
                break
        else:
            logger.debug(
                f"Failed to batched allocate memory for "
                f"{batch_size} tensor({shape}, {dtype}) because "
                "no memory is available"
            )
            return None

        # Do not add the block back if `block.size == aligned_size`
        self.explicit_list.remove(block)
        # Update the explicit list
        if block.size > total_aligned_size:
            self.explicit_list.add(
                FreeBlock(
                    start=block.start + total_aligned_size,
                    size=block.size - total_aligned_size,
                )
            )

        # TODO (Jiayi): need a flag to drop these debug ops
        # Update debug status
        self.total_allocated_size += total_aligned_size
        self.num_active_allocations += batch_size
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

        raw_datas = torch.chunk(
            self.buffer[block.start : block.start + total_aligned_size],
            batch_size,
        )
        tensor_mem_objs = []
        temp_start = block.start
        for raw_data in raw_datas:
            tensor_mem_objs.append(
                TensorMemoryObj(
                    raw_data=raw_data,
                    metadata=MemoryObjMetadata(
                        shape, dtype, temp_start, unit_aligned_size, 1, False, fmt
                    ),
                    parent_allocator=parent_allocator,
                )
            )
            temp_start += unit_aligned_size

        return tensor_mem_objs

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj):
        if not memory_obj.is_valid():
            return

        new_free_block = FreeBlock(
            start=memory_obj.meta.address, size=memory_obj.meta.phy_size
        )
        index = self.explicit_list.bisect_right(new_free_block)
        prev_block = self.explicit_list[index - 1] if index > 0 else None
        succ_block = (
            self.explicit_list[index] if index < len(self.explicit_list) else None
        )

        coalesced = self._coalesce(new_free_block, prev_block, succ_block)

        if not coalesced:
            self.explicit_list.add(new_free_block)
        memory_obj.invalidate()

        # TODO (Jiayi): need a flag to drop these debug ops
        # Update debug status
        self.total_allocated_size -= memory_obj.meta.phy_size
        self.num_active_allocations = max(0, self.num_active_allocations - 1)
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

    @_lmcache_nvtx_annotate
    def batched_free(self, memory_objs: List[MemoryObj]):
        """
        Batched free memory objs.
        Unlike `batched_allocate`, this function does not
        assume that the memory objs are equal-sized.
        """

        new_free_block = None
        curr_start = memory_objs[0].meta.address
        new_free_blocks = []
        num_valid_blocks = 0
        total_freed_size = 0
        for memory_obj in memory_objs:
            if not memory_obj.is_valid():
                logger.warning("Trying to free an invalidated MemoryObj")
                continue
            num_valid_blocks += 1
            memory_obj.invalidate()
            total_freed_size += memory_obj.meta.phy_size
            if new_free_block is None:
                new_free_block = FreeBlock(
                    start=memory_obj.meta.address, size=memory_obj.meta.phy_size
                )
                curr_start += memory_obj.meta.phy_size
                continue

            if curr_start == memory_obj.meta.address:
                new_free_block.size += memory_obj.meta.phy_size
                curr_start += memory_obj.meta.phy_size
            else:
                new_free_blocks.append(new_free_block)
                new_free_block = FreeBlock(
                    start=memory_obj.meta.address, size=memory_obj.meta.phy_size
                )
                curr_start = memory_obj.meta.address + memory_obj.meta.phy_size
        new_free_blocks.append(new_free_block)

        for new_free_block in new_free_blocks:
            index = self.explicit_list.bisect_right(new_free_block)
            prev_block = self.explicit_list[index - 1] if index > 0 else None
            succ_block = (
                self.explicit_list[index] if index < len(self.explicit_list) else None
            )

            coalesced = self._coalesce(new_free_block, prev_block, succ_block)

            if not coalesced:
                self.explicit_list.add(new_free_block)

        # TODO (Jiayi): need a flag to drop these debug ops
        # Update debug status
        self.total_allocated_size -= total_freed_size
        self.num_active_allocations = max(
            0, self.num_active_allocations - num_valid_blocks
        )
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

    def memcheck(self):
        """For debug purposes.
        Returns True is everything is fine, otherwise False.
        """
        clear = True
        logger.info("Checking memory allocator consistency")
        logger.info(f" - Total active allocations: {self.num_active_allocations}")
        logger.info(
            f" - Total allocated size: {self.total_allocated_size / 1048576} MB"
        )

        # Check the real total free size
        total_free_size = sum([block.size for block in self.explicit_list])
        logger.info(f" - Total free size: {total_free_size / 1048576} MB")

        # Check if the numbers are consistent
        if total_free_size + self.total_allocated_size != self.buffer.numel():
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        # Check if the blocks are coalesced
        for prev, succ in zip(
            self.explicit_list[:-1], self.explicit_list[1:], strict=False
        ):
            if prev.can_be_coalesced(succ):
                logger.error("Memory allocator has non-coalesced blocks")
                logger.error("This implies a bug in the memory allocator")
                clear = False
        return clear

    def __del__(self):
        del self.buffer


class BufferAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, device="cpu"):
        """
        :param str device: The device of the buffer memory.
        """
        self.device = device

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
    ) -> BytesBufferMemoryObj:
        n = shape[0]
        byte_array = bytearray(n)
        return BytesBufferMemoryObj(byte_array)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
    ) -> List[BytesBufferMemoryObj]:
        n = shape[0]
        # TODO(Jiayi): Optimize the following loop.
        byte_arrays = [bytearray(n) for _ in range(batch_size)]
        return [BytesBufferMemoryObj(byte_array) for byte_array in byte_arrays]

    def free(self, memory_obj: MemoryObj):
        return

    def batched_free(self, memory_objs: List[MemoryObj]):
        return

    def memcheck(self):
        return True


class HostMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated Host memory."""

    def __init__(self, size: int):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, device="cpu")
        self.allocator = TensorMemoryAllocator(buffer)

        self.host_mem_lock = threading.Lock()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, self)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(shape, dtype, batch_size, fmt, self)

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(self, memory_objs: List[MemoryObj]):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()


class PinMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, size: int):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        self.allocator = TensorMemoryAllocator(buffer)

        self.host_mem_lock = threading.Lock()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, self)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(shape, dtype, batch_size, fmt, self)

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(self, memory_objs: List[MemoryObj]):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()


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

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shape, dtype, fmt)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
        ]:
            with self.host_mem_lock:
                return self.pin_allocator.allocate(shape, dtype, fmt, self)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[List[MemoryObj]]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.batched_allocate(shape, dtype, batch_size, fmt)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
        ]:
            with self.host_mem_lock:
                return self.pin_allocator.batched_allocate(
                    shape, dtype, batch_size, fmt, self
                )
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj):
        fmt = memory_obj.meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.free(memory_obj)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
        ]:
            with self.host_mem_lock:
                self.pin_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_free(self, memory_objs: List[MemoryObj]):
        # NOTE: fmts of all memory_objs should be the same
        fmt = memory_objs[0].meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.batched_free(memory_objs)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
        ]:
            with self.host_mem_lock:
                self.pin_allocator.batched_free(memory_objs)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def memcheck(self):
        with self.host_mem_lock:
            return self.pin_allocator.memcheck()


class GPUMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated GPU memory."""

    def __init__(self, size: int, device="cuda", align_bytes: Optional[int] = None):
        """
        :param int size: The size of the GPU memory in bytes.
        :param Optional[int] align_bytes: The byte alignment for allocations.
        """
        self.tensor = torch.empty(size, dtype=torch.uint8, device=device)
        kwargs = {}
        if align_bytes is not None:
            kwargs["align_bytes"] = align_bytes
        self.allocator = TensorMemoryAllocator(self.tensor, **kwargs)

        self.device_mem_lock = threading.Lock()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        with self.device_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, self)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[List[MemoryObj]]:
        with self.device_mem_lock:
            return self.allocator.batched_allocate(shape, dtype, batch_size, fmt, self)

    def free(self, memory_obj: MemoryObj):
        with self.device_mem_lock:
            self.allocator.free(memory_obj)

    def batched_free(self, memory_objs: List[MemoryObj]):
        with self.device_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.device_mem_lock:
            return self.allocator.memcheck()


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

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """
        Returns a dummy MemoryObj for testing purposes.
        """
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"

        # Return a dummy object with no actual memory allocation
        return TensorMemoryObj(
            raw_data=torch.empty(shape, dtype=dtype, device=self.device),
            metadata=MemoryObjMetadata(
                shape=shape,
                dtype=dtype,
                address=0,
                phy_size=0,
                ref_count=1,
                is_pin=False,
                fmt=fmt,
            ),
            parent_allocator=self,
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[List[MemoryObj]]:
        raise NotImplementedError(
            "Batched allocation is not supported in AdHocMemoryAllocator"
        )

    def free(self, memory_obj: MemoryObj):
        pass

    def batched_free(self, memory_objs: List[MemoryObj]):
        pass

    def ref_count_up(self, memory_obj: MemoryObj):
        pass

    def ref_count_down(self, memory_obj: MemoryObj):
        pass

    def get_ref_count(self, memory_obj: MemoryObj):
        return 0

    def memcheck(self):
        return True


class CuFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, device=None):
        # HACK(Jiayi): cufile import is buggy on some hardware
        # (e.g., without GPUDirect), so it's temporarily put here.
        # Third Party
        from cufile.bindings import cuFileBufDeregister, cuFileBufRegister

        self.cuFileBufDeregister = cuFileBufDeregister
        if device is None:
            # TODO(Serapheim): Ideally we'd get the device from the upper
            # layer - for now just use the current device.
            device = f"cuda:{torch.cuda.current_device()}"
        super().__init__(size, device, align_bytes=4096)
        self.base_pointer = self.tensor.data_ptr()
        cuFileBufRegister(ctypes.c_void_p(self.base_pointer), size, flags=0)

    def __del__(self):
        self.cuFileBufDeregister(ctypes.c_void_p(self.base_pointer))
