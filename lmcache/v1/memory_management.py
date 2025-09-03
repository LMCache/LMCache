# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, Union
import abc
import ctypes
import math
import threading

# Third Party
import sortedcontainers
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
import lmcache.c_ops as lmc_ops

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
        elif self == MemoryFormat.KV_T2D:
            return 1
        elif self == MemoryFormat.KV_2TD:
            return 0
        elif self == MemoryFormat.BINARY:
            return 0
        elif self == MemoryFormat.BINARY_BUFFER:
            return 0
        elif self == MemoryFormat.KV_MLA_FMT:
            return 2
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

    # Whether the object is pinned and cannot be evicted
    # lookup pins are temporary
    # cache controller pins are persistent
    pin_count: int = 0

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED

    # Positions when the cache is stored
    cached_positions: Optional[torch.Tensor] = None

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

    def get_size(self) -> int:
        num_elements = math.prod(self.shape)
        element_size = self.dtype.itemsize  # type: ignore
        size_in_bytes = num_elements * element_size
        return size_in_bytes


class MemoryObj(metaclass=abc.ABCMeta):
    """
    MemoryObj interface.
    """

    # subclasses should expose raw_data differently
    raw_data: Any

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
        Note that this number could be smaller than the physical size.
        The physical size is aligned to the allocator's alignment.
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

    @abc.abstractmethod
    def get_num_tokens(self) -> int:
        """
        Get token number for the given MemoryObj.
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
        The size is will be the physical size instead of the unaligned size.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_ptr(self) -> int:
        """
        Get the data pointer of the MemoryObj.
        This is used to access the raw data in the memory.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_pinned(self) -> bool:
        """
        Check whether the memory obj is pinned.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        """
        raise NotImplementedError


class TensorMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    monitor = LMCStatsMonitor.GetOrCreate()

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"],
    ):
        assert metadata.dtype is not None, "dtype must be specified for TensorMemoryObj"
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
        num_elements = math.prod(self.meta.shape)
        element_size = self.meta.dtype.itemsize  # type: ignore
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
            if self.meta.ref_count < 0:
                logger.warning(
                    f"Ref count of MemoryObj {self.meta.address}"
                    f"is negative: {self.meta.ref_count}."
                    "Double free occurred somewhere."
                    "Setting ref count back to 0 as a hack but please find the bug."
                )
                self.meta.ref_count = 0
            if (
                self.meta.ref_count == 0
                and self.parent_allocator is not None
                and self.meta.pin_count == 0
            ):
                self.parent_allocator.free(self)

    def get_ref_count(self) -> int:
        with self.lock:
            return self.meta.ref_count

    def get_num_tokens(self) -> int:
        with self.lock:
            token_dim = self.meta.fmt.token_dim()
            return self.meta.shape[token_dim]

    def pin(self) -> bool:
        with self.lock:
            # if pin_count is 0, indicates that the object is pinned for the first time
            if self.meta.pin_count == 0:
                TensorMemoryObj.monitor.update_pinned_memory_objs_count(1)

            self.meta.pin_count += 1
            return True

    def unpin(self) -> bool:
        with self.lock:
            self.meta.pin_count -= 1

            # if pin_count is 0, indicates that the object is unpinned
            if self.meta.pin_count == 0:
                TensorMemoryObj.monitor.update_pinned_memory_objs_count(-1)

            if self.meta.pin_count <= 0 and self.meta.ref_count <= 0:
                if self.parent_allocator is None:
                    logger.error(
                        "Parent allocator is None when trying to free MemoryObj."
                        "This could cause memory leak"
                    )
                else:
                    self.parent_allocator.free(self)

            if self.meta.pin_count < 0:
                logger.warning(
                    f"Pin count of MemoryObj {self.meta.address}"
                    f"is negative: {self.meta.pin_count}."
                    "Double unpin occurred somewhere."
                    "Setting pin count back to 0 as a hack but please find the bug."
                )
                self.meta.pin_count = 0
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
        # TODO(Jiayi): consider caching the `get_size()`
        return (
            self.raw_data[: self.get_size()].view(self.meta.dtype).view(self.meta.shape)
        )

    @property
    def byte_array(self) -> bytes:
        num_bytes = self.raw_data.numel() * self.raw_data.element_size()
        ptr = self.raw_data.data_ptr()
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = (ctypes.c_ubyte * num_bytes).from_address(
            ctypes.addressof(ubyte_ptr.contents)
        )
        return memoryview(byte_array)

    @property
    def data_ptr(self) -> int:
        return self.raw_data.data_ptr()

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A memory obj can be evicted if it is not pinned and ref_count=1.
        """
        return not self.is_pinned and self.get_ref_count() == 1


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
                pin_count=0,
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
        self.metadata.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.metadata.pin_count -= 1
        if self.metadata.pin_count < 0:
            logger.warning(
                f"Pin count of MemoryObj {self.meta.address}"
                f"is negative: {self.meta.pin_count}."
                "Double unpin occurred somewhere."
                "Setting pin count back to 0 as a hack but please find the bug."
            )
            self.metadata.pin_count = 0
        return True

    def ref_count_up(self):
        pass

    def ref_count_down(self):
        pass

    def get_ref_count(self) -> int:
        return 1

    def get_num_tokens(self) -> int:
        # TODO(Jiayi): record the number of tokens somehow
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
    def data_ptr(self) -> int:
        mv = memoryview(self.raw_data)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        return addr

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A buffer memory obj can be evicted if it is not pinned.
        """
        return not self.is_pinned


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
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
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
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
    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ):
        """
        Frees the memory allocated for the given MemoryObj.
        Note that this function shouldn't be explicitly called.
        Instead, use `ref_count_down` to decrease ref count.

        :param MemoryObj memory_obj: The MemoryObj to free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        """
        Frees the memory allocated for the given list of MemoryObjs.

        :param List[MemoryObj] memory_objs: The list of MemoryObjs
            to free.
        """
        raise NotImplementedError

    def close(self):
        """
        Closes the memory allocator.
        This is called when the LMCacheEngine is closed.
        """
        return

    def memcheck(self):
        """
        Checks the memory allocator for consistency.
        """
        return


class TensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a "explicit list" memory allocator.
    """

    ALIGN_BYTES = 4096

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
        allocator_type: Optional[str] = None,
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
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        return TensorMemoryObj(
            raw_data=self.buffer[block.start : block.start + raw_size],
            metadata=MemoryObjMetadata(
                shape, dtype, block.start, aligned_size, 1, False, fmt
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
        allocator_type: Optional[str] = None,
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
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

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
                    parent_allocator=self,
                )
            )
            temp_start += unit_aligned_size

        return tensor_mem_objs

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
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
        self.num_active_allocations -= 1
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
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

        if new_free_block is not None:
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

        if update_stats:
            # TODO (Jiayi): need a flag to drop these debug ops
            # Update debug status
            self.total_allocated_size -= total_freed_size
            self.num_active_allocations -= num_valid_blocks
            self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
            self.stats_monitor.update_active_memory_objs_count(
                self.num_active_allocations
            )

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

    def __str__(self):
        return "TensorMemoryAllocator"


class PagedTensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a paged memory allocator.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ):
        self.buffer = tensor.view(torch.uint8).flatten()
        self.buffer_size = self.buffer.numel() * self.buffer.element_size()
        self.buffer_ptr = self.buffer.data_ptr()

        self.shape = shape
        self.dtype = dtype
        self.fmt = fmt

        num_elements = shape.numel()
        self.bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        self.align_bytes = num_elements * self.bytes_per_element

        assert self.buffer_size % self.align_bytes == 0, (
            f"Buffer size {self.buffer_size} must be a"
            f" multiple of align bytes {self.align_bytes}"
            " in paged memory allocator."
        )

        self.paged_buffers = torch.split(self.buffer, self.align_bytes, dim=0)

        # NOTE: deque is used since thread-safety is not a concern here as
        # is implemented in C under the hood (in CPython), and operations
        # on deque are atomic.
        self.free_blocks: deque[TensorMemoryObj] = deque()

        for idx, buf in enumerate(self.paged_buffers):
            # NOTE: idx is the paged index
            # NOTE: the last unfull chunk's shape needs to be
            # adjusted during allocation.
            metadata = MemoryObjMetadata(
                self.shape,
                self.dtype,
                idx,
                self.align_bytes,  # 1 page
                1,  # ref_count=1
                0,  # pin_count=0
                self.fmt,
            )
            mem_obj = TensorMemoryObj(
                raw_data=buf,
                metadata=metadata,
                parent_allocator=self,
            )
            self.free_blocks.append(mem_obj)

        # For debugging purposes
        self.num_active_allocations = 0
        self.total_allocated_size = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @staticmethod
    @_lmcache_nvtx_annotate
    def _Compute_raw_size(shape: torch.Size, dtype: torch.dtype) -> int:
        return shape.numel() * dtype.itemsize

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[TensorMemoryObj]:
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"

        try:
            free_block = self.free_blocks.popleft()
        except IndexError:
            logger.debug(
                f"Failed to allocate memory for "
                f"tensor({shape}, {dtype}) because "
                "no free blocks is available"
            )
            return None

        # TODO (Jiayi): This is a bit redundant.
        free_block.meta.shape = shape
        free_block.meta.fmt = fmt
        free_block.meta.ref_count = 1

        if shape != self.shape:
            size_in_bytes = shape.numel() * self.bytes_per_element
            free_block.raw_data = free_block.raw_data[:size_in_bytes]

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.num_active_allocations += 1
        self.total_allocated_size += self.align_bytes
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        return free_block

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        """
        Batched allocate tensor memory objs with pre-defined equal sizes.
        """
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        assert dtype is not None, "dtype must be specified"

        allocated_blocks: list[TensorMemoryObj] = []
        for i in range(batch_size):
            try:
                free_block = self.free_blocks.popleft()
            except IndexError:
                logger.debug(
                    f"Failed to allocate memory for "
                    f"tensor({shape}, {dtype}) because "
                    "no free blocks is available"
                )
                self.batched_free(allocated_blocks, update_stats=False)
                return None

            # FIXME: think about whether pareant_allocator
            # should be updated here.
            free_block.meta.shape = shape
            free_block.meta.fmt = fmt
            free_block.meta.ref_count = 1

            if shape != self.shape:
                size_in_bytes = shape.numel() * self.bytes_per_element
                free_block.raw_data = free_block.raw_data[:size_in_bytes]

            allocated_blocks.append(free_block)

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.num_active_allocations += batch_size
        self.total_allocated_size = self.num_active_allocations * self.align_bytes
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        return allocated_blocks

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: TensorMemoryObj, allocator_type: Optional[str] = None):
        if not memory_obj.is_valid():
            return
        if memory_obj.meta.shape != self.shape:
            page_idx = memory_obj.meta.address
            memory_obj.raw_data = self.paged_buffers[page_idx]

        self.free_blocks.append(memory_obj)

        # memory_obj.invalidate()

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.total_allocated_size -= self.align_bytes
        self.num_active_allocations -= 1
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[TensorMemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        """
        Batched free memory objs.
        Unlike `batched_allocate`, this function does not
        assume that the memory objs are equal-sized.
        """

        for memory_obj in memory_objs:
            if not memory_obj.is_valid():
                logger.warning("Trying to free an invalidated MemoryObj")
                continue
            # memory_obj.invalidate()
            if memory_obj.meta.shape != self.shape:
                page_idx = memory_obj.meta.address
                memory_obj.raw_data = self.paged_buffers[page_idx]

            self.free_blocks.append(memory_obj)

        if update_stats:
            num_freed_blocks = len(memory_objs)
            # TODO (Jiayi): need a flag to drop these debug ops
            # NOTE (Jiayi): the following code is not thread-safe but
            # is tolerable as this is only used for debugging purposes.
            # Update debug status
            self.total_allocated_size -= self.align_bytes * num_freed_blocks
            self.num_active_allocations -= num_freed_blocks
            self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
            self.stats_monitor.update_active_memory_objs_count(
                self.num_active_allocations
            )

    def memcheck(self):
        """For debug purposes.
        Returns True is everything is fine, otherwise False.
        """

        logger.info("Checking memory allocator consistency")
        logger.info(f" - Total active allocations: {self.num_active_allocations}")
        logger.info(
            f" - Total allocated size: {self.total_allocated_size / 1048576} MB"
        )

        # Check the real total free size
        total_free_size = len(self.free_blocks) * self.align_bytes
        logger.info(f" - Total free size: {total_free_size / 1048576} MB")

        # Check if the numbers are consistent
        if total_free_size + self.total_allocated_size != self.buffer.numel():
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            return False

        return True

    def __str__(self):
        return "PagedTensorMemoryAllocator"

    def __del__(self):
        # FIXME: NIXL-related memory leak should be handled somewhere (else).
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
        allocator_type: Optional[str] = None,
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
        allocator_type: Optional[str] = None,
    ) -> List[BytesBufferMemoryObj]:
        n = shape[0]
        # TODO(Jiayi): Optimize the following loop.
        byte_arrays = [bytearray(n) for _ in range(batch_size)]
        return [BytesBufferMemoryObj(byte_array) for byte_array in byte_arrays]

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        return

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        return

    def __str__(self):
        return "BufferAllocator"

    def memcheck(self):
        return True


class HostMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated Host memory."""

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, device="cpu")

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shape" in kwargs, (
                "shape must be specified for paged memory allocator"
            )
            assert "dtype" in kwargs, (
                "dtype must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(
                shape, dtype, batch_size, fmt, str(self)
            )

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()

    def __str__(self):
        return "HostMemoryAllocator"


class PinMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """

        ptr = lmc_ops.alloc_pinned_ptr(size, 0)
        array_type = ctypes.c_uint8 * size
        buf = array_type.from_address(ptr)
        self.buffer = torch.frombuffer(buf, dtype=torch.uint8)

        self._unregistered = False

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shape" in kwargs, (
                "shape must be specified for paged memory allocator"
            )
            assert "dtype" in kwargs, (
                "dtype must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(
                shape, dtype, batch_size, fmt, str(self)
            )

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()

    def close(self):
        if not self._unregistered:
            torch.cuda.synchronize()
            lmc_ops.free_pinned_ptr(self.buffer.data_ptr())
            self._unregistered = True

    def __str__(self):
        return "PinMemoryAllocator"


class MixedMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates (1) memory in the pre-allocated pinned memory.
              (2) byte_array buffer memory.
    """

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """

        self.numa_mapping = kwargs.get("numa_mapping", None)

        self.size = size

        if self.numa_mapping:
            current_device_id = torch.cuda.current_device()
            gpu_to_numa_mapping = self.numa_mapping.gpu_to_numa_mapping
            assert current_device_id in gpu_to_numa_mapping, (
                f"Current device {current_device_id} is not in the GPU NUMA mapping."
            )
            numa_id = gpu_to_numa_mapping[current_device_id]
            ptr = lmc_ops.alloc_pinned_numa_ptr(size, numa_id)
        else:
            ptr = lmc_ops.alloc_pinned_ptr(size, 0)
        array_type = ctypes.c_uint8 * size
        buf = array_type.from_address(ptr)
        self.buffer = torch.frombuffer(buf, dtype=torch.uint8)
        self._unregistered = False

        self.pin_allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shape" in kwargs, (
                "shape must be specified for paged memory allocator"
            )
            assert "dtype" in kwargs, (
                "dtype must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.pin_allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            self.pin_allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

        self.buffer_allocator = BufferAllocator("cpu")

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
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
                return self.pin_allocator.allocate(shape, dtype, fmt, str(self))
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
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
                    shape, dtype, batch_size, fmt, str(self)
                )
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
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
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
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

    def close(self):
        if not self._unregistered:
            torch.cuda.synchronize()
            if self.numa_mapping:
                lmc_ops.free_pinned_numa_ptr(self.buffer.data_ptr(), self.size)
            else:
                lmc_ops.free_pinned_ptr(self.buffer.data_ptr())
            self._unregistered = True

    def __str__(self):
        return "MixedMemoryAllocator"


class GPUMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated GPU memory."""

    def __init__(
        self,
        size: int,
        device="cuda",
        align_bytes: Optional[int] = None,
        use_paging: bool = False,
        **kwargs,
    ):
        """
        :param int size: The size of the GPU memory in bytes.
        :param Optional[int] align_bytes: The byte alignment for allocations.
        """
        self.tensor = torch.empty(size, dtype=torch.uint8, device=device)

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shape" in kwargs, (
                "shape must be specified for paged memory allocator"
            )
            assert "dtype" in kwargs, (
                "dtype must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.tensor,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            kwargs = {}
            if align_bytes is not None:
                kwargs["align_bytes"] = align_bytes
            self.allocator = TensorMemoryAllocator(self.tensor, **kwargs)

        self.device_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.device_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.device_mem_lock:
            return self.allocator.batched_allocate(
                shape, dtype, batch_size, fmt, str(self)
            )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.device_mem_lock:
            self.allocator.free(memory_obj)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.device_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.device_mem_lock:
            return self.allocator.memcheck()

    def __str__(self):
        return "GPUMemoryAllocator"


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
        allocator_type: Optional[str] = None,
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
                pin_count=0,
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
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        raise NotImplementedError(
            "Batched allocation is not supported in AdHocMemoryAllocator"
        )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        pass

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        pass

    def ref_count_up(self, memory_obj: MemoryObj):
        pass

    def ref_count_down(self, memory_obj: MemoryObj):
        pass

    def get_ref_count(self, memory_obj: MemoryObj):
        return 0

    def memcheck(self):
        return True

    def __str__(self):
        return "AdHocMemoryAllocator"


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

    def __str__(self):
        return "CuFileMemoryAllocator"


class NixlCPUMemoryAllocator(MemoryAllocatorInterface):
    """
    NIXL + CPU Memory Allocator
    This is a special allocator makes pd and cpu compatible.
    """

    def __init__(self):
        pass

    def init_nixl_memory_allocator(
        self,
        tensor: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ):
        self.nixl_allocator = PagedTensorMemoryAllocator(
            tensor,
            shape,
            dtype,
            fmt,
        )

    def init_cpu_memory_allocator(
        self,
        size: int,
    ):
        self.cpu_allocator = MixedMemoryAllocator(size)

    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "cpu",
    ) -> Optional[MemoryObj]:
        if allocator_type == "nixl":
            return self.nixl_allocator.allocate(shape, dtype, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.allocate(shape, dtype, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "cpu",
    ) -> Optional[List[MemoryObj]]:
        if allocator_type == "nixl":
            return self.nixl_allocator.batched_allocate(shape, dtype, batch_size, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.batched_allocate(shape, dtype, batch_size, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = "cpu"):
        if allocator_type == "nixl":
            self.nixl_allocator.free(memory_obj)
        elif allocator_type == "cpu":
            self.cpu_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        if allocator_type == "nixl":
            self.nixl_allocator.batched_free(memory_objs, update_stats=update_stats)
        elif allocator_type == "cpu":
            self.cpu_allocator.batched_free(memory_objs, update_stats=update_stats)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def __str__(self):
        return "NixlCPUMemoryAllocator"
