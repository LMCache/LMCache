# SPDX-License-Identifier: Apache-2.0
"""Shared memory object contracts and metadata."""

# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union
import abc

# Third Party
import torch

# First Party
from lmcache.v1.common.utils import get_size_bytes


class MemoryFormat(Enum):
    UNDEFINED = 0
    """[2, num_layers, num_tokens, hidden_dim]
    """
    # KV_BLOB = 1
    KV_2LTD = auto()
    """[num_tokens, 2, hidden_dim]
    """
    # LAYER_KV_BLOB = 2
    KV_T2D = auto()
    """[2, num_tokens, hidden_dim]
    """

    KV_2TD = auto()
    """Compressed binary array format
    """
    BINARY = auto()

    BINARY_BUFFER = auto()

    KV_MLA_FMT = auto()
    """[1, num_layers, num_tokens, aligned_head_size]
    """

    # This is for the encoder cache (EC) tensor format
    EC_TD = auto()
    """[num_tokens, hidden_dim]
    """

    # Hidden-state store (HS) tensor format. Same logical shape as EC_TD
    # ([num_tokens, hidden_dim]) but tagged separately so the allocator and
    # any future mp/serialization paths can distinguish encoder-cache entries
    # from hidden-state entries.
    HS_TD = auto()
    """[num_tokens, hidden_dim]
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
        elif self == MemoryFormat.EC_TD:
            return 0
        elif self == MemoryFormat.HS_TD:
            return 0
        return 0


@dataclass
class MemoryObjMetadata:
    # TODO(chunxiaozheng): use shapes and dtypes to replace shape and dtype
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

    # shapes and dtypes should be used in the future
    shapes: Optional[list[torch.Size]] = None
    dtypes: Optional[list[torch.dtype]] = None

    def to_dict(self):
        # Note(Kuntai): this is used for serializing MemoryObjMetadata via
        # msgpack.
        return {
            "__type__": "MemoryObjMetadata",
            "shape": list(self.shape),  # torch.Size -> list
            "dtype": str(self.dtype) if self.dtype else None,
            "address": self.address,
            "phy_size": self.phy_size,
            "ref_count": self.ref_count,
            "fmt": self.fmt.value,
            "shapes": [list(shape) for shape in self.shapes] if self.shapes else None,
            "dtypes": [str(dtype) for dtype in self.dtypes] if self.dtypes else None,
        }

    @staticmethod
    def from_dict(d):
        dtype_str = d["dtype"]
        dtype = getattr(torch, dtype_str.replace("torch.", "")) if dtype_str else None
        shapes_list = d["shapes"]
        shapes = [torch.Size(s) for s in shapes_list] if shapes_list else None
        dtypes_list = d["dtypes"]
        dtypes = (
            [getattr(torch, d_str.replace("torch.", "")) for d_str in dtypes_list]
            if dtypes_list
            else None
        )
        return MemoryObjMetadata(
            shape=torch.Size(d["shape"]),
            dtype=dtype,
            address=d["address"],
            phy_size=d["phy_size"],
            ref_count=d["ref_count"],
            fmt=MemoryFormat(d["fmt"]),
            shapes=shapes,
            dtypes=dtypes,
        )

    def get_size(self) -> int:
        if self.shapes is not None and self.dtypes is not None:
            return get_size_bytes(self.shapes, self.dtypes)
        return self.shape.numel() * self.dtype.itemsize  # type: ignore


class MemoryObj(metaclass=abc.ABCMeta):
    """
    MemoryObj interface.
    """

    # subclasses should expose raw_data differently
    raw_data: Any

    def __init__(self, metadata: MemoryObjMetadata):
        self.meta = metadata
        self._l1_manager_id: int | None = None

    def set_l1_manager(self, owner_tag: int) -> None:
        """Assign the process-local L1 owner, not the writer's reservation tag.

        Args:
            owner_tag: Stable integer identity of the responsible L1 manager.

        Raises:
            ValueError: If this allocation already belongs to another manager.
        """
        if self._l1_manager_id is not None and self._l1_manager_id != owner_tag:
            raise ValueError("Memory object already belongs to another L1 manager")
        self._l1_manager_id = owner_tag

    def get_l1_manager(self) -> int | None:
        """Return the process-local L1 owner, or ``None`` outside the L1 path."""
        return self._l1_manager_id

    def reset_l1_manager(self) -> None:
        """Clear ownership when an allocator starts a recycled object's lifetime.

        Only call after the previous allocation and all its users have drained.
        This identity is intentionally separate from serialized metadata.
        """
        self._l1_manager_id = None

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
    def get_shapes(self) -> list[torch.Size]:
        """
        Get the shapes of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dtypes(self) -> list[torch.dtype]:
        """
        Get the dtypes of the MemoryObj.
        """
        raise NotImplementedError

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

    def set_used_size(self, n: int) -> None:  # noqa: B027
        """Narrow this buffer's logical size to the first ``n`` bytes.

        Optional hook for callers that have just written ``n`` bytes
        into a buffer originally allocated with an upper-bound size
        (e.g. the async serde processor, where the destination is sized
        from ``estimate_serialized_size`` but ``serialize`` returns the
        actual ``n`` it wrote). After this call, ``get_size()`` /
        ``byte_array`` / any downstream L2 adapter that reads the
        logical size will see exactly ``n`` bytes.

        Default is a no-op so subclasses without a "used vs allocated"
        distinction (e.g. :class:`BytesBufferMemoryObj`, where the raw
        bytes already are the actual contents) keep working unchanged.

        Args:
            n: bytes actually used in this buffer. Subclasses that
                implement this must validate ``n`` and raise
                ``ValueError`` on out-of-range or unsupported layouts.
        """
        pass

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
    def shm_offset(self) -> int:
        """Return the byte offset of this object inside the SHM pool."""
        return self.meta.address

    @property
    def shm_byte_length(self) -> int:
        """Return the byte length of this object inside the SHM pool."""
        return self.get_size()

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

    @property
    @abc.abstractmethod
    def raw_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the raw tensor from the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj at the given index(group).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        """
        Get the allocator that allocates this memory object
        """
        raise NotImplementedError


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param int batch_size: The number of tensors to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A list of MemoryObjs wrapping the allocated memory.
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

    def memcheck(self) -> bool:
        """
        Checks the memory allocator for consistency.

        Returns:
            True if everything is fine otherwise False
        """
        return True

    # TODO(chunxiaozheng): remove if after all params replaced by shapes/dtypes
    def _adapt_shapes_and_dtypes(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> Tuple[list[torch.Size], list[torch.dtype]]:
        if isinstance(shapes, torch.Size):
            shapes = [shapes]

        if isinstance(dtypes, torch.dtype):
            dtypes = [dtypes]

        assert len(shapes) == len(dtypes), (
            f"shapes and dtypes must have the same length, "
            f"got {len(shapes)} and {len(dtypes)}, "
            f"shapes: {shapes}, dtypes: {dtypes}"
        )
        return shapes, dtypes
