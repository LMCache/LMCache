# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Optional, Union

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)


class AdHocMemoryAllocator(MemoryAllocatorInterface):
    """
    AdHocMemoryAllocator is a simple allocator that does not actually
    allocate memory. It is used for testing purposes only.
    """

    def __init__(self, device: str = "cpu"):
        """
        :param str device: The device of the ad hoc memory allocator.
        """
        if not torch_dev.is_available():
            self.device = "cpu"
        else:
            self.device = device

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Returns a dummy MemoryObj for testing purposes.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        size = get_size_bytes(shapes, dtypes)

        # Return a dummy object with no actual memory allocation
        return TensorMemoryObj(
            raw_data=torch.empty(
                torch.Size([size]), dtype=torch.uint8, device=self.device
            ),
            metadata=MemoryObjMetadata(
                shape=shapes[0],
                dtype=dtypes[0],
                address=0,
                phy_size=0,
                ref_count=1,
                pin_count=0,
                fmt=fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=self,
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
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
