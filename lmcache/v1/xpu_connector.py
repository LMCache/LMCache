# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
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
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)


class VLLMPagedMemXPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemXPUConnectorV2":
        """Create a connector from LMCacheEngineMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemXPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemXPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemXPUConnector"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(slot_mapping.device)
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            for i, kvcache in enumerate(kvcaches):
                kvcache.view(total_blocks, head_size).index_copy_(0, slices, tmp[i])
        else:
            tmp_k = memory_obj.tensor[0].to(slot_mapping.device)
            tmp_v = memory_obj.tensor[1].to(slot_mapping.device)
            num_blocks, block_size, num_heads, head_size = kvcaches[0][0].shape
            total_blocks = num_blocks * block_size
            d = num_heads * head_size
            for i, (kcache, vcache) in enumerate(kvcaches):
                kcache.view(total_blocks, d).index_copy_(0, slices, tmp_k[i])
                vcache.view(total_blocks, d).index_copy_(0, slices, tmp_v[i])

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]

        if self.use_mla:
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, head_size).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
        else:
            num_blocks, block_size, num_heads, head_size = kvcaches[0][0].shape
            total_blocks = num_blocks * block_size
            d = num_heads * head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        memory_obj.tensor.copy_(tmp, non_blocking=True)

        if not memory_obj.tensor.is_xpu:
            # Force a synchronize if the target buffer is NOT XPU device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.xpu.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)
