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
from typing import List, Optional, Generator, Iterable, Sequence, Tuple, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2, GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

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
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemXPUConnectorV2":
        """Create a connector from LMCacheMetadata.

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

def _split_token2d_kv(token2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accepts either:
      - [2, T, D]
      - [T, 2, D]
    Returns:
      - k_tok: [T, D]
      - v_tok: [T, D]
    """
    if token2d.dim() != 3:
        raise ValueError(f"Expected token2d dim=3, got {token2d.shape}")
    if token2d.shape[0] == 2:  # [2, T, D]
        return token2d[0], token2d[1]
    if token2d.shape[1] == 2:  # [T, 2, D]
        return token2d[:, 0, :], token2d[:, 1, :]
    raise ValueError(f"Unrecognized token2d layout: {token2d.shape}")


def _get_paged_kv_views(
    kv_cache_layer: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    use_mla: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns flattened views for index_copy/index_select.

    MLA:
      kv_cache_layer: [num_pages, page_size, head_size]
      returns: flat: [num_pages*page_size, head_size]

    Non-MLA:
      kv_cache_layer either:
        - tensor [2, num_pages, page_size, num_heads, head_size]
        - tuple (k, v) each [num_pages, page_size, num_heads, head_size]
      returns:
        (k_flat, v_flat) each [num_pages*page_size, num_heads*head_size]
    """
    if use_mla:
        if not isinstance(kv_cache_layer, torch.Tensor):
            raise ValueError("MLA expects kv_cache_layer as Tensor")
        if kv_cache_layer.dim() != 3:
            raise ValueError(f"MLA expects dim=3, got {kv_cache_layer.shape}")
        num_pages, page_size, head_size = kv_cache_layer.shape
        return kv_cache_layer.view(num_pages * page_size, head_size)

    # non-MLA
    if isinstance(kv_cache_layer, torch.Tensor):
        # [2, num_pages, page_size, num_heads, head_size]
        if kv_cache_layer.dim() != 5 or kv_cache_layer.shape[0] != 2:
            raise ValueError(f"Expected [2, P, B, H, D], got {kv_cache_layer.shape}")
        k = kv_cache_layer[0]
        v = kv_cache_layer[1]
    else:
        k, v = kv_cache_layer
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(f"Expected (k,v) 4D, got {k.shape}, {v.shape}")

    num_pages, page_size, num_heads, head_size = k.shape
    total = num_pages * page_size
    d = num_heads * head_size
    return k.view(total, d), v.view(total, d)


class VLLMPagedMemLayerwiseXPUConnector(GPUConnectorInterface):
    """
    Layerwise paged KV connector for XPU.

    Implements the *same generator contract* as VLLMPagedMemLayerwiseGPUConnector:
      - batched_to_gpu(...) yields num_layers + 2 times
      - batched_from_gpu(...) yields num_layers + 1 times

    Transfer is implemented with pure torch ops (index_copy_/index_select).
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        assert "chunk_size" in kwargs, "chunk_size should be provided."
        assert "dtype" in kwargs, "dtype should be provided."
        assert "device" in kwargs, "device should be provided."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # XPU streams
        self.load_stream = torch.xpu.Stream()
        self.store_stream = torch.xpu.Stream()

        # Optional device staging buffer allocator (same pattern as CUDA class)
        self.gpu_buffer_allocator = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseXPUConnector":
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=metadata.kv_shape[2],
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _lazy_initialize_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        # Buffer allocator only needed when use_gpu=True (device staging)
        if self.use_gpu and self.gpu_buffer_allocator is None:
            # Import here to avoid circulars
            from lmcache.v1.memory_management import GPUMemoryAllocator

            self.gpu_buffer_allocator = GPUMemoryAllocator()

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError("Layerwise uses batched_to_gpu(generator).")

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError("Layerwise uses batched_from_gpu(generator).")

    def _batched_to_gpu_gen(self, starts: List[int], ends: List[int], **kwargs):
        """
        Generator: CPU token2d -> (optional XPU staging) -> XPU paged KV (per layer).
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = [slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)]
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        # Ensure mapping is on same device as paged cache
        slot_mapping_full = slot_mapping_full.to(self.device)

        num_tokens = int(slot_mapping_full.numel())
        offset = starts[0]

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            from lmcache.v1.memory_management import MemoryFormat

            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None and tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.xpu.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield  # List[MemoryObj] for this layer

            if sync:
                current_stream.wait_stream(self.load_stream)

            with torch.xpu.stream(self.load_stream):
                # Destination paged KV views
                dst_layer = self.kvcaches[layer_id]

                if self.use_mla:
                    dst_flat = _get_paged_kv_views(dst_layer, use_mla=True)
                else:
                    dst_k_flat, dst_v_flat = _get_paged_kv_views(dst_layer, use_mla=False)

                # For each chunk, scatter into paged by mapping
                for s, e, mem in zip(starts, ends, memory_objs_layer, strict=False):
                    assert mem.tensor is not None
                    src = mem.tensor.to(self.device, non_blocking=True)

                    if self.use_gpu:
                        # stage into tmp buffer first
                        tmp_gpu_buffer_obj.tensor[s - offset : e - offset].copy_(
                            src, non_blocking=True
                        )
                    else:
                        # direct scatter per chunk
                        sl = slot_mapping_full[s - offset : e - offset]
                        if self.use_mla:
                            # src expected [T, D] or [1, T, D]? keep it simple:
                            if src.dim() == 2:
                                dst_flat.index_copy_(0, sl, src)
                            elif src.dim() == 3 and src.shape[0] == 1:
                                dst_flat.index_copy_(0, sl, src[0])
                            else:
                                raise ValueError(f"Unexpected MLA token tensor: {src.shape}")
                        else:
                            k_tok, v_tok = _split_token2d_kv(src)
                            dst_k_flat.index_copy_(0, sl, k_tok)
                            dst_v_flat.index_copy_(0, sl, v_tok)

                # If staged, scatter once from the staged buffer
                if self.use_gpu:
                    staged = tmp_gpu_buffer_obj.tensor
                    sl = slot_mapping_full
                    if self.use_mla:
                        if staged.dim() == 2:
                            dst_flat.index_copy_(0, sl, staged)
                        elif staged.dim() == 3 and staged.shape[0] == 1:
                            dst_flat.index_copy_(0, sl, staged[0])
                        else:
                            raise ValueError(f"Unexpected MLA staged tensor: {staged.shape}")
                    else:
                        k_tok, v_tok = _split_token2d_kv(staged)
                        dst_k_flat.index_copy_(0, sl, k_tok)
                        dst_v_flat.index_copy_(0, sl, v_tok)

        yield  # after last layer

        if sync:
            current_stream.wait_stream(self.load_stream)

        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield  # final

    def batched_from_gpu(
        self,
        memory_objs: List[List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Generator: XPU paged KV -> (optional XPU staging) -> CPU token2d (per layer).
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        current_stream = torch.xpu.current_stream()

        # Optional staging buffer
        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            from lmcache.v1.memory_management import MemoryFormat

            slot_mapping_full = torch.cat([slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)], dim=0)
            slot_mapping_full = slot_mapping_full.to(self.device)
            num_tokens = int(slot_mapping_full.numel())
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None and tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            mem_layer = memory_objs[layer_id]

            with torch.xpu.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                src_layer = self.kvcaches[layer_id]

                # For each chunk, gather out by its mapping
                for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                    assert mem.tensor is not None
                    sl = slot_mapping[s:e].to(self.device)

                    if self.use_mla:
                        src_flat = _get_paged_kv_views(src_layer, use_mla=True)
                        gathered = src_flat.index_select(0, sl)
                        # write into mem.tensor on CPU
                        mem.tensor.copy_(gathered.to(mem.tensor.device), non_blocking=True)
                        mem.metadata.fmt = MemoryFormat.KV_MLA_FMT
                    else:
                        src_k_flat, src_v_flat = _get_paged_kv_views(src_layer, use_mla=False)
                        k = src_k_flat.index_select(0, sl)
                        v = src_v_flat.index_select(0, sl)

                        # mem.tensor may be [2, T, D] or [T, 2, D]
                        if mem.tensor.dim() == 3 and mem.tensor.shape[0] == 2:
                            mem.tensor[0].copy_(k.to(mem.tensor.device), non_blocking=True)
                            mem.tensor[1].copy_(v.to(mem.tensor.device), non_blocking=True)
                        elif mem.tensor.dim() == 3 and mem.tensor.shape[1] == 2:
                            mem.tensor[:, 0, :].copy_(k.to(mem.tensor.device), non_blocking=True)
                            mem.tensor[:, 1, :].copy_(v.to(mem.tensor.device), non_blocking=True)
                        else:
                            raise ValueError(f"Unexpected output token2d layout: {mem.tensor.shape}")

                        # keep consistent with CUDA layerwise connector behavior
                        mem.metadata.fmt = MemoryFormat.KV_MLA_FMT if self.use_mla else mem.metadata.fmt

            yield
            if sync:
                self.store_stream.synchronize()

        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_to_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj], List[int], None] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        assert starts is not None and ends is not None
        return self._batched_to_gpu_gen(starts, ends, **kwargs)


    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
