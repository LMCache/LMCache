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
from typing import List, Optional, Union
import os

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    VLLMPagedMemGPUConnectorV2,
)
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    discover_gpu_kv_format,
    ensure_contiguous_kv_caches,
    get_block_size,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    is_mla,
)
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

ALLOWED_FORMAT_TRANSITIONS = {
    (None, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_MLA_FMT, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_T2D, MemoryFormat.KV_MLA_FMT),
}


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
        use_gpu: bool = False,
        **kwargs,
    ):
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu
        self._kv_cache_pointers_on_xpu: Optional[torch.Tensor] = None
        self._kv_cache_ptrs_cpu: Optional["np.ndarray"] = None
        # Two-stage D2H staging buffer (allocated lazily on first from_gpu call)
        self._d2h_staging: Optional[torch.Tensor] = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemXPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemXPUConnectorV2.
        """
        return cls(
            use_gpu=use_gpu,
        )

    def _initialize_xpu_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        """Build a device tensor of raw data pointers for the Triton kernel.

        Caches the result and only rebuilds when the underlying data
        pointers change (e.g. when kvcaches are replaced by new tensors).
        """
        kv_caches = ensure_contiguous_kv_caches(kv_caches)

        # Device pointers may exceed signed int64 range on XPU.
        # Use numpy uint64 → view as int64 to avoid overflow.
        ptrs_np = np.array(
            [t.data_ptr() for t in kv_caches], dtype=np.uint64
        )
        ptrs_i64 = ptrs_np.view(np.int64)

        # Fast check: skip rebuild if pointers haven't changed.
        if (
            self._kv_cache_pointers_on_xpu is not None
            and self._kv_cache_ptrs_cpu is not None
            and len(ptrs_i64) == len(self._kv_cache_ptrs_cpu)
            and (ptrs_i64 == self._kv_cache_ptrs_cpu).all()
        ):
            return self._kv_cache_pointers_on_xpu

        self._kv_cache_ptrs_cpu = ptrs_i64.copy()
        ptrs_cpu = torch.from_numpy(ptrs_i64)
        self._kv_cache_pointers_on_xpu = ptrs_cpu.to(self.device)
        return self._kv_cache_pointers_on_xpu

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

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        kv_cache_pointers = self._initialize_xpu_pointers(self.kvcaches)

        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
            block_size=self.block_size,
            head_size=self.head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_MLA_FMT
        if use_mla is True.

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

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        kv_cache_pointers = self._initialize_xpu_pointers(self.kvcaches)

        if not memory_obj.tensor.is_xpu:
            # Two-stage: Triton gather → XPU staging (device-local),
            # then staging → CPU pinned (bulk DMA).
            # Direct scattered PCIe writes from GPU to CPU are very slow.
            target_shape = memory_obj.tensor.shape
            if (
                self._d2h_staging is None
                or self._d2h_staging.shape != target_shape
            ):
                self._d2h_staging = torch.empty(
                    target_shape,
                    dtype=memory_obj.tensor.dtype,
                    device=self.kvcaches[0].device,
                )
            lmc_ops.multi_layer_kv_transfer(
                self._d2h_staging,
                kv_cache_pointers,
                slot_mapping[start:end],
                self.kvcaches[0].device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
                block_size=self.block_size,
                head_size=self.head_size,
            )
            memory_obj.tensor.copy_(self._d2h_staging)
            torch.xpu.synchronize(self.device)
        else:
            lmc_ops.multi_layer_kv_transfer(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                self.kvcaches[0].device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
                block_size=self.block_size,
                head_size=self.head_size,
            )

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data.

        Raises:
            RuntimeError: If attributes have not been initialized yet
                (i.e., no kv_caches have been seen).
        """
        if not self._attributes_initialized:
            raise RuntimeError(
                "Cannot determine shape before attributes are initialized. "
                "Call to_gpu or from_gpu first so that _initialize_attributes "
                "can discover the KV cache layout."
            )
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        """Validate that the memory object has the expected format.

        Args:
            memory_obj: The memory object to validate.

        Raises:
            ValueError: If the memory format does not match the expected
                format based on whether MLA is in use.
        """
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemXPUConnectorV2"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemXPUConnectorV2"
                )

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]):
        """Initialize attributes from the kv_caches using utils functions.

        Uses format discovery and utility functions from utils.py to
        extract all KV cache parameters lazily on first use.

        Args:
            kv_caches: The KV cache tensors from which to discover
                the cache layout and parameters.
        """
        if self._attributes_initialized:
            return

        self.device = kv_caches[0].device
        assert self.device.type == "xpu", "The device should be XPU."

        self.gpu_kv_format = discover_gpu_kv_format(kv_caches, EngineType.VLLM)
        self.num_layers = get_num_layers(kv_caches, self.gpu_kv_format)
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        self.hidden_dim_size = get_hidden_dim_size(kv_caches, self.gpu_kv_format)
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)
        self.use_mla = is_mla(self.gpu_kv_format)
        self.dtype = get_dtype(kv_caches, self.gpu_kv_format)
        self.num_heads = (
            1 if self.use_mla else get_num_heads(kv_caches, self.gpu_kv_format)
        )

        self._attributes_initialized = True
        logger.info(
            "XPU: attributes initialized - format: %s, "
            "num_layers: %d, num_blocks: %d, block_size: %d, "
            "page_buffer_size: %d, hidden_dim_size: %d, head_size: %d, "
            "use_mla: %s, dtype: %s, num_heads: %d",
            self.gpu_kv_format,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.page_buffer_size,
            self.hidden_dim_size,
            self.head_size,
            self.use_mla,
            self.dtype,
            self.num_heads,
        )


class VLLMPagedMemLayerwiseXPUConnector(GPUConnectorInterface):
    """
    Layerwise paged KV connector for XPU.

    Implements the *same generator contract* as VLLMPagedMemLayerwiseGPUConnector:
      - batched_to_gpu(...) yields num_layers + 2 times
      - batched_from_gpu(...) yields num_layers + 1 times

    Transfer uses the Triton ``single_layer_kv_transfer`` kernel when
    available, falling back to PyTorch index_copy_/index_select otherwise.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_xpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_xpu = use_xpu

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
        self.gpu_buffer_allocator: Optional[MemoryAllocatorInterface] = None
        self.gpu_kv_format = None

    def initialize_kvcaches_ptr(self, **kwargs):
        """Override to discover gpu_kv_format from the KV cache tensors."""
        super().initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is not None and self.gpu_kv_format is None:
            self.gpu_kv_format = discover_gpu_kv_format(
                self.kvcaches, EngineType.VLLM
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_xpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseXPUConnector":
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_xpu=use_xpu,
            chunk_size=metadata.kv_shape[2],
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _validate_format_transition(self, mem, target_fmt):
        current_fmt = mem.metadata.fmt

        if (current_fmt, target_fmt) not in ALLOWED_FORMAT_TRANSITIONS:
            raise ValueError(
                f"Invalid KV format transition: {current_fmt} -> {target_fmt}"
            )

    def _lazy_initialize_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        # Buffer allocator only needed when use_xpu=True (device staging)
        if self.use_xpu and self.gpu_buffer_allocator is None:
            # First Party
            from lmcache.v1.memory_management import XPUMemoryAllocator

            # Derive size from first layer KV tensor
            layer0 = kv_caches[0]
            derived_bytes = layer0.numel() * layer0.element_size()

            # Allow override via env variable
            staging_bytes = int(
                os.getenv("LMCACHE_GPU_STAGING_BUFFER_BYTES", derived_bytes)
            )

            logger.info(
                "Initializing staging buffer (derived=%d bytes, final=%d bytes)",
                derived_bytes,
                staging_bytes,
            )

            self.gpu_buffer_allocator = XPUMemoryAllocator(
                size=staging_bytes,
                device=self.device,
            )

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

        def _ensure_xpu(t: torch.Tensor) -> torch.Tensor:
            # Handle both torch.device('xpu:0') and string devices consistently.
            if t is None:
                return t
            if t.device != self.device:
                # non_blocking is fine; will be blocking
                # if underlying memory isn't pinned
                return t.to(self.device, non_blocking=True)
            return t

        # Build a single contiguous mapping in the SAME order we will pack chunks.
        slot_mapping_chunks = [
            slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
        ]
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        # Move mapping ONCE to device (fixes multiple small H2D copies).
        slot_mapping_full = _ensure_xpu(slot_mapping_full)

        num_tokens = int(slot_mapping_full.numel())
        if num_tokens <= 0:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.xpu.current_stream().wait_stream(self.load_stream)
            yield
            return

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_xpu:
            # First Party
            from lmcache.v1.memory_management import MemoryFormat

            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            requested_bytes = (
                int(buffer_shape.numel())
                * torch.empty((), dtype=self.dtype).element_size()
            )
            allocator_tensor = getattr(self.gpu_buffer_allocator, "tensor", None)
            capacity_bytes: Optional[int] = None
            if isinstance(allocator_tensor, torch.Tensor):
                capacity_bytes = int(
                    allocator_tensor.numel() * allocator_tensor.element_size()
                )
            allocator_backend = getattr(self.gpu_buffer_allocator, "allocator", None)
            allocated_bytes = getattr(allocator_backend, "total_allocated_size", None)
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None or tmp_gpu_buffer_obj.tensor is None:
                raise RuntimeError(
                    "Failed to allocate XPU staging buffer for batched_to_gpu: "
                    f"requested_bytes={requested_bytes}, "
                    f"capacity_bytes={capacity_bytes}, "
                    f"allocated_bytes={allocated_bytes}, "
                    f"allocator_type={type(self.gpu_buffer_allocator).__name__}, "
                    f"allocator_tensor_device="
                    f"{getattr(allocator_tensor, 'device', None)}"
                )

        current_stream = torch.xpu.current_stream()

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield  # List[MemoryObj] for this layer

                if sync:
                    current_stream.wait_stream(self.load_stream)

                with torch.xpu.stream(self.load_stream):
                    dst_layer = self.kvcaches[layer_id]
                    cursor = 0

                    if self.use_xpu:
                        assert tmp_gpu_buffer_obj is not None
                        staged = tmp_gpu_buffer_obj.tensor
                        assert staged is not None

                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue

                            src = _ensure_xpu(mem.tensor)

                            staged[cursor : cursor + n].copy_(src, non_blocking=True)
                            cursor += n

                        sl = _ensure_xpu(slot_mapping_full)
                        lmc_ops.single_layer_kv_transfer(
                            staged[:num_tokens],
                            dst_layer,
                            sl,
                            lmc_ops.TransferDirection.H2D,
                            self.gpu_kv_format,
                            token_major=True,
                        )

                    else:
                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue

                            src = _ensure_xpu(mem.tensor)
                            sl = slot_mapping_full[cursor : cursor + n]
                            sl = _ensure_xpu(sl)
                            cursor += n

                            # Detect token_major from tensor shape
                            if self.use_mla or src.dim() != 3:
                                token_major = False
                            else:
                                token_major = src.shape[1] == 2

                            lmc_ops.single_layer_kv_transfer(
                                src,
                                dst_layer,
                                sl,
                                lmc_ops.TransferDirection.H2D,
                                self.gpu_kv_format,
                                token_major=token_major,
                            )

            yield

            if sync:
                current_stream.wait_stream(self.load_stream)
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(  # type: ignore[override]
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

        slot_mapping_on_device = slot_mapping.to(self.device)

        # Precompute “full” mapping for batched gather
        # NOTE: this assumes starts/ends partition slot_mapping contiguously.
        # If not contiguous, concatenation is still correct.
        slot_mapping_full = torch.cat(
            [slot_mapping_on_device[s:e] for s, e in zip(starts, ends, strict=False)],
            dim=0,
        )
        total_tokens = int(slot_mapping_full.numel())

        # Optional staging buffer (will be USED when self.use_xpu=True)
        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_xpu:
            # First Party
            from lmcache.v1.memory_management import MemoryFormat

            # buffer shape uses existing helper; must match how allocator expects KV_T2D
            buffer_shape = self.get_shape(total_tokens)
            assert self.gpu_buffer_allocator is not None
            requested_bytes = (
                int(buffer_shape.numel())
                * torch.empty((), dtype=self.dtype).element_size()
            )
            allocator_tensor = getattr(self.gpu_buffer_allocator, "tensor", None)
            capacity_bytes: Optional[int] = None
            if isinstance(allocator_tensor, torch.Tensor):
                capacity_bytes = int(
                    allocator_tensor.numel() * allocator_tensor.element_size()
                )
            allocator_backend = getattr(self.gpu_buffer_allocator, "allocator", None)
            allocated_bytes = getattr(allocator_backend, "total_allocated_size", None)
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None or tmp_gpu_buffer_obj.tensor is None:
                raise RuntimeError(
                    "Failed to allocate XPU staging buffer for batched_from_gpu: "
                    f"requested_bytes={requested_bytes}, "
                    f"capacity_bytes={capacity_bytes}, "
                    f"allocated_bytes={allocated_bytes}, "
                    f"allocator_type={type(self.gpu_buffer_allocator).__name__}, "
                    f"allocator_tensor_device="
                    f"{getattr(allocator_tensor, 'device', None)}"
                )
            tmp = tmp_gpu_buffer_obj.tensor  # staging tensor on device

        try:
            for layer_id in range(self.num_layers):
                mem_layer = memory_objs[layer_id]

                with torch.xpu.stream(self.store_stream):
                    self.store_stream.wait_stream(current_stream)

                    src_layer = self.kvcaches[layer_id]

                    if self.use_xpu:
                        assert tmp_gpu_buffer_obj is not None
                        tmp = tmp_gpu_buffer_obj.tensor

                        # Gather from paged KV cache into staging buffer
                        lmc_ops.single_layer_kv_transfer(
                            tmp[:total_tokens],
                            src_layer,
                            slot_mapping_full,
                            lmc_ops.TransferDirection.D2H,
                            self.gpu_kv_format,
                            token_major=not self.use_mla,
                        )

                        # Copy chunks from staging to mem tensors
                        off = 0
                        for s, e, mem in zip(
                            starts, ends, mem_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = e - s
                            chunk = tmp[off : off + n]
                            off += n
                            mem.tensor.copy_(
                                chunk.to(mem.tensor.device), non_blocking=True
                            )
                    else:
                        # Non-staged: per-chunk gather with D2H staging
                        for s, e, mem in zip(
                            starts, ends, mem_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            sl = slot_mapping_on_device[s:e]

                            if self.use_mla or mem.tensor.dim() != 3:
                                token_major = False
                            else:
                                token_major = mem.tensor.shape[1] == 2

                            if not mem.tensor.is_xpu:
                                # Two-stage: gather into device staging,
                                # then bulk DMA to CPU (avoids slow
                                # scattered PCIe writes).
                                d2h_stg = torch.empty_like(
                                    mem.tensor, device=self.device
                                )
                                lmc_ops.single_layer_kv_transfer(
                                    d2h_stg,
                                    src_layer,
                                    sl,
                                    lmc_ops.TransferDirection.D2H,
                                    self.gpu_kv_format,
                                    token_major=token_major,
                                )
                                mem.tensor.copy_(d2h_stg)
                            else:
                                lmc_ops.single_layer_kv_transfer(
                                    mem.tensor,
                                    src_layer,
                                    sl,
                                    lmc_ops.TransferDirection.D2H,
                                    self.gpu_kv_format,
                                    token_major=token_major,
                                )

                    if self.use_mla:
                        target_fmt = MemoryFormat.KV_MLA_FMT
                        for mem in mem_layer:
                            self._validate_format_transition(mem, target_fmt)
                            mem.metadata.fmt = target_fmt

                if sync:
                    self.store_stream.synchronize()
                yield
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        return self._batched_to_gpu_gen(starts=starts or [], ends=ends or [], **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
