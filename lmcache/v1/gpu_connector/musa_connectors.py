# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Union, cast
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    VLLMPagedMemGPUConnectorV2,
)
from lmcache.v1.gpu_connector.utils import (
    DiscoverableKVCache,
    LayoutHints,
    _get_head_size_view,
    _split_token2d_kv,
    get_block_size,
    get_device,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    normalize_kv_and_discover_format,
)
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.platform.musa.native_kv_transfer import (
    try_native_from_gpu,
    try_native_to_gpu,
)
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

_SUPPORTED_MUSA_KV_FORMATS = (
    lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
)

ALLOWED_FORMAT_TRANSITIONS = {
    (None, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_MLA_FMT, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_T2D, MemoryFormat.KV_MLA_FMT),
}


def _copy_tensor_at_pin_boundaries(
    dest: torch.Tensor,
    src: torch.Tensor,
    memory_obj: MemoryObj,
) -> None:
    """Copy a host tensor without crossing lazy registration ranges.

    Args:
        dest: Destination tensor.
        src: Source tensor with the same number of bytes as ``dest``.
        memory_obj: Host memory object participating in the transfer.

    Returns:
        None.

    Raises:
        ValueError: If a lazy transfer has mismatched sizes, noncontiguous
            tensors, or does not contain exactly one tensor in ``memory_obj``.
    """
    if not isinstance(memory_obj.parent(), LazyMemoryAllocator):
        dest.copy_(src, non_blocking=True)
        return
    if dest.nbytes != src.nbytes:
        raise ValueError(
            f"MUSA copy size mismatch: dest={dest.nbytes}, src={src.nbytes}"
        )
    if not dest.is_contiguous() or not src.is_contiguous():
        raise ValueError("Lazy MUSA transfers require contiguous tensors")

    memory_start = memory_obj.data_ptr
    memory_end = memory_start + memory_obj.get_size()
    src_start = src.data_ptr()
    dest_start = dest.data_ptr()
    src_is_host = memory_start <= src_start and src_start + src.nbytes <= memory_end
    dest_is_host = memory_start <= dest_start and dest_start + dest.nbytes <= memory_end
    if src_is_host == dest_is_host:
        raise ValueError(
            "Lazy MUSA copy must have exactly one tensor in the memory object"
        )

    host_start = src_start if src_is_host else dest_start
    host_offset = memory_obj.meta.address + host_start - memory_start
    chunk_size = LazyMemoryAllocator.PIN_CHUNK_SIZE
    dest_bytes = dest.view(torch.uint8).flatten()
    src_bytes = src.view(torch.uint8).flatten()
    copied = 0
    while copied < src.nbytes:
        bytes_to_boundary = chunk_size - ((host_offset + copied) % chunk_size)
        copy_size = min(src.nbytes - copied, bytes_to_boundary)
        dest_bytes[copied : copied + copy_size].copy_(
            src_bytes[copied : copied + copy_size],
            non_blocking=True,
        )
        copied += copy_size


def _to_musa_at_pin_boundaries(
    tensor: torch.Tensor,
    memory_obj: MemoryObj,
    device: torch.device,
) -> torch.Tensor:
    """Move a host tensor to MUSA without crossing lazy registration ranges.

    Args:
        tensor: Source tensor owned by ``memory_obj``.
        memory_obj: Memory object that identifies lazy allocator ownership and
            the host offset.
        device: Destination MUSA device.

    Returns:
        ``tensor`` when already on ``device``; otherwise, a tensor on ``device``
        containing the copied data.

    Raises:
        ValueError: If lazy-copy validation fails.
    """
    if tensor.device == device:
        return tensor
    if not isinstance(memory_obj.parent(), LazyMemoryAllocator):
        return tensor.to(device, non_blocking=True)
    musa_tensor = torch.empty_like(tensor, device=device)
    _copy_tensor_at_pin_boundaries(musa_tensor, tensor, memory_obj)
    return musa_tensor


class VLLMPagedMemMUSAConnectorV2(VLLMPagedMemGPUConnectorV2):
    """Non-layerwise paged KV connector for MUSA devices.

    Follows the same contract as VLLMPagedMemXPUConnectorV2: pure torch ops
    (index_copy_ / index_select) with ``torch.musa`` stream and sync APIs.

    Supported paged KV cache layouts:
      - Non-MLA vLLM flash-attention layout:
        ``NL x [2, NB, BS, NH, HS]`` with LMCache ``KV_2LTD`` memory shaped
        ``[2, NL, T, NH * HS]``.
      - MLA vLLM layout:
        ``NL x [NB, BS, HS]`` with LMCache ``KV_MLA_FMT`` memory shaped
        ``[1, NL, T, HS]``.

    Other vLLM layouts, including flash-infer, HND, cross-layer, connector
    v3, and MP GPU-transfer kernel layouts, are not implemented by this
    connector.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemMUSAConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemMUSAConnectorV2.
        """
        return cls(use_gpu=use_gpu)

    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """Store KV data from a memory object into MUSA paged KV caches.

        Args:
            memory_obj: The memory object containing KV data.
            start: Starting index in the token sequence.
            end: Ending index in the token sequence.

        Keyword Args:
            kvcaches: Nested tuple of K/V tensors for the whole sequence.
            slot_mapping: Full slot mapping tensor.

        Raises:
            ValueError: If slot_mapping is missing from kwargs.
            AssertionError: If memory_obj has no tensor.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)
        self._validate_supported_kv_format()

        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))
        transfer_start = start + skip_prefix_n_tokens
        if transfer_start >= end:
            return
        if try_native_to_gpu(
            use_mla=self.use_mla,
            memory_tensor=memory_obj.tensor,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            start=start,
            end=end,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
        ):
            return

        slices = slot_mapping[transfer_start:end].to(
            device=self.device, dtype=torch.long, non_blocking=True
        )

        if self.use_mla:
            tmp = _to_musa_at_pin_boundaries(
                memory_obj.tensor[0],
                memory_obj,
                self.device,
            )
            total_blocks = self.num_blocks * self.block_size
            for i, kvcache in enumerate(self.kvcaches):
                kvcache.view(total_blocks, self.head_size).index_copy_(
                    0, slices, tmp[i, skip_prefix_n_tokens:]
                )
        else:
            tmp_k = _to_musa_at_pin_boundaries(
                memory_obj.tensor[0],
                memory_obj,
                self.device,
            )
            tmp_v = _to_musa_at_pin_boundaries(
                memory_obj.tensor[1],
                memory_obj,
                self.device,
            )
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            for i, (kcache, vcache) in enumerate(self.kvcaches):
                kcache.view(total_blocks, d).index_copy_(
                    0, slices, tmp_k[i, skip_prefix_n_tokens:]
                )
                vcache.view(total_blocks, d).index_copy_(
                    0, slices, tmp_v[i, skip_prefix_n_tokens:]
                )

    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """Load KV data from MUSA paged KV caches into a memory object.

        Args:
            memory_obj: The memory object to populate.
            start: Starting index in the token sequence.
            end: Ending index in the token sequence.

        Keyword Args:
            kvcaches: Nested tuple of K/V tensors for the whole sequence.
            slot_mapping: Full slot mapping tensor.

        Raises:
            ValueError: If slot_mapping is missing from kwargs.
            AssertionError: If memory_obj has no tensor.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)
        self._validate_supported_kv_format()
        if start >= end:
            if self.use_mla:
                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return
        if try_native_from_gpu(
            use_mla=self.use_mla,
            memory_tensor=memory_obj.tensor,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            start=start,
            end=end,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
        ):
            if memory_obj.tensor.device.type != "musa" and hasattr(torch, "musa"):
                torch.musa.synchronize()  # type: ignore[attr-defined]
            if self.use_mla:
                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return

        slices = slot_mapping[start:end].to(
            device=self.device, dtype=torch.long, non_blocking=True
        )

        if self.use_mla:
            total_blocks = self.num_blocks * self.block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, self.head_size).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
        else:
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        _copy_tensor_at_pin_boundaries(memory_obj.tensor, tmp, memory_obj)

        if memory_obj.tensor.device.type != "musa":
            torch.musa.synchronize()  # type: ignore[attr-defined]

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends should be provided.")

        typed_memory_objs = cast(List[MemoryObj], memory_objs)
        for memory_obj, start, end in zip(
            typed_memory_objs, starts, ends, strict=False
        ):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data.

        Raises:
            RuntimeError: If attributes have not been initialized yet.
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
            ValueError: If the memory format does not match.
        """
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemMUSAConnectorV2"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemMUSAConnectorV2"
                )

    def _validate_supported_kv_format(self) -> None:
        """Validate that this connector implements the discovered engine KV format.

        Raises:
            ValueError: If the active KV cache layout is unsupported by the
                non-layerwise torch-based MUSA path.
        """
        if self.engine_kv_format not in _SUPPORTED_MUSA_KV_FORMATS:
            supported = ", ".join(fmt.name for fmt in _SUPPORTED_MUSA_KV_FORMATS)
            raise ValueError(
                "VLLMPagedMemMUSAConnectorV2 supports only vLLM MUSA layouts "
                f"{supported}; got {self.engine_kv_format.name}. Unsupported "
                "layouts include flash-infer, HND, cross-layer, connector v3, "
                "and MP GPU-transfer kernel layouts."
            )

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]) -> None:
        """Initialize attributes from KV caches using utils functions.

        Args:
            kv_caches: The KV cache tensors from which to discover layout.
        """
        if self._attributes_initialized:
            return

        self.device = kv_caches[0].device
        assert self.device.type == "musa", "The device should be MUSA."

        discoverable_kv_caches = cast(DiscoverableKVCache, kv_caches)
        normalized_kv_caches: DiscoverableKVCache
        self.engine_kv_format, normalized_kv_caches = normalize_kv_and_discover_format(
            discoverable_kv_caches, EngineType.VLLM
        )
        self.num_layers = get_num_layers(normalized_kv_caches, self.engine_kv_format)
        self.num_blocks = get_num_blocks(normalized_kv_caches, self.engine_kv_format)
        self.block_size = get_block_size(normalized_kv_caches, self.engine_kv_format)
        self.page_buffer_size = get_page_buffer_size(
            normalized_kv_caches, self.engine_kv_format
        )
        self.hidden_dim_size = get_hidden_dim_size(
            normalized_kv_caches, self.engine_kv_format
        )
        self.head_size = get_head_size(normalized_kv_caches, self.engine_kv_format)
        self.use_mla = lmcache_native.is_mla(self.engine_kv_format)
        self.dtype = get_dtype(normalized_kv_caches, self.engine_kv_format)
        self.num_heads = (
            1
            if self.use_mla
            else get_num_heads(normalized_kv_caches, self.engine_kv_format)
        )

        self._attributes_initialized = True
        logger.info(
            "MUSA: attributes initialized - format: %s, "
            "num_layers: %d, num_blocks: %d, block_size: %d, "
            "page_buffer_size: %d, hidden_dim_size: %d, head_size: %d, "
            "use_mla: %s, dtype: %s, num_heads: %d",
            self.engine_kv_format,
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


class VLLMPagedMemLayerwiseMUSAConnector(GPUConnectorInterface):
    """Layerwise paged KV connector for MUSA devices.

    Implements the same generator contract as VLLMPagedMemLayerwiseXPUConnector:
      - batched_to_gpu(...) yields num_layers + 2 times
      - batched_from_gpu(...) yields num_layers + 1 times

    Transfer is implemented with pure torch ops (index_copy_ / index_select).
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_musa: bool = False,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_musa = use_musa

        assert "chunk_size" in kwargs, "chunk_size should be provided."
        assert "dtype" in kwargs, "dtype should be provided."
        assert "device" in kwargs, "device should be provided."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self._load_stream: Optional[Any] = None
        self._store_stream: Optional[Any] = None

        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    @property
    def load_stream(self) -> Any:
        """Return the lazily-created MUSA load stream."""
        self._ensure_streams()
        return self._load_stream

    @property
    def store_stream(self) -> Any:
        """Return the lazily-created MUSA store stream."""
        self._ensure_streams()
        return self._store_stream

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_musa: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseMUSAConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model
                configuration.
            use_musa: Whether to use MUSA intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemLayerwiseMUSAConnector.
        """
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_musa=use_musa,
            chunk_size=metadata.kv_shape[2],
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _validate_format_transition(
        self, mem: MemoryObj, target_fmt: MemoryFormat
    ) -> None:
        current_fmt = mem.metadata.fmt
        if (current_fmt, target_fmt) not in ALLOWED_FORMAT_TRANSITIONS:
            raise ValueError(
                f"Invalid KV format transition: {current_fmt} -> {target_fmt}"
            )

    def _lazy_initialize_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        if self.use_musa and self.gpu_buffer_allocator is None:
            layer0 = kv_caches[0]
            derived_bytes = layer0.numel() * layer0.element_size()
            staging_bytes = int(
                os.getenv("LMCACHE_GPU_STAGING_BUFFER_BYTES", derived_bytes)
            )
            logger.info(
                "Initializing MUSA staging buffer (derived=%d bytes, final=%d bytes)",
                derived_bytes,
                staging_bytes,
            )
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                size=staging_bytes, device=self.device
            )

    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        raise NotImplementedError("Layerwise uses batched_to_gpu(generator).")

    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        raise NotImplementedError("Layerwise uses batched_from_gpu(generator).")

    def _batched_to_gpu_gen(
        self, starts: List[int], ends: List[int], **kwargs: Any
    ) -> Generator[Any, Any, None]:
        """Generator: CPU token2d -> (optional staging) -> MUSA paged KV."""
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        def _ensure_musa(t: torch.Tensor) -> torch.Tensor:
            if t.device != self.device:
                return t.to(self.device, non_blocking=True)
            return t

        def _ensure_musa_memory(mem: MemoryObj) -> torch.Tensor:
            assert mem.tensor is not None
            return _to_musa_at_pin_boundaries(mem.tensor, mem, self.device)

        slot_mapping_chunks = [
            slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
        ]
        if not slot_mapping_chunks:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.musa.current_stream().wait_stream(self.load_stream)  # type: ignore[attr-defined]
            yield
            return

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        slot_mapping_full = _ensure_musa(slot_mapping_full)

        num_tokens = int(slot_mapping_full.numel())
        if num_tokens <= 0:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.musa.current_stream().wait_stream(self.load_stream)  # type: ignore[attr-defined]
            yield
            return

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_musa:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None or tmp_gpu_buffer_obj.tensor is None:
                raise RuntimeError(
                    "Failed to allocate MUSA staging buffer for batched_to_gpu."
                )

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield

                if sync:
                    current_stream.wait_stream(self.load_stream)

                with torch.musa.stream(self.load_stream):  # type: ignore[attr-defined]
                    dst_layer = self.kvcaches[layer_id]
                    if self.use_mla:
                        dst_flat = cast(
                            torch.Tensor,
                            _get_head_size_view(dst_layer, use_mla=True),
                        )
                    else:
                        dst_k_flat, dst_v_flat = _get_head_size_view(  # type: ignore[misc]
                            dst_layer, use_mla=False
                        )

                    cursor = 0

                    if self.use_musa:
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
                            src = _ensure_musa_memory(mem)
                            staged[cursor : cursor + n].copy_(src, non_blocking=True)
                            cursor += n

                        sl = _ensure_musa(slot_mapping_full)

                        if self.use_mla:
                            staged_dev = _ensure_musa(staged)
                            if staged_dev.dim() == 2:
                                dst_flat.index_copy_(0, sl, staged_dev)
                            elif staged_dev.dim() == 3 and staged_dev.shape[0] == 1:
                                dst_flat.index_copy_(0, sl, staged_dev[0])
                            else:
                                raise ValueError(
                                    f"Unexpected MLA staged tensor: {staged_dev.shape}"
                                )
                        else:
                            k_tok, v_tok = _split_token2d_kv(staged)
                            k_tok = _ensure_musa(k_tok)
                            v_tok = _ensure_musa(v_tok)

                            if (
                                k_tok.dim() == 2
                                and dst_k_flat.dim() == 3
                                and k_tok.shape[1]
                                == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                            ):
                                k_tok = k_tok.reshape(
                                    k_tok.shape[0],
                                    dst_k_flat.shape[1],
                                    dst_k_flat.shape[2],
                                )
                            if (
                                v_tok.dim() == 2
                                and dst_v_flat.dim() == 3
                                and v_tok.shape[1]
                                == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                            ):
                                v_tok = v_tok.reshape(
                                    v_tok.shape[0],
                                    dst_v_flat.shape[1],
                                    dst_v_flat.shape[2],
                                )

                            dst_k_flat.index_copy_(0, sl, k_tok)
                            dst_v_flat.index_copy_(0, sl, v_tok)

                    else:
                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue
                            src = _ensure_musa_memory(mem)
                            sl = slot_mapping_full[cursor : cursor + n]
                            sl = _ensure_musa(sl)
                            cursor += n

                            if self.use_mla:
                                if src.dim() == 2:
                                    dst_flat.index_copy_(0, sl, src)
                                elif src.dim() == 3 and src.shape[0] == 1:
                                    dst_flat.index_copy_(0, sl, src[0])
                                else:
                                    raise ValueError(
                                        f"Unexpected MLA token tensor: {src.shape}"
                                    )
                            else:
                                k_tok, v_tok = _split_token2d_kv(src)
                                k_tok = _ensure_musa(k_tok)
                                v_tok = _ensure_musa(v_tok)

                                if (
                                    k_tok.dim() == 2
                                    and dst_k_flat.dim() == 3
                                    and k_tok.shape[1]
                                    == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                                ):
                                    k_tok = k_tok.reshape(
                                        k_tok.shape[0],
                                        dst_k_flat.shape[1],
                                        dst_k_flat.shape[2],
                                    )
                                if (
                                    v_tok.dim() == 2
                                    and dst_v_flat.dim() == 3
                                    and v_tok.shape[1]
                                    == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                                ):
                                    v_tok = v_tok.reshape(
                                        v_tok.shape[0],
                                        dst_v_flat.shape[1],
                                        dst_v_flat.shape[2],
                                    )

                                dst_k_flat.index_copy_(0, sl, k_tok)
                                dst_v_flat.index_copy_(0, sl, v_tok)

            yield

            if sync:
                current_stream.wait_stream(self.load_stream)
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs: Any,
    ) -> Generator[Any, Any, None]:
        """Generator: MUSA paged KV -> CPU token2d (per layer)."""
        typed_memory_objs = cast(List[List[MemoryObj]], memory_objs)
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        slot_mapping_on_device = slot_mapping.to(self.device)

        for layer_id in range(self.num_layers):
            mem_layer = typed_memory_objs[layer_id]

            with torch.musa.stream(self.store_stream):  # type: ignore[attr-defined]
                self.store_stream.wait_stream(current_stream)

                src_layer = self.kvcaches[layer_id]

                if self.use_mla:
                    src_flat = cast(
                        torch.Tensor,
                        _get_head_size_view(src_layer, use_mla=True),
                    )
                    for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                        assert mem.tensor is not None
                        sl = slot_mapping_on_device[s:e]
                        gathered = src_flat.index_select(0, sl)
                        if isinstance(mem.parent(), LazyMemoryAllocator):
                            _copy_tensor_at_pin_boundaries(mem.tensor, gathered, mem)
                        else:
                            mem.tensor.copy_(
                                gathered.to(mem.tensor.device),
                                non_blocking=True,
                            )

                    target_fmt = MemoryFormat.KV_MLA_FMT
                    for mem in mem_layer:
                        self._validate_format_transition(mem, target_fmt)
                        mem.metadata.fmt = target_fmt
                else:
                    src_k_flat, src_v_flat = _get_head_size_view(
                        src_layer, use_mla=False
                    )
                    for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                        assert mem.tensor is not None
                        sl = slot_mapping_on_device[s:e]
                        k = src_k_flat.index_select(0, sl)
                        v = src_v_flat.index_select(0, sl)

                        if isinstance(mem.parent(), LazyMemoryAllocator):
                            if mem.tensor.shape[0] == 2:
                                gathered = torch.stack((k, v))
                            elif mem.tensor.dim() >= 2 and mem.tensor.shape[1] == 2:
                                gathered = torch.stack((k, v), dim=1)
                            else:
                                raise ValueError(
                                    f"Unrecognized KV tensor layout: {mem.tensor.shape}"
                                )
                            _copy_tensor_at_pin_boundaries(mem.tensor, gathered, mem)
                        elif mem.tensor.shape[0] == 2:
                            mem.tensor[0].copy_(
                                k.to(mem.tensor.device), non_blocking=True
                            )
                            mem.tensor[1].copy_(
                                v.to(mem.tensor.device), non_blocking=True
                            )
                        elif mem.tensor.dim() >= 2 and mem.tensor.shape[1] == 2:
                            mem.tensor[:, 0].copy_(
                                k.to(mem.tensor.device), non_blocking=True
                            )
                            mem.tensor[:, 1].copy_(
                                v.to(mem.tensor.device), non_blocking=True
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized KV tensor layout: {mem.tensor.shape}"
                            )

            if sync:
                self.store_stream.synchronize()
            yield

        yield

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Generator[Any, Any, None]:
        return self._batched_to_gpu_gen(starts=starts or [], ends=ends or [], **kwargs)

    def _ensure_streams(self) -> None:
        """Lazily create MUSA streams on first transfer."""
        if self._load_stream is None:
            self._load_stream = torch.musa.Stream()  # type: ignore[attr-defined]
            self._store_stream = torch.musa.Stream()  # type: ignore[attr-defined]

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data for a single layer.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data for one layer.
        """
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])


def _layer_views(
    kvcaches: DiscoverableKVCache,
    *,
    engine_kv_format: lmcache_native.EngineKVFormat,
    hidden_dim_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Create token-major views of every SGLang KV-cache layer."""
    if engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NBBS_ONE_HS:
        layers = cast(list[torch.Tensor], kvcaches)
        return [(tensor.view(-1, hidden_dim_size), None) for tensor in layers]
    if engine_kv_format == lmcache_native.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        key_layers, value_layers = cast(list[list[torch.Tensor]], kvcaches)
        return [
            (
                key_tensor.view(-1, hidden_dim_size),
                value_tensor.view(-1, hidden_dim_size),
            )
            for key_tensor, value_tensor in zip(key_layers, value_layers, strict=True)
        ]
    raise ValueError(
        "SGLang MUSA in-process transfer supports only "
        "TWO_X_NL_X_NBBS_NH_HS and NL_X_NBBS_ONE_HS; "
        f"got {engine_kv_format!r}"
    )


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to the MUSA KV-cache device when needed."""
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _slot_slice(
    slot_mapping: torch.Tensor,
    start: int,
    end: int,
    device: torch.device,
    *,
    offset: int = 0,
) -> torch.Tensor:
    """Return a device-local slice from a possibly partial slot mapping."""
    mapping_start = start - offset
    mapping_end = end - offset
    if mapping_start < 0:
        raise ValueError("start must not precede the SGLang slot-map offset")
    return slot_mapping[mapping_start:mapping_end].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )


def _split_flat_mha_kvcaches(
    kvcaches: DiscoverableKVCache,
    *,
    num_layers: int,
) -> DiscoverableKVCache:
    """Convert the legacy flat SGLang MHA list into nested K/V lists.

    The in-process SGLang adapter passes ``[K0, ..., Kn, V0, ..., Vn]``,
    while the shared format detector expects ``[[K0, ..., Kn],
    [V0, ..., Vn]]``. Keep this compatibility conversion local to the MUSA
    connector so enabling MUSA does not change format detection for CUDA,
    XPU, or multiprocess users.
    """
    if (
        isinstance(kvcaches, list)
        and len(kvcaches) == 2 * num_layers
        and all(isinstance(tensor, torch.Tensor) for tensor in kvcaches)
    ):
        return [kvcaches[:num_layers], kvcaches[num_layers:]]
    return kvcaches


def _prepare_kvcaches(
    kvcaches: DiscoverableKVCache,
    *,
    hidden_dim_size: int,
    num_layers: int,
    use_mla: bool,
) -> tuple[
    torch.device,
    list[tuple[torch.Tensor, torch.Tensor | None]],
]:
    """Normalize, validate, and expose token-major SGLang layer views."""
    if not use_mla:
        kvcaches = _split_flat_mha_kvcaches(
            kvcaches,
            num_layers=num_layers,
        )
    engine_kv_format, normalized = normalize_kv_and_discover_format(
        kvcaches,
        EngineType.SGLANG,
    )
    expected_format = (
        lmcache_native.EngineKVFormat.NL_X_NBBS_ONE_HS
        if use_mla
        else lmcache_native.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS
    )
    if engine_kv_format != expected_format:
        raise ValueError(
            f"SGLang MUSA expected {expected_format!r}, got {engine_kv_format!r}"
        )
    discovered_layers = get_num_layers(normalized, engine_kv_format)
    if discovered_layers != num_layers:
        raise ValueError(
            f"Expected {num_layers} SGLang layers, got {discovered_layers}"
        )
    device = get_device(normalized)
    if device.type != "musa":
        raise ValueError(
            "SGLang MUSA connectors require MUSA KV-cache tensors; "
            f"got device type {device.type!r}"
        )
    return device, _layer_views(
        normalized,
        engine_kv_format=engine_kv_format,
        hidden_dim_size=hidden_dim_size,
    )


class SGLangMUSAConnector(GPUConnectorInterface):
    """Transfer complete SGLang KV chunks with pure TorchMUSA operations.

    MHA caches use separate K/V layer lists whose tensors have shape
    ``[page_buffer_size, num_heads, head_size]``. MLA caches use one tensor
    per layer. The connector flattens only the head dimensions and gathers or
    scatters rows using SGLang's slot mapping; it does not depend on CUDA
    transfer kernels. The SGLang adapter selects this MLA path from the
    model's ``attention_arch`` and passes the cache pool's actual latent width.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        *,
        device: torch.device,
        use_mla: bool = False,
        **_: Any,
    ) -> None:
        """Create an SGLang MUSA connector.

        Args:
            hidden_dim_size: Flattened KV head width.
            num_layers: Number of model KV-cache layers.
            use_gpu: Whether LMCache requests a device intermediate buffer.
                The pure-torch path moves each source tensor directly and does
                not require a persistent staging allocation.
            device: MUSA device that owns the SGLang KV cache.
            use_mla: Whether the cache uses the single-tensor MLA layout.
            **_: Ignored compatibility arguments such as ``chunk_size`` and
                ``dtype``.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.device = device
        self.use_mla = use_mla

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> "SGLangMUSAConnector":
        """Create a connector from LMCache engine metadata.

        Args:
            metadata: Engine metadata describing the SGLang KV layout.
            use_gpu: Whether LMCache requests a device intermediate buffer.
            device: MUSA device that owns the SGLang KV cache.
            **kwargs: Additional forward-compatible connector options.

        Returns:
            A configured :class:`SGLangMUSAConnector`.

        Raises:
            ValueError: If ``device`` is not provided.
        """
        if device is None:
            raise ValueError("device must be provided for SGLang on MUSA")
        num_layers, _, _, num_kv_heads, head_size = metadata.kv_shape
        return cls(
            hidden_dim_size=num_kv_heads * head_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            device=device,
            use_mla=metadata.use_mla,
            **kwargs,
        )

    def to_gpu(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        """Scatter a complete LMCache chunk into SGLang's MUSA KV cache.

        Args:
            memory_obj: LMCache memory object containing the chunk.
            start: Start token offset in the request.
            end: End token offset in the request.
            **kwargs: Must contain ``kvcaches`` and ``slot_mapping``; may
                contain SGLang's ``offset`` for a partial slot mapping.

        Raises:
            ValueError: If required inputs, formats, or shapes are invalid.
        """
        if memory_obj.tensor is None:
            raise ValueError("memory_obj must contain a tensor")
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")
        self._validate_memory_format(memory_obj)

        kvcaches = cast(DiscoverableKVCache, kwargs["kvcaches"])
        slot_mapping = cast(torch.Tensor, kwargs["slot_mapping"])
        offset = int(kwargs.get("offset", 0))

        device, views = _prepare_kvcaches(
            kvcaches,
            hidden_dim_size=self.hidden_dim_size,
            num_layers=self.num_layers,
            use_mla=self.use_mla,
        )
        slots = _slot_slice(slot_mapping, start, end, device, offset=offset)
        self._validate_memory_shape(memory_obj.tensor, int(slots.numel()))

        if self.use_mla:
            for layer_id, (key_view, _) in enumerate(views):
                source = _to_device(memory_obj.tensor[layer_id], device)
                key_view.index_copy_(0, slots, source)
            return

        for layer_id, (key_view, value_view) in enumerate(views):
            if value_view is None:
                raise ValueError("SGLang MHA cache is missing a value layer")
            key_source = _to_device(memory_obj.tensor[0, layer_id], device)
            value_source = _to_device(memory_obj.tensor[1, layer_id], device)
            key_view.index_copy_(0, slots, key_source)
            value_view.index_copy_(0, slots, value_source)

    def from_gpu(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        """Gather a complete SGLang MUSA KV chunk into LMCache memory.

        Args:
            memory_obj: Destination LMCache memory object.
            start: Start token offset in the full slot mapping.
            end: End token offset in the full slot mapping.
            **kwargs: Must contain ``kvcaches`` and ``slot_mapping``.

        Raises:
            ValueError: If required inputs, formats, or shapes are invalid.
        """
        if memory_obj.tensor is None:
            raise ValueError("memory_obj must contain a tensor")
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")

        kvcaches = cast(DiscoverableKVCache, kwargs["kvcaches"])
        slot_mapping = cast(torch.Tensor, kwargs["slot_mapping"])
        offset = int(kwargs.get("offset", 0))
        device, views = _prepare_kvcaches(
            kvcaches,
            hidden_dim_size=self.hidden_dim_size,
            num_layers=self.num_layers,
            use_mla=self.use_mla,
        )
        slots = _slot_slice(slot_mapping, start, end, device, offset=offset)
        self._validate_memory_shape(memory_obj.tensor, int(slots.numel()))

        if self.use_mla:
            gathered = torch.stack(
                [key_view.index_select(0, slots) for key_view, _ in views]
            )
            memory_obj.tensor.copy_(gathered, non_blocking=True)
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
        else:
            keys = torch.stack(
                [key_view.index_select(0, slots) for key_view, _ in views]
            )
            values = torch.stack(
                [
                    value_view.index_select(0, slots)
                    for _, value_view in views
                    if value_view is not None
                ]
            )
            if values.shape[0] != self.num_layers:
                raise ValueError("SGLang MHA cache is missing a value layer")
            memory_obj.tensor.copy_(
                torch.stack((keys, values)),
                non_blocking=True,
            )
            memory_obj.metadata.fmt = MemoryFormat.KV_2LTD

        if memory_obj.tensor.device.type != "musa":
            torch.musa.synchronize()  # type: ignore[attr-defined]

    def batched_to_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj] | list[int] | None = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Scatter several LMCache chunks into SGLang's MUSA KV cache.

        Args:
            memory_objs: Flat list of complete-chunk memory objects.
            starts: Per-chunk start offsets.
            ends: Per-chunk end offsets.
            **kwargs: Forwarded to :meth:`to_gpu`.

        Raises:
            ValueError: If any required batch argument is missing.
        """
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends must be provided")
        for memory_obj, start, end in zip(
            cast(list[MemoryObj], memory_objs), starts, ends, strict=True
        ):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> None:
        """Gather several SGLang MUSA KV chunks into LMCache memory.

        Args:
            memory_objs: Flat list of complete-chunk memory objects.
            starts: Per-chunk start offsets.
            ends: Per-chunk end offsets.
            **kwargs: Forwarded to :meth:`from_gpu`.
        """
        for memory_obj, start, end in zip(
            cast(list[MemoryObj], memory_objs), starts, ends, strict=True
        ):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return the LMCache memory-object shape for ``num_tokens``.

        Args:
            num_tokens: Number of tokens in the memory object.

        Returns:
            MLA shape ``[layers, tokens, hidden]`` or MHA shape
            ``[2, layers, tokens, hidden]``.
        """
        if self.use_mla:
            return torch.Size([self.num_layers, num_tokens, self.hidden_dim_size])
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        expected = MemoryFormat.KV_MLA_FMT if self.use_mla else MemoryFormat.KV_2LTD
        if memory_obj.metadata.fmt != expected:
            raise ValueError(
                f"SGLang MUSA expected memory format {expected}, "
                f"got {memory_obj.metadata.fmt}"
            )

    def _validate_memory_shape(
        self,
        tensor: torch.Tensor,
        num_tokens: int,
    ) -> None:
        expected = self.get_shape(num_tokens)
        if tensor.shape != expected:
            raise ValueError(
                f"SGLang MUSA expected memory shape {tuple(expected)}, "
                f"got {tuple(tensor.shape)}"
            )


class SGLangLayerwiseMUSAConnector(GPUConnectorInterface):
    """Transfer MHA SGLang KV cache one layer at a time on MUSA.

    The connector follows LMCache's layerwise generator protocol and uses
    token-major memory objects shaped ``[tokens, 2, hidden]``. Operations run
    on TorchMUSA's current stream, preserving ordering with SGLang attention
    without relying on CUDA-only transfer kernels.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        *,
        device: torch.device,
        use_mla: bool = False,
        **_: Any,
    ) -> None:
        """Create a layerwise SGLang MUSA connector.

        Args:
            hidden_dim_size: Flattened KV head width.
            num_layers: Number of model KV-cache layers.
            use_gpu: Whether LMCache requests a device intermediate buffer.
            device: MUSA device that owns the SGLang KV cache.
            use_mla: Whether the model uses MLA.
            **_: Ignored forward-compatible connector options.

        Raises:
            NotImplementedError: If MLA layerwise mode is requested.
        """
        if use_mla:
            raise NotImplementedError(
                "Layerwise SGLang on MUSA does not support MLA; set use_layerwise=False"
            )
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.device = device
        self.use_mla = False

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> "SGLangLayerwiseMUSAConnector":
        """Create a layerwise connector from LMCache engine metadata.

        Args:
            metadata: Engine metadata describing the SGLang KV layout.
            use_gpu: Whether LMCache requests a device intermediate buffer.
            device: MUSA device that owns the SGLang KV cache.
            **kwargs: Additional forward-compatible connector options.

        Returns:
            A configured :class:`SGLangLayerwiseMUSAConnector`.

        Raises:
            ValueError: If ``device`` is not provided.
            NotImplementedError: If metadata requests MLA layerwise mode.
        """
        if device is None:
            raise ValueError("device must be provided for SGLang on MUSA")
        num_layers, _, _, num_kv_heads, head_size = metadata.kv_shape
        return cls(
            hidden_dim_size=num_kv_heads * head_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            device=device,
            use_mla=metadata.use_mla,
            **kwargs,
        )

    def to_gpu(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        """Reject non-generator use of the layerwise connector.

        Args:
            memory_obj: Unused memory object.
            start: Unused start offset.
            end: Unused end offset.
            **kwargs: Unused connector options.

        Raises:
            NotImplementedError: Always; use :meth:`batched_to_gpu`.
        """
        raise NotImplementedError("Layerwise SGLang uses batched_to_gpu")

    def from_gpu(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        """Reject non-generator use of the layerwise connector.

        Args:
            memory_obj: Unused memory object.
            start: Unused start offset.
            end: Unused end offset.
            **kwargs: Unused connector options.

        Raises:
            NotImplementedError: Always; use :meth:`batched_from_gpu`.
        """
        raise NotImplementedError("Layerwise SGLang uses batched_from_gpu")

    def batched_to_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj] | list[int] | None = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs: Any,
    ) -> Generator[None, list[MemoryObj], None]:
        """Yield a consumer that scatters one layer per ``send`` call.

        Args:
            memory_objs: Positional compatibility slot containing start offsets
                when LMCache calls ``batched_to_gpu(starts, ends)``.
            starts: Positional compatibility slot containing end offsets, or
                explicit start offsets when ``ends`` is also provided.
            ends: Explicit end offsets for interface-style keyword calls.
            **kwargs: Must contain ``kvcaches`` and ``slot_mapping``.

        Yields:
            ``None`` before each layer and once after the final layer. Send a
            list of that layer's memory objects into each layer yield.

        Raises:
            ValueError: If required inputs, formats, or shapes are invalid.
        """
        if ends is None:
            if memory_objs is None or starts is None:
                raise ValueError("starts and ends must be provided")
            token_starts = cast(list[int], memory_objs)
            token_ends = starts
        else:
            if starts is None:
                raise ValueError("starts and ends must be provided")
            token_starts = starts
            token_ends = ends

        slot_mapping, offset, device, views = self._transfer_inputs(kwargs)

        for layer_id, (key_view, value_view) in enumerate(views):
            if value_view is None:
                raise ValueError("SGLang MHA cache is missing a value layer")
            memory_objs_layer = yield
            for start, end, memory_obj in zip(
                token_starts, token_ends, memory_objs_layer, strict=True
            ):
                if memory_obj.tensor is None:
                    raise ValueError("memory_obj must contain a tensor")
                expected = self.get_shape(end - start)
                if memory_obj.tensor.shape != expected:
                    raise ValueError(
                        f"Layer {layer_id} expected memory shape "
                        f"{tuple(expected)}, got {tuple(memory_obj.tensor.shape)}"
                    )
                if memory_obj.metadata.fmt != MemoryFormat.KV_T2D:
                    raise ValueError(
                        "Layerwise SGLang on MUSA requires KV_T2D memory objects"
                    )
                slots = _slot_slice(
                    slot_mapping,
                    start,
                    end,
                    device,
                    offset=offset,
                )
                source = _to_device(memory_obj.tensor, device)
                key_view.index_copy_(0, slots, source[:, 0])
                value_view.index_copy_(0, slots, source[:, 1])
        yield

    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        """Gather one SGLang MUSA KV layer before each generator yield.

        Args:
            memory_objs: Layer-major lists of destination memory objects.
            starts: Per-chunk start offsets.
            ends: Per-chunk end offsets.
            **kwargs: Must contain ``kvcaches`` and ``slot_mapping``.

        Yields:
            ``None`` after each layer and once after the final layer.

        Raises:
            ValueError: If required inputs, formats, or shapes are invalid.
        """
        layer_memory_objs = cast(list[list[MemoryObj]], memory_objs)
        slot_mapping, offset, device, views = self._transfer_inputs(kwargs)
        if len(layer_memory_objs) != self.num_layers:
            raise ValueError(
                f"Expected memory objects for {self.num_layers} layers, "
                f"got {len(layer_memory_objs)}"
            )

        for layer_id, (key_view, value_view) in enumerate(views):
            if value_view is None:
                raise ValueError("SGLang MHA cache is missing a value layer")
            copied_to_host = False
            for start, end, memory_obj in zip(
                starts, ends, layer_memory_objs[layer_id], strict=True
            ):
                if memory_obj.tensor is None:
                    raise ValueError("memory_obj must contain a tensor")
                expected = self.get_shape(end - start)
                if memory_obj.tensor.shape != expected:
                    raise ValueError(
                        f"Layer {layer_id} expected memory shape "
                        f"{tuple(expected)}, got {tuple(memory_obj.tensor.shape)}"
                    )
                slots = _slot_slice(
                    slot_mapping,
                    start,
                    end,
                    device,
                    offset=offset,
                )
                keys = key_view.index_select(0, slots)
                values = value_view.index_select(0, slots)
                memory_obj.tensor.copy_(
                    torch.stack((keys, values), dim=1),
                    non_blocking=True,
                )
                memory_obj.metadata.fmt = MemoryFormat.KV_T2D
                copied_to_host = copied_to_host or (
                    memory_obj.tensor.device.type != "musa"
                )
            if copied_to_host:
                torch.musa.synchronize()  # type: ignore[attr-defined]
            yield
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return the per-layer token-major memory shape.

        Args:
            num_tokens: Number of tokens in the memory object.

        Returns:
            Shape ``[tokens, 2, hidden]``.
        """
        return torch.Size([num_tokens, 2, self.hidden_dim_size])

    def _transfer_inputs(
        self,
        kwargs: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        int,
        torch.device,
        list[tuple[torch.Tensor, torch.Tensor | None]],
    ]:
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")
        kvcaches = cast(DiscoverableKVCache, kwargs["kvcaches"])
        slot_mapping = cast(torch.Tensor, kwargs["slot_mapping"])
        offset = int(kwargs.get("offset", 0))
        device, views = _prepare_kvcaches(
            kvcaches,
            hidden_dim_size=self.hidden_dim_size,
            num_layers=self.num_layers,
            use_mla=False,
        )
        return slot_mapping, offset, device, views
