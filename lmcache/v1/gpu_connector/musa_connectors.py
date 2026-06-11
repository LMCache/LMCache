# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, List, Optional, Union, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_block_size,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


_SUPPORTED_MUSA_KV_FORMATS = (
    lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
    lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
)


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

    Other vLLM layouts, including flash-infer, HND, cross-layer, layerwise,
    connector v3, and MP GPU-transfer kernel layouts, are not implemented by
    this connector.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the non-layerwise MUSA connector.

        Args:
            use_gpu: Whether to use a GPU intermediate buffer.
            **kwargs: Reserved for API compatibility with other connectors.
        """
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
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
        slices = slot_mapping[transfer_start:end].to(
            device=self.device, dtype=torch.long, non_blocking=True
        )
        if slices.numel() == 0:
            return

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(self.device, non_blocking=True)
            total_blocks = self.num_blocks * self.block_size
            for i, kvcache in enumerate(self.kvcaches):
                kvcache.view(total_blocks, self.head_size).index_copy_(
                    0, slices, tmp[i, skip_prefix_n_tokens:]
                )
        else:
            tmp_k = memory_obj.tensor[0].to(self.device, non_blocking=True)
            tmp_v = memory_obj.tensor[1].to(self.device, non_blocking=True)
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
        memory_obj.tensor.copy_(tmp, non_blocking=True)

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
        """Store multiple memory objects into MUSA paged KV caches.

        Args:
            memory_objs: Memory objects containing KV data.
            starts: Start offsets for each memory object.
            ends: End offsets for each memory object.
            **kwargs: Arguments forwarded to :meth:`to_gpu`.
        """
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

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]) -> None:
        """Initialize attributes from KV caches using utils functions.

        Args:
            kv_caches: The KV cache tensors from which to discover layout.
        """
        if self._attributes_initialized:
            return

        self.device = kv_caches[0].device
        assert self.device.type == "musa", "The device should be MUSA."

        self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM
        )
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
            "MUSA: attributes initialized - format: %s, "
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

    def _validate_supported_kv_format(self) -> None:
        """Reject KV layouts that this torch-based MUSA path cannot index."""
        if self.gpu_kv_format in _SUPPORTED_MUSA_KV_FORMATS:
            return

        supported = (
            "NL x [2, NB, BS, NH, HS] for non-MLA vLLM flash attention, "
            "or NL x [NB, BS, HS] for vLLM MLA"
        )
        raise ValueError(
            "VLLMPagedMemMUSAConnectorV2 supports only "
            f"{supported}; got {self.gpu_kv_format}."
        )
