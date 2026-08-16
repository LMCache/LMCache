# SPDX-License-Identifier: Apache-2.0
"""RBLN (Rebellions NPU) paged-memory connector for vLLM.

vLLM-RBLN hands LMCache one 6-D tensor per layer,
``[2, num_blocks, num_kv_heads, 1, block_size, head_size]`` -- the HND layout
with a singleton axis between heads and block tokens that the RBLN attention
backend requires.

That layout is its own ``EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS``, so this
connector hands the caches to discovery exactly as vLLM-RBLN registered them
and needs no layout hint -- the format is HND by definition.

Axis 3 is always 1, so squeezing it is a free view onto identical bytes. This
connector applies :func:`squeeze_singleton_axis` only where it indexes those
bytes, which is the same split the multiprocess path uses in
:class:`~lmcache.v1.platform.rbln.device_ops.RblnDeviceOps`: detection sees the
registered layout, and the singleton is dropped at the point of transfer.

The connector produces and consumes ``KV_2LTD`` memory objects
(``[2, num_layers, num_tokens, num_heads * head_size]``), the same contract
every other vLLM connector uses.

Because the layout is HND, tokens are *not* contiguous within a layer: the
head axis sits between blocks and block tokens. The flat
``view(num_blocks * block_size, hidden_dim)`` reshape the NHD connectors use
would therefore address the wrong slots. Instead each transfer resolves the
slot mapping into ``(block, offset)`` pairs and uses advanced indexing, which
touches only the tokens in the request rather than materialising a permuted
copy of the whole KV cache.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, List, Optional, Union, cast

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
)
from lmcache.v1.gpu_connector.utils import (
    get_block_size,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    normalize_kv_and_discover_format,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.platform.rbln.kv_layout import squeeze_singleton_axis

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class VLLMPagedMemRBLNConnectorV2(GPUConnectorInterface):
    """Move KV between RBLN paged memory and ``KV_2LTD`` memory objects.

    The KV cache is a ``list`` of per-layer tensors, each
    ``[2, num_blocks, num_kv_heads, 1, block_size, head_size]``.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs: object,
    ) -> None:
        """Take from the model what the model knows; discover the rest.

        The split is the same one :class:`VLLMPagedMemGPUConnectorV2` makes.
        ``num_layers`` and ``hidden_dim_size`` come from the metadata because
        they are properties of the *model*, and they are all
        :meth:`get_shape` needs -- so a memory object can be sized before any
        cache is registered. Everything else (the KV format, ``num_blocks``,
        ``block_size``) is a property of the *allocated cache*, so it is
        discovered from the tensors on first use.

        Args:
            hidden_dim_size: ``num_kv_heads * head_size``.
            num_layers: Layers the model has.
            use_gpu: Whether an intermediate device buffer is in use.
            **kwargs: Ignored; accepted for parity with the other connectors.
        """
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu
        self.num_layers = num_layers
        self.hidden_dim_size = hidden_dim_size

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional["LayoutHints"] = None,
    ) -> "VLLMPagedMemRBLNConnectorV2":
        """Create a connector from :class:`LMCacheMetadata`.

        Args:
            metadata: Engine metadata carrying the model configuration.
                ``kv_shape`` is ``(num_layer, kv, chunk_size, num_kv_head,
                head_size)``.
            use_gpu: Whether an intermediate device buffer is in use.
            device: Device the connector runs on. Accepted for signature
                parity with the other connectors; the device is taken from the
                registered KV caches instead.
            layout_hints: Engine layout hints. Accepted for the same parity and
                never read: the RBLN cache is always HND, which the format
                itself records.

        Returns:
            VLLMPagedMemRBLNConnectorV2: A new connector.
        """
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        return cls(
            hidden_dim_size=num_kv_head * head_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def initialize_kvcaches_ptr(self, **kwargs: object) -> None:
        """Record the engine's KV caches when the caller supplies them."""
        kvcaches = kwargs.get("kvcaches")
        if kvcaches is not None:
            self.kvcaches = kvcaches  # type: ignore[assignment]

    def register_kv_caches(self, kv_caches: List[torch.Tensor]) -> None:
        """Register the per-layer KV caches and discover their geometry.

        Transfers do this lazily on first use, so registering is optional --
        it just moves the discovery, and the check against the metadata that
        comes with it, to a point the caller controls.

        Args:
            kv_caches: Per-layer tensors shaped ``[2, NB, NH, 1, BS, HS]``, in
                layer order.

        Raises:
            ValueError: If ``kv_caches`` is empty, or a tensor is not 6-D with
                a singleton at axis 3.
        """
        if not kv_caches:
            raise ValueError("kv_caches must be non-empty")
        self.kvcaches = list(kv_caches)
        self._initialize_attributes(self.kvcaches)

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]) -> None:
        """Discover the KV geometry once, from the registered 6-D caches."""
        if self._attributes_initialized or not kv_caches:
            return

        self.device = kv_caches[0].device

        self.engine_kv_format, discovered = normalize_kv_and_discover_format(
            cast(DiscoverableKVCache, list(kv_caches)),
            EngineType.VLLM,
        )
        self.num_blocks = get_num_blocks(discovered, self.engine_kv_format)
        self.block_size = get_block_size(discovered, self.engine_kv_format)
        self.head_size = get_head_size(discovered, self.engine_kv_format)
        self.num_heads = get_num_heads(discovered, self.engine_kv_format)
        self.dtype = get_dtype(discovered, self.engine_kv_format)

        # The two the metadata already fixed are checked, not overwritten: a
        # cache that disagrees with the model it was built for would otherwise
        # be transferred against get_shape()'s sizing and read past its end.
        for name, from_model, from_cache in (
            (
                "num_layers",
                self.num_layers,
                get_num_layers(discovered, self.engine_kv_format),
            ),
            (
                "hidden_dim_size",
                self.hidden_dim_size,
                get_hidden_dim_size(discovered, self.engine_kv_format),
            ),
        ):
            if from_model != from_cache:
                raise ValueError(
                    f"registered KV caches disagree with the engine metadata on "
                    f"{name}: metadata says {from_model}, the caches say "
                    f"{from_cache}"
                )

        self._attributes_initialized = True
        logger.info(
            "RBLN: attributes initialized - format: %s, num_layers: %d, "
            "num_blocks: %d, block_size: %d, hidden_dim_size: %d, "
            "head_size: %d, num_heads: %d, dtype: %s",
            self.engine_kv_format,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.hidden_dim_size,
            self.head_size,
            self.num_heads,
            self.dtype,
        )

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    def _prepare(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: object
    ) -> tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Validate a transfer request and resolve its slot mapping.

        Args:
            memory_obj: Memory object being read or written.
            start: First token index of the slice.
            end: One past the last token index of the slice.
            **kwargs: Must carry ``slot_mapping``; may carry ``kvcaches``.

        Returns:
            tuple: Squeezed per-layer views, per-token block indices, and
            per-token offsets within a block.

        Raises:
            ValueError: If ``slot_mapping`` or the KV caches are missing, or
                the memory object is not ``KV_2LTD``.
        """
        if memory_obj.tensor is None:
            raise ValueError("memory_obj must carry a tensor")

        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise ValueError(
                "'kvcaches' must be provided in kwargs or registered beforehand"
            )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        slot_mapping = kwargs["slot_mapping"]
        if not isinstance(slot_mapping, torch.Tensor):
            raise ValueError("'slot_mapping' must be a torch.Tensor")
        slices = slot_mapping[start:end].to(dtype=torch.long)

        views = squeeze_singleton_axis(self.kvcaches)
        blocks = torch.div(slices, self.block_size, rounding_mode="floor")
        offsets = slices % self.block_size
        return views, blocks, offsets

    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: object
    ) -> None:
        """Gather the requested tokens out of RBLN paged memory.

        Args:
            memory_obj: Destination, filled in ``KV_2LTD`` layout.
            start: First token index of the slice.
            end: One past the last token index of the slice.
            **kwargs: Must carry ``slot_mapping``; may carry ``kvcaches``.
        """
        views, blocks, offsets = self._prepare(memory_obj, start, end, **kwargs)
        num_tokens = int(blocks.numel())

        assert memory_obj.tensor is not None
        for layer_idx, layer in enumerate(views):
            # Advanced indices separated by a slice put the gathered axis
            # first: [num_tokens, num_heads, head_size].
            k = layer[0][blocks, :, offsets, :]
            v = layer[1][blocks, :, offsets, :]
            memory_obj.tensor[0, layer_idx].copy_(
                k.reshape(num_tokens, self.hidden_dim_size), non_blocking=True
            )
            memory_obj.tensor[1, layer_idx].copy_(
                v.reshape(num_tokens, self.hidden_dim_size), non_blocking=True
            )
        torch_dev.synchronize()

    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: object
    ) -> None:
        """Scatter a ``KV_2LTD`` memory object back into RBLN paged memory.

        Args:
            memory_obj: Source, laid out as ``KV_2LTD``.
            start: First token index of the slice.
            end: One past the last token index of the slice.
            **kwargs: Must carry ``slot_mapping``; may carry ``kvcaches``.
        """
        views, blocks, offsets = self._prepare(memory_obj, start, end, **kwargs)
        num_tokens = int(blocks.numel())

        assert memory_obj.tensor is not None
        shape = (num_tokens, self.num_heads, self.head_size)
        for layer_idx, layer in enumerate(views):
            src_k = memory_obj.tensor[0, layer_idx].to(self.device).reshape(shape)
            src_v = memory_obj.tensor[1, layer_idx].to(self.device).reshape(shape)
            layer[0][blocks, :, offsets, :] = src_k
            layer[1][blocks, :, offsets, :] = src_v
        torch_dev.synchronize()

    @staticmethod
    def _unpack_batch(
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj], List[int], None],
        starts: Optional[List[int]],
        ends: Optional[List[int]],
    ) -> tuple[List[MemoryObj], List[int], List[int]]:
        """Narrow the loosely-typed batch arguments the interface declares.

        Args:
            memory_objs: Batch of memory objects.
            starts: Per-object slice starts.
            ends: Per-object slice ends.

        Returns:
            tuple: The batch as flat, non-optional lists.

        Raises:
            ValueError: If any argument is missing or not a flat
                ``list[MemoryObj]``.
        """
        if memory_objs is None or starts is None or ends is None:
            raise ValueError(
                "memory_objs, starts and ends are all required for batched "
                "RBLN transfers"
            )
        if any(isinstance(item, (list, int)) for item in memory_objs):
            raise ValueError(
                "VLLMPagedMemRBLNConnectorV2 expects a flat list[MemoryObj]; "
                "nested and pointer batches are not supported"
            )
        return cast(List[MemoryObj], memory_objs), starts, ends

    def batched_from_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Gather each ``(memory_obj, start, end)`` triple in turn."""
        objs, begins, finishes = self._unpack_batch(memory_objs, starts, ends)
        for memory_obj, start, end in zip(objs, begins, finishes, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Scatter each ``(memory_obj, start, end)`` triple in turn."""
        objs, begins, finishes = self._unpack_batch(memory_objs, starts, ends)
        for memory_obj, start, end in zip(objs, begins, finishes, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return the ``KV_2LTD`` memory-object shape for ``num_tokens``.

        Args:
            num_tokens: Tokens the memory object must hold.

        Returns:
            torch.Size: ``[2, num_layers, num_tokens, num_heads * head_size]``.
        """
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        """Reject memory objects that are not in ``KV_2LTD`` layout.

        Args:
            memory_obj: The memory object to check.

        Raises:
            ValueError: If the format is not ``KV_2LTD``.
        """
        if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
            raise ValueError(
                "The memory object should be in KV_2LTD format in order to be "
                "processed by VLLMPagedMemRBLNConnectorV2"
            )
