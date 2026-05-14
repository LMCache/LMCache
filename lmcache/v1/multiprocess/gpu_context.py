# SPDX-License-Identifier: Apache-2.0
"""
GPU Cache Context management for LMCache multiprocessing.

This module provides GPU-side KV cache management functionality, including:
- GPUCacheContext: Manages shape and pointers to vLLM GPU KV cache tensors
- Helper functions for tensor operations and key resolution
"""

# Standard
from typing import Any
import array

# Third Party
import cupy
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_attention_backend,
    get_concrete_gpu_kv_shape,
    get_device,
    get_dtype,
    get_gpu_kv_shape_description,
    get_group_data_ptrs,
    get_num_blocks,
    get_num_layers,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import (
    KVCache,
)

# Backend selection (c_ops when CUDA is available, otherwise a pure-Python
# fallback) is handled once in ``lmcache/__init__.py`` via ``_get_backend``,
# which aliases the chosen module as ``lmcache.c_ops`` in ``sys.modules``.
# Importing it here transparently works in both CUDA and CPU-only envs.
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    unwrapped_tensors = []
    for ipc_wrapper in kv_caches:
        tensor = ipc_wrapper.to_tensor()
        unwrapped_tensors.append(tensor)
    return unwrapped_tensors


def list_to_gpu_tensor(lis: list[int], device: torch.device) -> torch.Tensor:
    return torch.frombuffer(array.array("l", lis), dtype=torch.long).to(
        device, non_blocking=True
    )


class GPUCacheContext:
    """
    Manages the shape and pointers to vLLM GPU KV cache tensors.
    """

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_logical_chunk_size: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_type: EngineType = EngineType.VLLM,
    ):
        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        self.gpu_kv_format_, self.kv_caches_ = normalize_kv_and_discover_format(
            unwrapped,
            engine_type,
            layout_hints=layout_hints,
        )
        self.device_ = get_device(self.kv_caches_)
        self.is_mla_ = is_mla(self.gpu_kv_format_)
        self.num_layers_ = get_num_layers(self.kv_caches_, self.gpu_kv_format_)
        self.num_blocks_ = get_num_blocks(self.kv_caches_, self.gpu_kv_format_)
        self.lmcache_logical_chunk_size = lmcache_logical_chunk_size

        self.kv_layer_groups_manager_ = KVLayerGroupsManager(
            self.kv_caches_,
            gpu_kv_format=self.gpu_kv_format_,
            num_blocks=self.num_blocks_,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=lmcache_logical_chunk_size,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        for group in self.kv_layer_groups_manager_.kv_layer_groups:
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.gpu_kv_format_, group.layer_indices
            )
            self.group_kv_pointers_.append(list_to_gpu_tensor(ptrs, self.device_))

        # Pre-allocated GPU buffer for block IDs (up to 1M elements).
        # The caller copies block_ids into this buffer before launching the
        # block-level kernel. Single-thread assumption: no lock needed.
        _MAX_BLOCK_IDS = 1_000_000
        self.block_ids_buffer_ = torch.empty(
            _MAX_BLOCK_IDS, dtype=torch.long, device=self.device_
        )

        # Temporary GPU buffer for transfers — a single flat uint8 buffer
        # laid out in chunk-major order so that each chunk's data matches
        # the layout of a MemoryObj.raw_data (all groups concatenated):
        #
        #   [ chunk_0: group_0_bytes | group_1_bytes | ... ]
        #   [ chunk_1: group_0_bytes | group_1_bytes | ... ]
        #   ...
        #
        # This lets callers copy an entire chunk to/from a MemoryObj with a
        # single memcpy, without needing to know the per-group layout.
        # max_batch_size is the max number of chunks processed concurrently.
        self.max_batch_size = 4
        # Byte size of one chunk entry (= one chunk across all groups).
        # tmp_chunk_group_offsets_[g] is the byte offset of group g within
        # a single chunk; tmp_chunk_group_offsets_[num_groups] ==
        # tmp_chunk_bytes_.
        self.tmp_chunk_group_offsets_: list[int] = [0]
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            # ``get_kv_buffer_shape`` takes *logical* tokens; for
            # compressed groups it folds ``compress_ratio`` logical
            # tokens into one physical slot internally.
            shape = self.get_kv_buffer_shape(lmcache_logical_chunk_size, group_idx)
            byte_size = shape.numel() * group.dtype.itemsize
            self.tmp_chunk_group_offsets_.append(
                self.tmp_chunk_group_offsets_[-1] + byte_size
            )
        self.tmp_chunk_bytes_ = self.tmp_chunk_group_offsets_[-1]
        self.tmp_gpu_buffer_ = torch.empty(
            self.tmp_chunk_bytes_ * self.max_batch_size,
            dtype=torch.uint8,
            device=self.device_,
        )

        # GPU streams
        self.cuda_stream_ = torch_dev.Stream(device=self.device_)
        self.cupy_stream_ = cupy.cuda.ExternalStream(
            self.cuda_stream_.cuda_stream, self.device_.index
        )

        _, high_priority = torch_dev.Stream.priority_range()
        self.high_priority_cuda_stream_ = torch_dev.Stream(
            device=self.device_, priority=high_priority
        )
        self.high_priority_cupy_stream_ = cupy.cuda.ExternalStream(
            self.high_priority_cuda_stream_.cuda_stream, self.device_.index
        )

        # Extra initialization
        self.cupy_stream_.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self.device_)
            ),
            logger,
        )

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype(self.kv_caches_, self.gpu_kv_format_)

    @property
    def device(self) -> torch.device:
        return self.device_

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        return self.kv_caches_

    @property
    def stream(self) -> Any:
        """
        Returns the GPU stream for KV cache operations
        """
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> cupy.cuda.Stream:
        return self.cupy_stream_

    @property
    def high_priority_stream(self) -> Any:
        return self.high_priority_cuda_stream_

    @property
    def high_priority_cupy_stream(self) -> cupy.cuda.Stream:
        return self.high_priority_cupy_stream_

    @property
    def group_physical_block_sizes(self) -> list[int]:
        """Per-group physical slot count (``shape_desc.bs``) in group
        order. For non-compressed groups this equals
        ``inference_engine_logical_block_size``; for compressed groups
        it equals
        ``inference_engine_logical_block_size // compress_ratio``.
        """
        return [
            group.shape_desc.bs
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]

    @property
    def group_compress_ratios(self) -> list[int]:
        """Per-group compression ratio
        (= ``inference_engine_logical_block_size // shape_desc.bs``)
        in group order. ``1`` for non-compressed groups.
        """
        return [
            group.compress_ratio
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]

    @property
    def num_layers(self) -> int:
        """
        Returns the number of layers in the model
        """
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        """
        Returns the number of blocks in the KV cache
        """
        return self.num_blocks_

    @property
    def is_mla(self) -> bool:
        """
        Returns whether the model uses MLA
        """
        return self.is_mla_

    @property
    def hidden_dim_sizes(self) -> list[int]:
        """Returns the hidden dimension sizes for each KV layer group."""
        return [
            group.hidden_dim_size
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Returns the PageBufferShapeDesc for the given KV layer group."""
        return self.kv_layer_groups_manager_.get_shape_desc(group_idx)

    def get_physical_chunk_size(self, group_idx: int) -> int:
        """Returns the per-chunk physical slot count for the given group.

        Equal to ``lmcache_logical_chunk_size // compress_ratio``; for
        non-compressed groups this is just ``lmcache_logical_chunk_size``.
        This is the value the block-level transfer kernel must be told.
        """
        return self.kv_layer_groups_manager_.get_physical_chunk_size(group_idx)

    @property
    def kv_layer_groups_manager(self) -> KVLayerGroupsManager:
        """Returns the KV layer groups manager."""
        return self.kv_layer_groups_manager_

    @property
    def gpu_kv_format_name(self) -> str:
        """Returns the GPU KV format enum name (e.g. ``'NL_X_TWO_NB_BS_NH_HS'``)."""
        return self.gpu_kv_format_.name

    @property
    def gpu_kv_shape(self) -> str:
        """Returns a human-readable shape description of the GPU KV cache layout."""
        return get_gpu_kv_shape_description(self.gpu_kv_format_)

    @property
    def attention_backend(self) -> str:
        """Returns the attention backend name."""
        return get_attention_backend(self.gpu_kv_format_)

    @property
    def concrete_gpu_kv_shape(self) -> str:
        """Returns the GPU KV shape with actual numeric values substituted."""
        return get_concrete_gpu_kv_shape(self.kv_caches_, self.gpu_kv_format_)

    def get_group_kv_pointers(self, group_idx: int) -> torch.Tensor:
        """Returns the pre-computed GPU tensor of KV cache pointers for the
        given group."""
        return self.group_kv_pointers_[group_idx]

    def get_tmp_gpu_buffer_flat(self, chunk_idx: int) -> torch.Tensor:
        """Returns the flat uint8 view of the temporary GPU buffer for the
        given chunk index, covering all KV layer groups.

        The returned tensor will fit a memory full object corresponding
        ``self.chunk_size`` tokens, so it can be copied to/from a MemoryObj
        with a single memcpy.

        Args:
            chunk_idx: Chunk index (0 <= chunk_idx < max_batch_size).
        """
        if chunk_idx >= self.max_batch_size:
            raise ValueError(
                f"chunk_idx {chunk_idx} exceeds max_batch_size {self.max_batch_size}"
            )
        start = chunk_idx * self.tmp_chunk_bytes_
        return self.tmp_gpu_buffer_[start : start + self.tmp_chunk_bytes_]

    def get_tmp_chunk_gpu_buffer(self, group_idx: int = 0) -> torch.Tensor:
        """
        Returns a view of the temporary GPU buffer for the given group,
        sized for a single chunk. The chunk holds
        ``lmcache_logical_chunk_size`` logical tokens which, for a
        compressed group, correspond to ``group.physical_chunk_size``
        physical slots.

        Args:
            group_idx: Index of the KV layer group (default 0).
        """
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_logical_chunk_size, group_idx)
        start = self.tmp_chunk_group_offsets_[group_idx]
        end = self.tmp_chunk_group_offsets_[group_idx + 1]
        return self.tmp_gpu_buffer_[start:end].view(group.dtype).view(shape)

    def get_tmp_chunk_gpu_buffer_batched(
        self, batch_size: int, group_idx: int = 0
    ) -> list[torch.Tensor]:
        """
        Returns a list of ``batch_size`` non-overlapping views into the
        pre-allocated temporary GPU buffer for the given group, each
        sized for ``lmcache_logical_chunk_size`` tokens.

        Args:
            batch_size: Number of concurrent requests (must be <= max_batch_size).
            group_idx: Index of the KV layer group (default 0).
        """
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}"
            )
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_logical_chunk_size, group_idx)
        g_start = self.tmp_chunk_group_offsets_[group_idx]
        g_end = self.tmp_chunk_group_offsets_[group_idx + 1]
        chunk = self.tmp_chunk_bytes_
        return [
            self.tmp_gpu_buffer_[i * chunk + g_start : i * chunk + g_end]
            .view(group.dtype)
            .view(shape)
            for i in range(batch_size)
        ]

    def stage_block_ids(self, block_ids: list[int]) -> torch.Tensor:
        """Copy block_ids into the pre-allocated GPU buffer and return a
        view of the occupied region. Uses non-blocking copy via a pinned
        CPU tensor created from the list's underlying buffer.

        Args:
            block_ids: Block indices as a Python list of ints.

        Returns:
            A GPU int64 tensor view into the pre-allocated buffer.
        """
        n = len(block_ids)
        cpu_tensor = torch.frombuffer(array.array("l", block_ids), dtype=torch.long)
        buf = self.block_ids_buffer_[:n]
        buf.copy_(cpu_tensor, non_blocking=True)
        return buf

    def get_kv_buffer_shape(
        self, logical_num_tokens: int, group_idx: int = 0
    ) -> torch.Size:
        """
        Returns the shape of the KV buffer for the given number of
        *logical* tokens.

        For a compressed group (``compress_ratio > 1``) every
        ``compress_ratio`` logical tokens are packed into a single
        physical slot, so the returned shape's token dimension is
        ``num_tokens // compress_ratio``. Callers therefore always
        pass logical-token counts and never need to know per-group
        compression ratios.

        Args:
            logical_num_tokens: Number of *logical* tokens. Must be a multiple
                of the group's ``compress_ratio``.
            group_idx: Index of the KV layer group (default 0).
        """
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        compress_ratio = group.compress_ratio
        if logical_num_tokens % compress_ratio != 0:
            raise ValueError(
                f"logical_num_tokens ({logical_num_tokens}) is not a multiple of "
                f"compress_ratio ({compress_ratio}) for group {group_idx}"
            )
        num_slots = logical_num_tokens // compress_ratio
        sd = group.shape_desc
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def cache_size_per_token(self) -> int:
        """
        Returns the cache size per *logical* token (in bytes), summed
        across all groups. For a compressed group, one physical slot
        stores ``compress_ratio`` logical tokens, so the per-logical-token
        contribution is ``physical_slot_bytes // compress_ratio``.

        Reporting-only metric (surfaced via the ``/api/status`` HTTP
        endpoint and the ``lmcache describe`` CLI); sub-byte truncation
        from integer division is acceptable.
        """
        total = 0
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            # ``get_kv_buffer_shape`` now takes *logical* tokens, so
            # query ``compress_ratio`` logical tokens (= 1 physical
            # slot) and then divide the resulting bytes back by
            # ``compress_ratio`` to recover the per-logical-token
            # contribution. Equivalent to the old
            # ``physical_slot_bytes // compress_ratio`` formulation.
            numels = self.get_kv_buffer_shape(group.compress_ratio, group_idx).numel()
            slot_bytes = numels * group.dtype.itemsize
            total += slot_bytes // group.compress_ratio
        return total


class PlainGPUCacheContext:
    """
    A plain GPU cache context that have a single contiguous 2LTD buffer
    """

    def __init__(self, kv_caches: KVCache, lmcache_chunk_size: int = 256):
        assert len(kv_caches) == 1, (
            "PlainGPUCacheContext only supports a single KV cache tensor"
        )

        # KV cache basics
        self._kv_cache = unwrap_kv_cache_tensors(kv_caches)[0]
        self._device = self._kv_cache.device

        # Shape related
        shape = self._kv_cache.shape
        assert len(shape) == 4, "Expected [2, L, T, D] for plain GPU cache"

        self._num_layers = shape[1]
        self._num_tokens = shape[2]
        self._hidden_dim_size = shape[3]

        # Temporary buffer
        tmp_buffer_shape = self.get_kv_buffer_shape(lmcache_chunk_size)
        self._tmp_gpu_buffer = torch.empty(
            tmp_buffer_shape, dtype=self.dtype, device=self.device
        )

        # GPU streams
        self._cuda_stream = torch_dev.Stream(device=self._device)
        self._cupy_stream = cupy.cuda.ExternalStream(
            self._cuda_stream.cuda_stream, self._device.index
        )

        _, high_priority = torch_dev.Stream.priority_range()
        self._high_priority_cuda_stream = torch_dev.Stream(
            device=self._device, priority=high_priority
        )
        self._high_priority_cupy_stream = cupy.cuda.ExternalStream(
            self._high_priority_cuda_stream.cuda_stream, self._device.index
        )

        # Extra initialization
        self._cupy_stream.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self._device)
            ),
            logger,
        )

    def get_kv_buffer_shape(self, num_tokens: int) -> torch.Size:
        """
        Returns the shape of the KV buffer for the given number of tokens
        """
        return torch.Size((2, self._num_layers, num_tokens, self._hidden_dim_size))

    def get_tmp_gpu_buffer(self, num_tokens: int) -> torch.Tensor:
        """
        Returns the temporary GPU buffer for transfers
        """
        return self._tmp_gpu_buffer[:, :, :num_tokens, :]

    def slice_kv_cache_on_tokens(self, start: int, end: int) -> torch.Tensor:
        """
        Slices the KV cache tensor on the token dimension
        """
        return self._kv_cache[:, :, start:end, :]

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_cache.dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def stream(self) -> Any:
        """Returns the device-specific GPU stream (e.g., torch_dev.Stream)."""
        return self._cuda_stream

    @property
    def cupy_stream(self) -> cupy.cuda.Stream:
        return self._cupy_stream

    @property
    def high_priority_stream(self) -> Any:
        return self._high_priority_cuda_stream

    @property
    def high_priority_cupy_stream(self) -> cupy.cuda.Stream:
        return self._high_priority_cupy_stream

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def hidden_dim_size(self) -> int:
        return self._hidden_dim_size

    @property
    def kv_cache_tensor(self) -> torch.Tensor:
        return self._kv_cache
