# SPDX-License-Identifier: Apache-2.0
"""
GPU Cache Context management for LMCache multiprocessing.

This module provides GPU-side KV cache management functionality, including:
- GPUCacheContext: Manages shape and pointers to vLLM GPU KV cache tensors
- Helper functions for tensor operations and key resolution
"""

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
import array

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    import cupy

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType, lmcache_deprecate
from lmcache.v1.gpu_connector.gds_context import get_gds_context
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_attention_backend,
    get_concrete_engine_kv_shape_from_shape_desc,
    get_device,
    get_dtype,
    get_engine_kv_shape_description,
    get_group_data_ptrs,
    get_num_blocks,
    get_num_layers,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

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


class _TempGPUBuffer:
    """
    Manages the temporary GPU buffer for GPUCacheContext

    The logical layout of the temp GPU buffer is (batch size,
    object group, kernel group).

    Here is an example of batch size = 4, with 2 object groups,
    and 2 kernel groups per object group:
    [
        batch 0:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 1:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 2:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 3:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...
    ]

    During the multi-layer copy kernel launch, we will do it at kernel
    group level, which means we will have:
    ```
    gpu_buffers = [
        get_temp_kernel_group_buffer(batch_idx, kernel_group_idx)
        for batch_idx in range(batch_size)
    ]
    ```

    During the lmcache_memcpy_async launch, we will do it at the object group
    level, which will be:
    ```
    for i in range(batch_size):
        gpu_buffer = get_temp_object_group_buffer(batch_idx, object_group_idx)
        lmcache_memcpy_async(...)
    ```
    """

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        self._temp_buffer = torch.empty(
            self._get_size_for_single_batch() * max_batch_size,
            dtype=torch.uint8,
            device=device,
        )

        # Offset map: (batch_idx, object_group_idx, kernel_group_idx) ->
        # (byte offset in the temp buffer, size of the buffer in bytes)
        self._offset_map: dict[tuple[int, int, int], tuple[int, int]] = {}

        # (batch_idx, kernel_group_idx) -> (byte offset for the kernel group,
        # size of the buffer in bytes).
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        # (batch_idx, object_group_idx) -> (byte offset for the object group,
        # size of the buffer in bytes)
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_size = 0
                object_group_start_offset = offset

                for kernel_group_idx in self._kv_groups_manager.object_groups[
                    object_group_idx
                ].kernel_group_indices:
                    key = (batch_idx, object_group_idx, kernel_group_idx)
                    key2 = (batch_idx, kernel_group_idx)

                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map[key] = (offset, size)
                    self._offset_map_kernel_group_only[key2] = (offset, size)

                    offset += size
                    object_group_size += size

                key3 = (batch_idx, object_group_idx)
                self._offset_map_object_group_only[key3] = (
                    object_group_start_offset,
                    object_group_size,
                )

        # Shape/dtype cache for kernel groups
        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk, kernel_group_idx
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            dtype = group.dtype
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, dtype)

    # Public APIs
    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks (batch slots) the buffer holds."""
        return self._max_batch_size

    @property
    def buffer(self) -> torch.Tensor:
        """The flat staging tensor (for GDS cuFile registration)."""
        return self._temp_buffer

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """
        Returns the temp GPU buffer for the given batch index and kernel group index.
        The returned buffer is with the correct shape and dtype for the kernel group.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.

        Returns:
            The temp GPU buffer for the given batch index and kernel group index.

        Raises:
            ValueError: If the batch_idx or kernel_group_idx is out of range.
        """
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """
        Returns the temp GPU buffer for the given batch index and object group index
        The returned buffer is a flat uint8 raw tensor.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            object_group_idx: Index of the object group.

        Returns:
            The temp GPU buffer for the given batch index and object group index.
        """
        key = (batch_idx, object_group_idx)
        if key not in self._offset_map_object_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or object_group_idx {object_group_idx}"
            )

        offset, size = self._offset_map_object_group_only[key]
        return self._temp_buffer[offset : offset + size]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """
        Returns the shape and dtype for the given kernel group index and
        number of tokens.

        Will be exported by GPUCacheContext and used to construct the
        MemoryLayoutDesc

        Args:
            num_tokens: Number of tokens. Must be a whole number of lmcache
                chunk size.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape and dtype for the given kernel group index and
            number of tokens.
        """
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        shape = self._get_shape_for_kernel_group(num_tokens, kernel_group_idx)

        return shape, dtype

    def get_cache_size_per_token(self) -> int:
        """
        Returns the cache size per token (in bytes), summed across all kernel groups.
        """
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    # Helper functions
    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        """
        Returns the shape of the temp GPU buffer for the given kernel group index

        Args:
            num_tokens: Number of tokens
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape of the temp GPU buffer for the given kernel group index.

        Raises:
            ValueError: If ``num_tokens`` is not a whole number of LMCache
                chunks.
        """
        if num_tokens % self._lmcache_tokens_per_chunk != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a multiple of "
                f"lmcache_tokens_per_chunk ({self._lmcache_tokens_per_chunk})"
            )

        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        sd = group.shape_desc

        num_chunks = num_tokens // self._lmcache_tokens_per_chunk
        num_slots = (
            self._kv_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)
            * num_chunks
        )

        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        """
        Returns the size in bytes of the temp GPU buffer for the given kernel group
        index

        **Assumes the size is lmcache_tokens_per_chunk

        Will only be called during initialization
        """
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk, kernel_group_idx
        )
        kernel_group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        dtype = kernel_group.dtype
        return shape.numel() * dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        """
        Returns the size in bytes of the temp GPU buffer for the given object group

        **Assumes the size is lmcache_tokens_per_chunk

        Will only be called during initialization
        """
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        """
        Returns the size in bytes of the temp GPU buffer for a single batch
        (i.e., a single chunk)

        **Assumes the size is lmcache_tokens_per_chunk
        """
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )


class GPUCacheContext:
    """
    Manages the shape and pointers to vLLM GPU KV cache tensors.
    """

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ):
        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        self.engine_kv_format_, self.kv_caches_ = normalize_kv_and_discover_format(
            unwrapped,
            engine_type,
            layout_hints=layout_hints,
        )
        self.device_ = get_device(self.kv_caches_)
        self.is_mla_ = is_mla(self.engine_kv_format_)
        self.num_layers_ = get_num_layers(self.kv_caches_, self.engine_kv_format_)
        self.num_blocks_ = get_num_blocks(self.kv_caches_, self.engine_kv_format_)
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk

        self.kv_layer_groups_manager_ = KVLayerGroupsManager(
            self.kv_caches_,
            engine_kv_format=self.engine_kv_format_,
            num_blocks=self.num_blocks_,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        for group in self.kv_layer_groups_manager_.kv_layer_groups:
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.engine_kv_format_, group.layer_indices
            )
            self.group_kv_pointers_.append(list_to_gpu_tensor(ptrs, self.device_))

        # Pre-allocated GPU buffer for block IDs (up to 1M elements).
        # The caller copies block_ids into this buffer before launching the
        # block-level kernel. Single-thread assumption: no lock needed.
        _MAX_BLOCK_IDS = 1 << 20
        self.block_ids_buffer_ = torch.empty(
            _MAX_BLOCK_IDS, dtype=torch.long, device=self.device_
        )

        # Temporary GPU buffer for transfers — a single flat uint8 buffer
        self._temp_buffer = _TempGPUBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )

        # GPU streams
        self.cuda_stream_ = torch_dev.Stream(device=self.device_)

        # Register the staging buffer with the GDS cuFile context on the
        # context's CUDA stream.
        with torch_dev.stream(self.cuda_stream_):
            get_gds_context().register_gpu_buffer(self._temp_buffer.buffer)

        # Third Party
        import cupy

        self.cupy_stream_: "cupy.cuda.Stream" = cupy.cuda.ExternalStream(
            self.cuda_stream_.cuda_stream, self.device_.index
        )

        # Extra initialization
        self.cupy_stream_.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self.device_)
            ),
            logger,
        )

    def close(self) -> None:
        """
        Deregister this context's GDS staging buffer (reverse of __init__).
        """
        with torch_dev.stream(self.cuda_stream_):
            get_gds_context().deregister_gpu_buffer(self._temp_buffer.buffer)

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype(self.kv_caches_, self.engine_kv_format_)

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
    def cupy_stream(self) -> "cupy.cuda.Stream":
        return self.cupy_stream_

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

    @property
    def kv_layer_groups_manager(self) -> KVLayerGroupsManager:
        """Returns the KV layer groups manager."""
        return self.kv_layer_groups_manager_

    def get_shape_desc(self, kernel_group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Returns the PageBufferShapeDesc for the given KV layer group."""
        return self.kv_layer_groups_manager_.get_shape_desc(kernel_group_idx)

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        """Returns the number of slots per lmcache chunk when doing
        D/H transfer.

        This function will take into account that for subchunk sliding window
        case, the transfer slots will be smaller than the "actual" slots per
        chunk (which is calculated based on the lmcache chunk size).

        Args:
            kernel_group_idx: Index of the kernel group.

        Returns:
            The number of used slots per lmcache chunk when doing D/H transfer.
        """
        return self.kv_layer_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the pre-computed GPU tensor of KV cache pointers for the
        given kernel group index.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns the temporary GPU buffer for the given batch index and kernel
        group index, with the correct shape and dtype for the kernel group.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.

        Returns:
            The temp GPU buffer for the given batch index and kernel group index.
        """
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx, kernel_group_idx
        )

    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks processed concurrently in one batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns the temporary GPU buffer for the given batch index and object
        group index, as a flat uint8 tensor.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            object_group_idx: Index of the object group.

        Returns:
            The temp GPU buffer for the given batch index and object group index.
        """
        return self._temp_buffer.get_temp_object_group_buffer(
            batch_idx, object_group_idx
        )

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns the shape and dtype for the given kernel group index and number
        of tokens.
        Will be exported by GPUCacheContext and used to construct the MemoryLayoutDesc

        Args:
            num_tokens: Number of tokens. Must be a whole number of lmcache
                chunk size.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape and dtype for the given kernel group index and number of tokens.
        """
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens, kernel_group_idx
        )

    def copy_view_block_ids_to_gpu(
        self, block_ids_per_group: list[list[int]]
    ) -> list[torch.Tensor]:
        """Copy block IDs for each LMCache KV layer group to GPU.

        The outer list is indexed by LMCache KV group index. All inner lists
        are packed into the shared GPU buffer once, and this returns one
        non-overlapping tensor view per LMCache group.
        """
        offsets = [0]
        flat: array.array = array.array("l")
        for view_block_ids in block_ids_per_group:
            flat.extend(view_block_ids)
            offsets.append(len(flat))

        total = offsets[-1]
        if total > self.block_ids_buffer_.shape[0]:
            raise ValueError(
                f"block ID total {total} exceeds the pre-allocated buffer "
                f"size {self.block_ids_buffer_.shape[0]}"
            )
        if total:
            cpu_tensor = torch.frombuffer(flat, dtype=torch.long)
            self.block_ids_buffer_[:total].copy_(cpu_tensor, non_blocking=True)

        return [
            self.block_ids_buffer_[offsets[i] : offsets[i + 1]]
            for i in range(len(block_ids_per_group))
        ]

    @lmcache_deprecate("will be refactored")
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
        # TODO: remove this!
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
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

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        """Calculate the number of blocks for a given number of tokens in a
        specified kernel group.

        Args:
            kernel_group_idx: 0-based index of the kernel group.
            num_tokens: The total number of tokens to be processed for the group.

        Returns:
            The number of blocks.

        Raises:
            IndexError: If *kernel_group_idx* is out of range.
        """
        return self.kv_layer_groups_manager.calculate_num_blocks(
            kernel_group_idx, num_tokens
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
        return self._temp_buffer.get_cache_size_per_token()

    def report_status(self) -> dict:
        """Return this context's KV cache layout metadata for ``/status``.

        Returns:
            A dict with these top-level fields:

            - ``num_layers`` (int): total layers in the model.
            - ``num_blocks`` (int)
            - ``cache_size_per_token`` (int): bytes per logical token,
              summed across groups.
            - ``kernel_groups`` (list[dict]): one entry per kernel group,
              each with:

              - ``kernel_group_idx`` (int): index into ``manager.kernel_groups``.
              - ``engine_group_idx`` (int): paged-block address space.
              - ``object_group_idx`` (int): owning object group.
              - ``num_layers`` (int): layers in this group.
              - ``layer_indices`` (list[int]): the group's layer indices.
              - ``tokens_per_block`` (int): logical tokens per paged chunk.
              - ``slots_per_block`` (int): physical slots per paged
                chunk (``slots_per_block``, i.e. ``shape_desc.bs``).
              - ``compress_ratio`` (int)
              - ``dtype`` (str): stringified torch dtype.
              - ``gpu_kv_concrete_shape`` (str): group-accurate numeric shape.
              - ``is_mla`` (bool)
              - ``engine_kv_format`` (str): engine KV format enum name.
              - ``engine_kv_shape`` (str): symbolic shape description.
              - ``attention_backend`` (str)
        """
        manager = self.kv_layer_groups_manager
        kernel_groups = manager.kernel_groups

        # Reverse-map each kernel group to its owning object group.
        kernel_group_to_object_group: dict[int, int] = {
            kg_idx: og_idx
            for og_idx, og in enumerate(manager.object_groups)
            for kg_idx in og.kernel_group_indices
        }

        engine_kv_format = self.engine_kv_format_
        group_reports: list[dict] = []
        for kernel_group_idx, group in enumerate(kernel_groups):
            group_reports.append(
                {
                    "kernel_group_idx": kernel_group_idx,
                    "engine_group_idx": group.engine_group_idx,
                    "object_group_idx": kernel_group_to_object_group.get(
                        kernel_group_idx, 0
                    ),
                    "num_layers": group.num_layers,
                    "layer_indices": list(group.layer_indices),
                    "tokens_per_block": group.tokens_per_block,
                    "slots_per_block": group.slots_per_block,
                    "dtype": str(group.dtype),
                    "engine_kv_concrete_shape": (
                        get_concrete_engine_kv_shape_from_shape_desc(
                            group.shape_desc, engine_kv_format
                        )
                    ),
                    "is_mla": is_mla(engine_kv_format),
                    "engine_kv_format": engine_kv_format.name,
                    "engine_kv_shape": get_engine_kv_shape_description(
                        engine_kv_format
                    ),
                    "attention_backend": get_attention_backend(engine_kv_format),
                }
            )

        return {
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "cache_size_per_token": self.cache_size_per_token(),
            "kernel_groups": group_reports,
        }


class PlainGPUCacheContext:
    """
    A plain GPU cache context that have a single contiguous 2LTD buffer
    """

    def __init__(self, kv_caches: KVCache, lmcache_tokens_per_chunk: int = 256):
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
        tmp_buffer_shape = self.get_kv_buffer_shape(lmcache_tokens_per_chunk)
        self._tmp_gpu_buffer = torch.empty(
            tmp_buffer_shape, dtype=self.dtype, device=self.device
        )

        # GPU streams
        self._cuda_stream = torch_dev.Stream(device=self._device)
        # Third Party
        import cupy

        self._cupy_stream: "cupy.cuda.Stream" = cupy.cuda.ExternalStream(
            self._cuda_stream.cuda_stream, self._device.index
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
    def cupy_stream(self) -> "cupy.cuda.Stream":
        return self._cupy_stream

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
