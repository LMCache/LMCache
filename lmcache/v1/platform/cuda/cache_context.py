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
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gds_context import get_gds_context
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    engine_group_layer_indices,
)
from lmcache.v1.multiprocess.modules.layer_major_plan import (
    carve_layer_major_object_group,
    layer_major_staging_enabled,
    layer_slices_from_kernel_groups,
    resolve_layer_major_placements,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    unwrapped_tensors = []
    for ipc_wrapper in kv_caches:
        tensor = ipc_wrapper.to_tensor()
        unwrapped_tensors.append(tensor)
    return unwrapped_tensors


def list_to_gpu_tensor(lis: list[int], device: torch.device) -> torch.Tensor:
    return torch.frombuffer(array.array("q", lis), dtype=torch.long).to(
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

        # (batch_idx, kernel_group_idx, local_layer_idx) -> (byte offset of
        # that one layer's slice, size in bytes).  Only the layer-wise path
        # addresses a single layer, so only its subclass fills this in.
        self._offset_map_layer: dict[tuple[int, int, int], tuple[int, int]] = {}

        # Staging layout of this buffer.  This class only ever produces the
        # kernel-group-major order below; the layer-wise subclass replaces
        # both flags when it re-carves the buffer in model-depth order.
        self._layer_major = False
        self._kernel_groups_contiguous = True

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
    def layer_major(self) -> bool:
        """Whether the object groups are staged in model-depth order.

        True for every layer-wise deployment and false for every
        per-chunk one: the layout follows from the staging gate, not
        from the model.  It says nothing about whether a kernel group is
        still contiguous -- see :attr:`kernel_groups_contiguous`.
        """
        return self._layer_major

    @property
    def kernel_groups_contiguous(self) -> bool:
        """Whether each kernel group is one contiguous run of bytes.

        Always true under the kernel-group-major layout, and true under
        the layer-major layout as well whenever every model layer owns
        its caches in a single kernel group -- then depth order and
        kernel-group-major order coincide.  Only a model whose layers
        span several kernel groups interleaves them, leaving no
        per-group range for callers that address a whole group.
        """
        return self._kernel_groups_contiguous

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
        if not self._kernel_groups_contiguous:
            raise ValueError(
                "A kernel group is not contiguous for a model whose layers "
                "span several kernel groups; use get_temp_layer_buffer() "
                "to address one layer"
            )

        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_layer_buffer(
        self, batch_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> torch.Tensor:
        """Returns the temp GPU buffer holding a single layer's caches.

        Valid under both layouts, which is what lets callers stay layout
        agnostic.  The returned view keeps the kernel group's shape with the
        layer axis pinned to length one.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.
            local_layer_idx: Position of the layer within that kernel group,
                i.e. an index into its ``layer_indices`` -- not a model depth.

        Raises:
            ValueError: If any index is out of range.
        """
        key = (batch_idx, kernel_group_idx, local_layer_idx)
        if key not in self._offset_map_layer:
            raise ValueError(
                f"Invalid batch_idx {batch_idx}, kernel_group_idx "
                f"{kernel_group_idx} or local_layer_idx {local_layer_idx}"
            )

        offset, size = self._offset_map_layer[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        layer_shape = torch.Size((shape[0], 1, *shape[2:]))
        return self._temp_buffer[offset : offset + size].view(dtype).view(layer_shape)

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

    def get_layer_offset_in_object(
        self, object_group_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> int:
        """Byte offset of one layer's caches from the start of its object.

        Layout-agnostic: it reads the same per-layer map both layouts fill,
        so callers do not have to know which one was selected.  Relative to
        the object group because that is what an LMCache memory object holds.
        """
        entry = self._offset_map_layer.get((0, kernel_group_idx, local_layer_idx))
        object_group = self._offset_map_object_group_only.get((0, object_group_idx))
        if entry is None or object_group is None:
            raise ValueError(
                "No staging offset for object_group="
                f"{object_group_idx}, kernel_group={kernel_group_idx}, "
                f"local_layer={local_layer_idx}"
            )
        return entry[0] - object_group[0]

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


class _TempLayerMajorGPUBuffer(_TempGPUBuffer):
    """Stages each object group in model-depth order.

    Built only when the layer-wise staging gate is on, so the per-chunk
    deployment never reaches this class and its base keeps the
    kernel-group-major carve it has always had.  The carve is re-run here
    rather than parameterised in the base so that the legacy ordering stays
    one unbroken block of code with no layer-wise branch threaded through
    it -- the same reason the ZMQ client carries a separate streaming
    submit instead of a flag on the per-chunk one.

    Re-carving after ``super().__init__`` is safe because the buffer's size
    comes from :meth:`_get_size_for_single_batch`, which depends on the
    registered geometry and not on the order the slices are laid out in:
    both layouts place the same slices in the same number of bytes.
    """

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        super().__init__(
            kv_layer_groups_manager,
            lmcache_tokens_per_chunk,
            device,
            max_batch_size,
        )
        self._recarve_in_depth_order()

    def _recarve_in_depth_order(self) -> None:
        """Replaces the base's kernel-group-major offsets with depth order.

        Object groups the layout module declines to place keep the base's
        ordering, so a deployment can mix the two.
        """
        (
            self._layer_placements,
            self._layer_major,
            self._kernel_groups_contiguous,
        ) = resolve_layer_major_placements(
            self._kv_groups_manager, self._get_size_for_kernel_group
        )

        self._offset_map.clear()
        self._offset_map_kernel_group_only.clear()
        self._offset_map_object_group_only.clear()
        self._offset_map_layer.clear()

        offset = 0
        for batch_idx in range(self._max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_size = 0
                object_group_start_offset = offset
                placement = self._layer_placements[object_group_idx]

                if placement is None:
                    for kernel_group_idx in self._kv_groups_manager.object_groups[
                        object_group_idx
                    ].kernel_group_indices:
                        size = self._get_size_for_kernel_group(kernel_group_idx)
                        self._offset_map[
                            (batch_idx, object_group_idx, kernel_group_idx)
                        ] = (offset, size)
                        self._offset_map_kernel_group_only[
                            (batch_idx, kernel_group_idx)
                        ] = (offset, size)

                        offset += size
                        object_group_size += size
                else:
                    offset, object_group_size = carve_layer_major_object_group(
                        placement, batch_idx, offset, self._offset_map_layer
                    )
                    if self._kernel_groups_contiguous:
                        # Depth order coincided with kernel-group-major
                        # order, so each group is still one run of bytes.
                        # Record it so whole-kernel-group callers keep
                        # working on the layer-wise path too.
                        for kernel_group_idx in self._kv_groups_manager.object_groups[
                            object_group_idx
                        ].kernel_group_indices:
                            size = self._get_size_for_kernel_group(kernel_group_idx)
                            first = self._offset_map_layer[
                                (batch_idx, kernel_group_idx, 0)
                            ][0]
                            self._offset_map[
                                (batch_idx, object_group_idx, kernel_group_idx)
                            ] = (first, size)
                            self._offset_map_kernel_group_only[
                                (batch_idx, kernel_group_idx)
                            ] = (first, size)

                self._offset_map_object_group_only[(batch_idx, object_group_idx)] = (
                    object_group_start_offset,
                    object_group_size,
                )

        self._offset_map_layer.update(
            layer_slices_from_kernel_groups(
                self._kv_groups_manager, self._offset_map_kernel_group_only
            )
        )


class GPUCacheContext(BaseCacheContext):
    """
    Manages the shape and pointers to vLLM GPU KV cache tensors.
    """

    device_type = "cuda"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
    ):
        # Kept for close(): raw-IPC wrappers hold driver-level mappings
        # that pin the exporter's device memory until explicitly closed.
        self._kv_wrappers: KVCache = list(kv_caches)
        try:
            self._init_impl(
                kv_caches,
                lmcache_tokens_per_chunk,
                layout_hints,
                engine_group_infos,
                engine_type,
                separate_object_groups,
                full_sw_kv,
            )
        except BaseException:
            # Partial construction may have imported some mappings
            # already; roll them back so a failed registration does not
            # pin the worker's KV pool.
            self._close_kv_wrappers()
            raise

    def _init_impl(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int,
        layout_hints: LayoutHints | None,
        engine_group_infos: Sequence[EngineGroupInfo],
        engine_type: EngineType,
        separate_object_groups: bool,
        full_sw_kv: bool,
    ) -> None:
        """Body of ``__init__``; split out so the constructor can roll
        back partially imported KV mappings on failure.
        """
        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        kv_caches_norm, engine_kv_formats = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        self.device_ = get_device(kv_caches_norm)
        num_layers_val = len(engine_kv_formats)

        kv_layer_groups_manager = KVLayerGroupsManager(
            kv_caches_norm,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=separate_object_groups,
        )
        # Set full_sw_kv before the temp buffer / object groups are built so
        # both are sized for the full (un-windowed) per-chunk geometry.
        if full_sw_kv:
            kv_layer_groups_manager.enable_full_sw_kv()

        # Pre-allocated GPU buffer for block IDs (up to 1M elements).
        # The caller copies block_ids into this buffer before launching the
        # block-level kernel. Single-thread assumption: no lock needed.
        _MAX_BLOCK_IDS = 1 << 20
        block_ids_buffer = torch.empty(
            _MAX_BLOCK_IDS, dtype=torch.long, device=self.device_
        )

        super().__init__(
            kv_caches=kv_caches_norm,
            device=self.device_,
            num_layers=num_layers_val,
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        for idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.get_engine_kv_format(idx), group.layer_indices
            )
            self.group_kv_pointers_.append(list_to_gpu_tensor(ptrs, self.device_))

        # Temporary GPU buffer for transfers — a single flat uint8 buffer
        temp_buffer_cls: type[_TempGPUBuffer] = (
            _TempLayerMajorGPUBuffer
            if layer_major_staging_enabled()
            else _TempGPUBuffer
        )
        self._temp_buffer = temp_buffer_cls(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )

        # CacheBlend addresses the staging buffer one kernel group at a time
        # and takes data pointers into those contiguous regions, which a model
        # whose layers span several kernel groups cannot offer: its groups are
        # interleaved by depth.  ``full_sw_kv`` is set from ``engine_type ==
        # "blend"`` when the server context is built, so it is the blend
        # signal available here.  Refuse the combination up front rather than
        # let blend read misaddressed bytes.
        if full_sw_kv and not self._temp_buffer.kernel_groups_contiguous:
            raise ValueError(
                "Blending is not supported for models whose layers span "
                "several kernel groups: their KV staging interleaves the "
                "groups by model depth, leaving no contiguous "
                "per-kernel-group region for blend to address. Disable "
                "blending for this model."
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
        """Drain the context stream, deregister the GDS staging buffer and
        release the imported KV mappings (reverse of __init__).

        The stream is synchronized first so no in-flight kernel still
        touches the mappings when they are unmapped. The wrapper close
        unmaps raw CUDA IPC imports; without it a dead worker's KV pool
        stays resident on this process's GPUs for the process lifetime.
        The tensors built over those mappings must not be dereferenced
        afterwards -- the caller (``_release_entries``) drops the context
        right after this call.
        """
        self.cuda_stream_.synchronize()
        with torch_dev.stream(self.cuda_stream_):
            get_gds_context().deregister_gpu_buffer(self._temp_buffer.buffer)
        self._close_kv_wrappers()

    def _close_kv_wrappers(self) -> None:
        """Close every KV wrapper, continuing past per-wrapper failures."""
        for wrapper in self._kv_wrappers:
            try:
                wrapper.close()
            except Exception:
                logger.warning("KV wrapper close failed", exc_info=True)
        self._kv_wrappers = []

    @property
    def stream(self) -> Any:
        """
        Returns the GPU stream for KV cache operations
        """
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> "cupy.cuda.Stream":
        return self.cupy_stream_

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

    @property
    def layer_major(self) -> bool:
        """Whether the staging buffer uses the layer-major layout."""
        return self._temp_buffer.layer_major

    @property
    def kernel_groups_contiguous(self) -> bool:
        """Whether each kernel group is one contiguous staging range."""
        return self._temp_buffer.kernel_groups_contiguous

    def get_temp_layer_buffer(
        self, batch_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> torch.Tensor:
        """Returns the temporary GPU buffer holding one layer's caches.

        Valid under both layouts; see
        :meth:`_TempGPUBuffer.get_temp_layer_buffer`.
        """
        return self._temp_buffer.get_temp_layer_buffer(
            batch_idx, kernel_group_idx, local_layer_idx
        )

    def get_layer_offset_in_object(
        self, object_group_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> int:
        """Byte offset of one layer's caches within its object group."""
        return self._temp_buffer.get_layer_offset_in_object(
            object_group_idx, kernel_group_idx, local_layer_idx
        )

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
