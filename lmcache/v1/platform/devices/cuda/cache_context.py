# SPDX-License-Identifier: Apache-2.0
"""
GPU Cache Context management for LMCache multiprocessing.

This module provides GPU-side KV cache management functionality, including:
- GPUCacheContext: Manages shape and pointers to vLLM GPU KV cache tensors
- Helper functions for tensor operations and key resolution
"""

# Standard
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
import array
import threading

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    import cupy

    # First Party
    from lmcache.v1.platform.base.object_group_transfer import ObjectGroupTransfer

# First Party
from lmcache import torch_dev
from lmcache.v1.common.buffer_registration import BufferRegistration, SlottedBufferRegistration
from lmcache.v1.common.custom_types import EngineType
from lmcache.v1.common.group_view import (
    EngineGroupInfo,
    engine_group_layer_indices,
)
from lmcache.v1.common.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.common.logging import init_logger
from lmcache.v1.common.utils import (
    LayoutHints,
    get_device,
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
    round_up,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.ipc_wrapper import KVCache

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
        staging_alignment: int = 0,
        staging_budget: int = 0,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        sizes = tuple(
            self._get_size_for_object_group(i)
            for i in range(kv_layer_groups_manager.num_object_groups)
        )
        padded_bytes = sum(round_up(size, staging_alignment or 1) for size in sizes) * max_batch_size
        aligned_backing = padded_bytes + staging_alignment - 1
        self.aligned_object_sizes = (
            sizes
            if staging_alignment and all(sizes) and aligned_backing <= staging_budget
            else ()
        )
        staging_bytes = (
            aligned_backing
            if self.aligned_object_sizes
            else sum(sizes) * max_batch_size
        )
        if staging_budget and staging_bytes > staging_budget:
            raise ValueError(f"Staging needs {staging_bytes} bytes, exceeding {staging_budget}")
        backing = torch.empty(staging_bytes, dtype=torch.uint8, device=device)
        if self.aligned_object_sizes:
            start = -backing.data_ptr() % staging_alignment
            self._temp_buffer = backing.narrow(0, start, padded_bytes)
        else:
            self._temp_buffer = backing

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

                if self.aligned_object_sizes:
                    offset = object_group_start_offset + round_up(
                        object_group_size, staging_alignment
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

    @property
    def slot_bytes(self) -> int:
        """Byte stride of one slot, including all of its object groups."""
        return self._temp_buffer.numel() // self._max_batch_size

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


class GPUCacheContext(BaseCacheContext):
    """
    Manages the shape and pointers to vLLM GPU KV cache tensors.

    ``object_group_transfer`` supplies the base full-chunk copy implementation;
    it is optional for layout-only users and device transfer overrides.
    """

    device_type = "cuda"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
        *,
        object_group_transfer: "ObjectGroupTransfer | None" = None,
        staging_alignment: int = 0,
        staging_budget: int = 0,
    ) -> None:
        # Kept for close(): raw-IPC wrappers hold driver-level mappings
        # that pin the exporter's device memory until explicitly closed.
        self._kv_wrappers: KVCache = list(kv_caches)
        self._buffer_registration: BufferRegistration | None = None
        self._staging_alignment = staging_alignment
        self._staging_budget = staging_budget
        self._gds_streams: dict[int, torch.cuda.Stream] = {}
        self._gds_ready_event: torch.cuda.Event | None = None
        self._gds_transfer_lock = threading.Lock()
        self._gds_closed = False
        try:
            self._init_impl(
                kv_caches,
                lmcache_tokens_per_chunk,
                layout_hints,
                engine_group_infos,
                engine_type,
                separate_object_groups,
                full_sw_kv,
                object_group_transfer,
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
        object_group_transfer: "ObjectGroupTransfer | None",
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
            object_group_transfer=object_group_transfer,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        self._group_kv_pointer_lists: list[list[int]] = []
        for idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.get_engine_kv_format(idx), group.layer_indices
            )
            self._group_kv_pointer_lists.append(list(ptrs))
            self.group_kv_pointers_.append(list_to_gpu_tensor(ptrs, self.device_))

        # Temporary GPU buffer for transfers — a single flat uint8 buffer
        self._temp_buffer = self._make_temp_buffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
            full_sw_kv=full_sw_kv,
        )

        # GPU streams
        self.cuda_stream_ = torch_dev.Stream(device=self.device_)

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

    def _make_temp_buffer(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int,
        full_sw_kv: bool,
    ) -> _TempGPUBuffer:
        """Builds the staging buffer for this context.

        This class always produces the kernel-group-major layout; a
        subclass overrides this to stage in a different layout without
        re-implementing ``__init__``.  ``full_sw_kv`` is passed through so a
        layout that cannot serve CacheBlend can refuse the combination while
        it is still being built.
        """
        return _TempGPUBuffer(
            kv_layer_groups_manager=kv_layer_groups_manager,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=device,
            max_batch_size=max_batch_size,
            staging_alignment=self._staging_alignment,
            staging_budget=self._staging_budget,
        )

    def register_staging_buffer(self, registration: BufferRegistration) -> None:
        """Register the full staging buffer on the context's transfer stream.

        Args:
            registration: External registration, retained after success until
                ``close``. Call once after construction and before transfers.

        Raises:
            RuntimeError: If registration fails; the adapter must roll back.
        """
        with torch_dev.stream(self.cuda_stream_):
            if self._staging_alignment and isinstance(registration, SlottedBufferRegistration):
                registration.register_slots(
                    self._temp_buffer.buffer,
                    self._temp_buffer.aligned_object_sizes,
                    self._temp_buffer.slot_bytes,
                )
            else:
                registration.register(self._temp_buffer.buffer)
        self._buffer_registration = registration

    def close(self) -> None:
        """Drain the context stream, release any staging registration and
        release the imported KV mappings (reverse of __init__).

        The stream is synchronized first so no in-flight kernel still
        touches the mappings when they are unmapped. The wrapper close
        unmaps raw CUDA IPC imports; without it a dead worker's KV pool
        stays resident on this process's GPUs for the process lifetime.
        The tensors built over those mappings must not be dereferenced
        afterwards -- the caller (``_release_entries``) drops the context
        right after this call.
        """
        with self._gds_transfer_lock:
            self._gds_closed = True
            self.cuda_stream_.synchronize()
            registration = self._buffer_registration
            if isinstance(registration, SlottedBufferRegistration):
                for stream in self._gds_streams.values():
                    registration.deregister_stream(self.device_, stream)
                self._gds_streams.clear()
            if registration is not None:
                with torch_dev.stream(self.cuda_stream_):
                    registration.deregister(self._temp_buffer.buffer)
                self._buffer_registration = None
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

    def get_gds_transfer_stream(self, slot: int) -> torch.cuda.Stream:
        """Return a persistent stream for one existing staging slot.

        Args:
            slot: Staging index in ``[0, max_batch_size)``.

        Returns:
            Stream on this context's device, created on first use.

        Raises:
            ValueError: The staging slot does not exist.

        Called only by the context's serialized affinity handler, before
        dispatching slot work. No new staging allocation is made.
        """
        if not 0 <= slot < self.max_batch_size:
            raise ValueError("GDS slot is outside the staging buffer")
        if slot not in self._gds_streams:
            self._gds_streams[slot] = torch.cuda.Stream(device=self.device_)
        return self._gds_streams[slot]

    @contextmanager
    def gds_transfer_scope(self) -> Iterator[None]:
        """Lease staging and engine mappings until all slot streams drain.

        Serializes concurrent transfers with unregister/reaping. Raises RuntimeError
        if close has already started; the caller must not submit slot work then.
        """
        with self._gds_transfer_lock:
            if self._gds_closed:
                raise RuntimeError("GPU cache context is closed")
            yield

    def record_gds_transfer_ready(self) -> torch.cuda.Event:
        """Record the persistent input event on this context's original stream.

        Returns:
            The recorded event for the current slot-work generation.

        The caller holds ``gds_transfer_scope`` until all slot waits finish.
        That lease makes re-recording safe and keeps the event out of the engine
        IPC handoff protocol, which continues to use its own recorded events.
        """
        if self._gds_ready_event is None:
            self._gds_ready_event = torch.cuda.Event()
        self._gds_ready_event.record(self.cuda_stream_)
        return self._gds_ready_event

    @property
    def cupy_stream(self) -> "cupy.cuda.Stream":
        return self.cupy_stream_

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the pre-computed GPU tensor of KV cache pointers for the
        given kernel group index.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_kernel_group_kv_pointer_list(self, kernel_group_idx: int) -> list[int]:
        """Returns the host-side KV cache pointers of a kernel group, in the
        same order as :meth:`get_kernel_group_kv_pointers`, without touching
        the device.
        """
        return self._group_kv_pointer_lists[kernel_group_idx]

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

    def get_temp_staging_buffer(self, num_bytes: int) -> torch.Tensor | None:
        """Return the start of the transfer staging buffer as one flat region.

        Args:
            num_bytes: Bytes the region must hold.

        Returns:
            A flat uint8 view of the buffer's first ``num_bytes`` bytes, or
            None when the staging buffer is smaller than ``num_bytes``.
        """
        staging_buffer = self._temp_buffer.buffer
        if num_bytes > staging_buffer.numel():
            return None
        return staging_buffer[:num_bytes]

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
