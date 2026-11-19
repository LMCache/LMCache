# SPDX-License-Identifier: Apache-2.0
"""GCU (Enflame) cache context for LMCache multiprocessing.

This is a self-contained :class:`BaseCacheContext` implementation (like the
MUSA/NPU contexts) rather than a :class:`GPUCacheContext` subclass. GCU's
transfer path is CUDA-like, so the staging-buffer and pointer-table machinery
mirrors ``devices/cuda/cache_context.py``; the one piece that differs is the
observability stream: torch_gcu has no ``cupy`` binding, so the cupy
``ExternalStream`` is replaced by the cupy-free
:class:`~lmcache.v1.platform.devices.cpu.stub_cpu_device.StubStream`. This is
correct on GCU because GDS DMA via the topsfile path is synchronous, so
every transfer has completed by the time a metrics callback would fire.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.common.group_view import engine_group_layer_indices
from lmcache.v1.common.utils import (
    LayoutHints,
    get_device,
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.gpu_connector.gds_context import (
    deregister_gds_gpu_buffer,
    register_gds_gpu_buffer,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.ipc_wrapper import KVCache
from lmcache.v1.platform.devices.cpu.stub_cpu_device import StubStream

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.group_view import EngineGroupInfo

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    """Return tensors reconstructed from GCU IPC wrappers.

    Args:
        kv_caches: GCU IPC wrappers received over the multiprocess wire.

    Returns:
        The reconstructed GCU KV-cache tensors.
    """
    return [wrapper.to_tensor() for wrapper in kv_caches]


class _TempGcuBuffer:
    """Owns the temporary GCU staging buffer for MP block-transfer batches.

    The logical layout of the temp GPU buffer is (batch size, object group,
    kernel group); the per-group regions are sized for one LMCache chunk and
    laid out kernel-group-major, exactly like the CUDA context.
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

        # (batch_idx, kernel_group_idx) -> (byte offset, size in bytes).
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        # (batch_idx, object_group_idx) -> (byte offset, size in bytes).
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_size = 0
                object_group_start_offset = offset

                for kernel_group_idx in self._kv_groups_manager.object_groups[
                    object_group_idx
                ].kernel_group_indices:
                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map_kernel_group_only[
                        (batch_idx, kernel_group_idx)
                    ] = (offset, size)

                    offset += size
                    object_group_size += size

                self._offset_map_object_group_only[(batch_idx, object_group_idx)] = (
                    object_group_start_offset,
                    object_group_size,
                )

        # Shape/dtype cache for kernel groups.
        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk, kernel_group_idx
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, group.dtype)

    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks (batch slots) the buffer holds."""
        return self._max_batch_size

    @property
    def buffer(self) -> torch.Tensor:
        """The flat staging tensor (for GDS registration)."""
        return self._temp_buffer

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Return a typed staging view for a (batch, kernel-group) pair.

        Args:
            batch_idx: Index of the batch (``0 <= batch_idx < max_batch_size``).
            kernel_group_idx: Index of the kernel group.

        Returns:
            The staging tensor for the pair, viewed with the kernel group's
            shape and dtype.

        Raises:
            ValueError: If ``batch_idx`` or ``kernel_group_idx`` is out of range.
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
        """Return a flat ``uint8`` staging view for a batch/object-group pair.

        Args:
            batch_idx: Index of the batch (``0 <= batch_idx < max_batch_size``).
            object_group_idx: Index of the object group.

        Returns:
            A flat ``uint8`` view over the object group's staging region.

        Raises:
            ValueError: If ``batch_idx`` or ``object_group_idx`` is out of range.
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
        """Return ``(shape, dtype)`` for a kernel group and token count.

        Args:
            num_tokens: Number of tokens. Must be a whole number of LMCache
                chunk size.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape and dtype for the kernel group's staging allocation.
        """
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._get_shape_for_kernel_group(num_tokens, kernel_group_idx), dtype

    def get_cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        """Return the staging shape for a kernel group and token count.

        Args:
            num_tokens: Number of tokens.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The staging tensor shape.

        Raises:
            ValueError: If ``num_tokens`` is not a whole number of LMCache chunks.
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
        """Return the byte size of one kernel group's staging region.

        Assumes the size is ``lmcache_tokens_per_chunk``; called only during
        initialization.
        """
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk, kernel_group_idx
        )
        kernel_group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        return shape.numel() * kernel_group.dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        """Return the byte size of one object group's staging region."""
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        """Return the byte size of a single batch (one LMCache chunk)."""
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )


class GcuCacheContext(BaseCacheContext):
    """Cache context for GCU-backed KV tensors in MP handle mode."""

    device_type = "gcu"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ) -> None:
        """Build a GCU cache context from IPC-wrapped KV tensors.

        Args:
            kv_caches: GCU IPC wrappers received during server registration.
            lmcache_tokens_per_chunk: Tokens per LMCache object.
            layout_hints: Optional KV layout hints from the serving engine.
            engine_group_infos: Engine-neutral group metadata.
            engine_type: Serving engine that produced the KV cache.
            separate_object_groups: Whether to split object groups by window.
            full_sw_kv: Whether sliding-window groups transfer their full KV.

        Raises:
            ValueError: If reconstructed tensors are not GCU tensors.
        """
        # Kept for close(): raw-IPC wrappers hold driver-level mappings that
        # pin the exporter's device memory until explicitly closed.
        self._kv_wrappers: KVCache = list(kv_caches)
        try:
            self._initialize(
                kv_caches,
                lmcache_tokens_per_chunk,
                layout_hints,
                engine_group_infos,
                engine_type,
                separate_object_groups,
                full_sw_kv,
            )
        except BaseException:
            # Partial construction may have imported some mappings already;
            # roll them back so a failed registration does not pin the
            # worker's KV pool.
            self._close_kv_wrappers()
            raise

    def _initialize(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int,
        layout_hints: LayoutHints | None,
        engine_group_infos: "Sequence[EngineGroupInfo]",
        engine_type: EngineType,
        separate_object_groups: bool,
        full_sw_kv: bool,
    ) -> None:
        """Initialize reconstructed tensors and GCU transfer resources."""
        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        kv_caches_norm, engine_kv_formats = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        self.device_ = get_device(kv_caches_norm)
        if self.device_.type != "gcu":
            raise ValueError(
                f"GcuCacheContext expected GCU tensors, got {self.device_.type!r}"
            )
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
        self._group_kv_pointer_lists: list[list[int]] = []
        for idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.get_engine_kv_format(idx), group.layer_indices
            )
            ptr_list = list(ptrs)
            self._group_kv_pointer_lists.append(ptr_list)
            self.group_kv_pointers_.append(
                torch.tensor(ptr_list, dtype=torch.uint64, device="cpu")
            )

        # Temporary GCU buffer for transfers — a single flat uint8 buffer.
        self._temp_buffer = _TempGcuBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )

        # GPU streams
        self.cuda_stream_ = torch_dev.Stream(device=self.device_)

        # Register the staging buffer with the GDS context on the GCU stream.
        with torch_dev.stream(self.cuda_stream_):
            register_gds_gpu_buffer(self._temp_buffer.buffer)

        # Observability stream: a cupy-free stub whose ptr is the real
        # gcuStream_t, so recorded events carry the correct handle.
        stub = StubStream(device=self.device_)
        stub.ptr = self.cuda_stream_.gcu_stream
        self.cupy_stream_ = stub

    def close(self) -> None:
        """Drain the context stream, deregister the GDS staging buffer and
        release the imported KV mappings (reverse of __init__).

        The stream is synchronized first so no in-flight kernel still touches
        the mappings when they are unmapped. The wrapper close unmaps raw GCU
        IPC imports; without it a dead worker's KV pool stays resident on this
        process's GPUs for the process lifetime.
        """
        self.cuda_stream_.synchronize()
        with torch_dev.stream(self.cuda_stream_):
            deregister_gds_gpu_buffer(self._temp_buffer.buffer)
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
        """Return the GCU stream used for transfer work."""
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> StubStream:
        """Return a cupy-free stream adapter for shared MP code paths."""
        return self.cupy_stream_

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Return the pre-computed CPU tensor of KV pointers for a group.

        Args:
            kernel_group_idx: Index of the kernel group.

        Returns:
            A one-dimensional CPU ``uint64`` tensor in kernel order.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_kernel_group_kv_pointer_list(self, kernel_group_idx: int) -> list[int]:
        """Return the host-side KV pointers of a kernel group, in kernel order.

        Args:
            kernel_group_idx: Index of the kernel group.

        Returns:
            The paged buffer data pointers as Python ints.
        """
        return self._group_kv_pointer_lists[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Return the GCU staging buffer for a batch/kernel-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            kernel_group_idx: Index of the kernel group to transfer.

        Returns:
            A typed view into the GCU staging allocation.
        """
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx, kernel_group_idx
        )

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of chunks transferred per batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Return the GCU staging buffer for a batch/object-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            object_group_idx: Index of the object group to transfer.

        Returns:
            A flat view covering the object's GCU staging allocation.
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
        """Return the shape and dtype for a kernel-group allocation.

        Args:
            num_tokens: Number of tokens represented by the allocation.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The allocation shape and element dtype.
        """
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens, kernel_group_idx
        )

    def cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._temp_buffer.get_cache_size_per_token()
