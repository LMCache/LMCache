# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache context for platforms without CUDA GPUs.

This module lives in the ``platform.cpu`` sub-package because it is
the CPU-specific implementation of the cross-platform cache context
-- it provides the same public API as
:class:`~lmcache.v1.multiprocess.gpu_context.GPUCacheContext` but
keeps all tensors on CPU. Stream / Event objects are provided by
:class:`~lmcache.v1.platform.cpu.stub_cpu_device.StubStream` so
CPU-only hosts never import ``cupy`` or instantiate a real CUDA
stream object.

The platform-agnostic dispatcher ``create_cache_context`` lives in
:mod:`lmcache.v1.platform.cache_context`.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING
import array
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_attention_backend,
    get_concrete_engine_kv_shape_from_shape_desc,
    get_engine_kv_shape_description,
    get_group_data_ptrs,
    get_num_blocks,
    get_num_layers,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform.cpu.stub_cpu_device import StubStream
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


class CpuCacheContext:
    """CPU-only cache context with the same public API as
    :class:`GPUCacheContext`.

    All tensors live on CPU. CUDA streams and cupy streams are
    replaced by :class:`StubStream` no-op objects so callers can keep
    using ``stream.synchronize()`` / ``wait_event(...)`` etc. without
    branching on the active backend.

    KV cache tensors are reconstructed from the
    :class:`CpuShmTensorWrapper` instances sent by the client over
    POSIX shared memory -- the server does **not** allocate the KV
    cache itself. This mirrors the GPU-mode CUDA-IPC flow where the
    client owns the buffers and the server only maps them.
    """

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        if not kv_caches:
            raise ValueError(
                "CpuCacheContext requires a non-empty list of "
                "CpuShmTensorWrapper; the legacy server-side "
                "self-allocation path has been removed."
            )

        # First Party
        from lmcache.v1.multiprocess.gpu_context import (
            unwrap_kv_cache_tensors,
        )

        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        self.device_ = torch.device("cpu")
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk

        # Discover layout & build KV layer groups via the same path
        # GPUCacheContext uses, so we don't need to hand-roll any
        # PageBufferShapeDesc here. ``layout_hints`` / ``engine_type``
        # are forwarded so the signature matches GPUCacheContext.
        (
            self._engine_kv_format,
            kv_caches_normalized,
        ) = normalize_kv_and_discover_format(
            unwrapped,
            engine_type,
            layout_hints=layout_hints,
        )
        self.kv_caches_: list[torch.Tensor] = list(kv_caches_normalized)
        self.is_mla_ = is_mla(self._engine_kv_format)
        self.num_layers_ = get_num_layers(self.kv_caches_, self._engine_kv_format)
        self.num_blocks_ = get_num_blocks(self.kv_caches_, self._engine_kv_format)
        self.kv_layer_groups_manager_ = KVLayerGroupsManager(
            self.kv_caches_,
            engine_kv_format=self._engine_kv_format,
            num_blocks=self.num_blocks_,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        # Per-group KV pointer tensors (CPU). Reuse the same helper
        # GPUCacheContext relies on so the layout matches exactly.
        self.group_kv_pointers_: list[torch.Tensor] = [
            torch.tensor(
                get_group_data_ptrs(
                    self.kv_caches_,
                    self.engine_kv_format_,
                    group.layer_indices,
                ),
                dtype=torch.long,
            )
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]

        # Backwards-compat aliases (a few callers still expect these).
        self.hidden_dim_sizes_: list[int] = [
            group.hidden_dim_size
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]
        self.kv_cache_pointers_ = torch.tensor(
            [t.data_ptr() for t in self.kv_caches_], dtype=torch.long
        )

        # Pre-allocated block IDs buffer (CPU).
        _MAX_BLOCK_IDS = 1_000_000
        self.block_ids_buffer_ = torch.empty(_MAX_BLOCK_IDS, dtype=torch.long)

        # Temporary buffer for transfers (same layout as
        # GPUCacheContext but on CPU).
        self.max_batch_size = 4
        self.tmp_chunk_group_offsets_: list[int] = [0]
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            shape = self.get_kv_buffer_shape(lmcache_tokens_per_chunk, group_idx)
            byte_size = shape.numel() * group.dtype.itemsize
            self.tmp_chunk_group_offsets_.append(
                self.tmp_chunk_group_offsets_[-1] + byte_size
            )
        self.tmp_chunk_bytes_ = self.tmp_chunk_group_offsets_[-1]
        # Buffer lives on CPU; keep the attribute name aligned with the
        # context to avoid GPU-prefixed naming bleeding into a CPU-only
        # class. The public ``get_tmp_gpu_buffer_flat`` method name is
        # preserved so ``server.py`` can duck-type across backends.
        self.tmp_cpu_buffer_ = torch.empty(
            self.tmp_chunk_bytes_ * self.max_batch_size,
            dtype=torch.uint8,
        )

        # Mock streams. ``StubStream`` already implements the small
        # subset of the API server-side code uses (``synchronize``,
        # ``wait_event``, ``record_event`` ...), so we never import
        # cupy or instantiate a real CUDA stream object here.
        self.cuda_stream_: StubStream = StubStream(device="cpu")
        self.cupy_stream_: StubStream = self.cuda_stream_
        self.high_priority_cuda_stream_: StubStream = StubStream(
            device="cpu", priority=0
        )
        self.high_priority_cupy_stream_: StubStream = self.high_priority_cuda_stream_

        # Sanity-check: warn if /dev/shm looks too small for the
        # registered KV cache. Only meaningful on Linux where
        # /dev/shm is the default tmpfs backing POSIX SHM.
        self._check_shm_capacity()

        logger.info(
            "CpuCacheContext: %d layers, %d blocks, dtype=%s (shm-backed)",
            self.num_layers_,
            self.num_blocks_,
            self.kv_caches_[0].dtype,
        )

    # -- Internal helpers --

    _SHM_PATH = "/dev/shm"

    def _check_shm_capacity(self) -> None:
        """Warn if /dev/shm free space is smaller than the KV cache."""
        if not os.path.isdir(self._SHM_PATH):
            return
        try:
            st = os.statvfs(self._SHM_PATH)
        except OSError:
            return
        free_bytes = st.f_bavail * st.f_frsize
        kv_bytes = sum(t.numel() * t.element_size() for t in self.kv_caches_)
        if kv_bytes > free_bytes:
            logger.warning(
                "Insufficient /dev/shm space for CPU KV cache: "
                "need %d bytes but only %d bytes available. "
                "Consider increasing the size of /dev/shm "
                "(e.g. mount -o remount,size=<N>G /dev/shm).",
                kv_bytes,
                free_bytes,
            )

    def close(self) -> None:
        """Release resources. No-op for CPU context (no GDS staging buffer)."""
        pass

    # -- Properties (same API as GPUCacheContext) --

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the KV cache tensors."""
        return self.kv_caches_[0].dtype

    @property
    def device(self) -> torch.device:
        """Returns the device (always CPU)."""
        return self.device_

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        """Returns the list of per-layer KV cache tensors."""
        return self.kv_caches_

    @property
    def kv_pointers(self) -> torch.Tensor:
        """Returns a tensor of KV cache data pointers."""
        return self.kv_cache_pointers_

    @property
    def stream(self) -> StubStream:
        """Returns the (mock) CUDA stream."""
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> StubStream:
        """Returns the (mock) external stream."""
        return self.cupy_stream_

    @property
    def high_priority_stream(self) -> StubStream:
        """Returns the (mock) high-priority CUDA stream."""
        return self.high_priority_cuda_stream_

    @property
    def high_priority_cupy_stream(self) -> StubStream:
        """Returns the (mock) high-priority external stream."""
        return self.high_priority_cupy_stream_

    @property
    def block_size(self) -> int:
        """Returns the block size (tokens per block)."""
        return self.kv_layer_groups_manager_.kv_layer_groups[0].shape_desc.bs

    @property
    def num_layers(self) -> int:
        """Returns the number of layers in the model."""
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        """Returns the number of blocks in the KV cache."""
        return self.num_blocks_

    @property
    def is_mla(self) -> bool:
        """Returns whether the model uses MLA."""
        return self.is_mla_

    @property
    def hidden_dim_sizes(self) -> list[int]:
        """Returns hidden dimension sizes per KV layer group."""
        return self.hidden_dim_sizes_

    @property
    def group_slots_per_blocks(self) -> list[int]:
        """Per-group physical slot count (``shape_desc.bs``) in group
        order."""
        return [
            group.shape_desc.bs
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]

    @property
    def kv_layer_groups_manager(self) -> KVLayerGroupsManager:
        """Returns the KV layer groups manager."""
        return self.kv_layer_groups_manager_

    @property
    def engine_kv_format_(self):
        """Returns the GPU KV format enum (API parity with GPUCacheContext)."""
        return self._engine_kv_format

    @property
    def engine_kv_shape(self) -> str:
        """Returns the symbolic GPU KV cache layout description."""
        return get_engine_kv_shape_description(self._engine_kv_format)

    @property
    def attention_backend(self) -> str:
        """Returns the attention backend name."""
        return get_attention_backend(self._engine_kv_format)

    @property
    def concrete_engine_kv_shape(self) -> str:
        """Returns the engine KV shape with actual numeric values."""
        group = self.kv_layer_groups_manager_.kv_layer_groups[0]
        return get_concrete_engine_kv_shape_from_shape_desc(
            group.shape_desc, self._engine_kv_format
        )

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        """Calculate the number of blocks for a given number of tokens.

        Mirrors :meth:`GPUCacheContext.calculate_num_blocks`.

        Args:
            num_tokens: The total number of tokens to be processed.
            kernel_group_idx: 0-based index of the kernel group.

        Returns:
            The number of blocks.
        """
        return self.kv_layer_groups_manager_.calculate_num_blocks(
            kernel_group_idx, num_tokens
        )

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Returns the PageBufferShapeDesc for the given group."""
        return self.kv_layer_groups_manager_.get_shape_desc(group_idx)

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        """Returns the number of slots per lmcache chunk when doing D/H transfer.

        Mirrors :meth:`GPUCacheContext.get_slots_per_chunk_in_sw`.

        Args:
            kernel_group_idx: Index of the kernel group.

        Returns:
            The number of used slots per lmcache chunk when doing D/H transfer.
        """
        return self.kv_layer_groups_manager_.get_slots_per_chunk_in_sw(kernel_group_idx)

    def blocks_for_tokens(self, num_logical_tokens: int, group_idx: int) -> int:
        """Number of blocks that span *num_logical_tokens* for a group.

        Mirrors :meth:`GPUCacheContext.blocks_for_tokens`.
        """
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        physical_slots = (
            num_logical_tokens * group.slots_per_block // group.tokens_per_block
        )
        return physical_slots // group.shape_desc.bs

    def get_group_kv_pointers(self, group_idx: int) -> torch.Tensor:
        """Returns the KV cache pointer tensor for the given group."""
        return self.group_kv_pointers_[group_idx]

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the KV pointer tensor for the given kernel group.

        Mirrors :meth:`GPUCacheContext.get_kernel_group_kv_pointers`.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns the shape and dtype for the given kernel group index and
        number of tokens.

        Mirrors :meth:`GPUCacheContext.get_kernel_group_shape_dtype` so
        callers such as ``lmcache_driven_transfer.get_layout_desc`` can duck-type
        across GPU and CPU backends.

        Args:
            num_tokens: Number of tokens.
            kernel_group_idx: Index of the kernel group.

        Returns:
            A ``(shape, dtype)`` tuple for the given kernel group.
        """
        group = self.kv_layer_groups_manager_.kv_layer_groups[kernel_group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
        if num_tokens % compress_ratio != 0:
            raise ValueError(
                "num_tokens (%d) is not a multiple of compress_ratio (%d) "
                "for kernel_group_idx %d"
                % (num_tokens, compress_ratio, kernel_group_idx)
            )
        num_slots = num_tokens // compress_ratio
        sd = group.shape_desc
        shape = torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )
        return shape, group.dtype

    def get_kv_buffer_shape(
        self, logical_num_tokens: int, group_idx: int = 0
    ) -> torch.Size:
        """Returns the KV buffer shape for the given number of
        *logical* tokens.

        Mirrors :meth:`GPUCacheContext.get_kv_buffer_shape`:
        divides by ``compress_ratio`` and uses ``sd.kv_size`` so
        compressed groups (MLA etc.) get the correct shape.
        """
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
        if logical_num_tokens % compress_ratio != 0:
            raise ValueError(
                "logical_num_tokens (%d) is not a multiple of "
                "compress_ratio (%d) for group %d"
                % (logical_num_tokens, compress_ratio, group_idx)
            )
        num_slots = logical_num_tokens // compress_ratio
        sd = group.shape_desc
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def get_tmp_gpu_buffer_flat(self, chunk_idx: int) -> torch.Tensor:
        """Returns the flat uint8 temp buffer for the given chunk."""
        if chunk_idx >= self.max_batch_size:
            raise ValueError(
                "chunk_idx %d >= max_batch_size %d" % (chunk_idx, self.max_batch_size)
            )
        start = chunk_idx * self.tmp_chunk_bytes_
        return self.tmp_cpu_buffer_[start : start + self.tmp_chunk_bytes_]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns the typed temp buffer for the given batch and kernel group.

        Mirrors :meth:`GPUCacheContext.get_temp_kernel_group_buffer`.

        Args:
            batch_idx: Batch slot index (0 <= batch_idx < max_batch_size).
            kernel_group_idx: Index of the kernel group.

        Returns:
            A typed tensor view with the correct shape and dtype.
        """
        if batch_idx >= self.max_batch_size:
            raise ValueError(
                "batch_idx %d >= max_batch_size %d" % (batch_idx, self.max_batch_size)
            )
        group = self.kv_layer_groups_manager_.kv_layer_groups[kernel_group_idx]
        shape = self.get_kv_buffer_shape(
            self.lmcache_tokens_per_chunk, kernel_group_idx
        )
        g_start = self.tmp_chunk_group_offsets_[kernel_group_idx]
        g_end = self.tmp_chunk_group_offsets_[kernel_group_idx + 1]
        chunk = self.tmp_chunk_bytes_
        return (
            self.tmp_cpu_buffer_[
                batch_idx * chunk + g_start : batch_idx * chunk + g_end
            ]
            .view(group.dtype)
            .view(shape)
        )

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns the flat uint8 temp buffer for the given batch and object
        group.

        Mirrors :meth:`GPUCacheContext.get_temp_object_group_buffer`.

        Args:
            batch_idx: Batch slot index (0 <= batch_idx < max_batch_size).
            object_group_idx: Index of the object group.

        Returns:
            A flat uint8 tensor view covering the object group's byte range.
        """
        if batch_idx >= self.max_batch_size:
            raise ValueError(
                "batch_idx %d >= max_batch_size %d" % (batch_idx, self.max_batch_size)
            )
        manager = self.kv_layer_groups_manager_
        object_group = manager.object_groups[object_group_idx]
        kg_indices = object_group.kernel_group_indices
        # Object group spans from the first to the last kernel group's range.
        g_start = self.tmp_chunk_group_offsets_[kg_indices[0]]
        g_end = self.tmp_chunk_group_offsets_[kg_indices[-1] + 1]
        chunk = self.tmp_chunk_bytes_
        return self.tmp_cpu_buffer_[
            batch_idx * chunk + g_start : batch_idx * chunk + g_end
        ]

    def get_tmp_chunk_gpu_buffer(self, group_idx: int = 0) -> torch.Tensor:
        """Returns a typed view of the temp buffer for one chunk."""
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_tokens_per_chunk, group_idx)
        start = self.tmp_chunk_group_offsets_[group_idx]
        end = self.tmp_chunk_group_offsets_[group_idx + 1]
        return self.tmp_cpu_buffer_[start:end].view(group.dtype).view(shape)

    def get_tmp_chunk_gpu_buffer_batched(
        self, batch_size: int, group_idx: int = 0
    ) -> list[torch.Tensor]:
        """Returns a list of non-overlapping temp buffer views."""
        if batch_size > self.max_batch_size:
            raise ValueError(
                "batch_size %d > max_batch_size %d" % (batch_size, self.max_batch_size)
            )
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_tokens_per_chunk, group_idx)
        g_start = self.tmp_chunk_group_offsets_[group_idx]
        g_end = self.tmp_chunk_group_offsets_[group_idx + 1]
        chunk = self.tmp_chunk_bytes_
        return [
            self.tmp_cpu_buffer_[i * chunk + g_start : i * chunk + g_end]
            .view(group.dtype)
            .view(shape)
            for i in range(batch_size)
        ]

    def stage_block_ids(self, block_ids: list[int]) -> torch.Tensor:
        """Copy block IDs into the pre-allocated buffer."""
        if not block_ids:
            raise ValueError("stage_block_ids requires a non-empty block_ids list")
        n = len(block_ids)
        capacity = self.block_ids_buffer_.shape[0]
        if n > capacity:
            raise ValueError(
                "stage_block_ids: %d block IDs exceeds buffer capacity %d"
                % (n, capacity)
            )
        cpu_tensor = torch.tensor(block_ids, dtype=torch.long)
        buf = self.block_ids_buffer_[:n]
        buf.copy_(cpu_tensor)
        return buf

    def copy_view_block_ids_to_gpu(
        self, block_ids_per_group: list[list[int]]
    ) -> list[torch.Tensor]:
        """CPU-side counterpart to ``GPUCacheContext.copy_view_block_ids_to_gpu``.

        Packs all per-group block IDs into the shared CPU buffer and
        returns one non-overlapping view per LMCache group. The name
        is kept for API parity; on a CPU-only host the buffer simply
        lives on the host.
        """
        offsets = [0]
        flat: array.array = array.array("l")
        for view_block_ids in block_ids_per_group:
            flat.extend(view_block_ids)
            offsets.append(len(flat))

        total = offsets[-1]
        if total > self.block_ids_buffer_.shape[0]:
            raise ValueError(
                "block ID total %d exceeds the pre-allocated buffer "
                "size %d" % (total, self.block_ids_buffer_.shape[0])
            )
        if total:
            cpu_tensor = torch.frombuffer(flat, dtype=torch.long)
            self.block_ids_buffer_[:total].copy_(cpu_tensor)

        return [
            self.block_ids_buffer_[offsets[i] : offsets[i + 1]]
            for i in range(len(block_ids_per_group))
        ]

    def report_status(self) -> dict:
        """Return this context's KV cache layout metadata.

        Mirrors :meth:`GPUCacheContext.report_status` so
        ``LMCacheDrivenTransferModule.report_status`` can duck-type across backends.
        """
        manager = self.kv_layer_groups_manager_
        kernel_groups = manager.kernel_groups

        kernel_group_to_object_group: dict[int, int] = {
            kg_idx: og_idx
            for og_idx, og in enumerate(manager.object_groups)
            for kg_idx in og.kernel_group_indices
        }

        engine_kv_format = self._engine_kv_format
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
            "num_layers": self.num_layers_,
            "num_blocks": self.num_blocks_,
            "cache_size_per_token": self.cache_size_per_token(),
            "kernel_groups": group_reports,
        }

    def cache_size_per_token(self) -> int:
        """Returns cache size per *logical* token in bytes,
        summed across all groups.

        Mirrors :meth:`GPUCacheContext.cache_size_per_token`.
        """
        total = 0
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            compress_ratio = group.tokens_per_block // group.slots_per_block
            numels = self.get_kv_buffer_shape(compress_ratio, group_idx).numel()
            slot_bytes = numels * group.dtype.itemsize
            total += slot_bytes // compress_ratio
        return total
