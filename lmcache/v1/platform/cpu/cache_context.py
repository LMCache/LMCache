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

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_group_data_ptrs,
    normalize_kv_and_discover_format,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform.cpu.stub_cpu_device import StubStream
import lmcache.c_ops as lmc_ops

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
        lmcache_logical_chunk_size: int = 256,
        layout_hints: LayoutHints | None = None,
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
        self.lmcache_logical_chunk_size = lmcache_logical_chunk_size
        self.is_mla_ = False

        # Discover layout & build KV layer groups via the same path
        # GPUCacheContext uses, so we don't need to hand-roll any
        # PageBufferShapeDesc here. ``layout_hints`` / ``engine_type``
        # are forwarded so the signature matches GPUCacheContext.
        (
            self.gpu_kv_format_,
            kv_caches_normalized,
        ) = normalize_kv_and_discover_format(
            unwrapped,
            engine_type,
            layout_hints=layout_hints,
        )
        self.kv_caches_: list[torch.Tensor] = list(kv_caches_normalized)
        self.num_layers_ = len(self.kv_caches_)
        # ``[2, num_blocks, block_size, num_heads, head_size]`` (NHD).
        first = self.kv_caches_[0]
        self.num_blocks_ = int(first.shape[1])
        self.block_size_ = int(first.shape[2])
        self.kv_layer_groups_manager_ = KVLayerGroupsManager(
            self.kv_caches_,
            gpu_kv_format=self.gpu_kv_format_,
            num_blocks=self.num_blocks_,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=lmcache_logical_chunk_size,
        )

        # Per-group KV pointer tensors (CPU). Reuse the same helper
        # GPUCacheContext relies on so the layout matches exactly.
        self.group_kv_pointers_: list[torch.Tensor] = [
            torch.tensor(
                get_group_data_ptrs(
                    self.kv_caches_,
                    self.gpu_kv_format_,
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
            shape = self.get_kv_buffer_shape(lmcache_logical_chunk_size, group_idx)
            byte_size = shape.numel() * group.dtype.itemsize
            self.tmp_chunk_group_offsets_.append(
                self.tmp_chunk_group_offsets_[-1] + byte_size
            )
        self.tmp_chunk_bytes_ = self.tmp_chunk_group_offsets_[-1]
        self.tmp_gpu_buffer_ = torch.empty(
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

        logger.info(
            "CpuCacheContext: %d layers, blocks=%dx%d, dtype=%s (shm-backed)",
            self.num_layers_,
            self.num_blocks_,
            self.block_size_,
            self.kv_caches_[0].dtype,
        )

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
        return self.block_size_

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
    def kv_layer_groups_manager(self) -> KVLayerGroupsManager:
        """Returns the KV layer groups manager."""
        return self.kv_layer_groups_manager_

    @property
    def gpu_kv_format_name(self) -> str:
        """Returns the GPU KV format enum name."""
        return self.gpu_kv_format_.name

    @property
    def gpu_kv_shape(self) -> str:
        """Returns the GPU KV cache layout description."""
        return "NL_X_TWO_NB_BS_NH_HS"

    @property
    def attention_backend(self) -> str:
        """Returns the attention backend name."""
        return "cpu"

    @property
    def concrete_gpu_kv_shape(self) -> str:
        """Returns the GPU KV shape with actual values."""
        return "cpu-context"

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Returns the PageBufferShapeDesc for the given group."""
        return self.kv_layer_groups_manager_.get_shape_desc(group_idx)

    def get_physical_chunk_size(self, group_idx: int) -> int:
        """Returns the per-chunk physical slot count for the group."""
        return self.kv_layer_groups_manager_.get_physical_chunk_size(group_idx)

    def get_group_kv_pointers(self, group_idx: int) -> torch.Tensor:
        """Returns the KV cache pointer tensor for the given group."""
        return self.group_kv_pointers_[group_idx]

    def get_kv_buffer_shape(self, num_tokens: int, group_idx: int = 0) -> torch.Size:
        """Returns the KV buffer shape for the given token count."""
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        num_layers_in_group = group.num_layers
        hidden_dim = self.hidden_dim_sizes_[group_idx]
        return torch.Size((2, num_layers_in_group, num_tokens, hidden_dim))

    def get_tmp_gpu_buffer_flat(self, chunk_idx: int) -> torch.Tensor:
        """Returns the flat uint8 temp buffer for the given chunk."""
        if chunk_idx >= self.max_batch_size:
            raise ValueError(
                "chunk_idx %d >= max_batch_size %d" % (chunk_idx, self.max_batch_size)
            )
        start = chunk_idx * self.tmp_chunk_bytes_
        return self.tmp_gpu_buffer_[start : start + self.tmp_chunk_bytes_]

    def get_tmp_chunk_gpu_buffer(self, group_idx: int = 0) -> torch.Tensor:
        """Returns a typed view of the temp buffer for one chunk."""
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_logical_chunk_size, group_idx)
        start = self.tmp_chunk_group_offsets_[group_idx]
        end = self.tmp_chunk_group_offsets_[group_idx + 1]
        return self.tmp_gpu_buffer_[start:end].view(group.dtype).view(shape)

    def get_tmp_chunk_gpu_buffer_batched(
        self, batch_size: int, group_idx: int = 0
    ) -> list[torch.Tensor]:
        """Returns a list of non-overlapping temp buffer views."""
        if batch_size > self.max_batch_size:
            raise ValueError(
                "batch_size %d > max_batch_size %d" % (batch_size, self.max_batch_size)
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

    def cache_size_per_token(self) -> int:
        """Returns cache size per token in bytes, summed across groups."""
        total = 0
        for group_idx, _group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            numels = self.get_kv_buffer_shape(1, group_idx).numel()
            total += numels * _group.dtype.itemsize
        return total
