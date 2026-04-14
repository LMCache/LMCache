# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache context for platforms without CUDA GPUs.

This module lives in the ``platform`` package because it is part
of the cross-platform compatibility layer — it provides the same
public API as :class:`~lmcache.v1.multiprocess.gpu_context.GPUCacheContext`
but allocates all tensors on CPU and uses the mock CUDA / cupy
objects installed by :func:`_install_cuda_compat` /
:func:`_install_cupy_compat`.
"""

# Standard
from typing import Any, Mapping

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class CpuCacheContext:
    """CPU-only cache context with the same public API as
    :class:`GPUCacheContext`.

    All tensors live on CPU.  CUDA streams and cupy streams
    are replaced by the platform compat layer's mock objects
    (installed at import time when CUDA is unavailable).

    This allows ``store`` / ``retrieve`` in
    :class:`MPCacheEngine` to run the same code path on
    CPU-only machines — ``lmc_ops`` falls back to the pure-
    Python ``non_cuda_equivalents`` automatically.
    """

    def __init__(
        self,
        layout_hints: Mapping[str, Any],
        lmcache_chunk_size: int = 256,
    ):
        num_layers: int = layout_hints.get("num_layers", 32)
        num_heads: int = layout_hints.get("num_heads", 8)
        head_size: int = layout_hints.get("head_size", 128)
        num_blocks: int = layout_hints.get("num_blocks", 1024)
        block_size: int = layout_hints.get("block_size", 16)
        dtype_str: str = layout_hints.get("dtype", "float16")
        dtype = _DTYPE_MAP.get(dtype_str, torch.float16)

        self.device_ = torch.device("cpu")
        self.num_layers_ = num_layers
        self.num_blocks_ = num_blocks
        self.block_size_ = block_size
        self.is_mla_ = False
        self.lmcache_chunk_size = lmcache_chunk_size

        # Allocate paged KV cache tensors on CPU
        # Shape: [2, num_blocks, block_size, num_heads, head_size]
        # (NL_X_TWO_NB_BS_NH_HS layout)
        self.kv_caches_: list[torch.Tensor] = [
            torch.zeros(
                (2, num_blocks, block_size, num_heads, head_size),
                dtype=dtype,
                device="cpu",
            )
            for _ in range(num_layers)
        ]

        # Pointers
        pointers_list = [t.data_ptr() for t in self.kv_caches_]
        self.kv_cache_pointers_ = torch.tensor(pointers_list, dtype=torch.long)

        # GPU KV format: use NL_X_TWO_NB_BS_NH_HS (value 1)
        self.gpu_kv_format_ = lmc_ops.GPUKVFormat(1)

        # Build KV layer groups
        self.kv_layer_groups_manager_ = KVLayerGroupsManager()
        self.kv_layer_groups_manager_.build_kv_layer_groups_from_list(self.kv_caches_)

        # Per-group attributes
        kv_size = 2
        self.hidden_dim_sizes_: list[int] = []
        self.group_num_heads_: list[int] = []
        self.group_head_sizes_: list[int] = []
        self.shape_descs_: list[lmc_ops.PageBufferShapeDesc] = []
        self.group_kv_pointers_: list[torch.Tensor] = []
        for group in self.kv_layer_groups_manager_.kv_layer_groups:
            hidden_dim = num_heads * head_size
            self.hidden_dim_sizes_.append(hidden_dim)
            self.group_num_heads_.append(num_heads)
            self.group_head_sizes_.append(head_size)

            sd = lmc_ops.PageBufferShapeDesc()
            sd.kv_size = kv_size
            sd.nl = group.num_layers
            sd.nb = num_blocks
            sd.bs = block_size
            sd.nh = num_heads
            sd.hs = head_size
            sd.element_size = dtype.itemsize
            sd.dtype = dtype
            self.shape_descs_.append(sd)

            self.group_kv_pointers_.append(
                torch.tensor(
                    [self.kv_caches_[i].data_ptr() for i in group.layer_indices],
                    dtype=torch.long,
                )
            )

        # Pre-allocated block IDs buffer (CPU)
        _MAX_BLOCK_IDS = 1_000_000
        self.block_ids_buffer_ = torch.empty(_MAX_BLOCK_IDS, dtype=torch.long)

        # Temporary buffer for transfers (same layout as
        # GPUCacheContext but on CPU)
        self.max_batch_size = 4
        self.tmp_chunk_group_offsets_: list[int] = [0]
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            shape = self.get_kv_buffer_shape(lmcache_chunk_size, group_idx)
            byte_size = shape.numel() * group.dtype.itemsize
            self.tmp_chunk_group_offsets_.append(
                self.tmp_chunk_group_offsets_[-1] + byte_size
            )
        self.tmp_chunk_bytes_ = self.tmp_chunk_group_offsets_[-1]
        self.tmp_gpu_buffer_ = torch.empty(
            self.tmp_chunk_bytes_ * self.max_batch_size,
            dtype=torch.uint8,
        )

        # Mock CUDA / cupy streams (platform compat layer)
        self.cuda_stream_ = torch.cuda.Stream(device=self.device_)
        self.cupy_stream_ = cupy.cuda.ExternalStream(0, 0)

        _, high_priority = torch.cuda.Stream.priority_range()
        self.high_priority_cuda_stream_ = torch.cuda.Stream(
            device=self.device_, priority=high_priority
        )
        self.high_priority_cupy_stream_ = cupy.cuda.ExternalStream(0, 0)

        logger.info(
            "CpuCacheContext: %d layers, %dx%d heads, blocks=%dx%d, dtype=%s",
            num_layers,
            num_heads,
            head_size,
            num_blocks,
            block_size,
            dtype,
        )

    # -- Properties (same API as GPUCacheContext) --

    @property
    def dtype(self) -> torch.dtype:
        return self.kv_caches_[0].dtype

    @property
    def device(self) -> torch.device:
        return self.device_

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        return self.kv_caches_

    @property
    def kv_pointers(self) -> torch.Tensor:
        return self.kv_cache_pointers_

    @property
    def stream(self) -> torch.cuda.Stream:
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> cupy.cuda.Stream:
        return self.cupy_stream_

    @property
    def high_priority_stream(self) -> torch.cuda.Stream:
        return self.high_priority_cuda_stream_

    @property
    def high_priority_cupy_stream(self) -> cupy.cuda.Stream:
        return self.high_priority_cupy_stream_

    @property
    def block_size(self) -> int:
        return self.block_size_

    @property
    def num_layers(self) -> int:
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        return self.num_blocks_

    @property
    def is_mla(self) -> bool:
        return self.is_mla_

    @property
    def hidden_dim_sizes(self) -> list[int]:
        return self.hidden_dim_sizes_

    @property
    def kv_layer_groups_manager(
        self,
    ) -> KVLayerGroupsManager:
        return self.kv_layer_groups_manager_

    @property
    def gpu_kv_format_name(self) -> str:
        return self.gpu_kv_format_.name

    @property
    def gpu_kv_shape(self) -> str:
        return "NL_X_TWO_NB_BS_NH_HS"

    @property
    def attention_backend(self) -> str:
        return "cpu"

    @property
    def concrete_gpu_kv_shape(self) -> str:
        return "cpu-context"

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        return self.shape_descs_[group_idx]

    def get_group_kv_pointers(self, group_idx: int) -> torch.Tensor:
        return self.group_kv_pointers_[group_idx]

    def get_kv_buffer_shape(self, num_tokens: int, group_idx: int = 0) -> torch.Size:
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        num_layers_in_group = group.num_layers
        hidden_dim = self.hidden_dim_sizes_[group_idx]
        return torch.Size((2, num_layers_in_group, num_tokens, hidden_dim))

    def get_tmp_gpu_buffer_flat(self, chunk_idx: int) -> torch.Tensor:
        if chunk_idx >= self.max_batch_size:
            raise ValueError(
                "chunk_idx %d >= max_batch_size %d" % (chunk_idx, self.max_batch_size)
            )
        start = chunk_idx * self.tmp_chunk_bytes_
        return self.tmp_gpu_buffer_[start : start + self.tmp_chunk_bytes_]

    def get_tmp_chunk_gpu_buffer(self, group_idx: int = 0) -> torch.Tensor:
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_chunk_size, group_idx)
        start = self.tmp_chunk_group_offsets_[group_idx]
        end = self.tmp_chunk_group_offsets_[group_idx + 1]
        return self.tmp_gpu_buffer_[start:end].view(group.dtype).view(shape)

    def get_tmp_chunk_gpu_buffer_batched(
        self, batch_size: int, group_idx: int = 0
    ) -> list[torch.Tensor]:
        if batch_size > self.max_batch_size:
            raise ValueError(
                "batch_size %d > max_batch_size %d" % (batch_size, self.max_batch_size)
            )
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_chunk_size, group_idx)
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
        n = len(block_ids)
        cpu_tensor = torch.tensor(block_ids, dtype=torch.long)
        buf = self.block_ids_buffer_[:n]
        buf.copy_(cpu_tensor)
        return buf

    def cache_size_per_token(self) -> int:
        total = 0
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            numels = self.get_kv_buffer_shape(1, group_idx).numel()
            total += numels * group.dtype.itemsize
        return total


def create_cache_context(
    kv_caches: Any,
    chunk_size: int,
    layout_hints: Any = None,
) -> Any:
    """Create the appropriate cache context.

    On CUDA platforms with non-empty *kv_caches*,
    ``GPUCacheContext`` is returned.  Otherwise a
    ``CpuCacheContext`` is created from *layout_hints*.
    """
    if torch.cuda.is_available() and kv_caches:
        # First Party
        from lmcache.v1.multiprocess.gpu_context import (
            GPUCacheContext,
        )

        return GPUCacheContext(
            kv_caches,
            chunk_size,
            layout_hints=layout_hints or None,
        )

    # CPU mode — forward layout_hints as-is
    hints = layout_hints if isinstance(layout_hints, dict) else {}
    return CpuCacheContext(
        layout_hints=hints,
        lmcache_chunk_size=chunk_size,
    )
