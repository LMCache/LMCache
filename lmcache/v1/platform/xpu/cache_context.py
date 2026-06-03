# SPDX-License-Identifier: Apache-2.0
"""XPU cache context for multiprocess transfer modules."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_dtype,
    get_num_blocks,
    get_num_layers,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.gpu_context import (
    GPUCacheContext,
    _TempGPUBuffer,
    list_to_gpu_tensor,
    unwrap_kv_cache_tensors,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.gpu_connector.utils import (
    get_attention_backend,
    get_concrete_engine_kv_shape_from_shape_desc,
    get_device,
    get_engine_kv_shape_description,
    get_group_data_ptrs,
)


class XpuCacheContext(GPUCacheContext):
    """GPUCacheContext-compatible context without CUDA-only GDS/CuPy setup.

    Args:
        kv_caches: XPU IPC wrappers containing engine KV cache tensors.
        lmcache_tokens_per_chunk: LMCache logical chunk size in tokens.
        layout_hints: Optional KV layout hints from the engine.
        engine_group_infos: Optional engine group metadata.
        engine_type: Serving engine type that produced the KV cache.
    """

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
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

        max_block_ids = 1 << 20
        self.block_ids_buffer_ = torch.empty(
            max_block_ids, dtype=torch.long, device=self.device_
        )
        self._temp_buffer = _TempGPUBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )
        self.cuda_stream_ = torch_dev.Stream(device=self.device_)
        self.cupy_stream_ = None

    def close(self) -> None:
        """Release XPU context resources."""

    @property
    def cupy_stream(self) -> None:
        """XPU does not expose a CuPy stream."""
        return None

    @property
    def group_physical_block_sizes(self) -> list[int]:
        """Per-group physical slot count in KV layer-group order."""
        return [
            group.shape_desc.bs
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]

    @property
    def group_compress_ratios(self) -> list[int]:
        """Per-group compression ratio in KV layer-group order."""
        return [
            group.compress_ratio
            for group in self.kv_layer_groups_manager_.kv_layer_groups
        ]

    def get_physical_chunk_size(self, kernel_group_idx: int) -> int:
        """Returns the physical slot count for one LMCache chunk."""
        return self.kv_layer_groups_manager_.get_physical_chunk_size(kernel_group_idx)

    @property
    def gpu_kv_format_name(self) -> str:
        """Returns the engine KV format enum name."""
        return self.engine_kv_format_.name

    @property
    def gpu_kv_shape(self) -> str:
        """Returns a human-readable engine KV cache shape description."""
        return get_engine_kv_shape_description(self.engine_kv_format_)

    @property
    def concrete_gpu_kv_shape(self) -> str:
        """Returns the engine KV shape with numeric dimensions substituted."""
        return get_concrete_engine_kv_shape_from_shape_desc(
            self.kv_layer_groups_manager_.get_shape_desc(0),
            self.engine_kv_format_,
        )

    @property
    def attention_backend(self) -> str:
        """Returns the attention backend name."""
        return get_attention_backend(self.engine_kv_format_)

    def cache_size_per_token(self) -> int:
        """Returns the total KV cache size per token in bytes."""
        return self._temp_buffer.get_cache_size_per_token()

    @property
    def dtype(self) -> torch.dtype:
        """Returns the KV cache dtype."""
        return get_dtype(self.kv_caches_, self.engine_kv_format_)
