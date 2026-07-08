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
from lmcache.v1.platform.base_cache_context import BaseCacheContext
from lmcache.v1.platform.xpu.ipc_wrapper import RawSyclIPCWrapper


def _unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    return [ipc_wrapper.to_tensor() for ipc_wrapper in kv_caches]


class _TempXPUBuffer:
    """Temporary XPU staging buffer for KV transfer batches."""

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

        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}
        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_size = 0
                object_group_start_offset = offset
                object_group = self._kv_groups_manager.object_groups[object_group_idx]
                for kernel_group_idx in object_group.kernel_group_indices:
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

        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk,
                kernel_group_idx,
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, group.dtype)

    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks processed concurrently in one batch."""
        return self._max_batch_size

    def get_temp_kernel_group_buffer(
        self,
        batch_idx: int,
        kernel_group_idx: int,
    ) -> torch.Tensor:
        """Return the typed staging buffer view for a kernel group."""
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )
        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return the flat uint8 staging buffer view for an object group."""
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
        """Return the shape and dtype for a kernel group."""
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._get_shape_for_kernel_group(num_tokens, kernel_group_idx), dtype

    def get_cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token."""
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
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
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk,
            kernel_group_idx,
        )
        dtype = self._kv_groups_manager.kernel_groups[kernel_group_idx].dtype
        return shape.numel() * dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )


class XpuCacheContext(BaseCacheContext):
    """XPU cache context for engine KV tensors.

    Args:
        kv_caches: XPU IPC wrappers containing engine KV cache tensors.
        lmcache_tokens_per_chunk: LMCache logical chunk size in tokens.
        layout_hints: Optional KV layout hints from the engine.
        engine_group_infos: Optional engine group metadata.
        engine_type: Serving engine type that produced the KV cache.
        separate_object_groups: Whether to split kernel groups by object group.
    """

    device_type = "xpu"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = True,
    ) -> None:
        unwrapped = _unwrap_kv_cache_tensors(kv_caches)
        kv_caches_norm, engine_kv_formats = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        device = get_device(kv_caches_norm)
        kv_layer_groups_manager = KVLayerGroupsManager(
            kv_caches_norm,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=separate_object_groups,
        )
        block_ids_buffer = torch.empty(1 << 20, dtype=torch.long, device=device)

        super().__init__(
            kv_caches=kv_caches_norm,
            device=device,
            num_layers=len(engine_kv_formats),
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self.group_kv_pointers_: list[list[int]] = []
        for idx, group in enumerate(self.kv_layer_groups_manager_.kv_layer_groups):
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.get_engine_kv_format(idx), group.layer_indices
            )
            self.group_kv_pointers_.append(ptrs)

        self._temp_buffer = _TempXPUBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=device,
            max_batch_size=4,
        )
        self.xpu_stream_ = torch_dev.Stream(device=device)

    def close(self) -> None:
        """Release XPU context resources."""
        RawSyclIPCWrapper.clear_opened_ipc_tensors()

    @property
    def stream(self) -> object:
        """Return the XPU stream for KV cache operations."""
        return self.xpu_stream_

    @property
    def cupy_stream(self) -> None:
        """Return the stream adapter used by optional array integrations."""
        return None

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> list[int]:
        """Return the pre-computed KV cache data pointers for an XPU group."""
        return self.group_kv_pointers_[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self,
        batch_idx: int,
        kernel_group_idx: int,
    ) -> torch.Tensor:
        """Return the temporary XPU buffer for a kernel group."""
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx,
            kernel_group_idx,
        )

    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks processed concurrently in one batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return the temporary XPU buffer for an object group."""
        return self._temp_buffer.get_temp_object_group_buffer(
            batch_idx,
            object_group_idx,
        )

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return the shape and dtype for a kernel group."""
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens,
            kernel_group_idx,
        )

    def cache_size_per_token(self) -> int:
        """Return the total KV cache size per logical token in bytes."""
        return self._temp_buffer.get_cache_size_per_token()
