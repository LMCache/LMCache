# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache context for platforms without CUDA GPUs.

This module lives in the ``platform`` package because it is part
of the cross-platform compatibility layer — it provides the same
public API as :class:`~lmcache.v1.multiprocess.gpu_context.GPUCacheContext`
but keeps all tensors on CPU and routes any CUDA / cupy stream
request through :func:`make_external_stream` so CPU-only hosts
never import ``cupy`` or instantiate a real ``torch.cuda.Stream``.
"""

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    get_group_data_ptrs,
    normalize_kv_and_discover_format,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform.stream import (
    ExternalStreamLike,
    make_external_stream,
)
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class CpuCacheContext:
    """CPU-only cache context with the same public API as
    :class:`GPUCacheContext`.

    All tensors live on CPU.  CUDA streams and cupy streams
    are replaced by the platform compat layer's mock objects
    (installed at import time when CUDA is unavailable).

    KV cache tensors are reconstructed from the
    :class:`CpuShmTensorWrapper` instances sent by the client over
    POSIX shared memory — the server does **not** allocate the KV
    cache itself. This mirrors the GPU-mode CUDA-IPC flow where the
    client owns the buffers and the server only maps them.
    """

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_chunk_size: int = 256,
    ):
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
        self.lmcache_chunk_size = lmcache_chunk_size
        self.is_mla_ = False

        # Discover layout & build KV layer groups via the same path
        # GPUCacheContext uses, so we don't need to hand-roll any
        # PageBufferShapeDesc here. ``EngineType.VLLM`` is fine for
        # the CPU buffers wired through CpuShmTensorWrapper.
        self.gpu_kv_format_, kv_caches_normalized = normalize_kv_and_discover_format(
            unwrapped,
            EngineType.VLLM,
            layout_hints=None,
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
            block_size=self.block_size_,
        )

        # Per-group KV pointer tensors (CPU). Reuse the same helper
        # GPUCacheContext relies on so the layout matches exactly.
        self.group_kv_pointers_: list[torch.Tensor] = [
            torch.tensor(
                get_group_data_ptrs(
                    self.kv_caches_, self.gpu_kv_format_, group.layer_indices
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

        # External streams via the platform dispatcher: on CUDA hosts
        # this returns a cupy-backed stream, on CPU-only hosts the
        # pure-Python mock. Either way we never directly import
        # ``cupy`` or instantiate ``torch.cuda.Stream`` here.
        self.cuda_stream_: torch.cuda.Stream | None = None
        self.cupy_stream_: ExternalStreamLike = make_external_stream(
            None,  # type: ignore[arg-type]
            0,
        )
        self.high_priority_cuda_stream_: torch.cuda.Stream | None = None
        self.high_priority_cupy_stream_: ExternalStreamLike = make_external_stream(
            None,  # type: ignore[arg-type]
            0,
        )

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
    def stream(self) -> torch.cuda.Stream | None:
        """Returns the (mock) CUDA stream — ``None`` on CPU-only hosts."""
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> ExternalStreamLike:
        """Returns the platform-dispatched external stream."""
        return self.cupy_stream_

    @property
    def high_priority_stream(self) -> torch.cuda.Stream | None:
        """Returns the high-priority CUDA stream — ``None`` on CPU."""
        return self.high_priority_cuda_stream_

    @property
    def high_priority_cupy_stream(self) -> ExternalStreamLike:
        """Returns the high-priority platform-dispatched stream."""
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
    def kv_layer_groups_manager(
        self,
    ) -> KVLayerGroupsManager:
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
        """Returns the PageBufferShapeDesc for the given group.

        Args:
            group_idx: Index of the KV layer group.
        """
        return self.kv_layer_groups_manager_.get_shape_desc(group_idx)

    def get_group_kv_pointers(self, group_idx: int) -> torch.Tensor:
        """Returns the KV cache pointer tensor for the given group.

        Args:
            group_idx: Index of the KV layer group.
        """
        return self.group_kv_pointers_[group_idx]

    def get_kv_buffer_shape(self, num_tokens: int, group_idx: int = 0) -> torch.Size:
        """Returns the KV buffer shape for the given token count.

        Args:
            num_tokens: Number of tokens.
            group_idx: Index of the KV layer group.
        """
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        num_layers_in_group = group.num_layers
        hidden_dim = self.hidden_dim_sizes_[group_idx]
        return torch.Size((2, num_layers_in_group, num_tokens, hidden_dim))

    def get_tmp_gpu_buffer_flat(self, chunk_idx: int) -> torch.Tensor:
        """Returns the flat uint8 temp buffer for the given chunk.

        Args:
            chunk_idx: Chunk index (< max_batch_size).

        Raises:
            ValueError: If chunk_idx >= max_batch_size.
        """
        if chunk_idx >= self.max_batch_size:
            raise ValueError(
                "chunk_idx %d >= max_batch_size %d" % (chunk_idx, self.max_batch_size)
            )
        start = chunk_idx * self.tmp_chunk_bytes_
        return self.tmp_gpu_buffer_[start : start + self.tmp_chunk_bytes_]

    def get_tmp_chunk_gpu_buffer(self, group_idx: int = 0) -> torch.Tensor:
        """Returns a typed view of the temp buffer for one chunk.

        Args:
            group_idx: Index of the KV layer group.
        """
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_chunk_size, group_idx)
        start = self.tmp_chunk_group_offsets_[group_idx]
        end = self.tmp_chunk_group_offsets_[group_idx + 1]
        return self.tmp_gpu_buffer_[start:end].view(group.dtype).view(shape)

    def get_tmp_chunk_gpu_buffer_batched(
        self, batch_size: int, group_idx: int = 0
    ) -> list[torch.Tensor]:
        """Returns a list of non-overlapping temp buffer views.

        Args:
            batch_size: Number of concurrent chunks.
            group_idx: Index of the KV layer group.

        Raises:
            ValueError: If batch_size > max_batch_size.
        """
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
        """Copy block IDs into the pre-allocated buffer.

        Args:
            block_ids: Block indices as a Python list.

        Returns:
            A tensor view into the pre-allocated buffer.
        """
        n = len(block_ids)
        cpu_tensor = torch.tensor(block_ids, dtype=torch.long)
        buf = self.block_ids_buffer_[:n]
        buf.copy_(cpu_tensor)
        return buf

    def cache_size_per_token(self) -> int:
        """Returns cache size per token in bytes, summed across groups."""
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
    engine_type: Any = None,
) -> Any:
    """Create the appropriate cache context.

    Selection is driven by the wrapper type of *kv_caches*:

    * Any element is a :class:`CpuShmTensorWrapper` \u2192
      :class:`CpuCacheContext` is built and the underlying CPU
      tensors are mapped from the client-owned POSIX shared-memory
      segments.
    * Otherwise (real ``CudaIPCWrapper`` instances) \u2192
      :class:`GPUCacheContext` is built.

    The legacy ``kv_caches=[]`` + ``layout_hints`` server-side
    self-allocation path has been removed \u2014 the client always
    owns the buffers now.

    Args:
        kv_caches: Non-empty list of KV cache wrappers.
        chunk_size: LMCache chunk size in tokens.
        layout_hints: See :class:`LayoutHints`.  Forwarded to
            ``GPUCacheContext`` only.
        engine_type: Forwarded to ``GPUCacheContext`` for serving-
            engine-specific layout detection.  Ignored in CPU mode.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import CpuShmTensorWrapper

    if not kv_caches:
        raise ValueError("create_cache_context requires a non-empty kv_caches list")

    if any(isinstance(w, CpuShmTensorWrapper) for w in kv_caches):
        return CpuCacheContext(
            kv_caches=kv_caches,
            lmcache_chunk_size=chunk_size,
        )

    # First Party
    from lmcache.v1.multiprocess.gpu_context import (
        GPUCacheContext,
    )

    kwargs: dict[str, Any] = {"layout_hints": layout_hints or None}
    if engine_type is not None:
        kwargs["engine_type"] = engine_type
    return GPUCacheContext(kv_caches, chunk_size, **kwargs)
