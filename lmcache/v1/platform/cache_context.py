# SPDX-License-Identifier: Apache-2.0
"""Cross-platform cache context abstraction.

Defines ``CacheContextBase`` — the interface that ``server.py``
programs against — and ``CpuCacheContext``, a concrete
implementation that uses plain CPU tensors to emulate a paged
GPU KV cache.  On CUDA platforms the existing
``GPUCacheContext`` (in ``multiprocess/gpu_context.py``) is
used instead; both share the same duck-typed interface.

Public API::

    from lmcache.v1.platform.cache_context import (
        CacheContextBase,
        CpuCacheContext,
    )
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any
import abc
import array

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import (
    KVLayerGroupsManager,
)
from lmcache.v1.platform.gpu_connector import (
    MockCudaStream,
)
from lmcache.v1.platform.ops import lmc_ops

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


# ------------------------------------------------------------------
# ABC
# ------------------------------------------------------------------


class CacheContextBase(abc.ABC):
    """Interface that ``MPCacheEngine.store / retrieve`` rely on.

    Both ``GPUCacheContext`` (CUDA) and ``CpuCacheContext``
    (CPU-only) satisfy this contract.
    """

    # Attributes that concrete classes must set
    lmcache_chunk_size: int
    max_batch_size: int
    gpu_kv_format_: Any  # GPUKVFormat enum or None

    # -- scalar properties --

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abc.abstractmethod
    def stream(self) -> Any: ...

    @property
    @abc.abstractmethod
    def cupy_stream(self) -> Any: ...

    @property
    @abc.abstractmethod
    def high_priority_stream(self) -> Any: ...

    @property
    @abc.abstractmethod
    def high_priority_cupy_stream(self) -> Any: ...

    @property
    @abc.abstractmethod
    def block_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abc.abstractmethod
    def num_blocks(self) -> int: ...

    @property
    @abc.abstractmethod
    def is_mla(self) -> bool: ...

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abc.abstractmethod
    def kv_tensors(self) -> list[torch.Tensor]: ...

    @property
    @abc.abstractmethod
    def kv_pointers(self) -> torch.Tensor: ...

    @property
    @abc.abstractmethod
    def hidden_dim_sizes(self) -> list[int]: ...

    @property
    @abc.abstractmethod
    def kv_layer_groups_manager(
        self,
    ) -> KVLayerGroupsManager: ...

    # -- methods --

    @abc.abstractmethod
    def get_kv_buffer_shape(
        self,
        num_tokens: int,
        group_idx: int = 0,
    ) -> torch.Size: ...

    @abc.abstractmethod
    def get_shape_desc(
        self,
        group_idx: int,
    ) -> Any: ...

    @abc.abstractmethod
    def get_group_kv_pointers(
        self,
        group_idx: int,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def get_tmp_gpu_buffer_flat(
        self,
        chunk_idx: int = 0,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def get_tmp_chunk_gpu_buffer(
        self,
        group_idx: int = 0,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def get_tmp_chunk_gpu_buffer_batched(
        self,
        batch_size: int,
        group_idx: int = 0,
    ) -> list[torch.Tensor]: ...

    @abc.abstractmethod
    def stage_block_ids(
        self,
        block_ids: list[int],
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def gpu_kv_format_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def gpu_kv_shape(self) -> str: ...

    @property
    @abc.abstractmethod
    def attention_backend(self) -> str: ...

    @property
    @abc.abstractmethod
    def concrete_gpu_kv_shape(self) -> str: ...

    @abc.abstractmethod
    def cache_size_per_token(self) -> int: ...


# ------------------------------------------------------------------
# CPU implementation
# ------------------------------------------------------------------


class CpuCacheContext(CacheContextBase):
    """CPU-only cache context with **real** KV cache tensors.

    Allocates CPU tensors that mirror the paged GPU KV cache
    layout so that the full ``store`` / ``retrieve`` code-path
    (including ``lmc_ops`` calls and ``memcpy``) executes with
    real data movement — just on CPU instead of GPU.

    Args:
        chunk_size: LMCache chunk size (tokens per chunk).
        num_layers: Number of KV layers to simulate.
        num_heads: Number of attention heads per layer.
        head_size: Size of each attention head.
        num_blocks: Number of paged blocks.
        block_size: Tokens per block.
        dtype: KV cache data type.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        num_layers: int = 32,
        num_heads: int = 8,
        head_size: int = 128,
        num_blocks: int = 1024,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        max_block_ids: int = 1_000_000,
    ) -> None:
        self.lmcache_chunk_size = chunk_size
        self.max_batch_size = 4

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_size = head_size
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._dtype = dtype
        self._hidden_dim = num_heads * head_size

        # Allocate real CPU KV cache tensors — one per layer,
        # shape: [2, num_blocks, block_size, num_heads, head_size]
        # matching NL_X_TWO_NB_BS_NH_HS format.
        # Use deterministic random data so that different blocks
        # carry distinguishable content (important for checksum
        # verification in tests).
        rng = torch.Generator(device="cpu")
        rng.manual_seed(42)
        self.kv_caches_: list[torch.Tensor] = []
        for _ in range(num_layers):
            t = torch.randn(
                2,
                num_blocks,
                block_size,
                num_heads,
                head_size,
                dtype=dtype,
                device="cpu",
                generator=rng,
            )
            self.kv_caches_.append(t)

        # Pointers
        pointers_list = [t.data_ptr() for t in self.kv_caches_]
        self.kv_cache_pointers_ = torch.frombuffer(
            array.array("q", pointers_list),
            dtype=torch.long,
        ).clone()

        # Build real KV layer groups
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
            self.hidden_dim_sizes_.append(self._hidden_dim)
            self.group_num_heads_.append(num_heads)
            self.group_head_sizes_.append(head_size)

            sd = lmc_ops.PageBufferShapeDesc()
            sd.kv_size = kv_size
            sd.nl = group.num_layers
            sd.nb = num_blocks
            sd.bs = block_size
            sd.nh = num_heads
            sd.hs = head_size
            sd.element_size = torch.tensor([], dtype=dtype).element_size()
            sd.dtype = dtype
            self.shape_descs_.append(sd)

            self.group_kv_pointers_.append(
                torch.frombuffer(
                    array.array(
                        "q",
                        [self.kv_caches_[i].data_ptr() for i in group.layer_indices],
                    ),
                    dtype=torch.long,
                ).clone()
            )

        # GPU KV format — use the non-CUDA enum
        self.gpu_kv_format_ = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS

        # Pre-allocated block IDs buffer
        self.max_block_ids_ = max_block_ids
        self.block_ids_buffer_ = torch.empty(max_block_ids, dtype=torch.long)

        # Temporary transfer buffer — same layout as
        # GPUCacheContext but on CPU
        self.tmp_chunk_group_offsets_: list[int] = [0]
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            shape = self.get_kv_buffer_shape(chunk_size, group_idx)
            byte_size = shape.numel() * dtype.itemsize
            self.tmp_chunk_group_offsets_.append(
                self.tmp_chunk_group_offsets_[-1] + byte_size
            )
        self.tmp_chunk_bytes_ = self.tmp_chunk_group_offsets_[-1]
        self.tmp_gpu_buffer_ = torch.zeros(
            self.tmp_chunk_bytes_ * self.max_batch_size,
            dtype=torch.uint8,
        )

        # Mock streams (synchronous on CPU)
        self._mock_stream = MockCudaStream()

        logger.info(
            "CpuCacheContext created: %d layers, "
            "%d heads x %d head_size, "
            "block_size=%d, num_blocks=%d, chunk_size=%d",
            num_layers,
            num_heads,
            head_size,
            block_size,
            num_blocks,
            chunk_size,
        )

    # -- scalar properties --

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def stream(self) -> MockCudaStream:
        return self._mock_stream

    @property
    def cupy_stream(self) -> MockCudaStream:
        return self._mock_stream

    @property
    def high_priority_stream(self) -> MockCudaStream:
        return self._mock_stream

    @property
    def high_priority_cupy_stream(self) -> MockCudaStream:
        return self._mock_stream

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def is_mla(self) -> bool:
        return False

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        return self.kv_caches_

    @property
    def kv_pointers(self) -> torch.Tensor:
        return self.kv_cache_pointers_

    @property
    def hidden_dim_sizes(self) -> list[int]:
        return self.hidden_dim_sizes_

    @property
    def kv_layer_groups_manager(
        self,
    ) -> KVLayerGroupsManager:
        return self.kv_layer_groups_manager_

    # -- methods --

    def get_kv_buffer_shape(
        self,
        num_tokens: int,
        group_idx: int = 0,
    ) -> torch.Size:
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        num_layers_in_group = group.num_layers
        hidden_dim = self.hidden_dim_sizes[group_idx]
        return torch.Size((2, num_layers_in_group, num_tokens, hidden_dim))

    def get_shape_desc(
        self,
        group_idx: int,
    ) -> lmc_ops.PageBufferShapeDesc:
        return self.shape_descs_[group_idx]

    def get_group_kv_pointers(
        self,
        group_idx: int,
    ) -> torch.Tensor:
        return self.group_kv_pointers_[group_idx]

    def get_tmp_gpu_buffer_flat(
        self,
        chunk_idx: int = 0,
    ) -> torch.Tensor:
        if chunk_idx >= self.max_batch_size:
            raise ValueError(
                "chunk_idx %d exceeds max_batch_size %d"
                % (chunk_idx, self.max_batch_size)
            )
        start = chunk_idx * self.tmp_chunk_bytes_
        end = start + self.tmp_chunk_bytes_
        return self.tmp_gpu_buffer_[start:end]

    def get_tmp_chunk_gpu_buffer(
        self,
        group_idx: int = 0,
    ) -> torch.Tensor:
        group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_chunk_size, group_idx)
        start = self.tmp_chunk_group_offsets_[group_idx]
        end = self.tmp_chunk_group_offsets_[group_idx + 1]
        return self.tmp_gpu_buffer_[start:end].view(group.dtype).view(shape)

    def get_tmp_chunk_gpu_buffer_batched(
        self,
        batch_size: int,
        group_idx: int = 0,
    ) -> list[torch.Tensor]:
        if batch_size > self.max_batch_size:
            raise ValueError(
                "batch_size %d exceeds max_batch_size %d"
                % (batch_size, self.max_batch_size)
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

    def stage_block_ids(
        self,
        block_ids: list[int],
    ) -> torch.Tensor:
        n = len(block_ids)
        if n == 0:
            return self.block_ids_buffer_[:0]
        cpu_tensor = torch.frombuffer(array.array("q", block_ids), dtype=torch.long)
        buf = self.block_ids_buffer_[:n]
        buf.copy_(cpu_tensor)
        return buf

    def gpu_kv_format_name(self) -> str:
        return self.gpu_kv_format_.name

    @property
    def gpu_kv_shape(self) -> str:
        return "2 x NL x BS x NH*HS (cpu)"

    @property
    def attention_backend(self) -> str:
        return "cpu"

    @property
    def concrete_gpu_kv_shape(self) -> str:
        return "2 x %d x BS x %d (cpu)" % (
            self._num_layers,
            self._hidden_dim,
        )

    def cache_size_per_token(self) -> int:
        total = 0
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            numels = self.get_kv_buffer_shape(1, group_idx).numel()
            total += numels * group.dtype.itemsize
        return total
