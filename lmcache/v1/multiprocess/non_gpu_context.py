# SPDX-License-Identifier: Apache-2.0
"""Non-GPU context abstractions and utilities for multiprocess mode.

This module provides:
- ``NonGpuContextMetadata``: layout metadata dataclass for non-CUDA workers.
- ``NonGpuContext``: abstract base class with a two-phase prepare/commit
  interface for CPU-side KV data transfer. Concrete implementations (e.g.
  ``NonGpuContextPickle``) each decide *how* data is serialised and transported.
- ``create_non_gpu_context()``: factory that returns the appropriate
  ``NonGpuContext`` subclass (currently always ``NonGpuContextPickle``).
- ``compute_kv_layout``, ``gather_paged_kv_to_cpu``, ``scatter_cpu_to_paged_kv``:
  shared gather/scatter utilities used by all concrete implementations.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc


@dataclass
class NonGpuContextMetadata:
    """Non-GPU context layout metadata for non-CUDA workers.

    Attributes:
        layout_desc: Memory layout descriptor used to interpret chunk payloads.
        block_size: Number of tokens per paged block.
        use_mla: Whether the worker KV format is MLA.
    """

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool


class NonGpuContext(ABC):
    """Abstract base class for CPU-side KV data transfer contexts.

    All concrete implementations share a common message-queue client and
    expose a uniform two-phase ``prepare/commit`` interface so that the
    worker adapter is implementation-agnostic.

    Args:
        metadata: Layout metadata describing the chunk format.
        mq_client: Message-queue client used for server communication.
        mq_timeout: Timeout in seconds for blocking MQ requests.
    """

    def __init__(
        self,
        metadata: NonGpuContextMetadata,
        mq_client: Any,
        mq_timeout: float,
    ) -> None:
        self.metadata = metadata
        self.mq_client = mq_client
        self.mq_timeout = mq_timeout

    @property
    def layout_desc(self) -> MemoryLayoutDesc:
        """The memory layout descriptor for this context."""
        return self.metadata.layout_desc

    @abstractmethod
    def prepare_store(self, key: Any, instance_id: int) -> list[torch.Tensor] | None:
        """Prepare store. Returns pre-allocated out buffers (shm) or None (pickle)."""
        ...

    @abstractmethod
    def commit_store(
        self, key: Any, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Commit store. Pickle: serialize and send. Shm: notify server."""
        ...

    @abstractmethod
    def prepare_retrieve(self, key: Any, instance_id: int) -> list[torch.Tensor] | None:
        """Prepare retrieve. Returns chunks or shm views, or None on miss."""
        ...

    @abstractmethod
    def commit_retrieve(self, key: Any, instance_id: int) -> bool:
        """Commit retrieve. Pickle: no-op. Shm: release read locks."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this context."""
        ...


def create_non_gpu_context(
    metadata: NonGpuContextMetadata,
    mq_client: Any,
    mq_timeout: float,
) -> NonGpuContext:
    """Factory that returns the appropriate :class:`NonGpuContext` implementation.

    Currently always returns a pickle-based implementation
    (``NonGpuContextPickle``). A future SHM-capable PR
    may probe for shared-memory availability and fall back to pickle.

    Args:
        metadata: Layout metadata for the non-GPU context.
        mq_client: Message-queue client for server communication.
        mq_timeout: Timeout in seconds for blocking MQ requests.

    Returns:
        A concrete :class:`NonGpuContext` instance.
    """
    # Local
    from .non_gpu_context_pickle import NonGpuContextPickle

    return NonGpuContextPickle(metadata, mq_client, mq_timeout)


# ---------------------------------------------------------------------------
# Shared gather / scatter utilities
# ---------------------------------------------------------------------------


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: Any | None = None,
) -> tuple[int, int, int, str, Any]:
    """Compute KV layout metadata from KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        layout_hints: Optional engine layout hints.

    Returns:
        Tuple of ``(block_size, num_layers, hidden_dim_size, dtype_str,``
        ``gpu_kv_format)``.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_hidden_dim_size,
        get_num_layers,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors:
        raise ValueError("kv_caches is empty. Cannot compute KV layout.")

    gpu_kv_format, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    block_size = get_block_size(normalized, gpu_kv_format)
    num_layers = get_num_layers(normalized, gpu_kv_format)
    hidden_dim_size = get_hidden_dim_size(normalized, gpu_kv_format)
    dtype_str = str(tensors[0].dtype).replace("torch.", "")
    return block_size, num_layers, hidden_dim_size, dtype_str, gpu_kv_format


def gather_paged_kv_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: Any | None = None,
    gpu_kv_format: Any | None = None,
    out: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Gather paged KV blocks into CPU chunk tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.

    Returns:
        List of CPU tensors, one per chunk. For non-MLA each chunk has shape
        ``[2, num_layers, chunk_tokens, hidden_dim]`` where dimension ``0``
        stores ``(K, V)``. For MLA (multi-head latent attention) each chunk
        has shape ``[num_layers, chunk_tokens, hidden_dim]``.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt
    use_mla = is_mla(gpu_kv_format)
    is_hnd = gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    )

    block_size = get_block_size(normalized, gpu_kv_format)
    num_chunks = len(block_ids) // blocks_per_chunk

    # After normalization the structure is always a list of per-layer
    # tensors. Cast once so all downstream indexing is typed correctly.
    layer_tensors = cast(list[torch.Tensor], normalized)

    chunks: list[torch.Tensor] = [] if out is None else out
    for chunk_idx in range(num_chunks):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        if use_mla:
            mla_layers: list[torch.Tensor] = []
            idx = torch.tensor(chunk_block_ids, dtype=torch.long)
            for layer in layer_tensors:
                layer_blocks = layer[idx]
                mla_layers.append(
                    layer_blocks.reshape(
                        len(chunk_block_ids) * block_size, layer_blocks.shape[-1]
                    )
                )
            chunk_tensor = torch.stack(mla_layers, dim=0)
            if out is not None:
                out[chunk_idx].copy_(chunk_tensor, non_blocking=True)
            else:
                chunks.append(chunk_tensor.cpu())
        else:
            k_layers: list[torch.Tensor] = []
            v_layers: list[torch.Tensor] = []
            for layer in layer_tensors:
                if is_hnd:
                    if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
                        k_t = layer[0]
                        v_t = layer[1]
                    else:
                        k_t = layer[:, 0]
                        v_t = layer[:, 1]
                    _num_blocks, num_heads, _block_size, head_size = k_t.shape
                    k_blocks = k_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    v_blocks = v_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    # HND blocks are [NB, NH, BS, HS]; convert to token-major
                    # [NB, BS, NH, HS] before flattening to [tokens, NH*HS].
                    k_layers.append(
                        k_blocks.permute(0, 2, 1, 3).reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
                    v_layers.append(
                        v_blocks.permute(0, 2, 1, 3).reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
                else:
                    if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                        k_t = layer[0]
                        v_t = layer[1]
                    else:
                        k_t = layer[:, 0]
                        v_t = layer[:, 1]
                    _num_blocks, _block_size, num_heads, head_size = k_t.shape
                    k_blocks = k_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    v_blocks = v_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    k_layers.append(
                        k_blocks.reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
                    v_layers.append(
                        v_blocks.reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
            k_stacked = torch.stack(k_layers, dim=0)
            v_stacked = torch.stack(v_layers, dim=0)
            chunk_tensor = torch.stack([k_stacked, v_stacked], dim=0)
            if out is not None:
                out[chunk_idx].copy_(chunk_tensor, non_blocking=True)
            else:
                chunks.append(chunk_tensor.cpu())
    return chunks


def scatter_cpu_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: Any | None = None,
    gpu_kv_format: Any | None = None,
) -> None:
    """Scatter CPU chunk tensors back into paged KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.
        chunks: List of CPU chunk tensors (as returned by
            :func:`gather_paged_kv_to_cpu`).
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

    if not chunks:
        return

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt
    use_mla = is_mla(gpu_kv_format)

    block_size = get_block_size(normalized, gpu_kv_format)
    device = tensors[0].device
    is_hnd = gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    )

    # After normalization the structure is always a list of per-layer
    # tensors. Cast once so all downstream indexing is typed correctly.
    layer_tensors = cast(list[torch.Tensor], normalized)

    for chunk_idx, chunk_cpu in enumerate(chunks):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        if not chunk_block_ids:
            continue

        chunk_start_token = chunk_idx * blocks_per_chunk * block_size
        chunk_end_token = chunk_start_token + len(chunk_block_ids) * block_size
        effective_start = max(chunk_start_token, skip_first_n_tokens)
        if effective_start >= chunk_end_token:
            continue

        skip_blocks_in_chunk = (effective_start - chunk_start_token) // block_size
        effective_block_ids = chunk_block_ids[skip_blocks_in_chunk:]
        if not effective_block_ids:
            continue

        skip_tokens = skip_blocks_in_chunk * block_size
        chunk_device = chunk_cpu.to(device)

        if use_mla:
            eff_idx = torch.tensor(effective_block_ids, dtype=torch.long)
            for layer_idx, layer in enumerate(layer_tensors):
                mla_src = chunk_device[layer_idx, skip_tokens:]
                hidden_size = layer.shape[-1]
                mla_src_3d = mla_src.reshape(
                    len(effective_block_ids), block_size, hidden_size
                )
                layer[eff_idx] = mla_src_3d
        elif is_hnd:
            for layer_idx, layer in enumerate(layer_tensors):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
                if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
                    k_t = layer[0]
                    v_t = layer[1]
                else:
                    k_t = layer[:, 0]
                    v_t = layer[:, 1]
                _nb, nh, _bs, hs = k_t.shape
                k_blocks = k_src.reshape(
                    len(effective_block_ids), block_size, nh, hs
                ).permute(0, 2, 1, 3)
                v_blocks = v_src.reshape(
                    len(effective_block_ids), block_size, nh, hs
                ).permute(0, 2, 1, 3)
                k_t[effective_block_ids] = k_blocks
                v_t[effective_block_ids] = v_blocks
        else:
            for layer_idx, layer in enumerate(layer_tensors):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
                if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                    k_t = layer[0]
                    v_t = layer[1]
                else:
                    k_t = layer[:, 0]
                    v_t = layer[:, 1]
                _num_blocks, _block_size, num_heads, head_size = k_t.shape
                k_src_4d = k_src.reshape(
                    len(effective_block_ids), block_size, num_heads, head_size
                )
                v_src_4d = v_src.reshape(
                    len(effective_block_ids), block_size, num_heads, head_size
                )
                k_t[effective_block_ids] = k_src_4d
                v_t[effective_block_ids] = v_src_4d
