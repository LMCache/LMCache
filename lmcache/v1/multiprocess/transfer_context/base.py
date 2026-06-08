# SPDX-License-Identifier: Apache-2.0
"""Non-GPU context abstractions and utilities for multiprocess transport.

This module provides:
- ``NonGpuContextMetadata``: layout metadata dataclass for non-CUDA workers.
- ``NonGpuContext``: abstract base class with a two-phase prepare/commit
  interface for CPU-side KV data transfer. Concrete implementations (e.g.
  ``NonGpuContextPickle``) each decide *how* data is serialised and transported.
- ``create_non_gpu_context()``: factory that returns the appropriate
  ``NonGpuContext`` subclass.
- ``compute_kv_layout``, ``gather_paged_kv_to_cpu``, ``scatter_cpu_to_paged_kv``:
  shared gather/scatter utilities used by all concrete implementations.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import MessageQueueClient

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


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
        mq_client: MessageQueueClient,
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
    def prepare_store(
        self, key: IPCCacheEngineKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Prepare SHM buffers for a store operation.

        Returns:
            None: pickle mode — no pre-allocated buffers. Caller gathers all
                chunks to CPU itself and sends the serialized data via
                commit_store.
            ([], []): SHM mode but all chunks already cached. Caller should
                skip gather and commit entirely.
            (tensors, chunk_indices): SHM mode with new chunks to write.
                - tensors[i] is a writable SHM-backed buffer for one chunk.
                - chunk_indices[i] is the position of that chunk in the full
                  block_ids sequence (e.g. [0, 2] means only chunks 0 and 2
                  need writing; chunk 1 is already cached).
                Caller gathers only these chunks into the provided tensors,
                then calls commit_store with empty payload.
        """
        ...

    @abstractmethod
    def commit_store(
        self, key: IPCCacheEngineKey, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Commit store. Pickle: serialize and send. Shm: notify server."""
        ...

    @abstractmethod
    def prepare_retrieve(
        self, key: IPCCacheEngineKey, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Prepare retrieve. Returns chunks or shm views, or None on miss."""
        ...

    @abstractmethod
    def commit_retrieve(self, key: IPCCacheEngineKey, instance_id: int) -> bool:
        """Commit retrieve. Pickle: no-op. Shm: release read locks."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this context."""
        ...


def create_non_gpu_context(
    metadata: NonGpuContextMetadata,
    mq_client: MessageQueueClient,
    mq_timeout: float,
    shm_name: str,
    pool_size: int,
    *,
    use_pickle: bool = False,
) -> NonGpuContext:
    """Factory that returns the appropriate :class:`NonGpuContext` implementation.

    Returns SHM-based implementation when shared-memory pool information is
    available; otherwise falls back to the pickle-based implementation.
    If SHM initialization fails for any reason (e.g. segment not found,
    permission error), gracefully falls back to pickle transport.

    Args:
        metadata: Layout metadata for the non-GPU context.
        mq_client: Message-queue client for server communication.
        mq_timeout: Timeout in seconds for blocking MQ requests.
        shm_name: Shared-memory segment name. Empty values force pickle mode.
        pool_size: Shared-memory pool size in bytes. Non-positive values force
            pickle mode.
        use_pickle: Explicitly use pickle transport even when SHM info is
            available.

    Returns:
        A concrete :class:`NonGpuContext` instance.
    """
    if not shm_name or pool_size <= 0:
        use_pickle = True

    if not use_pickle:
        # Local
        from .shm import NonGpuContextShm

        try:
            logger.info(
                "Creating NonGpuContextShm (shm_name=%s, pool_size=%d)",
                shm_name,
                pool_size,
            )
            return NonGpuContextShm(
                metadata, mq_client, mq_timeout, shm_name, pool_size
            )
        except Exception:
            logger.warning(
                "Failed to initialize SHM context (shm_name=%s), "
                "falling back to pickle transport",
                shm_name,
                exc_info=True,
            )

    # Local
    from .pickle import NonGpuContextPickle

    logger.info("Creating NonGpuContextPickle (pickle transport)")
    return NonGpuContextPickle(metadata, mq_client, mq_timeout)


# ---------------------------------------------------------------------------
# Shared gather / scatter utilities
# ---------------------------------------------------------------------------


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> tuple[int, int, int, str, "lmc_ops.GPUKVFormat"]:
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
    layout_hints: LayoutHints | None = None,
    gpu_kv_format: "lmc_ops.GPUKVFormat" | None = None,
    out: list[torch.Tensor] | None = None,
    chunk_indices: list[int] | None = None,
) -> list[torch.Tensor]:
    """Gather paged KV blocks into CPU chunk tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.
        out: Optional pre-allocated output tensors.  If provided, length
            must be at least ``len(chunk_indices)`` when ``chunk_indices``
            is given, or the total number of chunks otherwise.  Any extra
            buffers beyond the number of gathered chunks are ignored.
        chunk_indices: Optional list of chunk positions (into the full
            ``block_ids`` sequence) to gather.  When provided together with
            ``out``, only those chunks are gathered and written into
            ``out[i]`` in order.  When ``None``, all chunks are gathered
            (backward-compatible behaviour).

    Returns:
        List of CPU tensors, one per chunk. For non-MLA each chunk has shape
        ``[2, num_layers, chunk_tokens, hidden_dim]`` where dimension ``0``
        stores ``(K, V)``. For MLA (multi-head latent attention) each chunk
        has shape ``[num_layers, chunk_tokens, hidden_dim]``.

    Raises:
        ValueError: If ``out`` is provided with fewer buffers than the number
            of gathered chunks.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_hidden_dim_size,
        get_num_blocks,
        get_num_layers,
        is_mla,
        make_page_buffer_shape_desc,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt

    block_size = get_block_size(normalized, gpu_kv_format)
    num_layers = get_num_layers(normalized, gpu_kv_format)
    hidden_dim_size = get_hidden_dim_size(normalized, gpu_kv_format)
    num_blocks = get_num_blocks(normalized, gpu_kv_format)
    num_chunks = len(block_ids) // blocks_per_chunk
    chunk_tokens = blocks_per_chunk * block_size

    shape_desc = make_page_buffer_shape_desc(
        normalized,
        gpu_kv_format,
        layer_idx=0,
        num_layers_in_group=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    iter_indices = (
        list(chunk_indices) if chunk_indices is not None else list(range(num_chunks))
    )
    # Require at least one output buffer per gathered chunk. Extra trailing
    # buffers are ignored (see ``chunks = out[: len(iter_indices)]`` below),
    # mirroring the scatter-side length check for consistency.
    if out is not None and len(out) < len(iter_indices):
        raise ValueError(
            f"out length ({len(out)}) must be at least the number of "
            f"gathered chunks ({len(iter_indices)})"
        )

    if out is None:
        use_mla = is_mla(gpu_kv_format)
        if use_mla:
            chunks = [
                torch.empty(
                    (num_layers, chunk_tokens, hidden_dim_size),
                    dtype=tensors[0].dtype,
                    device=torch.device("cpu"),
                )
                for _ in iter_indices
            ]
        else:
            chunks = [
                torch.empty(
                    (2, num_layers, chunk_tokens, hidden_dim_size),
                    dtype=tensors[0].dtype,
                    device=torch.device("cpu"),
                )
                for _ in iter_indices
            ]
    else:
        # Ignore any extra trailing buffers beyond what we actually gather so
        # the kernel's ``total_blocks % num_objects`` invariant still holds.
        # Return ``out`` unchanged when no trimming is needed so the in-place
        # fill contract (result is out) is preserved.
        if len(out) == len(iter_indices):
            chunks = out
        else:
            chunks = out[: len(iter_indices)]

    selected_block_ids: list[int] = []
    for chunk_idx in iter_indices:
        selected_block_ids.extend(
            block_ids[chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk]
        )

    if selected_block_ids:
        lmc_ops.multi_layer_block_kv_transfer(
            cast("DiscoverableKVCache", normalized),
            chunks,
            selected_block_ids,
            tensors[0].device,
            lmc_ops.TransferDirection.D2H,
            shape_desc,
            chunk_tokens,
            gpu_kv_format,
            0,
        )
    return chunks


def scatter_cpu_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
    gpu_kv_format: "lmc_ops.GPUKVFormat" | None = None,
) -> None:
    """Scatter CPU chunk tensors back into paged KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.  Length
            must be at least ``len(chunks) * blocks_per_chunk``; any extra
            trailing block IDs are ignored.
        chunks: List of CPU chunk tensors (as returned by
            :func:`gather_paged_kv_to_cpu`).
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.  Must be a
            multiple of ``block_size``; non-aligned values are rounded down
            to the nearest whole block and an error is logged (matching the
            GPU transfer path).
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.

    Raises:
        ValueError: If ``block_ids`` is shorter than
            ``len(chunks) * blocks_per_chunk``.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_num_blocks,
        get_num_layers,
        make_page_buffer_shape_desc,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

    if not chunks:
        return

    # Require enough block IDs to cover every chunk. Extra trailing block IDs
    # are ignored by the per-chunk slicing below, mirroring the gather-side
    # ``out`` length check for consistency.
    if len(block_ids) < len(chunks) * blocks_per_chunk:
        raise ValueError(
            f"block_ids length ({len(block_ids)}) must be at least "
            f"len(chunks) ({len(chunks)}) * blocks_per_chunk "
            f"({blocks_per_chunk})"
        )

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt

    block_size = get_block_size(normalized, gpu_kv_format)
    num_layers = get_num_layers(normalized, gpu_kv_format)
    num_blocks = get_num_blocks(normalized, gpu_kv_format)
    chunk_tokens = blocks_per_chunk * block_size

    # Block-level transfer can only skip whole blocks. A non-aligned prefix is
    # rounded down to the nearest block (matching the GPU transfer path in
    # gpu_transfer.py) rather than raising, so a slightly misaligned skip
    # degrades gracefully instead of failing the whole retrieve.
    if skip_first_n_tokens % block_size != 0:
        logger.error(
            "skip_first_n_tokens (%d) is not block-aligned (block_size=%d); "
            "rounding down to %d blocks",
            skip_first_n_tokens,
            block_size,
            skip_first_n_tokens // block_size,
        )
    skip_prefix_n_blocks = skip_first_n_tokens // block_size

    shape_desc = make_page_buffer_shape_desc(
        normalized,
        gpu_kv_format,
        layer_idx=0,
        num_layers_in_group=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    selected_block_ids: list[int] = []
    for chunk_idx in range(len(chunks)):
        selected_block_ids.extend(
            block_ids[chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk]
        )

    if not selected_block_ids:
        return

    lmc_ops.multi_layer_block_kv_transfer(
        cast("DiscoverableKVCache", normalized),
        chunks,
        selected_block_ids,
        tensors[0].device,
        lmc_ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        gpu_kv_format,
        skip_prefix_n_blocks,
    )
