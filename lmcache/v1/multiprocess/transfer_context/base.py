# SPDX-License-Identifier: Apache-2.0
"""Non-GPU context abstractions and utilities for multiprocess transport.

This module provides:
- ``EngineDrivenContextMetadata``: layout metadata dataclass for non-CUDA workers.
- ``EngineDrivenContext``: abstract base class with a two-phase prepare/commit
  interface for CPU-side KV data transfer. Concrete implementations each decide
  *how* data is serialised and transported.
- ``create_engine_driven_context()``: factory that returns the appropriate
  ``EngineDrivenContext`` subclass.
- ``compute_kv_layout``, ``gather_paged_kv_to_cpu``, ``scatter_cpu_to_paged_kv``:
  shared gather/scatter utilities used by all concrete implementations.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
import inspect

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform.native_block_transfer import transfer_with_address_kernel
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    pass

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def _detect_block_transfer_accepts_tensor(transfer_fn: Callable[..., None]) -> bool:
    """Inspect the selected transfer op, caching capabilities per callable.

    Tensor-capable ops receive tensor lists directly. Native pointer-only ops
    (including uninspectable pybind functions) need pointer conversion and
    pinned host buffers. A process may use both CPU and accelerator ops.
    """
    try:
        # TODO: Replace annotation inspection with an explicit ops capability.
        sig = inspect.signature(transfer_fn)
        param = sig.parameters.get("lmcache_objects_ptrs")
        if param is not None and param.annotation is not inspect.Parameter.empty:
            return "Tensor" in str(param.annotation)
    except (ValueError, TypeError):
        pass
    return False


# ---------------------------------------------------------------------------


@dataclass
class EngineDrivenContextMetadata:
    """Non-GPU context layout metadata for non-CUDA workers.

    Attributes:
        layout_desc: Memory layout descriptor used to interpret chunk payloads
            (group 0's layout for multi-group workers).
        block_size: Number of tokens per paged block.
        use_mla: Whether the worker KV format is MLA.
        group_layouts: Per-LMCache-group layouts for multi-group (hybrid-KV)
            workers, in protocol group order. ``None`` means single-group.
        group_windows: Per-group trailing-chunk windows parallel to
            ``group_layouts`` (``-1`` = full attention). A retrieve serves
            only each group's in-window suffix. ``None`` means every group is
            full attention.
    """

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool
    group_layouts: "list[MemoryLayoutDesc] | None" = None
    group_windows: "list[int] | None" = None


class EngineDrivenContext(ABC):
    """Abstract base class for CPU-side KV data transfer contexts.

    All concrete implementations share a common request client and
    expose a uniform two-phase ``prepare/commit`` interface so that the
    worker adapter is implementation-agnostic.

    Args:
        metadata: Layout metadata describing the chunk format.
        req_client: Transport-neutral client used for server requests.
        mq_timeout: Timeout in seconds for blocking MQ requests.
    """

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        req_client: RequestClient,
        mq_timeout: float,
    ) -> None:
        self.metadata = metadata
        self.req_client = req_client
        self.mq_timeout = mq_timeout

    @property
    def layout_desc(self) -> MemoryLayoutDesc:
        """The memory layout descriptor for this context."""
        return self.metadata.layout_desc

    @abstractmethod
    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Prepare SHM buffers for a store operation.

        Returns:
            None: inline mode — no pre-allocated buffers. Caller gathers all
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
        self, key: IPCCacheServerKey, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Commit store. Inline: serialize and send. Shm: notify server."""
        ...

    @abstractmethod
    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int, expected_count: int
    ) -> list[torch.Tensor] | None:
        """Prepare retrieve for exactly ``expected_count`` chunks.

        Args:
            key: Cache key to retrieve.
            instance_id: Worker instance identifier.
            expected_count: Required number of LMCache chunks.

        Returns:
            Chunks or SHM views on a hit, or ``None`` on a miss.
        """
        ...

    def prepare_store_grouped(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        skipped_chunks: tuple[tuple[int, ...], ...] = (),
    ) -> tuple[list[torch.Tensor], list[int], list[int]] | None:
        """Multi-group store prepare (SHM transport only).

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            skipped_chunks: Per group, the chunk positions the worker will not
                store (all-null engine blocks); the server reserves no slot for
                them. Empty means none.

        Returns:
            Parallel per-slot ``(tensors, chunk_indices, group_ids)`` lists,
            ``([], [], [])`` when fully cached, or ``None`` on a malformed
            response.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support multi-group transfers"
        )

    def prepare_retrieve_grouped(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int], list[int]] | None:
        """Multi-group retrieve prepare (SHM transport only).

        Returns:
            Parallel per-slot ``(tensors, group_ids, chunk_indices)`` lists
            covering each group's in-window chunks, or ``None`` on a miss.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support multi-group transfers"
        )

    @abstractmethod
    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit retrieve. Inline: no-op. Shm: release read locks."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this context."""
        ...


def create_engine_driven_context(
    metadata: EngineDrivenContextMetadata,
    req_client: RequestClient,
    mq_timeout: float,
    shm_name: str,
    pool_size: int,
) -> EngineDrivenContext:
    """Factory that returns the appropriate :class:`EngineDrivenContext` implementation.

    Returns SHM-based implementation when shared-memory pool information is
    available; otherwise falls back to the safe inline implementation.
    If SHM initialization fails for any reason (e.g. segment not found,
    permission error), gracefully falls back to inline transport.

    Args:
        metadata: Layout metadata for the non-GPU context.
        req_client: Transport-neutral client used for server requests.
        mq_timeout: Timeout in seconds for blocking MQ requests.
        shm_name: Shared-memory segment name. Empty values force inline mode.
        pool_size: Shared-memory pool size in bytes. Non-positive values force
            inline mode.
    Returns:
        A concrete :class:`EngineDrivenContext` instance.
    """
    if shm_name and pool_size > 0:
        # Local
        from .shm import EngineDrivenContextShm

        try:
            logger.info(
                "Creating EngineDrivenContextShm (shm_name=%s, pool_size=%d)",
                shm_name,
                pool_size,
            )
            return EngineDrivenContextShm(
                metadata, req_client, mq_timeout, shm_name, pool_size
            )
        except Exception:
            logger.warning(
                "Failed to initialize SHM context (shm_name=%s), "
                "falling back to inline tensor transport",
                shm_name,
                exc_info=True,
            )

    # Local
    from .inline import EngineDrivenContextInline

    logger.info("Creating EngineDrivenContextInline")
    return EngineDrivenContextInline(metadata, req_client, mq_timeout)


# ---------------------------------------------------------------------------
# Shared gather / scatter utilities
# ---------------------------------------------------------------------------


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
    engine_type: EngineType = EngineType.VLLM,
) -> tuple[int, int, int, str, "lmcache_native.EngineKVFormat", int]:
    """Compute KV layout metadata from KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        layout_hints: Optional engine layout hints.
        engine_type: Serving engine that produced the KV tensors.

    Returns:
        Tuple of ``(block_size, num_layers, hidden_dim_size, dtype_str,``
        ``engine_kv_format, kv_size)``. ``hidden_dim_size`` is the per-token
        object width (``num_heads * head_size``; for fused-K/V formats the
        head size is the packed ``2 * head_size``) and ``kv_size`` is the
        object plane count (1 for MLA and fused-K/V, else 2).

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_dtype,
        get_head_size,
        get_kv_size,
        get_num_heads,
        get_num_layers,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors:
        raise ValueError("kv_caches is empty. Cannot compute KV layout.")

    engine_kv_format, normalized = normalize_kv_and_discover_format(
        tensors, engine_type, layout_hints=layout_hints
    )
    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    hidden_dim_size = get_num_heads(normalized, engine_kv_format) * get_head_size(
        normalized, engine_kv_format
    )
    kv_size = get_kv_size(normalized, engine_kv_format)
    dtype_str = str(get_dtype(normalized, engine_kv_format)).replace("torch.", "")
    return (
        block_size,
        num_layers,
        hidden_dim_size,
        dtype_str,
        engine_kv_format,
        kv_size,
    )


def gather_paged_kv_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmcache_native.EngineKVFormat" | None = None,
    out: list[torch.Tensor] | None = None,
    chunk_indices: list[int] | None = None,
    engine_type: EngineType = EngineType.VLLM,
    blocks_per_window: int | None = None,
) -> list[torch.Tensor]:
    """Gather paged KV blocks into CPU chunk tensors.

    The KV tensor device selects the transfer backend and its buffer
    requirements, independently of process-wide accelerator availability.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        engine_kv_format: Optional pre-detected KV format.
        out: Optional pre-allocated output tensors.  If provided, length
            must be at least ``len(chunk_indices)`` when ``chunk_indices``
            is given, or the total number of chunks otherwise.  Any extra
            buffers beyond the number of gathered chunks are ignored.
        chunk_indices: Optional list of chunk positions (into the full
            ``block_ids`` sequence) to gather.  When provided together with
            ``out``, only those chunks are gathered and written into
            ``out[i]`` in order.  When ``None``, all chunks are gathered
            (backward-compatible behaviour).
        engine_type: Serving engine that produced the KV tensors.

    Returns:
        List of CPU tensors, one per chunk. For split-K/V formats each chunk
        has shape ``[2, num_layers, chunk_tokens, hidden_dim]`` where
        dimension ``0`` stores ``(K, V)``. For single-plane formats
        (``kv_size == 1``: MLA and fused-K/V) each chunk has shape
        ``[num_layers, chunk_tokens, num_heads * head_size]`` — for fused
        formats ``head_size`` is the packed ``2 * head_size``.

    Raises:
        ValueError: If ``out`` is provided with fewer buffers than the number
            of gathered chunks.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_device,
        get_dtype,
        get_head_size,
        get_kv_size,
        get_num_blocks,
        get_num_heads,
        get_num_layers,
        make_page_buffer_shape_desc,
        normalize_kv_and_discover_format,
        resolve_block_stride_and_log_layout,
    )

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, engine_type, layout_hints=layout_hints
    )
    # KV tensors, not process-wide accelerator availability, select the ops.
    device = get_device(normalized)
    device_ops = resolve_device_ops(device.type)
    accepts_tensor = _detect_block_transfer_accepts_tensor(
        device_ops.multi_layer_block_kv_transfer
    )
    if engine_kv_format is None:
        engine_kv_format = fmt

    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    content_size = get_num_heads(normalized, engine_kv_format) * get_head_size(
        normalized, engine_kv_format
    )
    num_blocks = get_num_blocks(normalized, engine_kv_format)
    num_chunks = len(block_ids) // blocks_per_chunk
    # Sliding-window groups store only the trailing window of each chunk; for
    # full attention bpw == blocks_per_chunk so window_tokens == chunk_tokens
    # and every branch below collapses to the full-coverage behaviour.
    bpw = blocks_per_chunk if blocks_per_window is None else blocks_per_window
    window_tokens = bpw * block_size

    shape_desc = make_page_buffer_shape_desc(
        normalized,
        engine_kv_format,
        layer_idx=0,
        num_layers_in_group=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        block_stride_elems=resolve_block_stride_and_log_layout(
            normalized, engine_kv_format, layer_idx=0, group_idx=0
        ),
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

    # The address-taking native kernel reads host chunks directly, so chunks
    # allocated here for it are pinned; ``transfer_with_address_kernel``
    # stages an unpinned ``out``.
    requires_pinned = not accepts_tensor

    if out is None:
        # One object plane per K/V entry: MLA and fused-K/V formats
        # (kv_size == 1) store a single plane, split formats store K and V.
        if get_kv_size(normalized, engine_kv_format) == 1:
            chunks = [
                torch.empty(
                    (num_layers, window_tokens, content_size),
                    dtype=get_dtype(normalized, engine_kv_format),
                    device=torch.device("cpu"),
                    pin_memory=requires_pinned,
                )
                for _ in iter_indices
            ]
        else:
            chunks = [
                torch.empty(
                    (2, num_layers, window_tokens, content_size),
                    dtype=get_dtype(normalized, engine_kv_format),
                    device=torch.device("cpu"),
                    pin_memory=requires_pinned,
                )
                for _ in iter_indices
            ]
    elif len(out) == len(iter_indices):
        # Return ``out`` itself so the in-place fill contract (result is out)
        # holds.
        chunks = out
    else:
        # Ignore extra trailing buffers beyond what we actually gather so the
        # kernel's ``total_blocks % num_objects`` invariant still holds.
        chunks = out[: len(iter_indices)]

    selected_block_ids: list[int] = []
    for chunk_idx in iter_indices:
        chunk = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        # Keep only the trailing window blocks of the chunk (whole chunk when
        # bpw == blocks_per_chunk), matching downsample_and_stage_block_ids.
        selected_block_ids.extend(chunk[-bpw:])

    if selected_block_ids:
        if accepts_tensor:
            # Python fallback: accepts tensor list directly for all params.
            paged_arg = normalized
            objs_arg = chunks
            block_ids_arg = selected_block_ids

            # call kernel in one shot
            device_ops.multi_layer_block_kv_transfer(
                paged_arg,
                objs_arg,
                block_ids_arg,
                device,
                lmcache_native.TransferDirection.D2H,
                shape_desc,
                window_tokens,
                engine_kv_format,
                0,
            )

        else:
            # Compiled CUDA/XPU/NPU kernel: it takes addresses, so the platform
            # helper builds them and stages an unpinned ``out`` in place.
            transfer_with_address_kernel(
                device_ops.multi_layer_block_kv_transfer,
                normalized,
                chunks,
                selected_block_ids,
                device,
                lmcache_native.TransferDirection.D2H,
                shape_desc,
                window_tokens,
                engine_kv_format,
                0,
            )

    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly synchronize the KV tensor device
    # before consuming these chunks to ensure data validity.

    return chunks


def scatter_cpu_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmcache_native.EngineKVFormat" | None = None,
    engine_type: EngineType = EngineType.VLLM,
    blocks_per_window: int | None = None,
    tokens_per_block: int | None = None,
) -> None:
    """Scatter CPU chunk tensors back into paged KV tensors.

    The destination KV tensor device selects the transfer backend, not the
    CPU chunk device or process-wide accelerator availability.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.  Length
            must be at least ``len(chunks) * blocks_per_chunk``; any extra
            trailing block IDs are ignored.
        chunks: List of CPU chunk tensors (as returned by
            :func:`gather_paged_kv_to_cpu`).
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Logical token prefix to skip when scattering.
            Non-aligned values are rounded down
            to the nearest whole block and an error is logged (matching the
            GPU transfer path).
        layout_hints: Optional engine layout hints.
        engine_kv_format: Optional pre-detected KV format.
        engine_type: Serving engine that produced the KV tensors.
        tokens_per_block: Logical tokens covered by one block ID. Defaults to
            the tensor's physical slot count for uncompressed caches.

    Raises:
        ValueError: If ``block_ids`` is shorter than
            ``len(chunks) * blocks_per_chunk``, or ``tokens_per_block`` is not
            a positive multiple of the physical slots per block.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_device,
        get_num_blocks,
        get_num_layers,
        make_page_buffer_shape_desc,
        normalize_kv_and_discover_format,
        resolve_block_stride_and_log_layout,
    )

    # Local like the rest: object_group_transfer pulls in the GPU connector and
    # distributed stack (serde needs `cryptography`), which would otherwise load
    # for every importer of transfer_context, e.g. the ATOM integration.
    from lmcache.v1.multiprocess.object_group_transfer import (
        recalculate_blocks_to_skip,
    )

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
        tensors, engine_type, layout_hints=layout_hints
    )
    device = get_device(normalized)
    device_ops = resolve_device_ops(device.type)
    accepts_tensor = _detect_block_transfer_accepts_tensor(
        device_ops.multi_layer_block_kv_transfer
    )
    if engine_kv_format is None:
        engine_kv_format = fmt

    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    num_blocks = get_num_blocks(normalized, engine_kv_format)

    # Block-level transfer can only skip whole blocks. A non-aligned prefix is
    # rounded down to the nearest block (matching the lmcache-driven path in
    # lmcache_driven_transfer.py) rather than raising, so a slightly misaligned skip
    # degrades gracefully instead of failing the whole retrieve.
    logical_block_size = block_size if tokens_per_block is None else tokens_per_block
    if logical_block_size <= 0 or logical_block_size % block_size:
        raise ValueError(
            f"tokens_per_block ({logical_block_size}) must be a positive "
            f"multiple of slots_per_block ({block_size})"
        )
    if skip_first_n_tokens % logical_block_size != 0:
        logger.error(
            "skip_first_n_tokens (%d) is not block-aligned (tokens_per_block=%d); "
            "rounding down to %d blocks",
            skip_first_n_tokens,
            logical_block_size,
            skip_first_n_tokens // logical_block_size,
        )
    skip_prefix_n_blocks = skip_first_n_tokens // logical_block_size
    bpw = blocks_per_chunk if blocks_per_window is None else blocks_per_window
    window_tokens = bpw * block_size
    # For a sliding-window group the leading blocks of each chunk were dropped
    # by the trailing-keep below, so an APC skip-prefix expressed in full-chunk
    # blocks must be re-expressed in window blocks. No-op when window == chunk.
    skip_prefix_window = recalculate_blocks_to_skip(
        blocks_per_chunk, bpw, skip_prefix_n_blocks
    )

    shape_desc = make_page_buffer_shape_desc(
        normalized,
        engine_kv_format,
        layer_idx=0,
        num_layers_in_group=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        block_stride_elems=resolve_block_stride_and_log_layout(
            normalized, engine_kv_format, layer_idx=0, group_idx=0
        ),
    )

    selected_block_ids: list[int] = []
    for chunk_idx in range(len(chunks)):
        chunk = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        # Trailing window blocks only (whole chunk when bpw == blocks_per_chunk).
        selected_block_ids.extend(chunk[-bpw:])

    if not selected_block_ids:
        return

    if accepts_tensor:
        # Python fallback: accepts tensor list directly for all params.
        paged_arg = normalized
        objs_arg = chunks
        block_ids_arg = selected_block_ids

        device_ops.multi_layer_block_kv_transfer(
            paged_arg,
            objs_arg,
            block_ids_arg,
            device,
            lmcache_native.TransferDirection.H2D,
            shape_desc,
            window_tokens,
            engine_kv_format,
            skip_prefix_window,
        )
    else:
        # Compiled CUDA/XPU/NPU kernel: it takes addresses, so the platform
        # helper builds them; it pins unpinned chunks into temporaries and
        # synchronizes before returning, so they outlive the async reads.
        transfer_with_address_kernel(
            device_ops.multi_layer_block_kv_transfer,
            normalized,
            chunks,
            selected_block_ids,
            device,
            lmcache_native.TransferDirection.H2D,
            shape_desc,
            window_tokens,
            engine_kv_format,
            skip_prefix_window,
        )
    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly synchronize the KV tensor device
    # before consuming these chunks to ensure data validity.
