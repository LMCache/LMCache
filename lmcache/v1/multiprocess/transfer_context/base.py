# SPDX-License-Identifier: Apache-2.0
"""Non-GPU context abstractions and utilities for multiprocess transport.

This module provides:
- ``EngineDrivenContextMetadata``: layout metadata dataclass for non-CUDA workers.
- ``EngineDrivenContext``: abstract base class with a two-phase prepare/commit
  interface for CPU-side KV data transfer. Concrete implementations (e.g.
  ``EngineDrivenContextPickle``) each decide *how* data is serialised and transported.
- ``create_engine_driven_context()``: factory that returns the appropriate
  ``EngineDrivenContext`` subclass.
- ``compute_kv_layout``, ``gather_paged_kv_to_cpu``, ``scatter_cpu_to_paged_kv``:
  shared gather/scatter utilities used by all concrete implementations.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import inspect

# Third Party
import numpy as np
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Global capability flag: does lmc_ops.multi_layer_block_kv_transfer accept
# list[torch.Tensor] directly for lmcache_objects_ptrs, or only list[int]?
#
# We inspect the function signature once at import time. If the annotation
# for ``lmcache_objects_ptrs`` includes ``Tensor``, the op can handle tensors
# natively and we pass them through. Otherwise (annotation is list[int], or
# inspect fails entirely) we must convert tensors to data pointers before
# calling.
# ---------------------------------------------------------------------------
def _detect_block_transfer_accepts_tensor() -> bool:
    """Return True if lmc_ops.multi_layer_block_kv_transfer accepts
    list[torch.Tensor] for its lmcache_objects_ptrs parameter."""
    try:
        # First Party
        import lmcache.c_ops as _lmc_ops

        fn = _lmc_ops.multi_layer_block_kv_transfer

        # Attempt: use inspect.signature (works on newer pybind11 builds)
        # Assumptions: if lmcache_objects_ptrs accepts tensors,
        # it's fallback path, and we do not convert tensors to ptrs explicitly.
        # TODO: String matching on annotations is fragile. Wait for lmc_ops to
        # expose a direct version flag (e.g., lmc_ops.__version__) or
        # an explicit capability boolean.
        try:
            sig = inspect.signature(fn)
            param = sig.parameters.get("lmcache_objects_ptrs")
            if param is not None and param.annotation is not inspect.Parameter.empty:
                ann_str = str(param.annotation)
                if "Tensor" in ann_str:
                    return True
                # Annotation exists but no Tensor mention -> ptr-only
                return False
        except (ValueError, TypeError):
            pass

    except Exception:
        # Import failed or any other error -> conservative: assume ptr-only
        pass

    # Default: inspect failed or lmc_ops not available -> assume ptr-only
    return False


_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR: bool = _detect_block_transfer_accepts_tensor()
"""If True, ``lmc_ops.multi_layer_block_kv_transfer`` accepts
``list[torch.Tensor]`` directly for ``lmcache_objects_ptrs``.
If False, callers must convert tensors to ``list[int]`` data pointers."""

logger.info(
    "multi_layer_block_kv_transfer mode: %s",
    "tensor" if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR else "ptr",
)


def _tensors_to_ptrs(tensors: list[torch.Tensor]) -> list[int]:
    """Convert a list of tensors to a list of their data_ptr() values."""
    return [t.data_ptr() for t in tensors]


# ---------------------------------------------------------------------------


@dataclass
class EngineDrivenContextMetadata:
    """Non-GPU context layout metadata for non-CUDA workers.

    Attributes:
        layout_desc: Memory layout descriptor (single-group, backward compat).
        block_size: Tokens per paged block (single-group, backward compat).
        use_mla: Whether MLA format is used (single-group, backward compat).
        group_layout_descs: Per-group layout descriptors for hybrid models.
        group_block_sizes: Per-group block sizes for hybrid models.
        group_use_mla: Per-group MLA flags for hybrid models.
        group_blocks_in_chunk: Per-group blocks-in-chunk counts for hybrid models.
    """

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool
    group_layout_descs: list[MemoryLayoutDesc] = field(default_factory=list)
    group_block_sizes: list[int] = field(default_factory=list)
    group_use_mla: list[bool] = field(default_factory=list)
    group_blocks_in_chunk: list[int] = field(default_factory=list)

    @property
    def is_multi_group(self) -> bool:
        """Return True if this metadata describes a hybrid multi-group model."""
        return len(self.group_layout_descs) > 1

class EngineDrivenContext(ABC):
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
        metadata: EngineDrivenContextMetadata,
        mq_client: MessageQueueClient,
        mq_timeout: float,
    ) -> None:
        self.metadata = metadata
        self.mq_client = mq_client
        self.mq_timeout = mq_timeout

    @abstractmethod
    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Prepare SHM buffers for a store operation.

        Returns:
            None: pickle mode -- no pre-allocated buffers. Caller gathers all
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
        """Commit store. Pickle: serialize and send. Shm: notify server."""
        ...

    @abstractmethod
    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Prepare retrieve. Returns chunks or shm views, or None on miss."""
        ...

    @abstractmethod
    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit retrieve. Pickle: no-op. Shm: release read locks."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this context."""
        ...
    def commit_store_raw(
        self, key: IPCCacheServerKey, instance_id: int, cpu_data: bytes
    ) -> bool:
        """Send pre-serialized bytes directly via COMMIT_STORE.

        Used by multi-group transfers where the caller has already serialized
        the chunks. No additional pickle.dumps() is performed.
        """
        from lmcache.v1.multiprocess.protocol import RequestType
        from lmcache.v1.multiprocess.protocol import get_response_class

        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id, cpu_data],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def commit_store_group_raw(
        self, key: IPCCacheServerKey, instance_id: int, group_idx: int, cpu_data: bytes
    ) -> bool:
        """Send one group's pre-serialized bytes via COMMIT_STORE_GROUP.

        Multi-group engine-driven transfer splits the per-key store into one
        COMMIT_STORE_GROUP message per group so that each individual
        ``cpu_data`` blob stays under the msgspec msgpack bin limit (4 GiB).
        """
        from lmcache.v1.multiprocess.protocol import RequestType
        from lmcache.v1.multiprocess.protocol import get_response_class

        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE_GROUP,
            [key, instance_id, group_idx, cpu_data],
            get_response_class(RequestType.COMMIT_STORE_GROUP),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def commit_store_group_raw_async(
        self, key: IPCCacheServerKey, instance_id: int, group_idx: int, cpu_data: bytes
    ):
        """Async version of :meth:`commit_store_group_raw`.

        Returns the underlying :class:`MessagingFuture` so the caller can
        pipeline multiple ``COMMIT_STORE_GROUP`` requests and ``join()``
        all responses at the end (avoids one round-trip wait per group).

        The server still serialises requests per connection via the
        affinity pool, so pipelining only saves CPU-side wait time on
        the worker -- the server processes them in submission order.  In
        practice this hides ~5-30 ms of latency per group on top of the
        CPU-side serialize cost.
        """
        from lmcache.v1.multiprocess.protocol import RequestType
        from lmcache.v1.multiprocess.protocol import get_response_class

        return self.mq_client.submit_request(
            RequestType.COMMIT_STORE_GROUP,
            [key, instance_id, group_idx, cpu_data],
            get_response_class(RequestType.COMMIT_STORE_GROUP),
        )
    def prepare_retrieve_raw(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> bytes | None:
        """Send PREPARE_RETRIEVE and return raw bytes (no pickle.loads).

        Used by multi-group transfers where the caller deserializes with
        ``_deserialize_multi_group_chunks``. Returns ``None`` on cache-miss
        or timeout.
        """
        from lmcache.v1.multiprocess.protocol import RequestType
        from lmcache.v1.multiprocess.protocol import get_response_class

        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success or not response.data:
            return None
        return response.data


class PinnedBufferPool:
    """Pool of pinned (page-locked) CPU tensors keyed by (shape, dtype).

    Pinned memory is allocated in **system RAM** (via ``cudaHostAlloc``
    / ``mlock`` on Linux) -- it does **not** consume any GPU VRAM.  The
    pool is grown lazily on first ``acquire`` and never shrinks: the OS
    reclaims pinned pages only when the underlying physical pages are
    unpinned, so retaining them keeps the allocator cost amortised.

    Used by ``EngineDrivenTransferContext.submit_store`` to avoid per-
    chunk ``torch.empty(pin_memory=True)`` calls inside
    ``gather_paged_kv_to_cpu``.  Each pool entry is a single CPU tensor;
    the pool itself holds them in a FIFO deque per (shape, dtype).
    """

    def __init__(self) -> None:
        self._pools: dict[tuple[tuple[int, ...], torch.dtype], deque[torch.Tensor]] = {}
        self._acquired_count = 0

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        count: int = 1,
    ) -> list[torch.Tensor]:
        """Pop up to ``count`` tensors of the given shape from the pool.

        New tensors are allocated with ``pin_memory=True`` on cache miss.
        """
        key = (shape, dtype)
        pool = self._pools.setdefault(key, deque())
        out: list[torch.Tensor] = []
        for _ in range(count):
            if pool:
                out.append(pool.popleft())
            else:
                out.append(
                    torch.empty(shape, dtype=dtype,
                                device=torch.device("cpu"),
                                pin_memory=True)
                )
            self._acquired_count += 1
        return out

    def release(self, buffers: list[torch.Tensor]) -> None:
        """Return buffers to the pool for reuse on the next acquire."""
        for buf in buffers:
            key = (tuple(buf.shape), buf.dtype)
            self._pools.setdefault(key, deque()).append(buf)

    def clear(self) -> None:
        """Release all pinned memory.  Call on context destruction."""
        self._pools.clear()
        self._acquired_count = 0

    def stats(self) -> dict:
        """Snapshot of pool state (for diagnostics / benchmarks)."""
        return {
            "shapes": [(k[0], k[1], len(v)) for k, v in self._pools.items()],
            "acquired_total": self._acquired_count,
        }


def _pool() -> "PinnedBufferPool":
    """Module-level pool singleton.  Use ``_POOL`` directly if needed.

    Lifetime is the Python process: pinned memory grows with the process
    and is freed on interpreter shutdown.  Multi-group paths that
    benefit from per-context isolation can create their own
    ``PinnedBufferPool()`` and pass it through ``gather_paged_kv_*``.
    """
    global _POOL
    try:
        return _POOL
    except NameError:
        _POOL = PinnedBufferPool()
        return _POOL


def create_engine_driven_context(
    metadata: EngineDrivenContextMetadata,
    mq_client: MessageQueueClient,
    mq_timeout: float,
    shm_name: str,
    pool_size: int,
    *,
    use_pickle: bool = False,
) -> EngineDrivenContext:
    """Factory that returns the appropriate :class:`EngineDrivenContext` implementation.

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
    A concrete :class:`EngineDrivenContext` instance.
    """
    if metadata.is_multi_group:
        use_pickle = True
        logger.info(
            "Multi-group engine-driven context: forcing pickle transport "
            "(SHM pool is sized for single-group layout)"
        )

    if not shm_name or pool_size <= 0:
        use_pickle = True

    if not use_pickle:
        # Local
        from .shm import EngineDrivenContextShm

        try:
            logger.info(
                "Creating EngineDrivenContextShm (shm_name=%s, pool_size=%d)",
                shm_name,
                pool_size,
            )
            return EngineDrivenContextShm(
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
    from .pickle import EngineDrivenContextPickle

    logger.info("Creating EngineDrivenContextPickle (pickle transport)")
    return EngineDrivenContextPickle(metadata, mq_client, mq_timeout)


# ---------------------------------------------------------------------------
# Shared gather / scatter utilities
# ---------------------------------------------------------------------------


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> tuple[int, int, int, str, "lmc_ops.EngineKVFormat"]:
    """Compute KV layout metadata from KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        layout_hints: Optional engine layout hints.

    Returns:
        Tuple of ``(block_size, num_layers, hidden_dim_size, dtype_str,``
        ``engine_kv_format)``.

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

    engine_kv_format, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    hidden_dim_size = get_hidden_dim_size(normalized, engine_kv_format)
    dtype_str = str(tensors[0].dtype).replace("torch.", "")
    return block_size, num_layers, hidden_dim_size, dtype_str, engine_kv_format


def gather_paged_kv_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmc_ops.EngineKVFormat" | None = None,
    out: list[torch.Tensor] | None = None,
    chunk_indices: list[int] | None = None,
    pinned_pool: "PinnedBufferPool | None" = None,
) -> list[torch.Tensor]:
    """Gather paged KV blocks into CPU chunk tensors.
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
        pinned_pool: Optional ``PinnedBufferPool`` to source pre-allocated
            pinned CPU tensors.  When set and ``out is None``, the function
            draws buffers from the pool rather than calling
            ``torch.empty(pin_memory=True)`` for each chunk.  This skips
            the CUDA-allocator overhead on every store and avoids
            system-RAM page-locking cost on the second call onward
            (buffers are returned to the pool on the caller side).
            Pinned memory is **system RAM**, never GPU VRAM.
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
    if engine_kv_format is None:
        engine_kv_format = fmt

    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    hidden_dim_size = get_hidden_dim_size(normalized, engine_kv_format)
    num_blocks = get_num_blocks(normalized, engine_kv_format)
    num_chunks = len(block_ids) // blocks_per_chunk
    chunk_tokens = blocks_per_chunk * block_size

    shape_desc = make_page_buffer_shape_desc(
        normalized,
        engine_kv_format,
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

    # Determine if pinned memory is strictly required
    # (only for the compiled C++ path which does not accept tensor)
    requires_pinned = not _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR
    needs_staging = False
    staged_chunks = []

    if out is None:
        use_mla = is_mla(engine_kv_format)
        # Pull from pinned_pool when available -- avoids per-chunk
        # torch.empty(pin_memory=True) and the CUDA allocator round-trip.
        if pinned_pool is not None:
            if use_mla:
                shape = (num_layers, chunk_tokens, hidden_dim_size)
            else:
                shape = (2, num_layers, chunk_tokens, hidden_dim_size)
            chunks = pinned_pool.acquire(shape, tensors[0].dtype, len(iter_indices))
        else:
            if use_mla:
                chunks = [
                    torch.empty(
                        (num_layers, chunk_tokens, hidden_dim_size),
                        dtype=tensors[0].dtype,
                        device=torch.device("cpu"),
                        pin_memory=requires_pinned,
                    )
                    for _ in iter_indices
                ]
            else:
                chunks = [
                    torch.empty(
                        (2, num_layers, chunk_tokens, hidden_dim_size),
                        dtype=tensors[0].dtype,
                        device=torch.device("cpu"),
                        pin_memory=requires_pinned,
                    )
                    for _ in iter_indices
                ]
    else:
        _target_out = out[: len(iter_indices)]

        if requires_pinned and not all(t.is_pinned() for t in _target_out):
            # Core fallback: Unpinned memory (e.g., IPC Shared Memory) detected.
            # We cannot dynamically call `.pin_memory()` on `out` because it
            # would allocate new tensors, breaking the caller's expectation
            # of an in-place update. Instead, we allocate a temporary pinned
            # staging buffer for the C++ kernel to write to safely.
            logger.warning(
                "Unpinned memory detected in 'out' during "
                "gather_paged_kv_to_cpu (likely Shared Memory). "
                "Using an internal pinned staging buffer, which "
                "adds a CPU memory copy overhead."
            )
            needs_staging = True
            staged_chunks = [torch.empty_like(t, pin_memory=True) for t in _target_out]
            chunks = (
                staged_chunks  # Point to the safe staging buffer for the H2D transfer
            )
        else:
            # Ideal case: Memory is pinned, or we are using Python fallback.
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
        if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR:
            # Python fallback: accepts tensor list directly for all params.
            paged_arg = normalized
            objs_arg = chunks
            block_ids_arg = selected_block_ids

            # call kernel in one shot
            lmc_ops.multi_layer_block_kv_transfer(
                paged_arg,
                objs_arg,
                block_ids_arg,
                tensors[0].device,
                lmc_ops.TransferDirection.D2H,
                shape_desc,
                chunk_tokens,
                engine_kv_format,
                0,
            )

        else:
            # Compiled C++/CUDA/XPU: requires int64 pointer tensor and list[int].
            _ptrs_np = np.array(
                [t.data_ptr() for t in normalized],  # type: ignore[union-attr]
                dtype=np.uint64,
            ).view(np.int64)
            paged_arg = torch.from_numpy(_ptrs_np).to(device=tensors[0].device)

            # This safely points to either the pre-pinned chunks
            # OR the temporary staged_chunks
            objs_arg = _tensors_to_ptrs(chunks)

            block_ids_arg = torch.tensor(
                selected_block_ids, dtype=torch.int64, device=tensors[0].device
            )

            # Split transfer to respect CUDA kernel's object count limitation
            MAX_OBJECTS = 4
            req_blocks_per_obj = blocks_per_chunk
            total_objects = len(objs_arg)

            for i in range(0, total_objects, MAX_OBJECTS):
                # Slice object pointers and corresponding block IDs
                batch_objs_ptrs = objs_arg[i : i + MAX_OBJECTS]

                start_block = i * req_blocks_per_obj
                end_block = min(
                    (i + MAX_OBJECTS) * req_blocks_per_obj, len(selected_block_ids)
                )
                batch_blocks = block_ids_arg[start_block:end_block]

                # Execute batched transfer
                lmc_ops.multi_layer_block_kv_transfer(
                    paged_arg,
                    batch_objs_ptrs,
                    batch_blocks,
                    tensors[0].device,
                    lmc_ops.TransferDirection.D2H,
                    shape_desc,
                    chunk_tokens,
                    engine_kv_format,
                    0,
                )

    # --- Final reconciliation ---
    # If we used a staging buffer to protect unpinned shared memory,
    # we now copy the gathered data back into the caller's original tensors.
    if needs_staging:
        assert out is not None
        # The CPU MUST block and wait for the GPU ONLY when a temporary
        # staging buffer is used. This is because the CPU needs to immediately
        # read this data for the memory copy below.
        torch_dev.synchronize()

        for dst, src in zip(_target_out, staged_chunks, strict=False):
            dst.copy_(src)  # High-speed CPU-to-CPU memory copy

        if len(out) == len(iter_indices):
            chunks = out
        else:
            chunks = _target_out

    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly call `torch_dev.synchronize()`
    # before consuming these chunks to ensure data validity.

    return chunks


def scatter_cpu_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmc_ops.EngineKVFormat" | None = None,
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
        engine_kv_format: Optional pre-detected KV format.

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
    if engine_kv_format is None:
        engine_kv_format = fmt

    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    num_blocks = get_num_blocks(normalized, engine_kv_format)
    chunk_tokens = blocks_per_chunk * block_size

    # Block-level transfer can only skip whole blocks. A non-aligned prefix is
    # rounded down to the nearest block (matching the lmcache-driven path in
    # lmcache_driven_transfer.py) rather than raising, so a slightly misaligned skip
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
        engine_kv_format,
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

    if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR:
        # Python fallback: accepts tensor list directly for all params.
        paged_arg = normalized
        objs_arg = chunks
        block_ids_arg = selected_block_ids

        lmc_ops.multi_layer_block_kv_transfer(
            paged_arg,
            objs_arg,
            block_ids_arg,
            tensors[0].device,
            lmc_ops.TransferDirection.H2D,
            shape_desc,
            chunk_tokens,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
    else:
        # assuming this is c ops path which requires pin memory
        # TODO: may have a better approach here
        # Defensive check: Ensure all incoming CPU chunks are pinned memory.
        # Otherwise, the underlying CUDA kernel may throw an Illegal
        # Memory Access error during H2D transfer.
        if not all(chunk.is_pinned() for chunk in chunks):
            logger.warning(
                "Received unpinned CPU tensors in scatter_cpu_to_paged_kv. "
                "Dynamically pinning memory now, which may incur additional"
                "synchronization overhead."
            )
            chunks = [
                chunk.pin_memory() if not chunk.is_pinned() else chunk
                for chunk in chunks
            ]

        # Compiled C++/CUDA/XPU: requires int64 pointer tensor and list[int].
        _ptrs_np = np.array(
            [t.data_ptr() for t in normalized],  # type: ignore[union-attr]
            dtype=np.uint64,
        ).view(np.int64)
        paged_arg = torch.from_numpy(_ptrs_np).to(device=tensors[0].device)
        objs_arg = _tensors_to_ptrs(chunks)
        block_ids_arg = torch.tensor(
            selected_block_ids, dtype=torch.int64, device=tensors[0].device
        )

        # Batched transfer to satisfy cuda's limitation (max 4 objects)
        MAX_OBJECTS = 4
        req_blocks_per_obj = (
            blocks_per_chunk  # Each chunk corresponds to one object's blocks
        )
        total_chunks = len(chunks)

        for i in range(0, total_chunks, MAX_OBJECTS):
            # Slice objects and block IDs for this batch
            batch_objs_ptrs = objs_arg[i : i + MAX_OBJECTS]

            start_block = i * req_blocks_per_obj
            end_block = min(
                (i + MAX_OBJECTS) * req_blocks_per_obj, len(selected_block_ids)
            )
            batch_blocks = block_ids_arg[start_block:end_block]

            # Execute transfer for this batch
            lmc_ops.multi_layer_block_kv_transfer(
                paged_arg,
                batch_objs_ptrs,
                batch_blocks,
                tensors[0].device,
                lmc_ops.TransferDirection.H2D,
                shape_desc,
                chunk_tokens,
                engine_kv_format,
                skip_prefix_n_blocks if i == 0 else 0,
            )
    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly call `torch_dev.synchronize()`
    # before consuming these chunks to ensure data validity.

# ---------------------------------------------------------------------------
# Multi-group serialization helpers
# ---------------------------------------------------------------------------


def _serialize_single_group_chunks(
    group_chunks: list[torch.Tensor],
) -> bytes:
    """Serialize one group's CPU chunk tensors as a compact pickle blob.

    Optimization: pickles bf16/fp16/fp32/int* tensors natively via torch's
    ``__reduce_ex__`` (preserves dtype, no conversion). This previously
    cast bf16 to float32 for numpy compatibility, doubling serialization
    memory traffic and adding a full copy on deserialize.

    fp8 tensors (e4m3fn, e5m2) still go through a uint8 view because torch's
    legacy pickle loader returns them as ``UntypedStorage`` without dtype
    metadata -- see ``tests/v1/multiprocess/test_multi_group.py`` for the
    regression test.
    """
    import pickle

    def _to_serializable(t: torch.Tensor):
        # fp8 needs explicit view to uint8 + dtype tag because torch's
        # legacy unpickler drops the dtype info.
        if hasattr(torch, "float8_e4m3fn") and t.dtype == torch.float8_e4m3fn:
            return ("fp8", t.contiguous().view(torch.uint8).numpy(), "float8_e4m3fn")
        if hasattr(torch, "float8_e5m2") and t.dtype == torch.float8_e5m2:
            return ("fp8", t.contiguous().view(torch.uint8).numpy(), "float8_e5m2")
        # All other dtypes (bf16, fp16, fp32, int*, uint8, bool) are
        # All other dtypes (bf16, fp16, fp32, int*, uint8, bool) are
        # pickled natively by torch -- no conversion, dtype preserved.
        return ("tensor", t.contiguous())

    # Wrap in outer list so the blob is structurally identical to a single-
    # group slice of ``_serialize_multi_group_chunks`` output. The
    # deserializer iterates groups as the outer dimension.
    payload = [[_to_serializable(c) for c in group_chunks]]
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _to_serializable_chunk(t: torch.Tensor):
    """Tag a chunk for the dev24+ wire format.

    fp8 (e4m3fn / e5m2) needs an explicit uint8 view + dtype tag because
    torch's legacy unpickler drops the dtype metadata for these dtypes.
    All other dtypes (bf16, fp16, fp32, int*, uint8, bool) roundtrip via
    torch's native ``__reduce_ex__`` -- no conversion, dtype preserved.
    """
    if hasattr(torch, "float8_e4m3fn") and t.dtype == torch.float8_e4m3fn:
        return ("fp8", t.contiguous().view(torch.uint8).numpy(), "float8_e4m3fn")
    if hasattr(torch, "float8_e5m2") and t.dtype == torch.float8_e5m2:
        return ("fp8", t.contiguous().view(torch.uint8).numpy(), "float8_e5m2")
    return ("tensor", t.contiguous())


def _serialize_single_group_chunks(
    group_chunks: list[torch.Tensor],
) -> bytes:
    """Serialize one group's CPU chunk tensors as a compact pickle blob.

    The blob is structurally identical to a single-group slice of
    ``_serialize_multi_group_chunks`` output so the deserializer can use
    the same outer-loop logic.

    Performance: pickles bf16/fp16/fp32/int* natively (no bf16->float32
    conversion).  Combined with ``pickle.HIGHEST_PROTOCOL``, this gives
    ~1.9-- speedup and 50% smaller blobs on bf16 vs the previous
    numpy-based path.
    """
    import pickle
    payload = [[_to_serializable_chunk(c) for c in group_chunks]]
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _serialize_multi_group_chunks(
    group_chunks: list[list[torch.Tensor]],
) -> bytes:
    """Serialize multiple groups as a single pickle blob.

    Used as the fast path for the single ``COMMIT_STORE`` request when
    the combined size fits in the msgspec bin limit (4 GiB). When it does
    not, callers fall back to ``_serialize_single_group_chunks`` +
    ``COMMIT_STORE_GROUP`` per group.
    """
    import pickle
    payload = [[_to_serializable_chunk(t) for t in group] for group in group_chunks]
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_multi_group_chunks(
    cpu_data: bytes,
) -> list[list[torch.Tensor]]:
    """Deserialize a multi-group pickle blob back to tensors.

    Wire-format compatibility:

    - ``dev24+`` (new): each chunk is a tagged tuple
      ``("tensor", torch.Tensor)`` for native dtypes (bf16/fp16/fp32/int*),
      or ``("fp8", ndarray, dtype_str)`` for fp8 tensors that needed the
      uint8-view roundtrip. bf16 is stored as native bf16 (no conversion).
    - ``<=dev23`` (old): each chunk is ``(ndarray, dtype_str)`` with bf16
      upcast to float32 (or fp8 as uint8 view). Detected automatically
      by tuple length and tag presence; no cache wipe required when
      upgrading.
    """
    import pickle
    raw = pickle.loads(cpu_data)
    result = []
    for group_items in raw:
        out_group = []
        for item in group_items:
            tag, payload = _classify_chunk(item)
            if tag == "fp8_new":
                arr, dtype_name = payload
                t = torch.from_numpy(arr).view(getattr(torch, dtype_name))
                out_group.append(t)
            elif tag == "tensor_new":
                out_group.append(payload)
            else:  # "old" -- (ndarray, dtype_str) tuple
                arr, dtype_name = payload
                if dtype_name == "bfloat16":
                    # Old format stored bf16 as float32; cast back.
                    t = torch.from_numpy(arr).to(torch.bfloat16)
                elif dtype_name in ("float8_e4m3fn", "float8_e5m2"):
                    t = torch.from_numpy(arr).view(getattr(torch, dtype_name))
                else:
                    t = torch.from_numpy(arr)
                out_group.append(t)
        result.append(out_group)
    return result


def _classify_chunk(item: object) -> tuple[str, object]:
    """Distinguish dev24+ tagged tuples from <=dev23 legacy tuples.

    Returns:
        One of:
        - ``("tensor_new", torch_tensor)`` -- dev24+ native pickling
        - ``("fp8_new", (ndarray, dtype_name))`` -- dev24+ fp8 roundtrip
        - ``("old", (ndarray, dtype_name))`` -- <=dev23 legacy format
    """
    # Use type-checked string comparison -- `item[0]` may be a numpy array
    # (legacy format), where `==` returns an array of bools and breaks `if`.
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and item[0] == "tensor":
        return "tensor_new", item[1]
    if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str) and item[0] == "fp8":
        return "fp8_new", (item[1], item[2])
    if isinstance(item, tuple) and len(item) == 2:
        # Legacy: (ndarray, dtype_str)
        return "old", item
    # Bare tensor (defensive fallback)
    return "tensor_new", item
    return "tensor_new", item
# ---------------------------------------------------------------------------
# Multi-group gather / scatter utilities
# ---------------------------------------------------------------------------

def slice_kv_caches_for_group(
    kv_caches: dict[str, torch.Tensor],
    layer_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Extract a subset of KV tensors for a single group.

    Args:
        kv_caches: All layers, ordered as passed by the adapter.
        layer_indices: 0-based indices of the layers belonging to this group.

    Returns:
        Ordered dict with only the layers of this group.
    """
    all_values = list(kv_caches.values())
    return {str(i): all_values[idx] for i, idx in enumerate(sorted(layer_indices))}


def gather_paged_kv_multi_group_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    engine_group_infos: list[EngineGroupInfo],
    lmcache_tokens_per_chunk: int,
    layout_hints: LayoutHints | None = None,
    pinned_pool: "PinnedBufferPool | None" = None,
) -> list[list[torch.Tensor]]:
    """Gather all KV groups to CPU tensors.

    Args:
        kv_caches: All layers, ordered as passed by the adapter.
        block_ids: Block IDs per group.
        engine_group_infos: Group metadata from registration.
        lmcache_tokens_per_chunk: Global LMCache chunk size in tokens.
        layout_hints: Optional layout hints.
        pinned_pool: Optional ``PinnedBufferPool`` used to source pre-
            allocated pinned CPU tensors for each group's chunks.  When
            set, the allocator is bypassed and the per-group chunk buffers
            are reused across calls (no GPU VRAM cost).
    """
    result: list[list[torch.Tensor]] = []
    for group_idx, group_info in enumerate(engine_group_infos):
        group_kv = slice_kv_caches_for_group(kv_caches, group_info.layer_indices)
        tokens_per_block = group_info.tokens_per_block
        if tokens_per_block <= 0:
            block_size, _, _, _, _ = compute_kv_layout(
                group_kv, layout_hints=layout_hints
            )
            tokens_per_block = block_size
        blocks_in_chunk = lmcache_tokens_per_chunk // tokens_per_block
        if blocks_in_chunk == 0:
            raise ValueError(
                f"Group {group_idx}: tokens_per_block ({tokens_per_block}) > "
                f"lmcache_tokens_per_chunk ({lmcache_tokens_per_chunk}). "
                "Each block is larger than the chunk size."
            )
        group_chunks = gather_paged_kv_to_cpu(
            group_kv,
            block_ids[group_idx],
            blocks_in_chunk,
            layout_hints=layout_hints,
            pinned_pool=pinned_pool,
        )
        result.append(group_chunks)
    return result


def scatter_cpu_multi_group_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    group_chunks: list[list[torch.Tensor]],
    engine_group_infos: list[EngineGroupInfo],
    lmcache_tokens_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
) -> None:
    """Scatter CPU tensors back into paged KV for all groups.

    Args:
        kv_caches: All layers, ordered as passed by the adapter.
        block_ids: Block IDs per group.
        group_chunks: ``group_chunks[group_idx][chunk_idx]`` = CPU tensor.
        engine_group_infos: Group metadata from registration.
        lmcache_tokens_per_chunk: Global LMCache chunk size in tokens.
        skip_first_n_tokens: Tokens to skip at the start (for APC).
        layout_hints: Optional layout hints.
    """
    for group_idx, group_info in enumerate(engine_group_infos):
        group_kv = slice_kv_caches_for_group(kv_caches, group_info.layer_indices)
        tokens_per_block = group_info.tokens_per_block
        if tokens_per_block <= 0:
            block_size, _, _, _, _ = compute_kv_layout(
                group_kv, layout_hints=layout_hints
            )
            tokens_per_block = block_size
        blocks_in_chunk = lmcache_tokens_per_chunk // tokens_per_block
        if blocks_in_chunk == 0:
            raise ValueError(
                f"Group {group_idx}: tokens_per_block ({tokens_per_block}) > "
                f"lmcache_tokens_per_chunk ({lmcache_tokens_per_chunk}). "
                "Each block is larger than the chunk size."
            )
        scatter_cpu_to_paged_kv(
            group_kv,
            block_ids[group_idx],
            group_chunks[group_idx],
            blocks_in_chunk,
            skip_first_n_tokens=skip_first_n_tokens if group_idx == 0 else 0,
            layout_hints=layout_hints,
        )
