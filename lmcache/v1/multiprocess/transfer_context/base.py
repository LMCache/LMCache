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
import threading

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

# First Party
from lmcache.v1.multiprocess.transfer_context.wire_format import (
    deserialize_group_chunks_maybe as _wire_decode,
)
from lmcache.v1.multiprocess.transfer_context.wire_format import (
    is_lmcache_blob as _is_lmcache_blob,
)
from lmcache.v1.multiprocess.transfer_context.wire_format import (
    serialize_group_chunks_torchsave as _wire_encode,
)

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
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

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
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

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
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

        return self.mq_client.submit_request(
            RequestType.COMMIT_STORE_GROUP,
            [key, instance_id, group_idx, cpu_data],
            get_response_class(RequestType.COMMIT_STORE_GROUP),
        )

    def commit_store_group_delta_raw_async(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        group_idx: int,
        skip_count: int,
        cpu_data: bytes,
    ) -> "MessagingFuture":
        """Async variant of :meth:`commit_store_group_delta`.

        Used by multi-group engine-driven transfer where the worker
        already knows that the first ``skip_count`` chunks for this
        group are in L2 (caller is responsible for proving this via
        the prior lookup). The server writes only the chunks provided
        in ``cpu_data`` at offset ``skip_count``.
        """
        # First Party
        from lmcache.v1.multiprocess.protocol import (
            RequestType,
            get_response_class,
        )

        return self.mq_client.submit_request(
            RequestType.COMMIT_STORE_GROUP_DELTA,
            [key, instance_id, group_idx, skip_count, cpu_data],
            get_response_class(RequestType.COMMIT_STORE_GROUP_DELTA),
        )

    def commit_store_group_delta(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        group_idx: int,
        skip_count: int,
        cpu_data: bytes,
    ) -> bool:
        """Synchronous delta-store commit (waits for server response)."""
        future = self.commit_store_group_delta_raw_async(
            key,
            instance_id,
            group_idx,
            skip_count,
            cpu_data,
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            logger.error(
                "commit_store_group_delta timed out after %ss "
                "(group=%d, skip_count=%d)",
                self.mq_timeout,
                group_idx,
                skip_count,
            )
            return False

    def prepare_retrieve_raw(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> bytes | None:
        """Send PREPARE_RETRIEVE and return raw bytes (no pickle.loads).

        Used by multi-group transfers where the caller deserializes with
        ``_deserialize_multi_group_chunks``. Returns ``None`` on cache-miss
        or timeout.
        """
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

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

    def prepare_retrieve_group_raw_async(
        self, key: IPCCacheServerKey, instance_id: int, group_idx: int
    ) -> "MessagingFuture":
        """Send PREPARE_RETRIEVE_GROUP for one group, return the future.

        Mirror of :meth:`commit_store_group_raw_async` on the retrieve
        side: the caller pipelines one request per group so each
        response stays under the msgspec msgpack bin limit (4 GiB) and
        the server materializes only one group at a time.  The future
        resolves to a ``PrepareRetrieveResponse`` whose ``data`` is the
        wire-format blob for this group (decode with
        ``_deserialize_multi_group_chunks``).
        """
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

        return self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE_GROUP,
            [key, instance_id, group_idx],
            get_response_class(RequestType.PREPARE_RETRIEVE_GROUP),
        )


# Maximum amount of idle page-locked host RAM the PinnedBufferPool
# retains, in GiB.  Buffers beyond the cap are dropped (oldest first) on
# release so their pages get unpinned by the allocator.  A value <= 0
# disables the cap (old unbounded behaviour).
ENV_PINNED_POOL_GB = "LMCACHE_MP_PINNED_POOL_GB"
_DEFAULT_PINNED_POOL_GB = 12.0


def _pinned_pool_capacity_bytes() -> int:
    """Resolve the pinned-pool capacity from the environment (once per pool)."""
    # Standard
    import os

    raw = os.environ.get(ENV_PINNED_POOL_GB, "")
    try:
        gb = float(raw) if raw else _DEFAULT_PINNED_POOL_GB
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %.1f GiB",
            ENV_PINNED_POOL_GB,
            raw,
            _DEFAULT_PINNED_POOL_GB,
        )
        gb = _DEFAULT_PINNED_POOL_GB
    if gb <= 0:
        return 0  # 0 = unlimited
    return int(gb * (1 << 30))


class PinnedBufferPool:
    """Pool of pinned (page-locked) CPU tensors keyed by (shape, dtype).

    Pinned memory is allocated in **system RAM** (via ``cudaHostAlloc``
    / ``mlock`` on Linux) -- it does **not** consume any GPU VRAM.  The
    pool is grown lazily on first ``acquire``.  Idle buffers held by the
    pool are capped at ``LMCACHE_MP_PINNED_POOL_GB`` (default 12 GiB);
    on release, the oldest idle buffers are dropped until the pool is
    under the cap, so a single huge store does not permanently reserve
    tens of GiB of unswappable host RAM.

    Used by ``EngineDrivenTransferContext.submit_store`` to avoid per-
    chunk ``torch.empty(pin_memory=True)`` calls inside
    ``gather_paged_kv_to_cpu``.  Each pool entry is a single CPU tensor;
    the pool itself holds them in a FIFO deque per (shape, dtype).
    """

    def __init__(self, capacity_bytes: int | None = None) -> None:
        self._pools: dict[tuple[tuple[int, ...], torch.dtype], deque[torch.Tensor]] = {}
        self._acquired_count = 0
        self._lock = threading.Lock()
        # Bytes of idle buffers currently held by the pool (in-flight
        # buffers handed out via acquire() are not counted).
        self._idle_bytes = 0
        self._capacity_bytes = (
            capacity_bytes
            if capacity_bytes is not None
            else _pinned_pool_capacity_bytes()
        )

    @staticmethod
    def _nbytes(buf: torch.Tensor) -> int:
        return buf.numel() * buf.element_size()

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
        with self._lock:
            pool = self._pools.setdefault(key, deque())
            out: list[torch.Tensor] = []
            for _ in range(count):
                if pool:
                    buf = pool.popleft()
                    self._idle_bytes -= self._nbytes(buf)
                    out.append(buf)
                else:
                    out.append(
                        torch.empty(
                            shape,
                            dtype=dtype,
                            device=torch.device("cpu"),
                            pin_memory=True,
                        )
                    )
                self._acquired_count += 1
        return out

    def release(self, buffers: list[torch.Tensor]) -> None:
        """Return buffers to the pool for reuse on the next acquire.

        Trims the pool (oldest idle buffers first) when the retained
        idle bytes exceed the configured capacity.
        """
        with self._lock:
            for buf in buffers:
                key = (tuple(buf.shape), buf.dtype)
                self._pools.setdefault(key, deque()).append(buf)
                self._idle_bytes += self._nbytes(buf)
            self._trim_locked()

    def _trim_locked(self) -> None:
        """Drop oldest idle buffers until under capacity (lock held)."""
        if self._capacity_bytes <= 0:
            return
        while self._idle_bytes > self._capacity_bytes:
            # Pop the oldest buffer from the fullest shape-pool.
            fullest = max(self._pools.values(), key=len, default=None)
            if not fullest:
                break
            dropped = fullest.popleft()
            self._idle_bytes -= self._nbytes(dropped)

    def clear(self) -> None:
        """Release all pinned memory.  Call on context destruction."""
        with self._lock:
            self._pools.clear()
            self._acquired_count = 0
            self._idle_bytes = 0

    def stats(self) -> dict:
        """Snapshot of pool state (for diagnostics / benchmarks)."""
        with self._lock:
            return {
                "shapes": [(k[0], k[1], len(v)) for k, v in self._pools.items()],
                "acquired_total": self._acquired_count,
                "idle_bytes": self._idle_bytes,
                "capacity_bytes": self._capacity_bytes,
            }


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
# Multi-group serialization helpers (dev31+: torch.save wire format)
# ---------------------------------------------------------------------------


def _serialize_single_group_chunks(
    group_chunks: list[torch.Tensor],
) -> bytes:
    """Serialize one group's CPU chunk tensors to the wire-format blob.

    Uses :func:`wire_format.serialize_group_chunks_torchsave` which
    produces a ``b'L'`` + ``torch.save`` blob -- ~2x faster than
    ``pickle.HIGHEST_PROTOCOL`` on production-sized (~50 MiB) chunks.
    """
    return _wire_encode(group_chunks)


def _serialize_multi_group_chunks(
    group_chunks: list[list[torch.Tensor]],
) -> bytes:
    """Serialize multiple groups as a single wire-format blob.

    Returns the new ``torch.save`` format with the list-of-lists
    structure preserved -- the deserializer returns the original
    group boundaries.
    """
    return _wire_encode(group_chunks)


def _deserialize_multi_group_chunks(
    cpu_data: bytes,
) -> list[list[torch.Tensor]]:
    """Deserialize a wire-format blob back to ``list[list[Tensor]]``.

    Wire-format compatibility:

    - ``dev31+`` (new): ``b'L'`` magic + ``torch.save`` of a flat
      list of tensors (single-group) or list of lists (multi-group).
    - ``dev24..dev30`` (legacy): ``pickle.HIGHEST_PROTOCOL`` of a
      ``list[list[tagged_tuple]]``.
    - ``<=dev23`` (old): bf16 was upcast to float32; detected by the
      absence of the ``"tensor"`` tag in the tuple.
    """
    if _is_lmcache_blob(cpu_data):
        # New format.  Two shapes are possible:
        #   - flat list of tensors (single-group)
        #   - list of groups (multi-group)
        # Distinguish by inspecting the first element.
        decoded = _wire_decode(cpu_data)
        if decoded is None:
            return []
        if decoded and isinstance(decoded[0], torch.Tensor):
            return [decoded]
        return decoded
    # Legacy format: list of groups, each group is list of tagged tuples
    # Standard
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
            else:
                arr, dtype_name = payload
                if dtype_name == "bfloat16":
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
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and item[0] == "tensor"
    ):
        return "tensor_new", item[1]
    if (
        isinstance(item, tuple)
        and len(item) == 3
        and isinstance(item[0], str)
        and item[0] == "fp8"
    ):
        return "fp8_new", (item[1], item[2])
    if isinstance(item, tuple) and len(item) == 2:
        # Legacy: (ndarray, dtype_str)
        return "old", item
    # Bare tensor (defensive fallback)
    return "tensor_new", item


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


def gather_paged_kv_multi_group_to_cpu_streams(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    engine_group_infos: list[EngineGroupInfo],
    lmcache_tokens_per_chunk: int,
    layout_hints: LayoutHints | None = None,
    pinned_pool: "PinnedBufferPool | None" = None,
) -> list[list[torch.Tensor]]:
    """Multi-stream variant of :func:`gather_paged_kv_multi_group_to_cpu`.

    Runs each group's gather on its own CUDA stream so the GPU->CPU
    copies can in principle overlap.  Disabled by default because the
    PCIe bus is shared on most consumer GPUs and the per-stream
    synchronisation overhead dominates the small overlap win.  See
    ``LMCACHE_MP_MULTI_STREAM_D2D`` env var.
    """
    # Standard
    import threading

    streams = [torch.cuda.Stream() for _ in engine_group_infos]
    # The KV cache blocks were written by the forward pass on the
    # producer stream (the caller's current stream).  Each gather
    # stream must wait on it, otherwise the D2H copies may read stale
    # or half-written KV data.
    ready_event = torch.cuda.Event()
    ready_event.record(torch.cuda.current_stream())
    # Launch each group's gather on its own stream
    results: list[list[torch.Tensor] | None] = [None] * len(engine_group_infos)
    errors: list[BaseException | None] = [None] * len(engine_group_infos)

    def _do(g_idx: int) -> None:
        try:
            streams[g_idx].wait_event(ready_event)
            with torch.cuda.stream(streams[g_idx]):
                results[g_idx] = gather_paged_kv_to_cpu(
                    slice_kv_caches_for_group(
                        kv_caches, engine_group_infos[g_idx].layer_indices
                    ),
                    block_ids[g_idx],
                    _blocks_in_chunk_for(
                        kv_caches,
                        engine_group_infos[g_idx],
                        lmcache_tokens_per_chunk,
                        layout_hints,
                    ),
                    layout_hints=layout_hints,
                    pinned_pool=pinned_pool,
                )
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors[g_idx] = exc

    threads = [
        threading.Thread(target=_do, args=(i,)) for i in range(len(engine_group_infos))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Final sync: each stream + main
    for s in streams:
        s.synchronize()
    # Propagate worker-thread failures instead of silently dropping the
    # failed group: filtering out None entries would shift the group
    # indices of every following group and commit chunks under the
    # wrong group_idx (silent data corruption).
    for g_idx, exc in enumerate(errors):
        if exc is not None:
            raise RuntimeError(
                f"multi-stream gather failed for group {g_idx}"
            ) from exc
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


# Group indices for which a shape-guessed tokens_per_block warning was
# already emitted (warn once per group per process, not per transfer).
_GRAIN_GUESS_WARNED: set[int] = set()


def _guess_tokens_per_block(
    group_idx: int,
    group_kv: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None,
) -> int:
    """Fallback grain when the engine did not report tokens_per_block.

    Derives a block size from the registered tensor SHAPES. This is only
    safe for the legacy single-group case without context parallelism:
    for hybrid models the physical slot count of e.g. a Mamba state
    tensor has nothing to do with the scheduler's block-ID grain, and
    under (uneven) DCP the per-rank physical page differs from the
    virtual scheduler block. A wrong grain silently stores mis-keyed
    chunk data, so warn loudly.
    """
    block_size, _, _, _, _ = compute_kv_layout(group_kv, layout_hints=layout_hints)
    if group_idx not in _GRAIN_GUESS_WARNED:
        _GRAIN_GUESS_WARNED.add(group_idx)
        logger.warning(
            "Group %d has no tokens_per_block from the engine; guessing "
            "%d from the registered tensor shapes. This is unreliable "
            "for hybrid models and under context parallelism -- the "
            "engine should report the scheduler's block-ID grain "
            "(EngineGroupInfo.tokens_per_block).",
            group_idx,
            block_size,
        )
    return block_size


def _blocks_in_chunk_for(
    kv_caches: dict[str, torch.Tensor],
    info: EngineGroupInfo,
    lmcache_tokens_per_chunk: int,
    layout_hints: LayoutHints | None,
) -> int:
    """Canonical blocks-per-chunk for one group.

    Same computation as :func:`gather_paged_kv_multi_group_to_cpu`:
    ``tokens_per_block`` comes from the group info, falling back to the
    vLLM ``block_size`` derived from the KV layout when unknown.
    """
    tpb = info.tokens_per_block
    if tpb <= 0:
        g_kv = slice_kv_caches_for_group(kv_caches, info.layer_indices)
        tpb = _guess_tokens_per_block(info.engine_group_id, g_kv, layout_hints)
    blocks_in_chunk = lmcache_tokens_per_chunk // tpb
    if blocks_in_chunk == 0:
        raise ValueError(
            f"tokens_per_block ({tpb}) > lmcache_tokens_per_chunk "
            f"({lmcache_tokens_per_chunk}). Each block is larger than "
            "the chunk size."
        )
    return blocks_in_chunk


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
        blocks_in_chunk = _blocks_in_chunk_for(
            kv_caches, group_info, lmcache_tokens_per_chunk, layout_hints
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
        blocks_in_chunk = _blocks_in_chunk_for(
            kv_caches, group_info, lmcache_tokens_per_chunk, layout_hints
        )
        scatter_cpu_to_paged_kv(
            group_kv,
            block_ids[group_idx],
            group_chunks[group_idx],
            blocks_in_chunk,
            skip_first_n_tokens=skip_first_n_tokens if group_idx == 0 else 0,
            layout_hints=layout_hints,
        )
