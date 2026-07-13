# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Protocol
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints, is_mla
from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    PinnedBufferPool,
    _deserialize_multi_group_chunks,
    _serialize_single_group_chunks,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_multi_group_to_cpu,
    gather_paged_kv_multi_group_to_cpu_streams,
    gather_paged_kv_to_cpu,
    scatter_cpu_multi_group_to_paged_kv,
    scatter_cpu_to_paged_kv,
    slice_kv_caches_for_group,
)
from lmcache.v1.platform import _registry as platform_registry

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"

# Opt-in: parallelise per-group GPU->CPU D2D copies across CUDA streams.
# Default is OFF because in our benchmarks the PCIe bus is shared and
# multi-stream adds sync overhead (~4x slower for serial D2D on a
# typical consumer GPU).  Set to "1" to force-enable for A/B testing.
ENV_MULTI_STREAM_D2D = "LMCACHE_MP_MULTI_STREAM_D2D"

# Opt-in: delta-store. Worker passes ``skip_count`` per group derived
# from the prior lookup's prefix hit count. Saves 14 GiB of wire on
# the re-run of a cached prompt (60k-token Qwen3-27B-AWQ).
# Default is ON -- the lookup is a no-op when no STORE direction is
# requested, and the savings on hot-cache rebuilds are large.
ENV_DELTA_STORE = "LMCACHE_MP_DELTA_STORE"


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode %r (valid: %s)" % (raw, valid)
        ) from exc


def _build_lmcache_driven_context(device_type: str) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        platform_registry.get_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    return LMCacheDrivenTransferContext()


class IPCEvent(Protocol):
    """Protocol for IPC-capable CUDA events used by transport operations."""

    def ipc_handle(self) -> object:
        """Return an IPC handle consumable by the multiprocess server."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. CUDA paths return
    CUDA-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (IPC for CUDA,
    SHM wrappers for CPU with CUDA-IPC-like semantics) and the LMCache
    server performs direct device-side data transfer.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        # First Party
        from lmcache.integration.vllm.vllm_multi_process_adapter import wrap_kv_caches

        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._mq_client is None or self._send_request is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event.ipc_handle()],
        ).to_cuda_future()

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._mq_client is None or self._send_request is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event.ipc_handle(), skip_first_n_tokens],
        ).to_cuda_future()

    def close(self) -> None:
        self._mq_client = None
        self._send_request = None


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    Supports hybrid multi-group KV cache models (e.g. GDN + Attention).
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._engine_group_infos: list[EngineGroupInfo] = []
        self._lmcache_tokens_per_chunk: int = 0
        # System-RAM pinned buffer pool. Avoids per-store CUDA-allocator
        # overhead and amortises the page-locking cost across calls.
        # Pinned memory is **system RAM**, never GPU VRAM.
        self._pinned_pool: PinnedBufferPool = PinnedBufferPool()
        # Worker-thread pool that parallelises per-group pickle
        # serialisation.  Sized to the typical KV-group count (4 for
        # Qwen3-27B hybrid: GatedDeltaNet + Attention).  Each thread
        # runs ``_serialize_single_group_chunks`` on a different
        # group's chunks, then immediately submits the resulting
        # ``COMMIT_STORE_GROUP`` message -- so the LMCache server's
        # D2D commit and the worker's next group's serialisation overlap
        # instead of running serially.  This is the dominant TTFT
        # win for cache-warm requests with hybrid multi-group models.
        self._serialize_pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="lmcache-serialize",
        )
        # LRU prefetch hint predictor.  Tracks recent lookups so the
        # connector can speculatively pre-warm a likely-next key from
        # L2.  Inherently heuristic -- disabled by default because the
        # L1 cache already covers the same-prompt re-run case; useful
        # only for chat-session style workloads with overlapping
        # prefixes.  See ``PrefetchPredictor`` for the heuristic.
        # First Party
        from lmcache.v1.multiprocess.transfer_context.prefetch_predictor import (
            PrefetchPredictor,
        )

        self._prefetch_predictor: PrefetchPredictor = PrefetchPredictor(
            max_entries=8,
        )

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register KV caches with the non-GPU context server.

        ``engine_group_infos`` is accepted to satisfy the base interface.
        When multiple groups are provided, the multi-group path is used
        for hybrid KV cache models (e.g. GDN + Attention).
        """
        # First Party
        from lmcache.v1.multiprocess.custom_types import GroupLayoutInfo

        self._layout_hints = layout_hints
        self._engine_group_infos = list(engine_group_infos)
        num_groups = len(engine_group_infos)
        is_multi_group = num_groups > 1

        if not is_multi_group:
            # ── Single-group path (backward compatible) ─────────────────
            (
                block_size,
                num_layers,
                hidden_dim_size,
                dtype_str,
                engine_kv_format,
            ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
            self._engine_kv_format = engine_kv_format
            use_mla_flag = is_mla(engine_kv_format)
            shape = (
                torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
                if use_mla_flag
                else torch.Size(
                    [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
                )
            )
            dtype = getattr(torch, dtype_str)
            layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
            group_layouts: list[GroupLayoutInfo] = []
            self._lmcache_tokens_per_chunk = blocks_in_chunk * block_size
        else:
            # ── Multi-group path ─────────────────────────────────────────
            group_layouts_list: list[GroupLayoutInfo] = []
            group_layout_descs: list[MemoryLayoutDesc] = []
            group_block_sizes: list[int] = []
            group_use_mla_flags: list[bool] = []
            group_blocks_in_chunk: list[int] = []

            for group_info in engine_group_infos:
                group_kv = slice_kv_caches_for_group(
                    kv_caches, group_info.layer_indices
                )
                (
                    g_block_size,
                    g_num_layers,
                    g_hidden_dim,
                    g_dtype_str,
                    g_fmt,
                ) = compute_kv_layout(group_kv, layout_hints=layout_hints)

                tpb = (
                    group_info.tokens_per_block
                    if group_info.tokens_per_block > 0
                    else g_block_size
                )
                g_use_mla = is_mla(g_fmt)
                g_dtype = getattr(torch, g_dtype_str)

                # For multi-group, lmcache_tokens_per_chunk is global
                # Use first group's block_size to compute it
                if group_layouts_list:
                    # Already computed from first group
                    lmcache_tokens_per_chunk = self._lmcache_tokens_per_chunk
                else:
                    self._lmcache_tokens_per_chunk = blocks_in_chunk * g_block_size
                    lmcache_tokens_per_chunk = self._lmcache_tokens_per_chunk

                g_blocks_in_chunk = lmcache_tokens_per_chunk // tpb
                g_chunk_tokens = g_blocks_in_chunk * tpb
                g_shape = (
                    torch.Size([g_num_layers, g_chunk_tokens, g_hidden_dim])
                    if g_use_mla
                    else torch.Size([2, g_num_layers, g_chunk_tokens, g_hidden_dim])
                )
                g_layout_desc = MemoryLayoutDesc(shapes=[g_shape], dtypes=[g_dtype])
                group_layout_descs.append(g_layout_desc)
                group_block_sizes.append(g_block_size)
                group_use_mla_flags.append(g_use_mla)
                group_blocks_in_chunk.append(g_blocks_in_chunk)
                group_layouts_list.append(
                    GroupLayoutInfo(
                        block_size=g_block_size,
                        num_layers=g_num_layers,
                        hidden_dim_size=g_hidden_dim,
                        dtype_str=g_dtype_str,
                        use_mla=g_use_mla,
                        tokens_per_block=tpb,
                    )
                )

            # For single-group fields: use first group's values
            block_size = group_block_sizes[0]
            dtype_str = group_layouts_list[0].dtype_str
            num_layers = group_layouts_list[0].num_layers
            hidden_dim_size = group_layouts_list[0].hidden_dim_size
            use_mla_flag = group_use_mla_flags[0]
            layout_desc = group_layout_descs[0]
            group_layouts = group_layouts_list
            self._engine_kv_format = None

        # Send registration request
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    group_layouts=group_layouts,
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        if is_multi_group:
            metadata = EngineDrivenContextMetadata(
                layout_desc=layout_desc,
                block_size=block_size,
                use_mla=use_mla_flag,
                group_layout_descs=group_layout_descs,
                group_block_sizes=group_block_sizes,
                group_use_mla=group_use_mla_flags,
                group_blocks_in_chunk=group_blocks_in_chunk,
            )
        else:
            metadata = EngineDrivenContextMetadata(
                layout_desc=layout_desc,
                block_size=block_size,
                use_mla=use_mla_flag,
            )

        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s, groups=%d)",
            instance_id,
            supported_transfer_mode,
            num_groups,
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        # Synchronize only the current device stream. Avoids waiting for
        # unrelated kernels (model forward, NCCL collectives) that the
        # global torch_dev.synchronize() would block on. The D2D copy
        # kernel uses at::cuda::getCurrentCUDAStream() so this is
        # sufficient to ensure the pinned-memory tensors are visible.
        if len(self._engine_group_infos) > 1:
            # ── Multi-group path ─────────────────────────────────────────
            if len(block_ids) != len(self._engine_group_infos):
                raise ValueError(
                    f"block_ids has {len(block_ids)} groups, "
                    f"but {len(self._engine_group_infos)} engine_group_infos registered"
                )
            if os.environ.get(ENV_MULTI_STREAM_D2D) == "1":
                # Opt-in: parallelise D2D across CUDA streams.
                # Disabled by default -- PCIe is shared, the sync
                # overhead exceeds the overlap win on consumer GPUs.
                group_chunks = gather_paged_kv_multi_group_to_cpu_streams(
                    kv_caches,
                    block_ids,
                    self._engine_group_infos,
                    lmcache_tokens_per_chunk=self._lmcache_tokens_per_chunk,
                    layout_hints=self._layout_hints,
                    pinned_pool=self._pinned_pool,
                )
            else:
                group_chunks = gather_paged_kv_multi_group_to_cpu(
                    kv_caches,
                    block_ids,
                    self._engine_group_infos,
                    lmcache_tokens_per_chunk=self._lmcache_tokens_per_chunk,
                    layout_hints=self._layout_hints,
                    pinned_pool=self._pinned_pool,
                )
            # ── Parallel serialize + pipelined submit ───────────────────
            # GPU→CPU D2D copies (issued above, all queued on the same
            # stream) finish on the single ``current_stream().synchronize()``
            # below.  Once the GPU side is done, we serialize each group's
            # chunks on a worker-thread pool so the CPU-side pickle time
            # for group N+1 overlaps the zmq round-trip for group N.
            torch_dev.current_stream().synchronize()
            num_groups = len(group_chunks)
            mq_futures: list = [None] * num_groups
            ok = True

            def _serialize_and_submit(
                g_idx: int, chunks, skip_count: int = 0
            ) -> tuple[int, "Future"]:
                data = _serialize_single_group_chunks(chunks)
                if os.environ.get(ENV_DELTA_STORE, "1") == "1":
                    # Use the delta-store variant. The connector can
                    # supply the per-group skip_count via a future
                    # LoadStoreOp field; for now we use 0 which is
                    # semantically identical to the legacy full send.
                    fut = (
                        self._engine_driven_context.commit_store_group_delta_raw_async(
                            key,
                            instance_id,
                            g_idx,
                            skip_count,
                            data,
                        )
                    )
                else:
                    fut = self._engine_driven_context.commit_store_group_raw_async(
                        key,
                        instance_id,
                        g_idx,
                        data,
                    )
                return g_idx, fut

            active = [(g_idx, gc) for g_idx, gc in enumerate(group_chunks) if gc]
            # ``submit`` is portable across Python versions where
            # ``starmap`` may not exist.  Each worker thread calls
            # ``_serialize_and_submit(g_idx, chunks)`` and returns
            # ``(g_idx, mq_future)``.  We dispatch all 4 groups in
            # parallel, then collect MQ futures for the final join.
            serialized_futs: dict[int, "Future"] = {}
            for g_idx, chunks in active:
                serialized_futs[g_idx] = self._serialize_pool.submit(
                    _serialize_and_submit,
                    g_idx,
                    chunks,
                )
            for g_idx, fut in serialized_futs.items():
                _, mq_fut = fut.result()
                mq_futures[g_idx] = mq_fut
            # Now collect all responses -- any per-group failure marks the
            # overall store as failed, but the rest still complete.
            for fut in mq_futures:
                if fut is None:
                    continue
                try:
                    group_ok = bool(
                        fut.result(timeout=self._engine_driven_context.mq_timeout)
                    )
                except TimeoutError:
                    group_ok = False
                ok = ok and group_ok
            # Return pinned buffers to the pool now that serialize+send is
            # done -- subsequent calls reuse them without re-locking pages.
            for group_chunks_list in group_chunks:
                if group_chunks_list:
                    self._pinned_pool.release(group_chunks_list)
        else:
            # ── Single-group path (backward compatible) ──────────────────
            result = self._engine_driven_context.prepare_store(key, instance_id)
            out_buffers, chunk_indices = result if result is not None else (None, None)
            if chunk_indices is not None and len(chunk_indices) == 0:
                future: MessagingFuture[bool] = MessagingFuture()
                future.set_result(True)
                return future
            cpu_chunks = gather_paged_kv_to_cpu(
                kv_caches,
                block_ids[0],
                blocks_in_chunk,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
                out=out_buffers,
                chunk_indices=chunk_indices,
            )
            if out_buffers is not None:
                torch_dev.current_stream().synchronize()
            ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        if len(self._engine_group_infos) > 1:
            # ── Multi-group path ─────────────────────────────────────────
            if len(block_ids) != len(self._engine_group_infos):
                raise ValueError(
                    f"block_ids has {len(block_ids)} groups, "
                    f"but {len(self._engine_group_infos)} engine_group_infos registered"
                )
            raw = self._engine_driven_context.prepare_retrieve_raw(key, instance_id)
            ok = raw is not None
            if raw:
                group_chunks = _deserialize_multi_group_chunks(raw)
                scatter_cpu_multi_group_to_paged_kv(
                    kv_caches,
                    block_ids,
                    group_chunks,
                    self._engine_group_infos,
                    lmcache_tokens_per_chunk=self._lmcache_tokens_per_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                )
                torch_dev.current_stream().synchronize()
            self._engine_driven_context.commit_retrieve(key, instance_id)
        else:
            src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
            ok = src_buffers is not None
            if src_buffers is not None:
                try:
                    scatter_cpu_to_paged_kv(
                        kv_caches,
                        block_ids[0],
                        src_buffers,
                        blocks_in_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        engine_kv_format=self._engine_kv_format,
                    )
                except (RuntimeError, ValueError, TypeError, IndexError):
                    logger.exception("Failed to scatter retrieved CPU context chunks")
                    ok = False
                torch_dev.current_stream().synchronize()
        future = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None
        # Shutdown the serialisation thread pool to avoid leaking threads.
        if self._serialize_pool is not None:
            self._serialize_pool.shutdown(wait=True)
            self._serialize_pool = None
        # Release pinned memory back to the OS.
        self._pinned_pool.clear()


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    num_engine_groups: int = 1,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
        num_engine_groups: Number of KV cache groups (1 = standard, >1 = hybrid).
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s, num_engine_groups=%d)",
        device_type,
        resolved_mode.value,
        num_engine_groups,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        if num_engine_groups > 1:
            raise ValueError(
                "Transfer mode 'lmcache_driven' does not support hybrid models "
                f"(num_engine_groups={num_engine_groups}). "
                "Use 'engine_driven' or 'auto'."
            )
        return _build_lmcache_driven_context(device_type)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return EngineDrivenTransferContext()
    # AUTO: dispatch by device type
    if device_type == "cuda":
        if num_engine_groups > 1:
            logger.info(
                "AUTO mode: hybrid model (%d groups) detected → engine-driven "
                "(eliminates server-side VRAM)",
                num_engine_groups,
            )
            return EngineDrivenTransferContext()
        return LMCacheDrivenTransferContext()
    return EngineDrivenTransferContext()
