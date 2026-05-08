# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Generator
import argparse
import threading
import time

# Third Party
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle, StorageManager
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.mp_observability.trace import maybe_initialize_trace_recorder
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.gpu_context import (
    GPUCacheContext,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.protocols.controller import PING_SENTINEL_INSTANCE_ID
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
)
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Helper functions
def compute_extra_count(
    tp_size: int,
    world_size: int,
) -> int:
    """Compute extra count for MLA multi-reader locking.

    Non-MLA: each TP worker owns a distinct KV shard,
      so each ObjectKey is retrieved by exactly 1
      worker -> extra_count = 0.
    MLA: TP does not split KV caches, all TP workers
      share the same object. vLLM passes world_size
      already divided by tp_size (e.g. world_size=1
      for TP=4 PP=1), so ipc_keys_to_object_keys
      only produces 1 ObjectKey per chunk.  All TP
      workers retrieve that same ObjectKey, hence
      extra_count = tp_size - 1.

    Detection: tp > world_size means MLA (world_size
    was divided by tp on the vLLM side).

    Fallback: old vLLM (<= 0.8.5) does not send
    tp_size (defaults to 1); we fall back to
    world_size which gives extra_count = 0
    (safe but may under-lock for MLA).

    TODO: world_size currently carries an overloaded
    meaning (total ranks for non-MLA vs total/tp for
    MLA). Consider a dedicated field in the future.

    Args:
        tp_size: Tensor-parallel size from the client.
        world_size: World size from the cache key.

    Returns:
        Number of extra count (0 for non-MLA).
    """
    tp = tp_size if tp_size > 1 else world_size
    return tp - 1 if tp > world_size else 0


def get_layout_desc(gpu_context: GPUCacheContext, num_tokens: int) -> MemoryLayoutDesc:
    """Get the memory layout description for a given GPU context and number of tokens.

    Supports multiple KV layer groups with different shapes and dtypes.

    Args:
        gpu_context: The GPU cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and dtypes.
    """
    num_groups = gpu_context.kv_layer_groups_manager.num_groups
    shapes = [
        gpu_context.get_kv_buffer_shape(num_tokens, group_idx)
        for group_idx in range(num_groups)
    ]
    dtypes = [
        gpu_context.kv_layer_groups_manager.kv_layer_groups[group_idx].dtype
        for group_idx in range(num_groups)
    ]
    return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)


def batched_iteration(lst: list, batch_size: int) -> Generator[tuple, None, None]:
    """Utility function to iterate over a list in batches.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.

    Yields:
        Batches of the list as tuples.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    it = iter(lst)
    while batch := tuple(islice(it, batch_size)):
        yield batch


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    world_size: int
    request_id: str
    # Number of tokens submitted for lookup (denominator for the L1+L2
    # token-level hit-rate metric).  Equals ``len(chunk_hashes) * chunk_size``
    # on the happy path; 0 for early-exit paths (no GPU context matches
    # or chunk_hashes is empty).  Consumed at ``MP_LOOKUP_PREFETCH_END``
    # emission time in ``query_prefetch_status``.
    requested_tokens: int
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""


@dataclass
class _InstanceLiveness:
    """Liveness state tracked per registered worker `instance_id`.

    All times are ``time.monotonic()`` so that NTP/wall-clock jumps cannot
    fire false reaps.
    """

    last_seen: float
    """Monotonic time of the most recent activity from this instance
    (PING / register / store / retrieve)."""

    registered_at: float
    """Monotonic time of the original `register_kv_cache`. Diagnostic only."""


class _InstanceReaperThread(PeriodicThread):
    """Periodically evict workers whose last heartbeat exceeded the deadline.

    Runs on its own thread (not the MQ handler pool). Reaps each stale entry
    by calling ``MPCacheEngine.unregister_kv_cache``. The CAS-style re-check
    inside ``MPCacheEngine._reap_worker`` closes the window where a PING
    refreshed the entry between snapshot and cleanup.

    The reap-after threshold lives on the engine
    (``MPCacheEngine.reap_after_seconds``) — this thread just reads it.
    """

    def __init__(
        self,
        engine: "MPCacheEngine",
        reaper_interval_seconds: float,
    ):
        """
        Args:
            engine: The engine whose ``_liveness`` table we drive.
            reaper_interval_seconds: How often this thread wakes to scan.
        """
        super().__init__(
            name="lmcache-instance-reaper",
            interval=reaper_interval_seconds,
            level=ThreadLevel.HIGH,
        )
        self._engine = engine

    def _execute(self) -> ThreadRunSummary:
        now = time.monotonic()
        deadline = self._engine.reap_after_seconds
        # Snapshot stale candidates under the lock so the iteration can't
        # race a concurrent PING insert/refresh ("dictionary changed size
        # during iteration"). Cleanup runs outside the lock — it calls
        # `unregister_kv_cache` (which can trigger `torch.cuda.empty_cache`)
        # and we don't want to hold a fast lock across GPU work.
        with self._engine._liveness_lock:
            stale = [
                iid
                for iid, st in self._engine._liveness.items()
                if now - st.last_seen > deadline
            ]
        for iid in stale:
            self._engine._reap_worker(iid, now)
        return ThreadRunSummary(
            success=True,
            message=(f"reaped={len(stale)} alive={len(self._engine._liveness)}"),
        )


# Main class for the mp cache engine
class MPCacheEngine:
    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
        reap_after_seconds: float = 30.0,
        reaper_interval_seconds: float = 10.0,
    ):
        # GPU ID -> KV cache tensors
        self.gpu_contexts: dict[int, GPUCacheContext] = {}

        # GPU ID -> (model name, world size) as metadata
        # NOTE: This is mainly for determining the layout desc during prefetch
        # We assume that if the (model name, world size) is the same, then
        # the layout desc returned by the gpu context is the same.
        self.gpu_context_meta: dict[int, tuple[str, int]] = {}

        # chunk size
        self.chunk_size = chunk_size

        # Lock for clear() to avoid concurrent storage manager mutations
        self.lock = threading.Lock()

        # storage manager
        self.storage_manager = StorageManager(storage_manager_config)

        # Token hasher and session manager for token-based operations
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self.session_manager = SessionManager(self.token_hasher)

        # EventBus for observability
        self._event_bus = get_event_bus()

        # Prefetch job tracking for two-phase lookup, keyed by request_id.
        # TODO: implement periodic cleanup of stale _prefetch_jobs entries
        # for crash resilience (e.g., client calls lookup but never queries)
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()

        # Worker liveness tracking — keyed by the worker's `instance_id`
        # (an opaque random 63-bit int generated adapter-side). Refreshed
        # on every PING / register / store / retrieve from that worker;
        # entries past `reap_after_seconds` are evicted by the reaper.
        self.reap_after_seconds = reap_after_seconds
        self._liveness: dict[int, _InstanceLiveness] = {}
        self._liveness_lock = threading.Lock()

        self._setup_metrics()

        # Periodic reaper that evicts unreachable workers. Started here so
        # tests / callers that construct an engine directly don't need to
        # remember to start it; stopped by `close()`.
        self._reaper = _InstanceReaperThread(
            engine=self,
            reaper_interval_seconds=reaper_interval_seconds,
        )
        self._reaper.start()

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
    ) -> None:
        """
        Registers the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (an opaque random 63-bit
                int generated by the adapter). Replaces the legacy
                PID-derived id to avoid container-PID-namespace collisions.
            kv_caches (KVCache): The KV cache tensor wrappers from the
                serving engine.
            model_name (str): The name of the model associated with this KV cache.
            world_size (int): The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to :class:`GPUCacheContext` for format detection.
            layout_hints: See :class:`LayoutHints`.  Forwarded to
                :class:`GPUCacheContext` for GPU KV format detection.
        """
        # Defensive cleanup: with random 63-bit ids the duplicate case is
        # astronomically unlikely, but if it does occur we must release the
        # old `GPUCacheContext` (and its CUDA IPC handles) before installing
        # a new one. Cheap insurance.
        if instance_id in self.gpu_contexts:
            self.unregister_kv_cache(instance_id)

        gpu_context = GPUCacheContext(
            kv_caches,
            self.chunk_size,
            layout_hints=layout_hints or None,
            engine_type=engine_type,
        )
        self.gpu_contexts[instance_id] = gpu_context
        self.gpu_context_meta[instance_id] = (model_name, world_size)
        now = time.monotonic()
        with self._liveness_lock:
            self._liveness[instance_id] = _InstanceLiveness(
                last_seen=now, registered_at=now
            )
        logger.info(
            "Registered KV cache for instance_id=%d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """
        Unregisters the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID.
        """
        with self._liveness_lock:
            self._liveness.pop(instance_id, None)
        if instance_id in self.gpu_contexts:
            del self.gpu_contexts[instance_id]
            del self.gpu_context_meta[instance_id]
            logger.info("Unregistered KV cache for instance_id=%d", instance_id)
            torch.cuda.empty_cache()
        else:
            logger.warning(
                "No KV cache found for instance_id=%d to unregister", instance_id
            )

    def _refresh_liveness(self, instance_id: int) -> None:
        """Refresh the worker's last-seen timestamp.

        Called from `store`/`retrieve` so that an instance actively serving
        traffic is never reaped, even if its periodic PINGs are temporarily
        delayed (GIL-stall during NCCL collective, scheduling jitter).

        Args:
            instance_id: The worker's `instance_id`. Silently ignored if
                the instance is not (or no longer) in `_liveness` —
                concurrent reap may have already evicted it; the existing
                `assert instance_id in self.gpu_contexts` will fire next.
        """
        now = time.monotonic()
        with self._liveness_lock:
            entry = self._liveness.get(instance_id)
            if entry is not None:
                entry.last_seen = now

    def _reap_worker(self, instance_id: int, snapshot_now: float) -> None:
        """Reap a single stale worker.

        CAS-style: re-check `last_seen` under `_liveness_lock` before
        `del`. If a PING refreshed the entry between the reaper's snapshot
        and now, the entry is no longer stale and we leave it.

        Args:
            instance_id: The candidate worker's `instance_id`.
            snapshot_now: The `time.monotonic()` value the reaper used when
                it built its candidate list. Used both to skip races (CAS)
                and to log the staleness duration.
        """
        with self._liveness_lock:
            entry = self._liveness.get(instance_id)
            if entry is None:
                return
            if snapshot_now - entry.last_seen <= self.reap_after_seconds:
                return  # raced with a refresh
            staleness = snapshot_now - entry.last_seen
            del self._liveness[instance_id]
        # `unregister_kv_cache` is idempotent; benign if `gpu_contexts` was
        # already cleared via a concurrent UNREGISTER_KV_CACHE handler.
        self.unregister_kv_cache(instance_id)
        logger.warning(
            "Reaped worker instance_id=%d (%.1fs since last contact)",
            instance_id,
            staleness,
        )

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Stores the GPU KV cache blocks to CPU.

        Args:
            key (IPCCacheEngineKey): The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to store.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the store operation. The second
                element indicates whether the store operation was successful.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]

        st = time.perf_counter()

        assert key.worker_id is not None, "Must store with worker_id != None"
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        # Refresh liveness so an actively-serving worker is never reaped
        # even if its periodic PINGs are briefly delayed.
        self._refresh_liveness(instance_id)
        gpu_context = self.gpu_contexts[instance_id]
        model_name = self.gpu_context_meta[instance_id][0]

        blocks_per_chunk = self.chunk_size // gpu_context.block_size

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)

            # Stage all block_ids to GPU once before the loop
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            # Wait for vLLM to finish
            vllm_event = torch.cuda.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(gpu_context.device)},
                )
            )

            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(gpu_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                ),
            )

            reserved_dict: dict = {}
            try:
                layout_desc = get_layout_desc(gpu_context, self.chunk_size)
                reserved_dict = self.storage_manager.reserve_write(
                    obj_keys, layout_desc, "new"
                )

                # NOTE: Store is not batched because some obj_keys may be
                # skipped (not in reserved_dict), making block_ids
                # non-contiguous. Batching would require torch.cat to
                # reassemble block_ids, negating the benefit.
                num_groups = gpu_context.kv_layer_groups_manager.num_groups
                for idx, obj_key in enumerate(obj_keys):
                    if obj_key in reserved_dict:
                        memory_obj = reserved_dict[obj_key]
                    else:
                        continue

                    chunk_block_ids_gpu = all_block_ids_gpu[
                        idx * blocks_per_chunk : (idx + 1) * blocks_per_chunk
                    ]

                    # Copy from GPU paged buffer to tmp buffer, then to CPU — per group
                    for group_idx in range(num_groups):
                        tmp_buffer = gpu_context.get_tmp_chunk_gpu_buffer(group_idx)
                        group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tmp_buffer.data_ptr()],
                            chunk_block_ids_gpu,
                            gpu_context.device,
                            lmc_ops.TransferDirection.D2H,
                            gpu_context.get_shape_desc(group_idx),
                            self.chunk_size,
                            gpu_context.gpu_kv_format_,
                            0,
                        )
                    # Store is not batched, so we always use chunk_idx=0 (single slot)
                    lmcache_memcpy_async_d2h(
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=0), memory_obj
                    )
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event.record()
                if reserved_dict:
                    gpu_context.cupy_stream.launch_host_func(
                        self.storage_manager.finish_write,
                        list(reserved_dict.keys()),
                    )
                # All reserved MemoryObjs share one layout_desc, so per-object
                # size is identical — avoid summing N identical values.
                total_bytes = (
                    next(iter(reserved_dict.values())).get_size() * len(reserved_dict)
                    if reserved_dict
                    else 0
                )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": len(reserved_dict),
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                        },
                    ),
                )

        ed = time.perf_counter()
        if length := len(reserved_dict):
            logger.info(
                "Stored %d tokens in %.3f seconds",
                length * self.chunk_size,
                ed - st,
            )
        return event.ipc_handle(), True

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """
        Retrieves the CPU KV cache and put into GPU blocks.

        Args:
            key (IPCCacheEngineKey): The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to retrieve into.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.
            skip_first_n_tokens (int): Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the retrieve operation. The second
                element indicates whether the key was successfully retrieved.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]

        st = time.perf_counter()

        assert key.worker_id is not None, "Must retrieve with worker_id != None"
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        self._refresh_liveness(instance_id)
        gpu_context = self.gpu_contexts[instance_id]
        model_name = self.gpu_context_meta[instance_id][0]

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(gpu_context.device)},
            )
        )

        self._event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(gpu_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                },
            ),
        )

        blocks_per_chunk = self.chunk_size // gpu_context.block_size

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            _BATCH_SIZE = gpu_context.max_batch_size
            num_groups = gpu_context.kv_layer_groups_manager.num_groups
            for batch_idx, memory_obj_batch in enumerate(
                batched_iteration(memory_objs, batch_size=_BATCH_SIZE)
            ):
                batch_len = len(memory_obj_batch)
                chunk_start = batch_idx * self.chunk_size * _BATCH_SIZE
                chunk_end = chunk_start + self.chunk_size * batch_len

                effective_start = max(chunk_start, skip_first_n_tokens)
                if effective_start >= chunk_end:
                    # Entire batch is within APC range, skip it
                    continue

                skip_tokens_in_chunk = max(
                    0,
                    min(
                        effective_start - chunk_start,
                        self.chunk_size * batch_len - 1,
                    ),
                )
                if skip_tokens_in_chunk % gpu_context.block_size != 0:
                    logger.error(
                        "skip_first_n_tokens (%d) is not aligned to block_size (%d), "
                        "rounding down from %d tokens to %d blocks",
                        skip_first_n_tokens,
                        gpu_context.block_size,
                        skip_tokens_in_chunk,
                        skip_tokens_in_chunk // gpu_context.block_size,
                    )
                skip_blocks_in_chunk = skip_tokens_in_chunk // gpu_context.block_size

                start_chunk_id = batch_idx * _BATCH_SIZE
                end_chunk_id = start_chunk_id + batch_len
                chunk_block_ids_gpu = all_block_ids_gpu[
                    start_chunk_id * blocks_per_chunk : end_chunk_id * blocks_per_chunk
                ]

                # Copy from CPU to GPU tmp buffers, then scatter to paged KV — per group
                # H2D copy: each memory_obj maps to its own batch slot
                for chunk_idx, memory_obj in enumerate(memory_obj_batch):
                    lmcache_memcpy_async_h2d(
                        memory_obj,
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=chunk_idx),
                    )
                for group_idx in range(num_groups):
                    tmp_buffers = gpu_context.get_tmp_chunk_gpu_buffer_batched(
                        batch_len, group_idx
                    )
                    group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)

                    lmc_ops.multi_layer_block_kv_transfer(
                        group_kv_pointers,
                        [tb.data_ptr() for tb in tmp_buffers],
                        chunk_block_ids_gpu,
                        gpu_context.device,
                        lmc_ops.TransferDirection.H2D,
                        gpu_context.get_shape_desc(group_idx),
                        self.chunk_size,
                        gpu_context.gpu_kv_format_,
                        skip_blocks_in_chunk,
                    )

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            # Stage all block_ids to GPU once before the loop
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            event = torch.cuda.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            retrieve_succeeded = False
            total_bytes = 0
            try:
                with self.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), False

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    total_bytes = sum(mo.get_size() for mo in memory_objs)
                    _retrieve_loop(obj_keys, memory_objs)
                # Only set True when with-block exits normally
                retrieve_succeeded = True
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                if retrieve_succeeded:
                    gpu_context.cupy_stream.launch_host_func(
                        self.storage_manager.finish_read_prefetched,
                        prefetched_keys,
                    )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                        },
                    ),
                )
        tokens_retrieved = len(obj_keys) * self.chunk_size
        ed = time.perf_counter()
        logger.info(
            "Retrieved %d tokens in %.3f seconds",
            tokens_retrieved,
            ed - st,
        )

        return event.ipc_handle(), True

    def _find_layout_desc(
        self,
        model_name: str,
        world_size: int,
    ) -> MemoryLayoutDesc | None:
        """Find layout desc from a matching GPU context.

        Returns:
            The layout descriptor, or None if no context
            matches (model_name, world_size).
        """
        for gpu_id, (m, w) in self.gpu_context_meta.items():
            if m == model_name and w == world_size:
                return get_layout_desc(
                    self.gpu_contexts[gpu_id],
                    self.chunk_size,
                )
        return None

    def lookup(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Tensor-parallel size for MLA multi-reader locking.
        """
        model_name, world_size = key.model_name, key.world_size
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_desc = self._find_layout_desc(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        extra_count = compute_extra_count(tp_size, world_size)

        # Compute chunk hashes for all full chunks
        chunk_hashes = self.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        # Total chunk-aligned tokens submitted for lookup; surfaces as the
        # denominator of the L1+L2 token-level hit-rate via the
        # ``requested_tokens`` field on ``MP_LOOKUP_PREFETCH_END``.  Sub-chunk
        # trailing tokens are intentionally excluded — they cannot hit at
        # chunk granularity.
        requested_tokens = len(chunk_hashes) * self.chunk_size

        # Publish lookup event via EventBus for observability subscribers.
        # Guard with has_subscribers() to avoid allocating the metadata dict
        # (including dtype/shape list comprehensions) when no subscriber is
        # listening (e.g. lookup hash logger is disabled).
        if self._event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        # set lookup ipc key, for session manager to use and generate object keys
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        session.lookup_ipc_key = key

        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        handle = self.storage_manager.submit_prefetch_task(
            obj_keys,
            layout_desc,
            extra_count=extra_count,
            external_request_id=key.request_id,
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                world_size=key.world_size,
                request_id=key.request_id,
                requested_tokens=requested_tokens,
                model_name=model_name,
                cache_salt=key.cache_salt,
            )
        )

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        with self._prefetch_job_lock:
            self._prefetch_jobs[job.request_id] = job

    def query_prefetch_lookup_hits(
        self,
        request_id: str,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished.

        Returns:
            The number of hits for the prefetched keys if the lookup phase is
            done. None if the lookup phase is still in progress. 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)

        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_count = self.storage_manager.query_prefetch_lookup_hits(job.handle)
        if found_count is None:
            return None

        found_count = found_count // job.world_size
        return found_count

    def query_prefetch_status(
        self,
        request_id: str,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_count = self.storage_manager.query_prefetch_status(job.handle)
        if found_count is None:
            return None

        # NOTE(Kuntai): this assumes two things:
        # 1. the world size is the same between keys
        # 2. the lookup sort the keys in prefix order and breaks at the
        #    first failure
        found_count = found_count // job.world_size

        self._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={
                    "found_count": found_count,
                    "requested_tokens": job.requested_tokens,
                    "hit_tokens": found_count * self.chunk_size,
                    "model_name": job.model_name,
                    "cache_salt": job.cache_salt,
                },
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(request_id, None)

        return found_count

    def free_lookup_locks(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Computes the extra reader count from ``tp_size`` and
        ``world_size`` the same way :meth:`lookup` does, so
        the correct number of locks is released.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Tensor-parallel size for MLA
                multi-reader locking.
        """
        chunk_hashes = self.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=key.end
        )
        if not chunk_hashes:
            return
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        extra_count = compute_extra_count(tp_size, key.world_size)

        self.storage_manager.finish_read_prefetched(obj_keys, extra_count=extra_count)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def ping(self, instance_id: int) -> bool:
        """Respond to a ping request and refresh liveness.

        Args:
            instance_id: The sender's `instance_id`. Workers pass their
                random 63-bit id; the scheduler-side adapter passes
                ``PING_SENTINEL_INSTANCE_ID`` (0) — its PING is purely a
                server-health probe and is not tracked.

        Returns:
            ``True`` if the sender is the sentinel or is recognized in the
            liveness table (in which case `last_seen` is refreshed);
            ``False`` if `instance_id` is unknown to the server. A False
            return is the terminal "you've been reaped" signal — the
            adapter clears `_health_event`, stops its heartbeat thread,
            and waits for operator restart. No auto-resurrection.
        """
        if instance_id == PING_SENTINEL_INSTANCE_ID:
            return True
        now = time.monotonic()
        with self._liveness_lock:
            entry = self._liveness.get(instance_id)
            if entry is None:
                logger.warning(
                    "PING from unknown instance_id=%d; signaling adapter "
                    "to stop (operator must restart vLLM).",
                    instance_id,
                )
                return False
            entry.last_seen = now
        return True

    def get_chunk_size(self) -> int:
        """
        Returns the chunk size used for KV cache operations.

        Returns:
            int: The chunk size.
        """
        return self.chunk_size

    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self.session_manager.remove(request_id)
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=request_id,
            )
        )
        if session is None:
            logger.warning("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch", request_id
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        obj_keys = ipc_key_to_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self.storage_manager.touch_l1_keys(obj_keys)

    def report_status(self) -> dict:
        """Return a status dict for the entire cache engine."""
        sm = self.storage_manager.report_status()

        gpu_context_meta: dict[str, dict] = {}
        for gpu_id, meta in self.gpu_context_meta.items():
            entry: dict = {
                "model_name": meta[0],
                "world_size": meta[1],
            }
            ctx = self.gpu_contexts.get(gpu_id)
            if ctx is not None:
                entry["kv_cache_layout"] = {
                    "num_layers": ctx.num_layers,
                    "block_size": ctx.block_size,
                    "hidden_dim_sizes": str(ctx.hidden_dim_sizes),
                    "dtype": str(ctx.dtype),
                    "is_mla": ctx.is_mla,
                    "num_blocks": ctx.num_blocks,
                    "gpu_kv_format": ctx.gpu_kv_format_name,
                    "gpu_kv_shape": ctx.gpu_kv_shape,
                    "gpu_kv_concrete_shape": ctx.concrete_gpu_kv_shape,
                    "attention_backend": ctx.attention_backend,
                    "cache_size_per_token": ctx.cache_size_per_token(),
                }
            gpu_context_meta[str(gpu_id)] = entry

        return {
            "is_healthy": sm["is_healthy"],
            "engine_type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "hash_algorithm": self.token_hasher.hash_algorithm_name,
            "registered_gpu_ids": list(self.gpu_contexts.keys()),
            "gpu_context_meta": gpu_context_meta,
            "active_sessions": self.session_manager.active_count(),
            "active_prefetch_jobs": self._active_prefetch_count(),
            "storage_manager": sm,
        }

    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus.

        Args:
            instance_id: The scheduler instance ID.
            model_name: The model name from the adapter.
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "records": records,
                },
            )
        )

    def debug(self) -> str:
        return "OK"

    def clear(self) -> None:
        """
        Clears all stored KV cache data from the storage manager.
        """
        with self.lock:
            self.storage_manager.memcheck()
            self.storage_manager.clear(force=True)
            self.storage_manager.memcheck()

    def close(self) -> None:
        """
        Closes the MPCacheEngine and releases all resources.
        """
        # Stop the reaper FIRST so it can't fire mid-shutdown and call
        # `unregister_kv_cache`/`empty_cache` against half-closed state.
        self._reaper.stop()

        # Close storage manager
        self.storage_manager.close()
        logger.info("MPCacheEngine closed")

        # Release GPU contexts
        self.gpu_contexts.clear()
        with self._liveness_lock:
            self._liveness.clear()

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for MP engine metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_engine")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )


def add_handler_helper(
    server: MessageQueueServer, request_type: RequestType, handler_function
):
    payload_classes = get_payload_classes(request_type)
    handler_type = get_handler_type(request_type)
    server.add_handler(
        request_type,
        payload_classes,
        handler_type,
        handler_function,
    )


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        obs_config: Configuration for the observability stack
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive
        start_prometheus_http_server: Whether to start a standalone
            Prometheus HTTP server in a background thread.  Set to
            ``False`` when an external HTTP framework already serves
            ``/metrics`` to avoid port conflicts or redundant servers.

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheEngine)
        If return_engine is False: None (blocks until interrupted)
    """
    event_bus = init_observability(
        obs_config, start_prometheus_http_server=start_prometheus_http_server
    )

    # Wire up the trace recorder (no-op when --trace-level is unset).
    # Registered before the engine handlers are added so any
    # storage-manager calls during engine init are captured too.
    maybe_initialize_trace_recorder(event_bus, obs_config, storage_manager_config)

    # Initialize the engine (loggers self-register with the global controller)
    engine = MPCacheEngine(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
        reap_after_seconds=mp_config.reap_after_seconds,
        reaper_interval_seconds=mp_config.reaper_interval_seconds,
    )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=context,
    )

    # Add handlers
    add_handler_helper(server, RequestType.REGISTER_KV_CACHE, engine.register_kv_cache)
    add_handler_helper(
        server, RequestType.UNREGISTER_KV_CACHE, engine.unregister_kv_cache
    )
    add_handler_helper(server, RequestType.STORE, engine.store)
    add_handler_helper(server, RequestType.LOOKUP, engine.lookup)
    add_handler_helper(
        server, RequestType.QUERY_PREFETCH_STATUS, engine.query_prefetch_status
    )
    add_handler_helper(
        server,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        engine.query_prefetch_lookup_hits,
    )
    add_handler_helper(server, RequestType.FREE_LOOKUP_LOCKS, engine.free_lookup_locks)
    add_handler_helper(server, RequestType.RETRIEVE, engine.retrieve)
    add_handler_helper(server, RequestType.CLEAR, engine.clear)
    add_handler_helper(server, RequestType.GET_CHUNK_SIZE, engine.get_chunk_size)
    add_handler_helper(server, RequestType.PING, engine.ping)
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)
    add_handler_helper(
        server,
        RequestType.REPORT_BLOCK_ALLOCATION,
        engine.report_block_allocations,
    )

    # Assign thread pools
    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE],
        max_workers=mp_config.max_gpu_workers,
    )
    server.add_normal_thread_pool(
        [
            RequestType.LOOKUP,
            RequestType.QUERY_PREFETCH_STATUS,
            RequestType.QUERY_PREFETCH_LOOKUP_HITS,
            RequestType.FREE_LOOKUP_LOCKS,
            RequestType.END_SESSION,
            RequestType.CLEAR,
            RequestType.PING,
            RequestType.REPORT_BLOCK_ALLOCATION,
        ],
        max_workers=mp_config.max_cpu_workers,
    )

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )
    # Start the ZMQ server
    torch.cuda.init()
    server.start()

    logger.info("LMCache cache server is running...")

    # Return server and engine if requested (for HTTP server integration)
    if return_engine:
        return server, engine

    # Dummy loop to keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        event_bus.stop()
        server.close()
        engine.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )
