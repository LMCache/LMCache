# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
import argparse
import threading
import time

# Third Party
import prometheus_client
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_keys_to_object_keys,
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
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.config import (
    PrometheusConfig,
    add_prometheus_args,
    parse_args_to_prometheus_config,
)
from lmcache.v1.mp_observability.prometheus_controller import (
    get_prometheus_controller,
    init_prometheus_controller,
)
from lmcache.v1.mp_observability.telemetry import (
    TelemetryConfig,
    add_telemetry_args,
    get_telemetry_controller,
    init_telemetry_controller,
    log_telemetry,
    make_end_event,
    make_start_event,
    parse_args_to_telemetry_config,
)
from lmcache.v1.mp_observability.telemetry.config import (
    DEFAULT_TELEMETRY_CONFIG,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.custom_types import (
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
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Helper functions
def update_session_for_key(
    key: IPCCacheEngineKey,
    session_manager: SessionManager,
) -> None:
    """Update session state for a token-mode key.

    Sets the token sequence on the session and computes hashes so they
    are cached for resolve_keys.

    Args:
        key: An IPC cache engine key.
        session_manager: The session manager to use.
    """
    session = session_manager.get_or_create(key.request_id)
    session.set_tokens(list(key.token_ids))
    session.get_hashes(key.start, key.end)


def resolve_key(
    key: IPCCacheEngineKey,
    session_manager: SessionManager,
) -> list[IPCCacheEngineKey]:
    """Convert a token-mode key to hash-mode keys.

    Uses session to retrieve pre-computed rolling hashes, then creates
    hash-mode IPCCacheEngineKey instances.
    update_session_for_key must be called before this function.

    Args:
        key: An IPC cache engine key.
        session_manager: The session manager to use.

    Returns:
        List of IPCCacheEngineKey with hash, one per chunk.
    """
    session = session_manager.get_or_create(key.request_id)
    hashes = session.get_hashes(key.start, key.end)
    return [
        IPCCacheEngineKey(
            model_name=key.model_name,
            world_size=key.world_size,
            worker_id=key.worker_id,
            token_ids=key.token_ids,
            start=key.start,
            end=key.end,
            request_id=key.request_id,
            chunk_hash=TokenHasher.hash_to_bytes(h),
        )
        for h in hashes
    ]


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

    Args:
        gpu_context: The GPU cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and dtypes.
    """
    shape = gpu_context.get_kv_buffer_shape(num_tokens)
    dtype = gpu_context.dtype
    return MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    world_size: int
    request_id: str


# Main class for the mp cache engine
class MPCacheEngine:
    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
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

        # thread lock to avoid tmp buffer conflicts
        self.lock = threading.Lock()

        # storage manager
        self.storage_manager = StorageManager(storage_manager_config)

        # Token hasher and session manager for token-based operations
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self.session_manager = SessionManager(self.token_hasher)

        # Prefetch job tracking for two-phase lookup
        # TODO: implement periodic cleanup of stale _prefetch_jobs entries
        # for crash resilience (e.g., client calls lookup but never queries)
        self._prefetch_jobs: dict[int, _PrefetchJob] = {}
        self._next_prefetch_job_id: int = 0
        self._prefetch_job_lock = threading.Lock()

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
    ) -> None:
        """
        Registers the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
            kv_caches (KVCache): The KV cache tensor wrappers from vLLM.
            model_name (str): The name of the model associated with this KV cache.
            world_size (int): The world size associated with this KV cache.
        """
        gpu_context = GPUCacheContext(kv_caches, self.chunk_size)
        self.gpu_contexts[instance_id] = gpu_context
        self.gpu_context_meta[instance_id] = (model_name, world_size)
        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """
        Unregisters the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
        """
        if instance_id in self.gpu_contexts:
            del self.gpu_contexts[instance_id]
            del self.gpu_context_meta[instance_id]
            logger.info("Unregistered KV cache for GPU ID %d", instance_id)
            torch.cuda.empty_cache()
        else:
            logger.warning("No KV cache found for GPU ID %d to unregister", instance_id)

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
        update_session_for_key(key, self.session_manager)
        ipc_keys = resolve_key(key, self.session_manager)

        st = time.perf_counter()

        assert all(k.worker_id is not None for k in ipc_keys), (
            "Must store with worker_id != None"
        )
        obj_keys = ipc_keys_to_object_keys(ipc_keys)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)
            slot_mapping_tensor = gpu_context.get_slot_mapping_tensor(gpu_block_ids)

            # Wait for vLLM to finish
            vllm_event = torch.cuda.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            if get_telemetry_controller().is_enabled():
                gpu_context.cupy_stream.launch_host_func(
                    log_telemetry,
                    make_start_event(
                        "store",
                        key.request_id,
                        device=str(gpu_context.device),
                    ),
                )

            layout_desc = get_layout_desc(gpu_context, self.chunk_size)
            reserved_dict = self.storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )

            for idx, obj_key in enumerate(obj_keys):
                if obj_key in reserved_dict:
                    memory_obj = reserved_dict[obj_key]
                else:
                    continue

                start = idx * self.chunk_size
                end = start + self.chunk_size
                slot_mapping = slot_mapping_tensor[start:end]

                # Copy from GPU to CPU
                tmp_buffer = gpu_context.get_tmp_gpu_buffer(self.chunk_size)
                with self.lock:
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_buffer,
                        gpu_context.kv_pointers,
                        slot_mapping,
                        gpu_context.device,
                        gpu_context.block_size * gpu_context.num_blocks,
                        lmc_ops.TransferDirection.D2H,
                        gpu_context.gpu_kv_format_,
                        gpu_context.block_size,
                    )

                    assert memory_obj.tensor is not None
                    lmcache_memcpy_async_d2h(tmp_buffer, memory_obj)

            event.record()

        self.gpu_contexts[instance_id].cupy_stream.launch_host_func(
            self.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )

        if get_telemetry_controller().is_enabled():
            self.gpu_contexts[instance_id].cupy_stream.launch_host_func(
                log_telemetry,
                make_end_event(
                    "store",
                    key.request_id,
                    stored_count=len(reserved_dict),
                    device=str(gpu_context.device),
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
        update_session_for_key(key, self.session_manager)
        ipc_keys = resolve_key(key, self.session_manager)

        st = time.perf_counter()

        assert all(k.worker_id is not None for k in ipc_keys), (
            "Must retrieve with worker_id != None"
        )
        obj_keys = ipc_keys_to_object_keys(ipc_keys)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        if get_telemetry_controller().is_enabled():
            gpu_context.cupy_stream.launch_host_func(
                log_telemetry,
                make_start_event(
                    "retrieve",
                    key.request_id,
                    device=str(gpu_context.device),
                ),
            )

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            for idx, (key, memory_obj) in enumerate(
                zip(keys, memory_objs, strict=False)
            ):
                chunk_start = idx * self.chunk_size
                chunk_end = chunk_start + self.chunk_size

                # Skip tokens that overlap with APC-cached blocks to
                # avoid a data race: the retrieve writes on the LMCache
                # CUDA stream while concurrent requests may read from
                # those same APC-shared blocks on the vLLM CUDA stream.
                effective_start = max(chunk_start, skip_first_n_tokens)
                if effective_start >= chunk_end:
                    # Entire chunk is within APC range, skip it
                    continue
                # clamp to [0, chunk_size - 1]
                skip_in_chunk = max(
                    0, min(effective_start - chunk_start, self.chunk_size - 1)
                )
                slot_mapping = slot_mapping_tensor[chunk_start:chunk_end]

                # Copy from CPU to GPU
                tmp_gpu_buffer_ = gpu_context.get_tmp_gpu_buffer(self.chunk_size)
                with self.lock:
                    lmcache_memcpy_async_h2d(memory_obj, tmp_gpu_buffer_)
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_gpu_buffer_,
                        gpu_context.kv_pointers,
                        slot_mapping,
                        gpu_context.device,
                        gpu_context.block_size * gpu_context.num_blocks,
                        lmc_ops.TransferDirection.H2D,
                        gpu_context.gpu_kv_format_,
                        gpu_context.block_size,
                        skip_in_chunk,
                    )

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            slot_mapping_tensor = gpu_context.get_slot_mapping_tensor(gpu_block_ids)

            event = torch.cuda.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            try:
                with self.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), False

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    _retrieve_loop(obj_keys, memory_objs)
            except Exception as e:
                logger.warning("Cannot retrieve keys due to exception: %s", str(e))
                return event.ipc_handle(), False
            finally:
                event.record()
                gpu_context.cupy_stream.launch_host_func(
                    self.storage_manager.finish_read_prefetched,
                    prefetched_keys,
                )
                if get_telemetry_controller().is_enabled():
                    gpu_context.cupy_stream.launch_host_func(
                        log_telemetry,
                        make_end_event(
                            "retrieve",
                            key.request_id,
                            retrieved_count=len(prefetched_keys),
                            device=str(gpu_context.device),
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
    ) -> int:
        """Submit a prefix lookup and return a prefetch job ID.

        Hashes the key, submits a prefetch task to the storage manager,
        and returns a job ID that can be polled via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.

        Returns:
            Prefetch job ID for polling via query_prefetch_status.
        """
        ipc_keys: list[IPCCacheEngineKey] = []
        model_name, world_size = key.model_name, key.world_size
        log_telemetry(make_start_event("lookup_and_prefetch", key.request_id))

        layout_desc = self._find_layout_desc(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            return self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        request_id=-1,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                )
            )

        extra_count = compute_extra_count(tp_size, world_size)

        # Prepare for the obj keys
        ipc_keys.extend(key.to_hash_keys(self.token_hasher))
        if not ipc_keys:
            return self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        request_id=-1,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                )
            )

        obj_keys = ipc_keys_to_object_keys(ipc_keys)

        handle = self.storage_manager.submit_prefetch_task(
            obj_keys, layout_desc, extra_count=extra_count
        )
        return self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                world_size=ipc_keys[0].world_size,
                request_id=key.request_id,
            )
        )

    def _register_prefetch_job(self, job: _PrefetchJob) -> int:
        with self._prefetch_job_lock:
            job_id = self._next_prefetch_job_id
            self._next_prefetch_job_id += 1
            self._prefetch_jobs[job_id] = job
        return job_id

    def query_prefetch_status(
        self,
        prefetch_job_id: int,
    ) -> int | None:
        """Poll the status of a prefetch job.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            prefetch_job_id: Job ID returned by lookup().

        Returns:
            Chunk count (int) when done, None if still in progress
            or the job ID is unknown.
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(prefetch_job_id)
        if job is None:
            logger.warning(
                "Prefetch job %d not found (already completed or invalid)",
                prefetch_job_id,
            )
            return None

        found_count = self.storage_manager.query_prefetch_status(job.handle)
        if found_count is None:
            return None

        # NOTE(Kuntai): this assumes two things:
        # 1. the world size is the same between keys
        # 2. the lookup sort the keys in prefix order and breaks at the
        #    first failure
        found_count = found_count // job.world_size

        log_telemetry(
            make_end_event(
                "lookup_and_prefetch",
                job.request_id,
                found_count=found_count,
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(prefetch_job_id, None)

        return found_count

    def free_lookup_locks(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Computes the extra reader count from ``tp_size`` and
        ``world_size`` the same way :meth:`lookup` does, so
        the correct number of locks is released.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Tensor-parallel size for MLA
                multi-reader locking.
        """
        ipc_keys: list[IPCCacheEngineKey] = []
        all_hash_keys = key.to_hash_keys(self.token_hasher)
        chunk_size = self.token_hasher.chunk_size
        start_chunk = key.start // chunk_size
        end_chunk = key.end // chunk_size
        ipc_keys.extend(all_hash_keys[start_chunk:end_chunk])
        if not ipc_keys:
            return
        obj_keys = ipc_keys_to_object_keys(ipc_keys)

        extra_count = compute_extra_count(tp_size, key.world_size)

        self.storage_manager.finish_read_prefetched(obj_keys, extra_count=extra_count)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def ping(self) -> bool:
        """
        Respond to a ping request.

        Returns:
            bool: Always True.
        """
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
        self.session_manager.remove(request_id)

    def report_status(self) -> dict:
        """Return a status dict for the entire cache engine."""
        sm = self.storage_manager.report_status()
        return {
            "is_healthy": sm["is_healthy"],
            "engine_type": "MPCacheEngine",
            "chunk_size": self.chunk_size,
            "hash_algorithm": self.token_hasher.hash_algorithm_name,
            "registered_gpu_ids": list(self.gpu_contexts.keys()),
            "gpu_context_meta": {
                str(gpu_id): {"model_name": meta[0], "world_size": meta[1]}
                for gpu_id, meta in self.gpu_context_meta.items()
            },
            "active_sessions": self.session_manager.active_count(),
            "storage_manager": sm,
        }

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
        # Close storage manager
        self.storage_manager.close()
        logger.info("MPCacheEngine closed")

        # Release GPU contexts
        self.gpu_contexts.clear()


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
    prometheus_config: PrometheusConfig,
    telemetry_config: TelemetryConfig = DEFAULT_TELEMETRY_CONFIG,
    return_engine: bool = False,
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        prometheus_config: Configuration for the Prometheus observability stack
        telemetry_config: Configuration for the telemetry event system
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheEngine)
        If return_engine is False: None (blocks until interrupted)
    """
    # Initialize global prometheus controller
    init_prometheus_controller(prometheus_config)

    # Initialize global telemetry controller
    init_telemetry_controller(telemetry_config)

    # Start Prometheus metrics HTTP server if enabled
    if prometheus_config.enabled:
        prometheus_client.start_http_server(prometheus_config.port)
        logger.info(
            "Prometheus metrics available at http://0.0.0.0:%d/metrics",
            prometheus_config.port,
        )

    # Initialize the engine (loggers self-register with the global controller)
    engine = MPCacheEngine(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
    )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=context,
        max_workers=mp_config.max_workers,
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
    add_handler_helper(server, RequestType.FREE_LOOKUP_LOCKS, engine.free_lookup_locks)
    add_handler_helper(server, RequestType.RETRIEVE, engine.retrieve)
    add_handler_helper(server, RequestType.CLEAR, engine.clear)
    add_handler_helper(server, RequestType.GET_CHUNK_SIZE, engine.get_chunk_size)
    add_handler_helper(server, RequestType.PING, engine.ping)
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )
    # Start the ZMQ server
    torch.cuda.init()
    server.start()

    # Start prometheus controller after engine creation (loggers are registered)
    get_prometheus_controller().start()

    # Start telemetry controller
    get_telemetry_controller().start()
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
        get_telemetry_controller().stop()
        get_prometheus_controller().stop()
        server.close()
        engine.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_prometheus_args(parser)
    add_telemetry_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    prometheus_config = parse_args_to_prometheus_config(args)
    telemetry_config = parse_args_to_telemetry_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        prometheus_config=prometheus_config,
        telemetry_config=telemetry_config,
    )
