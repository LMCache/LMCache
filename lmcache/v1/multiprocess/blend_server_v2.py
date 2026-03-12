# SPDX-License-Identifier: Apache-2.0
"""
Overview
--------
This server enables KV cache reuse across requests that share token
sub-sequences at *arbitrary positions*, not only at a common prefix.

Workflow (example: chunk_size = 3)
-----------------------------------
1. cb_store_pre_computed([1,2,3,4,5,6])
   Tokens are split into full chunks ([1,2,3] and [4,5,6]).  Each chunk
   is stored in the underlying storage under its normal rolling prefix
   hash, and the chunk fingerprints are registered in
   BlendTokenRangeMatcher for fast sub-sequence lookup.  Because normal
   hashes are used, these chunks are also accessible via the standard
   lookup/retrieve path.

2. cb_lookup_pre_computed([x,y,z, a,b,c, 4,5,6, m,n,p])
   BlendTokenRangeMatcher slides a rolling polynomial hash over the new
   request's tokens and detects that the window at positions [6, 9)
   matches the stored chunk [4,5,6].  A prefetch task is submitted for
   that chunk using its stored hash as the storage key.  Only chunks
   confirmed present in storage are returned as CBMatchResult objects
   (with cur_st/cur_ed pointing to their location in the new request).

3. cb_retrieve_pre_computed(...)
   The (prefetched) KV cache for each matched chunk is copied (CPU→GPU)
   into the correct slot of the new request's KV cache buffer (at
   cur_st + offset), so the LLM can skip recomputing those tokens.

4. cb_store_final([x,y,z, a,b,c, 4,5,6, m,n,p])
   After inference completes on the new request, all its chunks are
   stored under normal prefix hashes.  Future requests sharing
   any prefix of the new request will get standard prefix-cache hits.
   Future requests sharing any prefix of the first request will also
   get hits because cb_store_pre_computed already stored those chunks
   under normal hashes.
"""

# Standard
import time

# Third Party
import numpy as np
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.mp_observability.config import (
    PrometheusConfig,
    parse_args_to_prometheus_config,
)
from lmcache.v1.mp_observability.prometheus_controller import (
    get_prometheus_controller,
    init_prometheus_controller,
)
from lmcache.v1.mp_observability.telemetry import (
    TelemetryConfig,
    get_telemetry_controller,
    init_telemetry_controller,
    parse_args_to_telemetry_config,
)
from lmcache.v1.mp_observability.telemetry.config import (
    DEFAULT_TELEMETRY_CONFIG,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.gpu_context import (
    PlainGPUCacheContext,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.server import MPCacheEngine, parse_args
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    unique_hits_direct_id_numba,
    update_table_id_numba,
)

logger = init_logger(__name__)


class BlendTokenRangeMatcher:
    # TODO(Jiayi): Needs thread-safety for this class.
    # TODO(Jiayi): Currently, the table size is fixed. We need to support
    # dynamic expanding or eviction.
    """Fast token-range matcher using polynomial rolling/chunk hashes and a
    direct-address lookup table.

    Table layout: poly_chunk_hash (u64) → compact_chunk_id (i64, sequential 0…N-1).

    Because compact IDs are bounded by _TABLE_SIZE, unique_hits_direct_id_numba
    can use a fixed `seen` array of _TABLE_SIZE bytes (~1 MB) rather than one
    sized by an arbitrary max hash — no memory explosion.

    Auxiliary storage:
      _chunk_token_hash[i] : caller-supplied token_hash for chunk i
      _token_hash_to_start : token_hash → start position in the registered seq

    on_new_token_hashes  – register a sequence; chunk_hash_windows_numba(token_hashes)
                        builds fingerprints, update_table_id_numba writes compact IDs.
    match_sub_sequence – rolling_hash_windows_numba + unique_hits_direct_id_numba
                         (num_ids=_TABLE_SIZE) → compact IDs → token_hash → start.
    """

    _TABLE_BITS: int = 20  # 2^20 ≈ 1 M entries
    _TABLE_SIZE: int = 1 << _TABLE_BITS
    _BASE: np.uint64 = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci-hashing constant

    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
        # poly_chunk_hash → compact_chunk_id; -1 = empty
        self._table_id = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        self._mask = np.uint64(self._TABLE_SIZE - 1)
        # compact_chunk_id → caller-supplied token_hash (full bytes)
        self._chunk_token_hash: list[bytes] = []
        # token_hash → start position in its registered sequence
        self._token_hash_to_start: dict[bytes, int] = {}

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
    ):
        """Register a new token sequence and index its non-overlapping chunks.

        Args:
            token_ids: Raw token IDs for the full sequence (num_tokens elements).
                       Used to compute polynomial chunk fingerprints that match
                       the rolling hashes computed in match_sub_sequence.
            token_hashes: Per-chunk bytes hashes supplied by the caller
                          (one per complete chunk of chunk_size tokens).
                          Stored as the storage key returned in CBMatchResult.hash.
        """
        arr = np.array(token_ids, dtype=np.uint64)
        # Polynomial fingerprints for non-overlapping chunks, built from raw token IDs
        # so they match the sliding-window rolling hashes in match_sub_sequence
        chunk_hashes = chunk_hash_windows_numba(arr, self.chunk_size, self._BASE)
        n = int(chunk_hashes.shape[0])
        if n == 0:
            return

        # Compact sequential IDs: bounded by _TABLE_SIZE, safe for seen-array sizing
        base_id = len(self._chunk_token_hash)
        compact_ids = np.arange(base_id, base_id + n, dtype=np.int64)

        # Write table: poly_chunk_hash → compact_chunk_id
        update_table_id_numba(chunk_hashes, self._table_id, compact_ids)

        # Persist compact_id → token_hash and token_hash → start
        for i in range(n):
            th = token_hashes[i]
            self._chunk_token_hash.append(th)
            self._token_hash_to_start[th] = i * self.chunk_size

    def match_sub_sequence(
        self,
        token_ids: list[int],
    ) -> list[CBMatchResult]:
        """Find stored chunks whose fingerprints appear anywhere in token_ids.

        Uses a sliding-window rolling hash so matches need not be aligned to
        chunk_size boundaries in the query.

        Args:
            token_ids: Query token sequence to probe (raw token IDs as uint64).

        Returns:
            One CBMatchResult per unique stored chunk that was hit.
              old_st/old_ed : positions in the originally registered sequence
              cur_st/cur_ed : positions in the query (token_ids) where
                              the match was found
              hash          : token_hash bytes (from registration) for cache key lookup
        """
        if not self._chunk_token_hash or len(token_ids) < self.chunk_size:
            return []

        arr = np.array(token_ids, dtype=np.uint64)

        # Sliding-window polynomial hashes over the query
        rolling = rolling_hash_windows_numba(arr, self.chunk_size, self._BASE)

        # Probe table; seen array is _TABLE_SIZE bytes (~1 MB), fixed and safe
        hit_ids = unique_hits_direct_id_numba(
            rolling, self._table_id, self._mask, self._TABLE_SIZE
        )

        if hit_ids.shape[0] == 0:
            return []

        # For each hit compact_id, find the first query position where it matched
        hit_id_set = set(int(cid) for cid in hit_ids)
        cid_to_query_pos: dict[int, int] = {}
        for q_pos in range(rolling.shape[0]):
            idx = int(rolling[q_pos]) & int(self._mask)
            cid = int(self._table_id[idx])
            if cid in hit_id_set and cid not in cid_to_query_pos:
                cid_to_query_pos[cid] = q_pos
                if len(cid_to_query_pos) == len(hit_id_set):
                    break

        results: list[CBMatchResult] = []
        for cid in hit_ids:
            cid_int = int(cid)
            th = self._chunk_token_hash[cid_int]
            old_st = self._token_hash_to_start.get(th)
            cur_st = cid_to_query_pos.get(cid_int)
            if old_st is None or cur_st is None:
                continue
            results.append(
                CBMatchResult(
                    old_st=old_st,
                    old_ed=old_st + self.chunk_size,
                    cur_st=cur_st,
                    cur_ed=cur_st + self.chunk_size,
                    hash=th,
                )
            )
        return results


# Main class and main functions
class BlendEngineV2(MPCacheEngine):
    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ):
        super().__init__(
            storage_manager_config, chunk_size, hash_algorithm=hash_algorithm
        )

        self._cb_gpu_contexts: dict[int, PlainGPUCacheContext] = {}

        # CB GPU ID -> (model name, world size) as metadata
        # NOTE: This is mainly for determining the layout desc during prefetch
        self._cb_gpu_context_meta: dict[int, tuple[str, int]] = {}

        # Fast local matcher: indexes pre-computed chunk hashes for sub-sequence lookup
        self._token_range_matcher = BlendTokenRangeMatcher(chunk_size)

    def cb_register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
    ) -> None:
        """
        Register the KV cache buffer from the blend engine

        Args:
            instance_id: Unique identifier for the blend engine instance
            kv_caches: KVCache object containing the GPU buffer pointers
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
        """
        gpu_context = PlainGPUCacheContext(kv_caches, self.chunk_size)
        self._cb_gpu_contexts[instance_id] = gpu_context
        self._cb_gpu_context_meta[instance_id] = (model_name, world_size)
        logger.info(
            "Registered CB KV cache for instance_id %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def cb_unregister_kv_cache(self, instance_id: int) -> None:
        """
        Unregister the KV cache buffer for the given instance_id

        Args:
            instance_id: Unique identifier for the blend engine instance to unregister
        """
        if instance_id in self._cb_gpu_contexts:
            del self._cb_gpu_contexts[instance_id]
            del self._cb_gpu_context_meta[instance_id]
            logger.info("Unregistered CB KV cache for instance_id %d", instance_id)
        else:
            logger.warning(
                "Attempted to unregister non-existent CB KV cache for instance_id %d",
                instance_id,
            )

    def cb_lookup_pre_computed(self, key: IPCCacheEngineKey) -> list[CBMatchResult]:
        """
        Lookup the pre-computed chunks in the underlying storage.

        Uses BlendTokenRangeMatcher for a fast local pre-filter, then submits
        prefetch tasks for matched chunks using their stored hashes directly.

        Args:
            key: IPCCacheEngineKey containing the token ids to lookup

        Returns:
            List of CBMatchResult for chunks that were actually found in storage,
            ready to be passed to cb_retrieve_pre_computed.
        """
        # Fast local pre-filter: find which stored chunks appear in this query
        cb_match_result = self._token_range_matcher.match_sub_sequence(
            list(key.token_ids)
        )
        if not cb_match_result:
            return []

        # Sort by query position and group consecutive matched chunks
        cb_match_result.sort(key=lambda r: r.cur_st)
        groups: list[list[CBMatchResult]] = []
        for result in cb_match_result:
            if groups and groups[-1][-1].cur_ed == result.cur_st:
                groups[-1].append(result)
            else:
                groups.append([result])

        prefetch_handles: list[PrefetchHandle] = []
        found_cb_match_result: list[CBMatchResult] = []
        model_name, world_size = key.model_name, key.world_size

        # Find the cb gpu context and calculate the layout desc
        layout_desc: MemoryLayoutDesc | None = None
        for gpu_id, (m_name, w_size) in self._cb_gpu_context_meta.items():
            if m_name == model_name and w_size == world_size:
                cb_ctx = self._cb_gpu_contexts[gpu_id]
                layout_desc = MemoryLayoutDesc(
                    shapes=[cb_ctx.get_kv_buffer_shape(self.chunk_size)],
                    dtypes=[cb_ctx.dtype],
                )
                break

        if layout_desc is None:
            logger.error(
                "No CB GPU context found for model %s with world size %d "
                "during cb_lookup_pre_computed!",
                model_name,
                world_size,
            )
            return []

        # Submit prefetch for each group using CBMatchResult.hash directly
        for group in groups:
            chunk_hashes = [r.hash for r in group]
            obj_keys = ipc_key_to_object_keys(key, chunk_hashes)
            handle = self.storage_manager.submit_prefetch_task(obj_keys, layout_desc)
            prefetch_handles.append(handle)

            logger.debug(
                "DEBUG: Submitted prefetch for %d chunks starting at %d",
                len(group),
                group[0].cur_st,
            )

        # TODO(Jiayi): We need to follow how lookup is handled in server.py
        # to optimize performance.
        # Collect only the CBMatchResults for chunks actually found in storage
        for handle, group in zip(prefetch_handles, groups, strict=False):
            found_count = None
            while True:
                found_count = self.storage_manager.query_prefetch_status(handle)
                if found_count is not None:
                    break

                # Standard
                import time

                time.sleep(0.001)

            # Real found count after dedup the TP
            found_count = found_count // world_size

            start = group[0].cur_st
            end = group[-1].cur_ed
            if found_count > 0:
                found_cb_match_result.extend(group[:found_count])
                logger.debug(
                    "Found %d pre-computed chunks for range (%d, %d)",
                    found_count,
                    start,
                    end,
                )
            else:
                logger.debug(
                    "No pre-computed chunks found for range (%d, %d)",
                    start,
                    end,
                )

        return found_cb_match_result

    def _cb_store_gpu_copy(
        self,
        obj_keys: list[ObjectKey],
        gpu_context: PlainGPUCacheContext,
        offset: int,
        event_ipc_handle: bytes,
    ) -> tuple[torch.cuda.Event, dict]:
        """
        Helper function to perform GPU-to-CPU copy operations for storing chunks.

        Args:
            obj_keys: List of object keys to store.
            gpu_context: GPU context for the blend engine instance.
            offset: The starting offset in the CB KV cache buffer.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            A tuple of (event, reserved_dict) where event is the CUDA event and
            reserved_dict is the dictionary of reserved memory objects.
        """
        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)

            # Wait for vLLM event to finish
            vllm_event = torch.cuda.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            # Prepare for the copy
            num_tokens = self.chunk_size
            cpu_shape = gpu_context.get_kv_buffer_shape(num_tokens)
            layout_desc = MemoryLayoutDesc(
                shapes=[cpu_shape], dtypes=[gpu_context.dtype]
            )

            reserved_dict = self.storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )

            for idx, obj_key in enumerate(obj_keys):
                if obj_key in reserved_dict:
                    memory_obj = reserved_dict[obj_key]
                else:
                    continue

                offset_start = idx * self.chunk_size + offset
                offset_end = offset_start + self.chunk_size

                # Copy from GPU to CPU
                tmp_buffer = gpu_context.get_tmp_gpu_buffer(offset_end - offset_start)
                gpu_kv_slice = gpu_context.slice_kv_cache_on_tokens(
                    offset_start, offset_end
                )
                with self.lock:
                    tmp_buffer.copy_(gpu_kv_slice, non_blocking=True)
                    lmcache_memcpy_async_d2h(tmp_buffer, memory_obj)

            event.record()

        # Call finish_write after the copy is done
        gpu_context.cupy_stream.launch_host_func(
            self.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )

        return event, reserved_dict

    def cb_store_pre_computed(
        self,
        key: IPCCacheEngineKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Store the pre-computed chunks in the underlying storage for later retrieval.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the pre-computed
                chunks are stored.
            offset: The starting offset in the CB KV cache buffer where the
                pre-computed
            instance_id: The instance_id of the blend engine instance to store the
                pre-computed chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of storing the
            pre-computed chunks, and a boolean flag indicating if the store is
            successful.

        Note:
            The input tokens should not have any separator in it. It should just be
            one "paragraph".
            This function will discard the last partial chunk and only store the full
            chunks
        """
        # Compute normal prefix hashes so these chunks are accessible both via
        # the CB lookup path and via the standard lookup/retrieve path.
        chunk_hashes = self.token_hasher.compute_chunk_hashes(list(key.token_ids))
        # convert to object key
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        assert instance_id in self._cb_gpu_contexts, (
            f"Instance ID {instance_id} not registered for CB KV cache"
        )
        gpu_context = self._cb_gpu_contexts[instance_id]

        event, reserved_dict = self._cb_store_gpu_copy(
            obj_keys, gpu_context, offset, event_ipc_handle
        )

        # Register chunk hashes with the local matcher for fast sub-sequence lookup
        token_hashes = list(chunk_hashes)

        # NOTE(Jiayi): We only register the token hashes for worker_id 0 or None to
        # avoid duplicate registration across workers.
        if key.worker_id in [0, None]:
            self._token_range_matcher.on_new_token_hashes(
                list(key.token_ids), token_hashes
            )

        logger.info(
            "Stored pre-computed doc with %d tokens, num stored chunks: %d",
            key.end - key.start,
            len(reserved_dict),
        )
        return event.ipc_handle(), True

    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheEngineKey,
        cb_match_result: list[CBMatchResult],
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Retrieve the pre-computed chunks from the underlying storage and copy them to
        the CB KV cache buffer.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the pre-computed
                chunks are retrieved.
            cb_match_result: List of CBMatchResult returned by cb_lookup_pre_computed,
                containing the per-chunk hashes and query positions.
            offset: The starting offset in the CB KV cache buffer to copy the retrieved
                chunks to.
            instance_id: The instance_id of the blend engine instance to retrieve the
                pre-computed chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of retrieving the
            pre-computed chunks, and a boolean flag indicating if the retrieval is
            successful.

        Note:
            We must call `cb_lookup_pre_computed` first before calling this function
        """
        assert instance_id in self._cb_gpu_contexts, (
            f"Instance ID {instance_id} not registered for CB KV cache"
        )
        gpu_context = self._cb_gpu_contexts[instance_id]

        # One obj_key per match_result, in cur_st order
        cb_match_result = sorted(cb_match_result, key=lambda r: r.cur_st)
        chunk_hashes = [r.hash for r in cb_match_result]
        all_obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        logger.debug("DEBUG object keys to retrieve: %s", all_obj_keys)

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)

            try:
                with self.storage_manager.read_prefetched_results(
                    all_obj_keys
                ) as memory_objs:
                    if memory_objs is None:
                        logger.error("Some keys not found during CB retrieve!")
                        return event.ipc_handle(), False

                    for r, memory_obj in zip(
                        cb_match_result, memory_objs, strict=False
                    ):
                        gpu_st = r.cur_st + offset
                        gpu_ed = gpu_st + self.chunk_size
                        tmp_buffer = gpu_context.get_tmp_gpu_buffer(self.chunk_size)
                        target_buffer = gpu_context.slice_kv_cache_on_tokens(
                            gpu_st, gpu_ed
                        )
                        with self.lock:
                            lmcache_memcpy_async_h2d(memory_obj, tmp_buffer)
                            target_buffer.copy_(tmp_buffer, non_blocking=True)

            except Exception as e:
                logger.error("Error during retrieving prefetched results: %s", e)
                return event.ipc_handle(), False

            finally:
                event.record()
                # TODO: here we simply "unlock" all the keys, which may cause
                # double-unlock if error happens during read_prefetched_results.
                # We should consider not unlocking objects in read_prefetched_results
                # if error happens.
                gpu_context.cupy_stream.launch_host_func(
                    self.storage_manager.finish_read_prefetched, all_obj_keys
                )

        logger.info(
            "Retrieved pre-computed for %d match results to GPU offset starting at %d",
            len(cb_match_result),
            offset,
        )
        return event.ipc_handle(), True

    def cb_store_final(
        self,
        key: IPCCacheEngineKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Store the final chunks in the underlying storage after processing. The stored
        chunk should be accessible for normal mode LLMs.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the final chunks
                are stored.
            offset: The starting offset in the CB KV cache buffer where the final
                chunks are stored.
            instance_id: The instance_id of the blend engine instance to store the final
                chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of storing the
            final chunks, and a boolean flag indicating if the store is successful.
        """
        # Compute normal hash for the keys
        chunk_hashes = self.token_hasher.compute_chunk_hashes(list(key.token_ids))

        # convert to object key
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        # Get GPU context
        assert instance_id in self._cb_gpu_contexts, (
            f"Instance ID {instance_id} not registered for CB KV cache"
        )
        gpu_context = self._cb_gpu_contexts[instance_id]

        event, reserved_dict = self._cb_store_gpu_copy(
            obj_keys, gpu_context, offset, event_ipc_handle
        )

        logger.info(
            "Stored final doc with %d tokens, num stored chunks: %d",
            key.end - key.start,
            len(reserved_dict),
        )
        return event.ipc_handle(), True


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
        If return_engine is True: tuple of (MessageQueueServer, BlendEngineV2)
        If return_engine is False: None (blocks until interrupted)
    """
    # Initialize global prometheus controller
    init_prometheus_controller(prometheus_config)

    # Initialize global telemetry controller
    init_telemetry_controller(telemetry_config)

    # Initialize the engine (loggers self-register with the global controller)
    engine = BlendEngineV2(
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

    # Add handlers for original server
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
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)

    # Add handler for blend operations
    add_handler_helper(
        server, RequestType.CB_REGISTER_KV_CACHE, engine.cb_register_kv_cache
    )
    add_handler_helper(
        server, RequestType.CB_UNREGISTER_KV_CACHE, engine.cb_unregister_kv_cache
    )
    add_handler_helper(
        server, RequestType.CB_LOOKUP_PRE_COMPUTED_V2, engine.cb_lookup_pre_computed
    )
    add_handler_helper(
        server, RequestType.CB_STORE_PRE_COMPUTED, engine.cb_store_pre_computed
    )
    add_handler_helper(
        server, RequestType.CB_RETRIEVE_PRE_COMPUTED_V2, engine.cb_retrieve_pre_computed
    )
    add_handler_helper(server, RequestType.CB_STORE_FINAL, engine.cb_store_final)

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
    logger.info("LMCache cache blend v2 server is running...")

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
