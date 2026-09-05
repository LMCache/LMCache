# SPDX-License-Identifier: Apache-2.0
"""Method-oriented ZMQ client for the multiprocess server."""

# Standard
from typing import Any

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_response_class,
)
from lmcache.v1.multiprocess.transport.base import RequestClient


class ZmqMultiprocessClient(RequestClient):
    """Expose named multiprocess RPC methods over the existing ZMQ client.

    The wrapper changes only the Python call surface. Every method delegates to
    :class:`MessageQueueClient` with the existing ``RequestType`` and positional
    payload list, so the ZMQ wire protocol and server remain unchanged.

    Args:
        message_queue_client: Existing ZMQ message queue client to wrap.
    """

    def __init__(self, message_queue_client: MessageQueueClient) -> None:
        self._message_queue_client = message_queue_client

    def register_layerwise_ipc_event_pool(
        self, instance_id: int
    ) -> MessagingFuture[Any]:
        """Import the server's per-layer IPC event pool."""
        return self._call(RequestType.REGISTER_LAYERWISE_IPC_EVENT_POOL, instance_id)

    def retrieve_layerwise(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: Any,
        skip_first_n_tokens: int,
        future: Any,
    ) -> MessagingFuture[Any]:
        """Retrieve one chunk layer by layer.

        The server answers with one frame per layer batch. Unlike the other
        methods the future is supplied by the caller, because it carries the
        per-layer state and has to be bound to the pending-request table
        before the first frame can arrive.
        """
        return self._message_queue_client.submit_streaming_request(
            RequestType.RETRIEVE_LAYERWISE,
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
            future,
        )

    def register_kv_cache(
        self,
        instance_id: int,
        kv_cache: Any,
        model_name: str,
        world_size: int,
        engine_type: Any,
        layout_hints: Any,
        engine_group_infos: list[Any],
    ) -> MessagingFuture[Any]:
        """Register a worker KV cache with the multiprocess server."""
        return self._call(
            RequestType.REGISTER_KV_CACHE,
            instance_id,
            kv_cache,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        )

    def unregister_kv_cache(self, instance_id: int) -> MessagingFuture[Any]:
        """Unregister a worker KV cache."""
        return self._call(RequestType.UNREGISTER_KV_CACHE, instance_id)

    def register_q_cache(
        self,
        instance_id: int,
        q_cache: Any,
        model_name: str,
        world_size: int,
        engine_type: Any,
        layout_hints: Any,
        engine_group_infos: list[Any],
    ) -> MessagingFuture[Any]:
        """Register a worker Q cache with the multiprocess server."""
        return self._call(
            RequestType.REGISTER_Q_CACHE,
            instance_id,
            q_cache,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        )

    def unregister_q_cache(self, instance_id: int) -> MessagingFuture[Any]:
        """Unregister a worker Q cache."""
        return self._call(RequestType.UNREGISTER_Q_CACHE, instance_id)

    def store_q(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]:
        """Store Q-cache blocks."""
        return self._call(
            RequestType.STORE_Q, key, instance_id, block_ids, event_ipc_handle
        )

    def store(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]:
        """Store KV-cache blocks."""
        return self._call(
            RequestType.STORE, key, instance_id, block_ids, event_ipc_handle
        )

    def retrieve(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
    ) -> MessagingFuture[Any]:
        """Retrieve KV-cache blocks."""
        return self._call(
            RequestType.RETRIEVE,
            key,
            instance_id,
            block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        )

    def lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Start a prefix lookup."""
        return self._call(RequestType.LOOKUP, key, tp_size)

    def query_prefetch_status(self, request_id: str) -> MessagingFuture[Any]:
        """Query a prefetch task without blocking for completion."""
        return self._call(RequestType.QUERY_PREFETCH_STATUS, request_id)

    def wait_prefetch_status(
        self, request_id: str, timeout: float
    ) -> MessagingFuture[Any]:
        """Wait for a prefetch task to complete."""
        return self._call(RequestType.WAIT_PREFETCH_STATUS, request_id, timeout)

    def query_prefetch_lookup_hits(self, request_id: str) -> MessagingFuture[Any]:
        """Query lookup hits while prefetch is in progress."""
        return self._call(RequestType.QUERY_PREFETCH_LOOKUP_HITS, request_id)

    def free_lookup_locks(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Release read locks acquired by lookup."""
        return self._call(RequestType.FREE_LOOKUP_LOCKS, key, tp_size)

    def end_session(self, request_id: str) -> MessagingFuture[Any]:
        """End a request session."""
        return self._call(RequestType.END_SESSION, request_id)

    def register_kv_cache_engine_driven_context(
        self, payload: Any
    ) -> MessagingFuture[Any]:
        """Register an engine-driven transfer context."""
        return self._call(RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT, payload)

    def unregister_kv_cache_engine_driven_context(
        self, instance_id: int
    ) -> MessagingFuture[Any]:
        """Unregister an engine-driven transfer context."""
        return self._call(
            RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT, instance_id
        )

    def prepare_store(self, key: Any, instance_id: int) -> MessagingFuture[Any]:
        """Prepare an engine-driven store."""
        return self._call(RequestType.PREPARE_STORE, key, instance_id)

    def commit_store(
        self, key: Any, instance_id: int, data: bytes
    ) -> MessagingFuture[Any]:
        """Commit an engine-driven store."""
        return self._call(RequestType.COMMIT_STORE, key, instance_id, data)

    def prepare_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]:
        """Prepare an engine-driven retrieve."""
        return self._call(RequestType.PREPARE_RETRIEVE, key, instance_id)

    def commit_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]:
        """Commit an engine-driven retrieve."""
        return self._call(RequestType.COMMIT_RETRIEVE, key, instance_id)

    def clear(self) -> MessagingFuture[Any]:
        """Clear all server caches."""
        return self._call(RequestType.CLEAR)

    def get_chunk_size(self) -> MessagingFuture[Any]:
        """Return the server chunk size."""
        return self._call(RequestType.GET_CHUNK_SIZE)

    def ping(self, instance_id: int | None) -> MessagingFuture[Any]:
        """Check server health and refresh worker liveness."""
        return self._call(RequestType.PING, instance_id)

    def report_block_allocation(
        self,
        instance_id: int,
        model_name: str,
        records: list[Any],
    ) -> MessagingFuture[Any]:
        """Report block-allocation changes."""
        return self._call(
            RequestType.REPORT_BLOCK_ALLOCATION,
            instance_id,
            model_name,
            records,
        )

    def noop(self) -> MessagingFuture[Any]:
        """Send a no-op request."""
        return self._call(RequestType.NOOP)

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[Any],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> MessagingFuture[Any]:
        """Register CacheBlend RoPE state."""
        return self._call(
            RequestType.CB_REGISTER_ROPE,
            instance_id,
            cos_sin_caches_ipc,
            head_size,
            is_neox_style,
            group_to_cache,
            group_rot,
        )

    def cb_unregister_rope(self, instance_id: int) -> MessagingFuture[Any]:
        """Unregister CacheBlend RoPE state."""
        return self._call(RequestType.CB_UNREGISTER_ROPE, instance_id)

    def cb_retrieve_pre_computed(
        self,
        key: Any,
        match_results: list[Any],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]:
        """Retrieve CacheBlend pre-computed blocks."""
        return self._call(
            RequestType.CB_RETRIEVE_PRE_COMPUTED,
            key,
            match_results,
            block_ids,
            instance_id,
            event_ipc_handle,
        )

    def cb_unified_lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Run a CacheBlend unified lookup."""
        return self._call(RequestType.CB_UNIFIED_LOOKUP, key, tp_size)

    def p2p_lookup_and_lock(
        self,
        keys: list[Any],
        group_layout_descs: dict[int, Any],
    ) -> MessagingFuture[Any]:
        """Look up and lock peer-transfer objects."""
        return self._call(RequestType.P2P_LOOKUP_AND_LOCK, keys, group_layout_descs)

    def p2p_query_lookup_results(self, task_id: int) -> MessagingFuture[Any]:
        """Query peer-transfer lookup results."""
        return self._call(RequestType.P2P_QUERY_LOOKUP_RESULTS, task_id)

    def p2p_unlock_objects(self, keys: list[Any]) -> MessagingFuture[Any]:
        """Release peer-transfer object locks."""
        return self._call(RequestType.P2P_UNLOCK_OBJECTS, keys)

    def get_experimental(self) -> MessagingFuture[Any]:
        """Return the server's experimental capabilities."""
        return self._call(RequestType.GET_EXPERIMENTAL)

    # Compatibility aliases used by older CacheBlend plugins.
    cb_register_rope_v3 = cb_register_rope
    cb_unregister_rope_v3 = cb_unregister_rope
    cb_retrieve_pre_computed_v3 = cb_retrieve_pre_computed

    def close(self) -> None:
        """Close the wrapped ZMQ client."""
        self._message_queue_client.close()

    def _call(
        self, request_type: RequestType, *request_payloads: Any
    ) -> MessagingFuture[Any]:
        return self._message_queue_client.submit_request(
            request_type,
            list(request_payloads),
            get_response_class(request_type),
        )
