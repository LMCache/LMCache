# SPDX-License-Identifier: Apache-2.0
"""Method-oriented ZMQ client for the multiprocess server."""

# Standard
from typing import Any

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RpcOperation
from lmcache.v1.multiprocess.rpc_messages import make_request_message
from lmcache.v1.multiprocess.transport.base import RequestClient


class ZmqMultiprocessClient(RequestClient):
    """Expose named multiprocess RPC methods over the existing ZMQ client.

    Every named method builds one transport-neutral Python request message and
    delegates it to :class:`MessageQueueClient` for ZMQ serialization.

    Args:
        message_queue_client: Existing ZMQ message queue client to wrap.
    """

    def __init__(self, message_queue_client: MessageQueueClient) -> None:
        self._message_queue_client = message_queue_client

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
            "register_kv_cache",
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
        return self._call("unregister_kv_cache", instance_id)

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
            "register_q_cache",
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
        return self._call("unregister_q_cache", instance_id)

    def store_q(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]:
        """Store Q-cache blocks."""
        return self._call("store_q", key, instance_id, block_ids, event_ipc_handle)

    def store(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]:
        """Store KV-cache blocks."""
        return self._call("store", key, instance_id, block_ids, event_ipc_handle)

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
            "retrieve",
            key,
            instance_id,
            block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        )

    def lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Start a prefix lookup."""
        return self._call("lookup", key, tp_size)

    def query_prefetch_status(self, request_id: str) -> MessagingFuture[Any]:
        """Query a prefetch task without blocking for completion."""
        return self._call("query_prefetch_status", request_id)

    def wait_prefetch_status(
        self, request_id: str, timeout: float
    ) -> MessagingFuture[Any]:
        """Wait for a prefetch task to complete."""
        return self._call("wait_prefetch_status", request_id, timeout)

    def query_prefetch_lookup_hits(self, request_id: str) -> MessagingFuture[Any]:
        """Query lookup hits while prefetch is in progress."""
        return self._call("query_prefetch_lookup_hits", request_id)

    def free_lookup_locks(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Release read locks acquired by lookup."""
        return self._call("free_lookup_locks", key, tp_size)

    def end_session(self, request_id: str) -> MessagingFuture[Any]:
        """End a request session."""
        return self._call("end_session", request_id)

    def register_kv_cache_engine_driven_context(
        self, payload: Any
    ) -> MessagingFuture[Any]:
        """Register an engine-driven transfer context."""
        return self._call("register_kv_cache_engine_driven_context", payload)

    def unregister_kv_cache_engine_driven_context(
        self, instance_id: int
    ) -> MessagingFuture[Any]:
        """Unregister an engine-driven transfer context."""
        return self._call("unregister_kv_cache_engine_driven_context", instance_id)

    def prepare_store(self, key: Any, instance_id: int) -> MessagingFuture[Any]:
        """Prepare an engine-driven store."""
        return self._call("prepare_store", key, instance_id)

    def commit_store(
        self, key: Any, instance_id: int, data: bytes
    ) -> MessagingFuture[Any]:
        """Commit an engine-driven store."""
        return self._call("commit_store", key, instance_id, data)

    def prepare_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]:
        """Prepare an engine-driven retrieve."""
        return self._call("prepare_retrieve", key, instance_id)

    def commit_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]:
        """Commit an engine-driven retrieve."""
        return self._call("commit_retrieve", key, instance_id)

    def clear(self) -> MessagingFuture[Any]:
        """Clear all server caches."""
        return self._call("clear")

    def get_chunk_size(self) -> MessagingFuture[Any]:
        """Return the server chunk size."""
        return self._call("get_chunk_size")

    def ping(self, instance_id: int | None) -> MessagingFuture[Any]:
        """Check server health and refresh worker liveness."""
        return self._call("ping", instance_id)

    def report_block_allocation(
        self,
        instance_id: int,
        model_name: str,
        records: list[Any],
    ) -> MessagingFuture[Any]:
        """Report block-allocation changes."""
        return self._call(
            "report_block_allocation",
            instance_id,
            model_name,
            records,
        )

    def noop(self) -> MessagingFuture[Any]:
        """Send a no-op request."""
        return self._call("noop")

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[Any],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> MessagingFuture[Any]:
        """Register blend RoPE state."""
        return self._call(
            "cb_register_rope",
            instance_id,
            cos_sin_caches_ipc,
            head_size,
            is_neox_style,
            group_to_cache,
            group_rot,
        )

    def cb_unregister_rope(self, instance_id: int) -> MessagingFuture[Any]:
        """Unregister blend RoPE state."""
        return self._call("cb_unregister_rope", instance_id)

    def cb_retrieve_pre_computed(
        self,
        key: Any,
        match_results: list[Any],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]:
        """Retrieve blend pre-computed blocks."""
        return self._call(
            "cb_retrieve_pre_computed",
            key,
            match_results,
            block_ids,
            instance_id,
            event_ipc_handle,
        )

    def cb_unified_lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Run a blend unified lookup."""
        return self._call("cb_unified_lookup", key, tp_size)

    def cb_protocol_handshake(self, client_version: int) -> MessagingFuture[Any]:
        """Exchange blend protocol versions with the server."""
        return self._call("cb_protocol_handshake", client_version)

    def p2p_lookup_and_lock(
        self,
        keys: list[Any],
        group_layout_descs: dict[int, Any],
    ) -> MessagingFuture[Any]:
        """Look up and lock peer-transfer objects."""
        return self._call("p2p_lookup_and_lock", keys, group_layout_descs)

    def p2p_query_lookup_results(self, task_id: int) -> MessagingFuture[Any]:
        """Query peer-transfer lookup results."""
        return self._call("p2p_query_lookup_results", task_id)

    def p2p_unlock_objects(self, keys: list[Any]) -> MessagingFuture[Any]:
        """Release peer-transfer object locks."""
        return self._call("p2p_unlock_objects", keys)

    def get_experimental(self) -> MessagingFuture[Any]:
        """Return the server's experimental capabilities."""
        return self._call("get_experimental")

    # Compatibility aliases used by older blend plugins.
    cb_register_rope_v3 = cb_register_rope
    cb_unregister_rope_v3 = cb_unregister_rope
    cb_retrieve_pre_computed_v3 = cb_retrieve_pre_computed

    def close(self) -> None:
        """Close the wrapped ZMQ client."""
        self._message_queue_client.close()

    def _call(
        self, operation: RpcOperation, *request_payloads: Any
    ) -> MessagingFuture[Any]:
        return self._message_queue_client.submit_request(
            operation,
            make_request_message(operation, *request_payloads),
        )
