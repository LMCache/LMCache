# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral multiprocess request client contract."""

# Standard
from typing import Any, Protocol

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture


class RequestClient(Protocol):
    """Define method-oriented multiprocess requests shared by transports."""

    def register_kv_cache(
        self,
        instance_id: int,
        kv_cache: Any,
        model_name: str,
        world_size: int,
        engine_type: Any,
        layout_hints: Any,
        engine_group_infos: list[Any],
    ) -> MessagingFuture[Any]: ...

    def unregister_kv_cache(self, instance_id: int) -> MessagingFuture[Any]: ...

    def register_q_cache(
        self,
        instance_id: int,
        q_cache: Any,
        model_name: str,
        world_size: int,
        engine_type: Any,
        layout_hints: Any,
        engine_group_infos: list[Any],
    ) -> MessagingFuture[Any]: ...

    def unregister_q_cache(self, instance_id: int) -> MessagingFuture[Any]: ...

    def store_q(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def store(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def retrieve(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
    ) -> MessagingFuture[Any]: ...

    def lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]: ...

    def query_prefetch_status(self, request_id: str) -> MessagingFuture[Any]: ...

    def wait_prefetch_status(
        self, request_id: str, timeout: float
    ) -> MessagingFuture[Any]: ...

    def query_prefetch_lookup_hits(self, request_id: str) -> MessagingFuture[Any]: ...

    def free_lookup_locks(self, key: Any, tp_size: int) -> MessagingFuture[Any]: ...

    def end_session(self, request_id: str) -> MessagingFuture[Any]: ...

    def register_kv_cache_engine_driven_context(
        self, payload: Any
    ) -> MessagingFuture[Any]: ...

    def unregister_kv_cache_engine_driven_context(
        self, instance_id: int
    ) -> MessagingFuture[Any]: ...

    def prepare_store(self, key: Any, instance_id: int) -> MessagingFuture[Any]: ...

    def commit_store(
        self, key: Any, instance_id: int, data: bytes
    ) -> MessagingFuture[Any]: ...

    def prepare_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]: ...

    def commit_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]: ...

    def clear(self) -> MessagingFuture[Any]: ...

    def get_chunk_size(self) -> MessagingFuture[Any]: ...

    def ping(self, instance_id: int | None) -> MessagingFuture[Any]: ...

    def report_block_allocation(
        self,
        instance_id: int,
        model_name: str,
        records: list[Any],
    ) -> MessagingFuture[Any]: ...

    def noop(self) -> MessagingFuture[Any]: ...

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[Any],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> MessagingFuture[Any]: ...

    def cb_unregister_rope(self, instance_id: int) -> MessagingFuture[Any]: ...

    def cb_retrieve_pre_computed(
        self,
        key: Any,
        match_results: list[Any],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def cb_unified_lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]: ...

    def p2p_lookup_and_lock(
        self, keys: list[Any], group_layout_descs: dict[int, Any]
    ) -> MessagingFuture[Any]: ...

    def p2p_query_lookup_results(self, task_id: int) -> MessagingFuture[Any]: ...

    def p2p_unlock_objects(self, keys: list[Any]) -> MessagingFuture[Any]: ...

    def get_experimental(self) -> MessagingFuture[Any]: ...

    def cb_register_rope_v3(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[Any],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> MessagingFuture[Any]: ...

    def cb_unregister_rope_v3(self, instance_id: int) -> MessagingFuture[Any]: ...

    def cb_retrieve_pre_computed_v3(
        self,
        key: Any,
        match_results: list[Any],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def close(self) -> None:
        """Close the client and release its transport resources."""
