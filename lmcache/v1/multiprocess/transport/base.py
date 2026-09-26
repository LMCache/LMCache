# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral multiprocess request contracts."""

# Standard
from typing import Protocol

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
    KVEventBatch,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.futures import MessagingFuture, MessagingStream
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.rpc import rpc_method


class RequestServer(Protocol):
    """Base interface for multiprocess request servers."""

    def start(self) -> None:
        """Start accepting request transport traffic."""
        ...

    def close(self) -> None:
        """Close the server and release its transport resources."""
        ...


class RequestClient(Protocol):
    """Typed, transport-neutral contract for multiprocess RPC clients."""

    @rpc_method
    def register_kv_cache(
        self,
        instance_id: int,
        kv_cache: list[DeviceIPCWrapper],
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> MessagingFuture[None]: ...

    @rpc_method
    def unregister_kv_cache(self, instance_id: int) -> MessagingFuture[None]: ...

    @rpc_method
    def register_q_cache(
        self,
        instance_id: int,
        q_cache: list[DeviceIPCWrapper],
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> MessagingFuture[None]: ...

    @rpc_method
    def unregister_q_cache(self, instance_id: int) -> MessagingFuture[None]: ...

    @rpc_method
    def store_q(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[tuple[bytes, bool]]: ...

    @rpc_method
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[tuple[bytes, bool]]: ...

    @rpc_method
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
    ) -> MessagingFuture[tuple[bytes, bool]]: ...

    @rpc_method
    def lookup(self, key: IPCCacheServerKey, tp_size: int) -> MessagingFuture[None]: ...

    @rpc_method
    def query_prefetch_status(self, request_id: str) -> MessagingFuture[int | None]: ...

    @rpc_method
    def wait_prefetch_status(
        self, request_id: str, timeout: float
    ) -> MessagingFuture[int | None]: ...

    @rpc_method
    def query_prefetch_lookup_hits(
        self, request_id: str
    ) -> MessagingFuture[int | None]: ...

    @rpc_method
    def free_lookup_locks(
        self, key: IPCCacheServerKey, tp_size: int
    ) -> MessagingFuture[None]: ...

    @rpc_method
    def end_session(self, request_id: str) -> MessagingFuture[None]: ...

    @rpc_method
    def register_kv_cache_engine_driven_context(
        self, payload: RegisterEngineDrivenContextPayload
    ) -> MessagingFuture[RegisterEngineDrivenContextResponse]: ...

    @rpc_method
    def unregister_kv_cache_engine_driven_context(
        self, instance_id: int
    ) -> MessagingFuture[None]: ...

    @rpc_method
    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> MessagingFuture[PrepareStoreResponse]: ...

    @rpc_method
    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, data: bytes
    ) -> MessagingFuture[bool]: ...

    @rpc_method
    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> MessagingFuture[PrepareRetrieveResponse]: ...

    @rpc_method
    def commit_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> MessagingFuture[bool]: ...

    @rpc_method
    def clear(self, force: bool = False) -> MessagingFuture[None]: ...

    @rpc_method
    def get_chunk_size(self) -> MessagingFuture[int]: ...

    @rpc_method
    def ping(self, instance_id: int | None) -> MessagingFuture[bool]: ...

    @rpc_method
    def report_block_allocation(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> MessagingFuture[None]: ...

    @rpc_method
    def noop(self) -> MessagingFuture[str]: ...

    @rpc_method
    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
        group_head_size: list[int],
    ) -> MessagingFuture[None]: ...

    @rpc_method
    def cb_unregister_rope(self, instance_id: int) -> MessagingFuture[None]: ...

    @rpc_method
    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheServerKey,
        match_results: list[CBMatchResult],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[tuple[bytes, bool]]: ...

    @rpc_method
    def cb_unified_lookup(
        self, key: IPCCacheServerKey, tp_size: int
    ) -> MessagingFuture[CBUnifiedLookupResult | None]: ...

    @rpc_method
    def cb_protocol_handshake(
        self, client_version: int
    ) -> MessagingFuture[tuple[int, bool]]: ...

    @rpc_method
    def p2p_lookup_and_lock(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> MessagingFuture[int]: ...

    @rpc_method
    def p2p_query_lookup_results(
        self, task_id: int
    ) -> MessagingFuture[list[TransferChannelAddress] | None]: ...

    @rpc_method
    def p2p_unlock_objects(self, keys: list[ObjectKey]) -> MessagingFuture[None]: ...

    @rpc_method
    def get_experimental(self) -> MessagingFuture[list[str]]: ...

    @rpc_method
    def subscribe_kv_events(
        self, instance_id: int, model_name: str, cursor: int, max_events: int
    ) -> MessagingStream[KVEventBatch]: ...

    def cb_register_rope_v3(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
        group_head_size: list[int],
    ) -> MessagingFuture[None]: ...

    def cb_unregister_rope_v3(self, instance_id: int) -> MessagingFuture[None]: ...

    def cb_retrieve_pre_computed_v3(
        self,
        key: IPCCacheServerKey,
        match_results: list[CBMatchResult],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[tuple[bytes, bool]]: ...

    def close(self) -> None:
        """Close the client and release its transport resources."""
        ...
