# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for engine-driven transfer RPCs."""

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types


@dataclass(frozen=True)
class RegisterKvCacheEngineDrivenContextRequest:
    """Register an engine-driven transfer context."""

    instance_id: int
    model_name: str
    world_size: int
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool
    num_physical_slots: int | None = None


@dataclass(frozen=True)
class RegisterKvCacheEngineDrivenContextResponse:
    """Return the shared-memory allocation for an engine-driven context."""

    shm_name: str = ""
    pool_size: int = 0


@dataclass(frozen=True)
class UnregisterKvCacheEngineDrivenContextRequest:
    """Unregister an engine-driven transfer context."""

    instance_id: int


@dataclass(frozen=True)
class UnregisterKvCacheEngineDrivenContextResponse:
    """Acknowledge engine-driven context removal."""


@dataclass(frozen=True)
class PrepareStoreRequest:
    """Prepare an engine-driven store operation."""

    key: IPCCacheServerKey
    instance_id: int


@dataclass(frozen=True)
class PrepareStoreResponse:
    """Return transport-independent store preparation context."""

    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CommitStoreRequest:
    """Commit an engine-driven store operation."""

    key: IPCCacheServerKey
    instance_id: int
    data: bytes


@dataclass(frozen=True)
class CommitStoreResponse:
    """Report whether an engine-driven store committed successfully."""

    success: bool


@dataclass(frozen=True)
class PrepareRetrieveRequest:
    """Prepare an engine-driven retrieve operation."""

    key: IPCCacheServerKey
    instance_id: int


@dataclass(frozen=True)
class PrepareRetrieveResponse:
    """Return transport-independent retrieve preparation state."""

    success: bool
    data: bytes = b""
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CommitRetrieveRequest:
    """Commit an engine-driven retrieve operation."""

    key: IPCCacheServerKey
    instance_id: int


@dataclass(frozen=True)
class CommitRetrieveResponse:
    """Report whether an engine-driven retrieve committed successfully."""

    success: bool


register_rpc_message_types(
    "register_kv_cache_engine_driven_context",
    RegisterKvCacheEngineDrivenContextRequest,
    RegisterKvCacheEngineDrivenContextResponse,
)
register_rpc_message_types(
    "unregister_kv_cache_engine_driven_context",
    UnregisterKvCacheEngineDrivenContextRequest,
    UnregisterKvCacheEngineDrivenContextResponse,
)
register_rpc_message_types("prepare_store", PrepareStoreRequest, PrepareStoreResponse)
register_rpc_message_types("commit_store", CommitStoreRequest, CommitStoreResponse)
register_rpc_message_types(
    "prepare_retrieve", PrepareRetrieveRequest, PrepareRetrieveResponse
)
register_rpc_message_types(
    "commit_retrieve", CommitRetrieveRequest, CommitRetrieveResponse
)


__all__ = [
    "CommitRetrieveRequest",
    "CommitRetrieveResponse",
    "CommitStoreRequest",
    "CommitStoreResponse",
    "PrepareRetrieveRequest",
    "PrepareRetrieveResponse",
    "PrepareStoreRequest",
    "PrepareStoreResponse",
    "RegisterKvCacheEngineDrivenContextRequest",
    "RegisterKvCacheEngineDrivenContextResponse",
    "UnregisterKvCacheEngineDrivenContextRequest",
    "UnregisterKvCacheEngineDrivenContextResponse",
]
