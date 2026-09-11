# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for CacheBlend RPCs."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.rpc_messages.lmcache_driven import EventIpcHandleResult
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


@dataclass(frozen=True)
class CbProtocolHandshakeRequest:
    """Exchange CacheBlend protocol versions."""

    client_version: int


@dataclass(frozen=True)
class CbProtocolHandshakeResponse:
    """Return the server protocol version and compatibility result."""

    server_version: int
    client_compatible: bool


@dataclass(frozen=True)
class CbRegisterRopeRequest:
    """Register CacheBlend RoPE state."""

    instance_id: int
    cos_sin_caches_ipc: list[DeviceIPCWrapper]
    head_size: int
    is_neox_style: bool
    group_to_cache: list[int]
    group_rot: list[list[int]]


@dataclass(frozen=True)
class CbRegisterRopeResponse:
    """Acknowledge CacheBlend RoPE registration."""


@dataclass(frozen=True)
class CbUnregisterRopeRequest:
    """Unregister CacheBlend RoPE state."""

    instance_id: int


@dataclass(frozen=True)
class CbUnregisterRopeResponse:
    """Acknowledge CacheBlend RoPE removal."""


@dataclass(frozen=True)
class CbRetrievePreComputedRequest:
    """Retrieve CacheBlend pre-computed segments."""

    key: IPCCacheServerKey
    cb_match_result: list[CBMatchResult]
    gpu_block_ids: list[list[int]]
    instance_id: int
    event_ipc_handle: bytes


@dataclass(frozen=True)
class CbRetrievePreComputedResponse:
    """Return the completion event for a CacheBlend retrieve."""

    result: EventIpcHandleResult


@dataclass(frozen=True)
class CbUnifiedLookupRequest:
    """Run a unified prefix and CacheBlend lookup."""

    key: IPCCacheServerKey
    tp_size: int


@dataclass(frozen=True)
class CbUnifiedLookupResponse:
    """Return a completed unified lookup result, if available."""

    payload: CBUnifiedLookupResult | None


register_rpc_message_types(
    "cb_protocol_handshake",
    CbProtocolHandshakeRequest,
    CbProtocolHandshakeResponse,
)
register_rpc_message_types(
    "cb_register_rope", CbRegisterRopeRequest, CbRegisterRopeResponse
)
register_rpc_message_types(
    "cb_unregister_rope",
    CbUnregisterRopeRequest,
    CbUnregisterRopeResponse,
)
register_rpc_message_types(
    "cb_retrieve_pre_computed",
    CbRetrievePreComputedRequest,
    CbRetrievePreComputedResponse,
)
register_rpc_message_types(
    "cb_unified_lookup", CbUnifiedLookupRequest, CbUnifiedLookupResponse
)


__all__ = [
    "CbProtocolHandshakeRequest",
    "CbProtocolHandshakeResponse",
    "CbRegisterRopeRequest",
    "CbRegisterRopeResponse",
    "CbRetrievePreComputedRequest",
    "CbRetrievePreComputedResponse",
    "CbUnifiedLookupRequest",
    "CbUnifiedLookupResponse",
    "CbUnregisterRopeRequest",
    "CbUnregisterRopeResponse",
]
