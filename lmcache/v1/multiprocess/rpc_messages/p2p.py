# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for P2P lookup and lock RPCs."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types


@dataclass(frozen=True)
class P2pLookupAndLockRequest:
    """Look up and read-lock local P2P objects."""

    keys: list[ObjectKey]
    group_layout_descs: dict[int, MemoryLayoutDesc]


@dataclass(frozen=True)
class P2pLookupAndLockResponse:
    """Return the asynchronous P2P lookup task identifier."""

    task_id: int


@dataclass(frozen=True)
class P2pQueryLookupResultsRequest:
    """Query the result of a P2P lookup task."""

    task_id: int


@dataclass(frozen=True)
class P2pQueryLookupResultsResponse:
    """Return transfer addresses when the P2P lookup has completed."""

    addresses: list[TransferChannelAddress] | None


@dataclass(frozen=True)
class P2pUnlockObjectsRequest:
    """Release P2P object read locks."""

    keys: list[ObjectKey]


@dataclass(frozen=True)
class P2pUnlockObjectsResponse:
    """Acknowledge P2P object unlocking."""


register_rpc_message_types(
    "p2p_lookup_and_lock", P2pLookupAndLockRequest, P2pLookupAndLockResponse
)
register_rpc_message_types(
    "p2p_query_lookup_results",
    P2pQueryLookupResultsRequest,
    P2pQueryLookupResultsResponse,
)
register_rpc_message_types(
    "p2p_unlock_objects", P2pUnlockObjectsRequest, P2pUnlockObjectsResponse
)


__all__ = [
    "P2pLookupAndLockRequest",
    "P2pLookupAndLockResponse",
    "P2pQueryLookupResultsRequest",
    "P2pQueryLookupResultsResponse",
    "P2pUnlockObjectsRequest",
    "P2pUnlockObjectsResponse",
]
