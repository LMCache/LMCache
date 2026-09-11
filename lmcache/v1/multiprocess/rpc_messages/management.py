# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for management and configuration RPCs."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types


@dataclass(frozen=True)
class ClearRequest:
    """Clear all server caches."""


@dataclass(frozen=True)
class ClearResponse:
    """Acknowledge cache clearing."""


@dataclass(frozen=True)
class GetChunkSizeRequest:
    """Query the configured cache chunk size."""


@dataclass(frozen=True)
class GetChunkSizeResponse:
    """Return the configured cache chunk size."""

    chunk_size: int


@dataclass(frozen=True)
class GetExperimentalRequest:
    """Query enabled experimental capabilities."""


@dataclass(frozen=True)
class GetExperimentalResponse:
    """Return enabled experimental capability names."""

    names: list[str]


@dataclass(frozen=True)
class PingRequest:
    """Refresh worker liveness or probe server health."""

    instance_id: int | None


@dataclass(frozen=True)
class PingResponse:
    """Return server health."""

    ok: bool


register_rpc_message_types("clear", ClearRequest, ClearResponse)
register_rpc_message_types("get_chunk_size", GetChunkSizeRequest, GetChunkSizeResponse)
register_rpc_message_types(
    "get_experimental", GetExperimentalRequest, GetExperimentalResponse
)
register_rpc_message_types("ping", PingRequest, PingResponse)


__all__ = [
    "ClearRequest",
    "ClearResponse",
    "GetChunkSizeRequest",
    "GetChunkSizeResponse",
    "GetExperimentalRequest",
    "GetExperimentalResponse",
    "PingRequest",
    "PingResponse",
]
