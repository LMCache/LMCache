# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for debug RPCs."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types


@dataclass(frozen=True)
class NoopRequest:
    """Issue a no-op request."""


@dataclass(frozen=True)
class NoopResponse:
    """Return the no-op diagnostic message."""

    message: str


register_rpc_message_types("noop", NoopRequest, NoopResponse)


__all__ = ["NoopRequest", "NoopResponse"]
