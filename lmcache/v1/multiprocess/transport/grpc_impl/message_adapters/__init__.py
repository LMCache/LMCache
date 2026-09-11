# SPDX-License-Identifier: Apache-2.0
"""Explicit registry of service-owned protobuf/Python message adapters."""

# Standard
from functools import lru_cache

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.message_adapters import common, p2p
from lmcache.v1.multiprocess.transport.grpc_impl.message_adapters.base import (
    MessageAdapterRegistry,
    RegisteredMessageAdapter,
)


@lru_cache(maxsize=1)
def get_message_adapter_registry() -> MessageAdapterRegistry:
    """Build the registry from each service's adapter declarations.

    Returns:
        The cached immutable message adapter registry.
    """
    return MessageAdapterRegistry(
        common.get_message_adapters() + p2p.get_message_adapters()
    )


__all__ = [
    "MessageAdapterRegistry",
    "RegisteredMessageAdapter",
    "get_message_adapter_registry",
]
