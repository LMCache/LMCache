# SPDX-License-Identifier: Apache-2.0
"""Explicit registry of service-owned protobuf/Python message codecs."""

# Standard
from functools import lru_cache

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.codecs import common, p2p
from lmcache.v1.multiprocess.transport.grpc_impl.codecs.base import (
    MessageCodecRegistry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.codecs.base import (
    RegisteredMessageCodec as RegisteredMessageCodec,
)


@lru_cache(maxsize=1)
def get_message_codec_registry() -> MessageCodecRegistry:
    """Build the registry from each service's codec declarations.

    Returns:
        The cached immutable message codec registry.
    """
    return MessageCodecRegistry(common.get_message_codecs() + p2p.get_message_codecs())
