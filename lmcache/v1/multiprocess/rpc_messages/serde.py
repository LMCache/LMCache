# SPDX-License-Identifier: Apache-2.0
"""Wire serialization shared by every transport-neutral RPC contract."""

# Standard
from typing import TypeVar

# First Party
from lmcache.v1.multiprocess.custom_types import (
    get_customized_decoder,
    get_customized_encoder,
)

MessageT = TypeVar("MessageT")


def serialize_rpc_message(message: MessageT, message_type: type[MessageT]) -> bytes:
    """Serialize one transport-neutral Python request or response message."""
    if not isinstance(message, message_type):
        raise TypeError(
            f"expected {message_type.__name__}, got {type(message).__name__}"
        )
    return get_customized_encoder(message_type).encode(message)


def deserialize_rpc_message(payload: bytes, message_type: type[MessageT]) -> MessageT:
    """Deserialize one transport-neutral Python request or response message."""
    return get_customized_decoder(message_type).decode(payload)


__all__ = ["deserialize_rpc_message", "serialize_rpc_message"]
