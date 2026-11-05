# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for protobuf codec registration primitives."""

# First Party
from lmcache.v1.transport.grpc.codec_registry import (
    MessageCodecRegistry as MessageCodecRegistry,
)
from lmcache.v1.transport.grpc.codec_registry import MessageReader as MessageReader
from lmcache.v1.transport.grpc.codec_registry import MessageWriter as MessageWriter
from lmcache.v1.transport.grpc.codec_registry import ProtoMessageT as ProtoMessageT
from lmcache.v1.transport.grpc.codec_registry import PythonValueT as PythonValueT
from lmcache.v1.transport.grpc.codec_registry import (
    RegisteredMessageCodec as RegisteredMessageCodec,
)

__all__ = [
    "MessageCodecRegistry",
    "MessageReader",
    "MessageWriter",
    "ProtoMessageT",
    "PythonValueT",
    "RegisteredMessageCodec",
]
