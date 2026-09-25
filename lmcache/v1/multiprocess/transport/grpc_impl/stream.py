# SPDX-License-Identifier: Apache-2.0
"""Persistent stream framing for the LMCache gRPC transport."""

# Future
from __future__ import annotations

# Standard
from functools import lru_cache
import struct

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import iter_methods
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    GrpcMethodCodec,
    get_method_codec_registry,
)

STREAM_SERVICE = "lmcache.mp.StreamService"
STREAM_METHOD = f"/{STREAM_SERVICE}/Dispatch"

_REQUEST_HEADER = struct.Struct("!QQH")
_RESPONSE_HEADER = struct.Struct("!QB")
_BATCH_COUNT = struct.Struct("!H")
_BATCH_ITEM_LENGTH = struct.Struct("!I")


@lru_cache(maxsize=1)
def get_stream_method_codecs() -> tuple[GrpcMethodCodec, ...]:
    """Return gRPC method codecs in the stable stream method-id order."""
    registry = get_method_codec_registry()
    return tuple(registry.by_full_name[method.full_name] for _, method in iter_methods())


@lru_cache(maxsize=1)
def get_stream_method_ids() -> dict[str, int]:
    """Return the stream method id for each generated gRPC method."""
    return {
        codec.full_name: method_id
        for method_id, codec in enumerate(get_stream_method_codecs())
    }


def identity_bytes(value: bytes) -> bytes:
    """Pass already-framed stream messages through gRPC unchanged."""
    return value


def pack_request_frame(
    request_id: int,
    client_key: int,
    method_id: int,
    payload: bytes,
) -> bytes:
    """Pack one stream request frame."""
    return _REQUEST_HEADER.pack(request_id, client_key, method_id) + payload


def unpack_request_frame(frame: bytes) -> tuple[int, int, int, bytes]:
    """Unpack one stream request frame."""
    request_id, client_key, method_id = _REQUEST_HEADER.unpack_from(frame)
    return request_id, client_key, method_id, frame[_REQUEST_HEADER.size :]


def pack_response_frame(request_id: int, payload: bytes) -> bytes:
    """Pack one successful stream response frame."""
    return _RESPONSE_HEADER.pack(request_id, 1) + payload


def pack_error_frame(request_id: int, message: str) -> bytes:
    """Pack one failed stream response frame."""
    return _RESPONSE_HEADER.pack(request_id, 0) + message.encode()


def unpack_response_frame(frame: bytes) -> tuple[int, bool, bytes]:
    """Unpack one stream response frame."""
    request_id, ok = _RESPONSE_HEADER.unpack_from(frame)
    return request_id, bool(ok), frame[_RESPONSE_HEADER.size :]


def pack_batch(frames: list[bytes]) -> bytes:
    """Pack multiple frames into one gRPC stream message."""
    chunks = [_BATCH_COUNT.pack(len(frames))]
    for frame in frames:
        chunks.append(_BATCH_ITEM_LENGTH.pack(len(frame)))
        chunks.append(frame)
    return b"".join(chunks)


def unpack_batch(batch: bytes) -> list[bytes]:
    """Unpack frames from one gRPC stream message."""
    (count,) = _BATCH_COUNT.unpack_from(batch)
    offset = _BATCH_COUNT.size
    frames: list[bytes] = []
    for _ in range(count):
        (size,) = _BATCH_ITEM_LENGTH.unpack_from(batch, offset)
        offset += _BATCH_ITEM_LENGTH.size
        frames.append(batch[offset : offset + size])
        offset += size
    return frames
