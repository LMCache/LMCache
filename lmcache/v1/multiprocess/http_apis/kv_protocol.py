# SPDX-License-Identifier: Apache-2.0
"""Versioned wire protocol for bytes-level KV cache HTTP transfers."""

# Standard
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast
import json
import struct

PROTOCOL_VERSION = 1
STREAM_MEDIA_TYPE = "application/x-lmcache-kv-stream; v=1"

_FRAME_PREFIX = struct.Struct("!IQ")
_MAX_HEADER_BYTES = 1024 * 1024

FrameValue = str | int | list[int]
FrameHeader = dict[str, FrameValue]


class FrameType(StrEnum):
    """Known frame types in the v1 KV transfer protocol."""

    STORE_MANIFEST = "store_manifest"
    STORE_CHUNK = "store_chunk"
    RETRIEVE_MANIFEST = "retrieve_manifest"
    RETRIEVE_SHARD = "retrieve_shard"


@dataclass(frozen=True)
class KVFrame:
    """A decoded protocol frame.

    Args:
        header: JSON metadata associated with the frame. Every header includes
            ``version``, ``type``, and ``payload_length``.
        payload: Binary payload bytes for this frame.
    """

    header: FrameHeader
    payload: bytes


@dataclass(frozen=True)
class StoreManifest:
    """Client-to-server store metadata carried by the first request frame."""

    model_name: str
    tokens: list[int]
    cache_salt: str
    shape: tuple[int, int, int, int]
    dtype: str


@dataclass(frozen=True)
class RetrieveRequest:
    """JSON request body for retrieve and lookup endpoints."""

    model_name: str
    tokens: list[int]
    cache_salt: str
    protocol_version: int


@dataclass(frozen=True)
class RetrieveManifest:
    """Server-to-client retrieve metadata carried by the first response frame."""

    model_name: str
    total_tokens: int
    total_chunks: int
    hit_tokens: int
    hit_chunks: int
    chunk_size: int
    world_size: int
    shape: tuple[int, int, int, int]
    shard_shape: tuple[int, int, int, int]
    dtype: str


def encode_frame(
    frame_type: FrameType,
    payload: bytes = b"",
    fields: Mapping[str, FrameValue] | None = None,
) -> bytes:
    """Encode a single v1 frame.

    Args:
        frame_type: Type tag for the frame.
        payload: Binary payload carried by the frame.
        fields: Additional JSON-serializable metadata. Keys must not
            override the protocol-managed ``version``, ``type``, or
            ``payload_length`` fields.

    Returns:
        Binary frame bytes suitable for HTTP streaming.

    Raises:
        ValueError: If ``fields`` contains a reserved header key.
    """
    header: FrameHeader = {
        "version": PROTOCOL_VERSION,
        "type": frame_type.value,
        "payload_length": len(payload),
    }
    if fields is not None:
        reserved = set(header).intersection(fields)
        if reserved:
            raise ValueError(f"reserved frame header fields: {sorted(reserved)}")
        header.update(fields)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _FRAME_PREFIX.pack(len(header_bytes), len(payload)) + header_bytes + payload


def iter_decode_frames(chunks: Iterable[bytes]) -> Iterator[KVFrame]:
    """Decode frames from a synchronous byte chunk iterable.

    Args:
        chunks: Byte chunks from an HTTP response body.

    Yields:
        Fully decoded frames in order.

    Raises:
        ValueError: If the stream is truncated or a frame header is invalid.
    """
    decoder = _FrameDecoder()
    for chunk in chunks:
        yield from decoder.feed(chunk)
    yield from decoder.finish()


async def aiter_decode_frames(chunks: AsyncIterable[bytes]) -> AsyncIterator[KVFrame]:
    """Decode frames from an asynchronous byte chunk iterable.

    Args:
        chunks: Byte chunks from an HTTP request body.

    Yields:
        Fully decoded frames in order.

    Raises:
        ValueError: If the stream is truncated or a frame header is invalid.
    """
    decoder = _FrameDecoder()
    async for chunk in chunks:
        for frame in decoder.feed(chunk):
            yield frame
    for frame in decoder.finish():
        yield frame


def encode_store_manifest(manifest: StoreManifest) -> bytes:
    """Encode a store manifest frame."""
    return encode_frame(
        FrameType.STORE_MANIFEST,
        fields={
            "model_name": manifest.model_name,
            "tokens": manifest.tokens,
            "cache_salt": manifest.cache_salt,
            "shape": list(manifest.shape),
            "dtype": manifest.dtype,
        },
    )


def decode_store_manifest(frame: KVFrame) -> StoreManifest:
    """Decode and validate a store manifest frame.

    Args:
        frame: Decoded frame.

    Returns:
        Store manifest metadata.

    Raises:
        ValueError: If ``frame`` is not a valid store manifest.
    """
    _require_frame_type(frame, FrameType.STORE_MANIFEST)
    return StoreManifest(
        model_name=_require_str(frame.header, "model_name"),
        tokens=_require_int_list(frame.header, "tokens"),
        cache_salt=_optional_str(frame.header, "cache_salt"),
        shape=_require_shape4(frame.header, "shape"),
        dtype=_require_str(frame.header, "dtype"),
    )


def encode_store_chunk(chunk_index: int, payload: bytes) -> bytes:
    """Encode one full-token-chunk store payload."""
    return encode_frame(
        FrameType.STORE_CHUNK,
        payload=payload,
        fields={"chunk_index": chunk_index},
    )


def decode_store_chunk(frame: KVFrame) -> tuple[int, bytes]:
    """Decode one full-token-chunk store frame."""
    _require_frame_type(frame, FrameType.STORE_CHUNK)
    return _require_int(frame.header, "chunk_index"), frame.payload


def encode_retrieve_request(request: RetrieveRequest) -> bytes:
    """Encode a retrieve or lookup JSON request body."""
    body = {
        "model_name": request.model_name,
        "tokens": request.tokens,
        "cache_salt": request.cache_salt,
        "protocol_version": request.protocol_version,
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def encode_retrieve_manifest(manifest: RetrieveManifest) -> bytes:
    """Encode a retrieve manifest frame."""
    return encode_frame(
        FrameType.RETRIEVE_MANIFEST,
        fields={
            "model_name": manifest.model_name,
            "total_tokens": manifest.total_tokens,
            "total_chunks": manifest.total_chunks,
            "hit_tokens": manifest.hit_tokens,
            "hit_chunks": manifest.hit_chunks,
            "chunk_size": manifest.chunk_size,
            "world_size": manifest.world_size,
            "shape": list(manifest.shape),
            "shard_shape": list(manifest.shard_shape),
            "dtype": manifest.dtype,
        },
    )


def decode_retrieve_manifest(frame: KVFrame) -> RetrieveManifest:
    """Decode and validate a retrieve manifest frame."""
    _require_frame_type(frame, FrameType.RETRIEVE_MANIFEST)
    return RetrieveManifest(
        model_name=_require_str(frame.header, "model_name"),
        total_tokens=_require_int(frame.header, "total_tokens"),
        total_chunks=_require_int(frame.header, "total_chunks"),
        hit_tokens=_require_int(frame.header, "hit_tokens"),
        hit_chunks=_require_int(frame.header, "hit_chunks"),
        chunk_size=_require_int(frame.header, "chunk_size"),
        world_size=_require_int(frame.header, "world_size"),
        shape=_require_shape4(frame.header, "shape"),
        shard_shape=_require_shape4(frame.header, "shard_shape"),
        dtype=_require_str(frame.header, "dtype"),
    )


def encode_retrieve_shard(chunk_index: int, worker_id: int, payload: bytes) -> bytes:
    """Encode one retrieved worker-shard payload frame."""
    return encode_frame(
        FrameType.RETRIEVE_SHARD,
        payload=payload,
        fields={"chunk_index": chunk_index, "worker_id": worker_id},
    )


def decode_retrieve_shard(frame: KVFrame) -> tuple[int, int, bytes]:
    """Decode one retrieved worker-shard payload frame."""
    _require_frame_type(frame, FrameType.RETRIEVE_SHARD)
    return (
        _require_int(frame.header, "chunk_index"),
        _require_int(frame.header, "worker_id"),
        frame.payload,
    )


def _decode_header(header_bytes: bytes, payload_length: int) -> FrameHeader:
    try:
        decoded = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid KV frame header: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("KV frame header must be a JSON object")
    header = cast(FrameHeader, decoded)
    version = _require_int(header, "version")
    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported KV protocol version {version}; expected {PROTOCOL_VERSION}"
        )
    declared_payload_length = _require_int(header, "payload_length")
    if declared_payload_length != payload_length:
        raise ValueError(
            "KV frame payload length mismatch: "
            f"header={declared_payload_length}, frame={payload_length}"
        )
    _require_str(header, "type")
    return header


def _require_frame_type(frame: KVFrame, frame_type: FrameType) -> None:
    actual = _require_str(frame.header, "type")
    if actual != frame_type.value:
        raise ValueError(f"expected {frame_type.value!r} frame, got {actual!r}")


def _require_int(header: FrameHeader, key: str) -> int:
    value = header.get(key)
    if not isinstance(value, int):
        raise ValueError(f"KV frame field {key!r} must be an int")
    return value


def _require_str(header: FrameHeader, key: str) -> str:
    value = header.get(key)
    if not isinstance(value, str):
        raise ValueError(f"KV frame field {key!r} must be a string")
    return value


def _optional_str(header: FrameHeader, key: str) -> str:
    value = header.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"KV frame field {key!r} must be a string")
    return value


def _require_int_list(header: FrameHeader, key: str) -> list[int]:
    value = header.get(key)
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"KV frame field {key!r} must be a list of ints")
    return list(value)


def _require_shape4(header: FrameHeader, key: str) -> tuple[int, int, int, int]:
    shape = _require_int_list(header, key)
    if len(shape) != 4:
        raise ValueError(f"KV frame field {key!r} must contain 4 dimensions")
    return (shape[0], shape[1], shape[2], shape[3])


class _FrameDecoder:
    """Incremental decoder for length-prefixed KV frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> Iterator[KVFrame]:
        """Feed bytes and yield every complete frame now available."""
        self._buffer.extend(chunk)
        while True:
            if len(self._buffer) < _FRAME_PREFIX.size:
                return
            header_len, payload_len = _FRAME_PREFIX.unpack(
                self._buffer[: _FRAME_PREFIX.size]
            )
            if header_len > _MAX_HEADER_BYTES:
                raise ValueError(f"KV frame header too large: {header_len} bytes")
            frame_len = _FRAME_PREFIX.size + header_len + payload_len
            if len(self._buffer) < frame_len:
                return
            header_start = _FRAME_PREFIX.size
            header_end = header_start + header_len
            payload_end = header_end + payload_len
            header = _decode_header(
                bytes(self._buffer[header_start:header_end]),
                payload_len,
            )
            payload = bytes(self._buffer[header_end:payload_end])
            del self._buffer[:payload_end]
            yield KVFrame(header=header, payload=payload)

    def finish(self) -> Iterator[KVFrame]:
        """Validate that the stream ended on a frame boundary."""
        if self._buffer:
            raise ValueError("truncated KV frame stream")
        return iter(())
