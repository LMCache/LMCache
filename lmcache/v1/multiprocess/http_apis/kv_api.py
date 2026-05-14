# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for chunk-streamed bytes-level KV cache access."""

# Standard
from collections.abc import AsyncIterator, Iterator
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.http_apis.kv_protocol import (
    PROTOCOL_VERSION,
    STREAM_MEDIA_TYPE,
    KVFrame,
    RetrieveManifest,
    aiter_decode_frames,
    decode_store_chunk,
    decode_store_manifest,
    encode_retrieve_manifest,
    encode_retrieve_shard,
)
from lmcache.v1.multiprocess.kv_bytes import RetrieveBytesResult
from lmcache.v1.multiprocess.server import MPCacheEngine

logger = init_logger(__name__)

router = APIRouter()


class _RetrieveBody(BaseModel):
    """JSON body for ``/api/kv/retrieve``."""

    model_name: str = Field(..., description="Registered model name.")
    tokens: list[int] = Field(..., description="Token sequence to address.")
    cache_salt: str = Field(
        default="",
        description="Optional per-namespace isolation salt.",
    )
    protocol_version: int = Field(
        default=PROTOCOL_VERSION,
        description="KV transfer protocol version requested by the client.",
    )


def _engine_or_503(request: Request) -> MPCacheEngine:
    """Resolve the MP engine off ``app.state`` or raise a 503 JSON error."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="engine not initialized",
        )
    return engine


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    """Resolve a ``torch`` dtype string such as ``"torch.float16"``."""
    normalized = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported KV dtype {dtype_name!r}")
    return dtype


def _check_protocol_version(protocol_version: int) -> None:
    """Reject clients using an unsupported KV transfer protocol version."""
    if protocol_version != PROTOCOL_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported KV protocol version {protocol_version}; "
                f"expected {PROTOCOL_VERSION}"
            ),
        )


async def _read_first_frame(frames: AsyncIterator[KVFrame]) -> KVFrame:
    """Read the first protocol frame or raise ``ValueError`` on an empty body."""
    async for frame in frames:
        return frame
    raise ValueError("KV store request body is missing a manifest frame")


async def _store_chunk_payloads(
    frames: AsyncIterator[KVFrame],
) -> AsyncIterator[bytes]:
    """Yield store chunk payloads in strict chunk-index order."""
    expected_chunk_index = 0
    async for frame in frames:
        chunk_index, payload = decode_store_chunk(frame)
        if chunk_index != expected_chunk_index:
            raise ValueError(
                f"expected store chunk {expected_chunk_index}, got {chunk_index}"
            )
        expected_chunk_index += 1
        yield payload


def _retrieve_manifest(
    model_name: str,
    result: RetrieveBytesResult,
    chunk_size: int,
) -> RetrieveManifest:
    """Build a retrieve manifest from an engine result."""
    shard_shape = result.per_shard_shape
    if result.hit_chunks == 0:
        full_shape = (0, 0, 0, 0)
    else:
        full_shape = (
            shard_shape[0],
            shard_shape[1],
            result.hit_tokens,
            shard_shape[3] * result.world_size,
        )
    return RetrieveManifest(
        model_name=model_name,
        total_tokens=result.total_tokens,
        total_chunks=result.total_chunks,
        hit_tokens=result.hit_tokens,
        hit_chunks=result.hit_chunks,
        chunk_size=chunk_size,
        world_size=result.world_size,
        shape=full_shape,
        shard_shape=shard_shape,
        dtype=str(result.dtype),
    )


def _retrieve_stream(
    model_name: str,
    result: RetrieveBytesResult,
    chunk_size: int,
) -> Iterator[bytes]:
    """Encode a retrieve result as a versioned shard stream."""
    try:
        yield encode_retrieve_manifest(
            _retrieve_manifest(model_name, result, chunk_size)
        )
        for shard in result.iter_shards():
            yield encode_retrieve_shard(
                shard.chunk_index,
                shard.worker_id,
                shard.data,
            )
    finally:
        result.close()


@router.post("/api/kv/store")
async def store(request: Request) -> dict[str, int | str]:
    """Store KV cache bytes from a versioned chunk stream.

    The request body starts with a v1 store manifest frame followed by one
    full KV chunk frame per chunk. Each chunk frame carries the complete
    unsharded hidden dimension for that token chunk. The engine splits those
    bytes into worker-shard ``MemoryObj`` buffers.
    """
    engine = _engine_or_503(request)
    frames = aiter_decode_frames(request.stream())
    try:
        manifest = decode_store_manifest(await _read_first_frame(frames))
        dtype = _dtype_from_name(manifest.dtype)
        result = await engine.store_kv_bytes_by_tokens(
            model_name=manifest.model_name,
            tokens=manifest.tokens,
            chunks=_store_chunk_payloads(frames),
            full_shape=manifest.shape,
            dtype=dtype,
            cache_salt=manifest.cache_salt,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail="model_name is not registered with this engine",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "total_tokens": result.total_tokens,
        "total_chunks": result.total_chunks,
        "stored_tokens": result.stored_tokens,
        "stored_chunks": result.stored_chunks,
    }


@router.post("/api/kv/retrieve")
async def retrieve(body: _RetrieveBody, request: Request) -> StreamingResponse:
    """Retrieve KV cache bytes as a versioned shard stream."""
    _check_protocol_version(body.protocol_version)
    engine = _engine_or_503(request)
    try:
        result = await asyncio.to_thread(
            engine.retrieve_kv_bytes_by_tokens,
            body.model_name,
            body.tokens,
            cache_salt=body.cache_salt,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"model_name {body.model_name!r} is not registered with this engine",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        _retrieve_stream(body.model_name, result, engine.chunk_size),
        media_type=STREAM_MEDIA_TYPE,
    )
