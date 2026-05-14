# SPDX-License-Identifier: Apache-2.0
"""Tests for the public ``lmcache.sdk`` KV-cache helpers."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

# Third Party
import httpx
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.http_apis.kv_protocol import (
    RetrieveManifest,
    decode_store_chunk,
    decode_store_manifest,
    encode_retrieve_manifest,
    encode_retrieve_shard,
    iter_decode_frames,
)
import lmcache.sdk as lmc_sdk

CHUNK_SIZE = 16
DTYPE = torch.float32


def _make_tensor() -> torch.Tensor:
    """Return a deterministic two-chunk KV tensor."""
    torch.manual_seed(123)
    return torch.randn((2, 2, CHUNK_SIZE * 2, 8), dtype=DTYPE)


def _shape4(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    """Return a tensor shape typed as the KV_2LTD 4-D contract."""
    return cast(tuple[int, int, int, int], tuple(tensor.shape))


def _response(
    method: str,
    url: str,
    *,
    json_body: dict[str, int | str] | None = None,
    content: bytes = b"",
) -> httpx.Response:
    """Build an ``httpx.Response`` with a request attached."""
    request = httpx.Request(method, url)
    if json_body is not None:
        return httpx.Response(200, json=json_body, request=request)
    return httpx.Response(200, content=content, request=request)


def test_store_streams_chunked_protocol(monkeypatch) -> None:
    """``lmc_sdk.store`` streams one manifest and one frame per chunk."""
    tensor = _make_tensor()
    tokens = list(range(CHUNK_SIZE * 2 + 3))
    package = lmc_sdk.KVCachePackage(
        kv=tensor,
        model_name="m",
        tokens=tokens,
    )

    captured_frames: list[bytes] = []

    def fake_get(url: str, timeout: float) -> httpx.Response:
        return _response("GET", url, json_body={"chunk_size": CHUNK_SIZE})

    def fake_post(
        url: str,
        content: Iterator[bytes],
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        captured_frames.extend(content)
        return _response(
            "POST",
            url,
            json_body={
                "status": "ok",
                "total_tokens": CHUNK_SIZE * 2,
                "total_chunks": 2,
                "stored_tokens": CHUNK_SIZE * 2,
                "stored_chunks": 2,
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)

    result = lmc_sdk.store(package, "localhost:8080")

    frames = iter_decode_frames(captured_frames)
    manifest = decode_store_manifest(next(frames))
    chunk0_index, chunk0_payload = decode_store_chunk(next(frames))
    chunk1_index, chunk1_payload = decode_store_chunk(next(frames))
    assert result.stored_chunks == 2
    assert manifest.model_name == "m"
    assert manifest.tokens == tokens
    assert manifest.shape == _shape4(tensor)
    assert chunk0_index == 0
    assert chunk1_index == 1
    assert chunk0_payload == _chunk_payload(tensor, 0)
    assert chunk1_payload == _chunk_payload(tensor, 1)


def test_store_accepts_in_memory_package(monkeypatch) -> None:
    """``lmc_sdk.store`` applies token overrides to an in-memory package."""
    tensor = _make_tensor()
    source_tokens = list(range(CHUNK_SIZE * 2 + 3))
    target_tokens = list(range(1000, 1000 + CHUNK_SIZE * 2 + 3))
    package = lmc_sdk.KVCachePackage(
        kv=tensor,
        model_name="m",
        tokens=source_tokens,
    )
    captured_frames: list[bytes] = []

    def fake_get(url: str, timeout: float) -> httpx.Response:
        return _response("GET", url, json_body={"chunk_size": CHUNK_SIZE})

    def fake_post(
        url: str,
        content: Iterator[bytes],
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        captured_frames.extend(content)
        return _response(
            "POST",
            url,
            json_body={
                "status": "ok",
                "total_tokens": CHUNK_SIZE * 2,
                "total_chunks": 2,
                "stored_tokens": CHUNK_SIZE * 2,
                "stored_chunks": 2,
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)

    result = lmc_sdk.store(package, "localhost:8080", tokens=target_tokens)

    frames = iter_decode_frames(captured_frames)
    manifest = decode_store_manifest(next(frames))
    assert result.stored_chunks == 2
    assert manifest.model_name == "m"
    assert manifest.tokens == target_tokens


def test_retrieve_assembles_shards_into_memory(monkeypatch) -> None:
    """``lmc_sdk.retrieve`` assembles streamed worker shards into memory."""
    tensor = _make_tensor()
    tokens = list(range(CHUNK_SIZE * 2))
    response_body = b"".join(
        [
            encode_retrieve_manifest(
                RetrieveManifest(
                    model_name="m",
                    total_tokens=CHUNK_SIZE * 2,
                    total_chunks=2,
                    hit_tokens=CHUNK_SIZE * 2,
                    hit_chunks=2,
                    chunk_size=CHUNK_SIZE,
                    world_size=2,
                    shape=_shape4(tensor),
                    shard_shape=(2, 2, CHUNK_SIZE, 4),
                    dtype=str(tensor.dtype),
                )
            ),
            *_retrieve_shard_frames(tensor, world_size=2),
        ]
    )

    @contextmanager
    def fake_stream(
        method: str,
        url: str,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> Iterator[httpx.Response]:
        yield _response(method, url, content=response_body)

    monkeypatch.setattr(httpx, "stream", fake_stream)

    result = lmc_sdk.retrieve(
        "http://localhost:8080",
        model_name="m",
        tokens=tokens,
    )

    assert result.hit_chunks == 2
    assert torch.equal(result.package.kv, tensor)
    assert result.package.model_name == "m"
    assert result.package.tokens == tokens


def test_retrieve_rejects_missing_shard(
    monkeypatch,
) -> None:
    """``lmc_sdk.retrieve`` rejects incomplete retrieve streams."""
    tensor = _make_tensor()[:, :, :CHUNK_SIZE, :].contiguous()
    tokens = list(range(CHUNK_SIZE))
    response_body = b"".join(
        [
            encode_retrieve_manifest(
                RetrieveManifest(
                    model_name="m",
                    total_tokens=CHUNK_SIZE,
                    total_chunks=1,
                    hit_tokens=CHUNK_SIZE,
                    hit_chunks=1,
                    chunk_size=CHUNK_SIZE,
                    world_size=2,
                    shape=_shape4(tensor),
                    shard_shape=(2, 2, CHUNK_SIZE, 4),
                    dtype=str(tensor.dtype),
                )
            ),
            _retrieve_shard_frames(tensor, world_size=2)[0],
        ]
    )

    @contextmanager
    def fake_stream(
        method: str,
        url: str,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> Iterator[httpx.Response]:
        yield _response(method, url, content=response_body)

    monkeypatch.setattr(httpx, "stream", fake_stream)

    with pytest.raises(lmc_sdk.KVCacheSDKError, match="missing 1 shard"):
        lmc_sdk.retrieve(
            "http://localhost:8080",
            model_name="m",
            tokens=tokens,
        )


def _chunk_payload(tensor: torch.Tensor, chunk_index: int) -> bytes:
    start = chunk_index * CHUNK_SIZE
    end = start + CHUNK_SIZE
    return tensor[:, :, start:end, :].contiguous().view(torch.uint8).numpy().tobytes()


def _retrieve_shard_frames(tensor: torch.Tensor, world_size: int) -> list[bytes]:
    frames: list[bytes] = []
    d_per_worker = tensor.shape[3] // world_size
    for chunk_index in range(tensor.shape[2] // CHUNK_SIZE):
        for worker_id in range(world_size):
            start = chunk_index * CHUNK_SIZE
            end = start + CHUNK_SIZE
            d_start = worker_id * d_per_worker
            d_end = d_start + d_per_worker
            payload = (
                tensor[:, :, start:end, d_start:d_end]
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            frames.append(encode_retrieve_shard(chunk_index, worker_id, payload))
    return frames
