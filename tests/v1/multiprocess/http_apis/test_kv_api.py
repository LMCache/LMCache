# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the chunk-streamed KV cache HTTP API."""

# Standard
from collections.abc import AsyncIterator
from typing import cast
import asyncio

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey, RetrieveBytesResult
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.http_apis.kv_api import router as kv_router
from lmcache.v1.multiprocess.http_apis.kv_protocol import (
    STREAM_MEDIA_TYPE,
    StoreManifest,
    decode_retrieve_manifest,
    decode_retrieve_shard,
    encode_store_chunk,
    encode_store_manifest,
    iter_decode_frames,
)
from lmcache.v1.multiprocess.server import MPCacheEngine

CHUNK_SIZE = 16
NUM_LAYERS = 2
HIDDEN_DIM_PER_WORKER = 8
DTYPE = torch.float32


class _FakeKVLayerGroup:
    """Small stand-in for the fields ``get_layout_desc`` reads in tests."""

    def __init__(self, shape: torch.Size, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.num_layers = shape[1]
        self.hidden_dim_size = shape[3]


class _FakeKVLayerGroupsManager:
    """Small stand-in for ``KVLayerGroupsManager`` used by ``get_layout_desc``."""

    def __init__(self, layout: MemoryLayoutDesc) -> None:
        self.kv_layer_groups = [
            _FakeKVLayerGroup(shape, dtype)
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
        ]

    @property
    def num_groups(self) -> int:
        """Return how many layer groups the fake context exposes."""
        return len(self.kv_layer_groups)


class _FakeGPUContext:
    """Minimal registered GPU context for CPU-only HTTP tests."""

    def __init__(self, layout: MemoryLayoutDesc) -> None:
        self._shapes = layout.shapes
        self.kv_layer_groups_manager = _FakeKVLayerGroupsManager(layout)

    def get_kv_buffer_shape(
        self,
        num_tokens: int,
        group_idx: int = 0,
    ) -> torch.Size:
        """Return the configured group shape with a caller-provided token dim."""
        shape = self._shapes[group_idx]
        return torch.Size((shape[0], shape[1], num_tokens, shape[3]))


def _make_engine(buffer_bytes: int = 16 * 1024 * 1024) -> MPCacheEngine:
    """Build a CPU-only ``MPCacheEngine`` for direct method testing."""
    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=buffer_bytes,
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    return MPCacheEngine(storage_manager_config=cfg, chunk_size=CHUNK_SIZE)


def _layout_for(world_size: int) -> MemoryLayoutDesc:
    """Synthetic per-shard layout with shape ``[2, L, chunk_size, D/W]``."""
    shape = torch.Size((2, NUM_LAYERS, CHUNK_SIZE, HIDDEN_DIM_PER_WORKER))
    return MemoryLayoutDesc(shapes=[shape], dtypes=[DTYPE])


def _install_resolver(engine: MPCacheEngine, models: dict[str, int]) -> None:
    """Register fake GPU contexts for models under test."""
    _register_fake_layouts(
        engine,
        {
            model_name: (_layout_for(world_size), world_size)
            for model_name, world_size in models.items()
        },
    )


def _register_fake_layouts(
    engine: MPCacheEngine,
    models: dict[str, tuple[MemoryLayoutDesc, int]],
) -> None:
    """Install fake registered contexts that drive the normal layout resolver."""
    engine.gpu_contexts.clear()
    engine.gpu_context_meta.clear()
    for gpu_id, (model_name, (layout, world_size)) in enumerate(models.items()):
        engine.gpu_contexts[gpu_id] = cast(GPUCacheContext, _FakeGPUContext(layout))
        engine.gpu_context_meta[gpu_id] = (model_name, world_size)


def _make_tensor(num_chunks: int, world_size: int, seed: int = 0) -> torch.Tensor:
    """Generate a deterministic canonical KV_2LTD tensor."""
    torch.manual_seed(seed)
    return torch.randn(
        (
            2,
            NUM_LAYERS,
            num_chunks * CHUNK_SIZE,
            HIDDEN_DIM_PER_WORKER * world_size,
        ),
        dtype=DTYPE,
    )


def _shape4(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    """Return a tensor shape typed as the KV_2LTD 4-D contract."""
    return cast(tuple[int, int, int, int], tuple(tensor.shape))


def _tokens_for(num_chunks: int, seed: int = 0) -> list[int]:
    """Tokens covering ``num_chunks`` whole chunks plus a stable partial tail."""
    return list(range(seed, seed + num_chunks * CHUNK_SIZE + 3))


def _chunk_payloads(tensor: torch.Tensor) -> list[bytes]:
    """Split a canonical KV tensor into full-token-chunk byte payloads."""
    return [
        tensor[:, :, start : start + CHUNK_SIZE, :]
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
        for start in range(0, tensor.shape[2], CHUNK_SIZE)
    ]


async def _async_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """Yield byte chunks through the async interface used by the engine."""
    for chunk in chunks:
        yield chunk


def _store_stream(
    model_name: str,
    tokens: list[int],
    tensor: torch.Tensor,
    *,
    cache_salt: str = "",
) -> bytes:
    """Encode a complete HTTP store stream for ``tensor``."""
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) != 4:
        raise ValueError("test tensor must be 4-D")
    frames = [
        encode_store_manifest(
            StoreManifest(
                model_name=model_name,
                tokens=tokens,
                cache_salt=cache_salt,
                shape=(shape[0], shape[1], shape[2], shape[3]),
                dtype=str(tensor.dtype),
            )
        )
    ]
    frames.extend(
        encode_store_chunk(chunk_index, payload)
        for chunk_index, payload in enumerate(_chunk_payloads(tensor))
    )
    return b"".join(frames)


def _tensor_from_engine_result(result: RetrieveBytesResult) -> torch.Tensor:
    """Assemble a retrieved tensor from the engine's public shard iterator."""
    try:
        if result.hit_chunks == 0:
            return torch.empty((0,), dtype=result.dtype)
        shard_shape = result.per_shard_shape
        output = torch.empty(
            (
                shard_shape[0],
                shard_shape[1],
                result.hit_tokens,
                shard_shape[3] * result.world_size,
            ),
            dtype=result.dtype,
        )
        for shard in result.iter_shards():
            t_start = shard.chunk_index * CHUNK_SIZE
            t_end = t_start + CHUNK_SIZE
            d_start = shard.worker_id * shard_shape[3]
            d_end = d_start + shard_shape[3]
            shard_tensor = torch.frombuffer(
                bytearray(shard.data),
                dtype=result.dtype,
            ).reshape(shard_shape)
            output[:, :, t_start:t_end, d_start:d_end] = shard_tensor
        return output
    finally:
        result.close()


def _tensor_from_retrieve_stream(content: bytes) -> tuple[int, int, torch.Tensor]:
    """Decode an HTTP retrieve stream into hit metadata and a tensor."""
    frames = iter_decode_frames([content])
    manifest = decode_retrieve_manifest(next(frames))
    dtype = getattr(torch, manifest.dtype.removeprefix("torch."))
    output = torch.empty(manifest.shape, dtype=dtype)
    for frame in frames:
        chunk_index, worker_id, payload = decode_retrieve_shard(frame)
        shard = torch.frombuffer(
            bytearray(payload),
            dtype=dtype,
        ).reshape(manifest.shard_shape)
        t_start = chunk_index * manifest.chunk_size
        t_end = t_start + manifest.chunk_size
        d_start = worker_id * manifest.shard_shape[3]
        d_end = d_start + manifest.shard_shape[3]
        output[:, :, t_start:t_end, d_start:d_end] = shard
    return manifest.hit_tokens, manifest.hit_chunks, output


def _object_keys_for(
    engine: MPCacheEngine,
    model_name: str,
    tokens: list[int],
    world_size: int,
    *,
    cache_salt: str = "",
) -> list[ObjectKey]:
    """Build the storage keys for a whole-token sequence."""
    chunk_hashes = engine.token_hasher.compute_chunk_hashes(tokens)
    total_tokens = len(chunk_hashes) * CHUNK_SIZE
    ipc_key = IPCCacheEngineKey(
        model_name=model_name,
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(tokens[:total_tokens]),
        start=0,
        end=total_tokens,
        request_id="test",
        cache_salt=cache_salt,
    )
    return ipc_key_to_object_keys(ipc_key, chunk_hashes)


def _store_empty_object(
    engine: MPCacheEngine,
    key: ObjectKey,
    layout: MemoryLayoutDesc,
) -> None:
    """Create an object without caring about its payload bytes."""
    reserved = engine.storage_manager.reserve_write([key], layout, "new")
    assert key in reserved
    engine.storage_manager.finish_write([key])


def _make_http_harness(
    models: dict[str, tuple[MemoryLayoutDesc, int]],
) -> tuple[TestClient, MPCacheEngine]:
    """Build a FastAPI test client backed by a real CPU engine."""
    engine = _make_engine()
    _register_fake_layouts(engine, models)
    app = FastAPI()
    app.state.engine = engine
    app.include_router(kv_router)
    return TestClient(app), engine


class TestStoreRetrieveBytes:
    """Direct ``MPCacheEngine`` round-trip tests for chunk streaming."""

    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_round_trip_byte_identity(self, world_size: int) -> None:
        engine = _make_engine()
        _install_resolver(engine, {"m": world_size})
        tokens = _tokens_for(num_chunks=3)
        tensor = _make_tensor(num_chunks=3, world_size=world_size, seed=1)

        store_result = asyncio.run(
            engine.store_kv_bytes_by_tokens(
                "m",
                tokens,
                _async_chunks(_chunk_payloads(tensor)),
                full_shape=_shape4(tensor),
                dtype=tensor.dtype,
            )
        )
        assert store_result.stored_chunks == 3

        retrieve_result = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        assert retrieve_result.hit_chunks == 3
        assert torch.equal(_tensor_from_engine_result(retrieve_result), tensor)

    def test_partial_hit(self) -> None:
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        store_tokens = _tokens_for(num_chunks=2)
        tensor = _make_tensor(num_chunks=2, world_size=1, seed=2)
        asyncio.run(
            engine.store_kv_bytes_by_tokens(
                "m",
                store_tokens,
                _async_chunks(_chunk_payloads(tensor)),
                full_shape=_shape4(tensor),
                dtype=tensor.dtype,
            )
        )

        full_tokens = list(store_tokens[: 2 * CHUNK_SIZE]) + list(range(99_000, 99_032))
        result = engine.retrieve_kv_bytes_by_tokens("m", full_tokens)
        assert result.total_chunks == 4
        assert result.hit_chunks == 2
        assert torch.equal(_tensor_from_engine_result(result), tensor)

    def test_retrieve_is_non_destructive(self) -> None:
        engine = _make_engine()
        _install_resolver(engine, {"m": 2})
        tokens = _tokens_for(num_chunks=2)
        tensor = _make_tensor(num_chunks=2, world_size=2, seed=3)
        asyncio.run(
            engine.store_kv_bytes_by_tokens(
                "m",
                tokens,
                _async_chunks(_chunk_payloads(tensor)),
                full_shape=_shape4(tensor),
                dtype=tensor.dtype,
            )
        )

        first = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        second = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        assert torch.equal(_tensor_from_engine_result(first), tensor)
        assert torch.equal(_tensor_from_engine_result(second), tensor)

    def test_unknown_model_raises(self) -> None:
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        tensor = _make_tensor(num_chunks=1, world_size=1)
        with pytest.raises(KeyError):
            asyncio.run(
                engine.store_kv_bytes_by_tokens(
                    "nope",
                    _tokens_for(1),
                    _async_chunks(_chunk_payloads(tensor)),
                    full_shape=_shape4(tensor),
                    dtype=tensor.dtype,
                )
            )
        with pytest.raises(KeyError):
            engine.retrieve_kv_bytes_by_tokens("nope", _tokens_for(1))

    def test_payload_shape_mismatch_rejected(self) -> None:
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        tensor = _make_tensor(num_chunks=1, world_size=1)
        with pytest.raises(ValueError, match="payload shape"):
            asyncio.run(
                engine.store_kv_bytes_by_tokens(
                    "m",
                    _tokens_for(1),
                    _async_chunks(_chunk_payloads(tensor)),
                    full_shape=(2, NUM_LAYERS, CHUNK_SIZE, 999),
                    dtype=tensor.dtype,
                )
            )

    def test_no_complete_chunks_returns_zero(self) -> None:
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        tokens = list(range(CHUNK_SIZE - 1))
        result = asyncio.run(
            engine.store_kv_bytes_by_tokens(
                "m",
                tokens,
                _async_chunks([]),
                full_shape=(2, NUM_LAYERS, 0, HIDDEN_DIM_PER_WORKER),
                dtype=DTYPE,
            )
        )
        assert result.stored_chunks == 0
        retrieve_result = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        assert retrieve_result.hit_chunks == 0
        retrieve_result.close()


@pytest.fixture
def http_harness() -> tuple[TestClient, MPCacheEngine]:
    """FastAPI test client wired to a real CPU engine with fake contexts."""
    return _make_http_harness(
        {
            "m": (_layout_for(world_size=1), 1),
            "other": (_layout_for(world_size=2), 2),
        }
    )


@pytest.fixture
def http_client(
    http_harness: tuple[TestClient, MPCacheEngine],
) -> TestClient:
    """Return only the FastAPI client for tests that do not inspect storage."""
    client, _ = http_harness
    return client


class TestKVApiHTTP:
    """HTTP-level tests against a FastAPI ``TestClient`` mounted with kv_api."""

    def test_store_retrieve_round_trip(self, http_client: TestClient) -> None:
        tokens = _tokens_for(num_chunks=2)
        tensor = _make_tensor(num_chunks=2, world_size=1, seed=10)

        store = http_client.post(
            "/api/kv/store",
            content=_store_stream("m", tokens, tensor),
            headers={"Content-Type": STREAM_MEDIA_TYPE},
        )
        assert store.status_code == 200, store.text
        assert store.json()["stored_chunks"] == 2

        retrieve = http_client.post(
            "/api/kv/retrieve",
            json={"model_name": "m", "tokens": tokens},
        )
        assert retrieve.status_code == 200
        hit_tokens, hit_chunks, retrieved = _tensor_from_retrieve_stream(
            retrieve.content
        )
        assert hit_tokens == 2 * CHUNK_SIZE
        assert hit_chunks == 2
        assert torch.equal(retrieved, tensor)

    def test_retrieve_miss_returns_empty_manifest(
        self,
        http_client: TestClient,
    ) -> None:
        tokens = _tokens_for(num_chunks=2, seed=999)
        retrieve = http_client.post(
            "/api/kv/retrieve",
            json={"model_name": "m", "tokens": tokens},
        )
        assert retrieve.status_code == 200
        hit_tokens, hit_chunks, retrieved = _tensor_from_retrieve_stream(
            retrieve.content
        )
        assert hit_tokens == 0
        assert hit_chunks == 0
        assert retrieved.numel() == 0

    def test_lookup_returns_hit_metadata(self, http_client: TestClient) -> None:
        tokens = _tokens_for(num_chunks=2, seed=20)
        tensor = _make_tensor(num_chunks=2, world_size=1, seed=20)
        http_client.post(
            "/api/kv/store",
            content=_store_stream("m", tokens, tensor),
            headers={"Content-Type": STREAM_MEDIA_TYPE},
        )
        lookup = http_client.post(
            "/api/kv/lookup",
            json={"model_name": "m", "tokens": tokens},
        )
        assert lookup.status_code == 200
        assert lookup.json()["hit_chunks"] == 2

    def test_store_reports_only_leading_complete_chunks(
        self,
        http_harness: tuple[TestClient, MPCacheEngine],
    ) -> None:
        client, engine = http_harness
        layout = _layout_for(world_size=1)
        tokens = _tokens_for(num_chunks=3)
        obj_keys = _object_keys_for(engine, "m", tokens, world_size=1)
        locked_middle_key = obj_keys[1]
        _store_empty_object(engine, locked_middle_key, layout)
        handle = engine.storage_manager.submit_prefetch_task(
            [locked_middle_key],
            layout,
            extra_count=0,
            external_request_id="test-lock-middle",
        )
        assert engine.storage_manager.query_prefetch_status(handle) == 1

        try:
            tensor = _make_tensor(num_chunks=3, world_size=1, seed=6)
            store = client.post(
                "/api/kv/store",
                content=_store_stream("m", tokens, tensor),
                headers={"Content-Type": STREAM_MEDIA_TYPE},
            )
            assert store.status_code == 200, store.text
            assert store.json()["stored_chunks"] == 1
        finally:
            engine.storage_manager.finish_read_prefetched([locked_middle_key])

    def test_retrieve_releases_partial_shard_hit_on_miss(
        self,
        http_harness: tuple[TestClient, MPCacheEngine],
    ) -> None:
        client, engine = http_harness
        layout = _layout_for(world_size=2)
        tokens = _tokens_for(num_chunks=1)
        partial_key = _object_keys_for(engine, "other", tokens, world_size=2)[0]
        _store_empty_object(engine, partial_key, layout)

        retrieve = client.post(
            "/api/kv/retrieve",
            json={"model_name": "other", "tokens": tokens},
        )
        assert retrieve.status_code == 200
        hit_tokens, _, _ = _tensor_from_retrieve_stream(retrieve.content)
        assert hit_tokens == 0

        reserved = engine.storage_manager.reserve_write(
            [partial_key],
            layout,
            mode="update",
        )
        assert partial_key in reserved
        engine.storage_manager.finish_write([partial_key])

    def test_retrieve_releases_partial_shard_hit_after_prefix(
        self,
        http_harness: tuple[TestClient, MPCacheEngine],
    ) -> None:
        client, engine = http_harness
        layout = _layout_for(world_size=2)
        tokens = _tokens_for(num_chunks=2)
        chunk0_tokens = tokens[:CHUNK_SIZE]
        chunk0_tensor = _make_tensor(num_chunks=1, world_size=2, seed=7)
        store = client.post(
            "/api/kv/store",
            content=_store_stream("other", chunk0_tokens, chunk0_tensor),
            headers={"Content-Type": STREAM_MEDIA_TYPE},
        )
        assert store.status_code == 200, store.text

        obj_keys = _object_keys_for(engine, "other", tokens, world_size=2)
        partial_next_chunk_key = obj_keys[2]
        _store_empty_object(engine, partial_next_chunk_key, layout)

        retrieve = client.post(
            "/api/kv/retrieve",
            json={"model_name": "other", "tokens": tokens},
        )
        assert retrieve.status_code == 200
        _, hit_chunks, retrieved = _tensor_from_retrieve_stream(retrieve.content)
        assert hit_chunks == 1
        assert torch.equal(retrieved, chunk0_tensor)

        reserved = engine.storage_manager.reserve_write(
            [partial_next_chunk_key],
            layout,
            mode="update",
        )
        assert partial_next_chunk_key in reserved
        engine.storage_manager.finish_write([partial_next_chunk_key])

    def test_multi_group_store_rejection_does_not_leave_write_lock(self) -> None:
        single_group_layout = _layout_for(world_size=1)
        multi_group_layout = MemoryLayoutDesc(
            shapes=[
                single_group_layout.shapes[0],
                torch.Size((2, 1, CHUNK_SIZE, 4)),
            ],
            dtypes=[DTYPE, DTYPE],
        )
        client, engine = _make_http_harness({"m": (multi_group_layout, 1)})
        tokens = _tokens_for(num_chunks=1)
        tensor = _make_tensor(num_chunks=1, world_size=1, seed=8)

        rejected = client.post(
            "/api/kv/store",
            content=_store_stream("m", tokens, tensor),
            headers={"Content-Type": STREAM_MEDIA_TYPE},
        )
        assert rejected.status_code == 400
        assert "single KV layer group" in rejected.text

        _register_fake_layouts(engine, {"m": (single_group_layout, 1)})
        stored = client.post(
            "/api/kv/store",
            content=_store_stream("m", tokens, tensor),
            headers={"Content-Type": STREAM_MEDIA_TYPE},
        )
        assert stored.status_code == 200, stored.text

    def test_unknown_model_returns_400(self, http_client: TestClient) -> None:
        retrieve = http_client.post(
            "/api/kv/retrieve",
            json={"model_name": "ghost", "tokens": _tokens_for(1)},
        )
        assert retrieve.status_code == 400

    @pytest.mark.parametrize("path", ["/api/kv/retrieve", "/api/kv/lookup"])
    def test_invalid_cache_salt_returns_400(
        self,
        http_client: TestClient,
        path: str,
    ) -> None:
        r = http_client.post(
            path,
            json={
                "model_name": "m",
                "tokens": _tokens_for(1),
                "cache_salt": "bad/salt",
            },
        )
        assert r.status_code == 400
        assert "cache_salt" in r.text

    def test_engine_not_initialized_returns_503(self) -> None:
        app = FastAPI()
        app.include_router(kv_router)
        client = TestClient(app)
        r = client.post(
            "/api/kv/retrieve",
            json={"model_name": "m", "tokens": _tokens_for(1)},
        )
        assert r.status_code == 503

    def test_store_missing_manifest_returns_400(self, http_client: TestClient) -> None:
        r = http_client.post("/api/kv/store", content=b"")
        assert r.status_code == 400

    def test_store_out_of_order_chunk_returns_400(
        self,
        http_client: TestClient,
    ) -> None:
        tokens = _tokens_for(num_chunks=1)
        tensor = _make_tensor(num_chunks=1, world_size=1)
        shape = tuple(int(dim) for dim in tensor.shape)
        body = b"".join(
            [
                encode_store_manifest(
                    StoreManifest(
                        model_name="m",
                        tokens=tokens,
                        cache_salt="",
                        shape=(shape[0], shape[1], shape[2], shape[3]),
                        dtype=str(tensor.dtype),
                    )
                ),
                encode_store_chunk(1, _chunk_payloads(tensor)[0]),
            ]
        )
        r = http_client.post(
            "/api/kv/store",
            content=body,
            headers={"Content-Type": STREAM_MEDIA_TYPE},
        )
        assert r.status_code == 400
        assert "expected store chunk" in r.text

    def test_store_malformed_stream_returns_400(
        self,
        http_client: TestClient,
    ) -> None:
        r = http_client.post("/api/kv/store", content=b"not a kv stream")
        assert r.status_code == 400
