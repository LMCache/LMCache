# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the bytes-level KV cache HTTP API.

The tests exercise both layers in one file:

- ``TestStoreRetrieveLookupBytes`` drives the ``MPCacheEngine``
  ``*_bytes_by_tokens`` methods directly. ``MPCacheEngine`` is built
  in-process with a small CPU L1 storage manager and fake registered GPU
  contexts, so the tests do not require CUDA.
- ``TestKVApiHTTP`` mounts ``kv_api.py`` on a FastAPI test client and
  verifies the HTTP envelope, headers, error paths, and regression cases
  through ``TestClient.post`` calls.

Wire format invariant exercised by every round-trip test: the payload is
canonical KV_2LTD ``[2, num_layers, num_tokens, hidden_dim]`` with
all-TP-shards aggregated along the hidden dim and all chunks concatenated
along the token dim.
"""

# Standard
from typing import cast

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
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.http_apis.kv_api import router as kv_router
from lmcache.v1.multiprocess.server import MPCacheEngine

CHUNK_SIZE = 16
NUM_LAYERS = 2
HIDDEN_DIM_PER_WORKER = 8
DTYPE = torch.float32  # picked for byte-level comparison ergonomics


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
    """Register fake GPU contexts for models under test.

    ``models`` maps model_name → world_size. All models share the same
    per-shard shape so the aggregated full hidden dim equals
    ``HIDDEN_DIM_PER_WORKER * world_size``.
    """

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


def _make_payload(num_chunks: int, world_size: int, seed: int = 0) -> bytes:
    """Generate a deterministic random KV_2LTD payload as bytes."""
    torch.manual_seed(seed)
    full_hidden = HIDDEN_DIM_PER_WORKER * world_size
    t = torch.randn(
        (2, NUM_LAYERS, num_chunks * CHUNK_SIZE, full_hidden),
        dtype=DTYPE,
    )
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _tokens_for(num_chunks: int, seed: int = 0) -> list[int]:
    """Tokens covering ``num_chunks`` whole chunks (plus a stable trailing partial)."""
    return list(range(seed, seed + num_chunks * CHUNK_SIZE + 3))


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


class TestStoreRetrieveLookupBytes:
    """Direct ``MPCacheEngine`` round-trip tests for the bytes API."""

    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_round_trip_byte_identity(self, world_size: int) -> None:
        """Store then retrieve must return byte-identical payload."""
        engine = _make_engine()
        _install_resolver(engine, {"m": world_size})

        tokens = _tokens_for(num_chunks=3)
        payload = _make_payload(num_chunks=3, world_size=world_size, seed=1)

        store_result = engine.store_kv_bytes_by_tokens("m", tokens, payload)
        assert store_result.total_chunks == 3
        assert store_result.stored_chunks == 3
        assert store_result.stored_tokens == 3 * CHUNK_SIZE

        retrieve_result = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        assert retrieve_result.hit_chunks == 3
        assert retrieve_result.hit_tokens == 3 * CHUNK_SIZE
        assert retrieve_result.payload == payload

    def test_partial_hit(self) -> None:
        """Retrieve with a longer token sequence returns only the cached prefix."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})

        # Store 2 chunks worth, query 4 chunks worth — only the first 2 hit.
        store_tokens = _tokens_for(num_chunks=2)
        store_payload = _make_payload(num_chunks=2, world_size=1, seed=2)
        engine.store_kv_bytes_by_tokens("m", store_tokens, store_payload)

        # The query starts with the same 2 chunks of tokens, then extends.
        full_tokens = list(store_tokens[: 2 * CHUNK_SIZE]) + list(range(99_000, 99_032))
        result = engine.retrieve_kv_bytes_by_tokens("m", full_tokens)
        assert result.total_chunks == 4
        assert result.hit_chunks == 2
        assert result.hit_tokens == 2 * CHUNK_SIZE
        assert result.payload == store_payload

    def test_retrieve_is_non_destructive(self) -> None:
        """Two retrievals in a row both succeed; the cache is not consumed."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 2})

        tokens = _tokens_for(num_chunks=2)
        payload = _make_payload(num_chunks=2, world_size=2, seed=3)
        engine.store_kv_bytes_by_tokens("m", tokens, payload)

        first = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        second = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        assert first.payload == payload
        assert second.payload == payload
        assert first.hit_chunks == 2
        assert second.hit_chunks == 2

    def test_lookup_matches_retrieve(self) -> None:
        """``lookup_*`` reports the same hit count as ``retrieve_*``."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})

        tokens = _tokens_for(num_chunks=2)
        payload = _make_payload(num_chunks=2, world_size=1, seed=4)
        engine.store_kv_bytes_by_tokens("m", tokens, payload)

        full_tokens = list(tokens[: 2 * CHUNK_SIZE]) + list(range(80_000, 80_032))
        lookup = engine.lookup_kv_bytes_by_tokens("m", full_tokens)
        assert lookup.total_chunks == 4
        assert lookup.hit_chunks == 2

        retrieve = engine.retrieve_kv_bytes_by_tokens("m", full_tokens)
        assert retrieve.hit_chunks == lookup.hit_chunks

    def test_multi_model_isolation(self) -> None:
        """Storing under model A must not surface under model B."""
        engine = _make_engine()
        _install_resolver(engine, {"a": 1, "b": 1})

        tokens = _tokens_for(num_chunks=2)
        payload = _make_payload(num_chunks=2, world_size=1, seed=5)
        engine.store_kv_bytes_by_tokens("a", tokens, payload)

        # Same tokens, different model — should be a clean miss.
        b_result = engine.retrieve_kv_bytes_by_tokens("b", tokens)
        assert b_result.hit_chunks == 0
        assert b_result.payload == b""

        # Original model still hits.
        a_result = engine.retrieve_kv_bytes_by_tokens("a", tokens)
        assert a_result.payload == payload

    def test_unknown_model_raises(self) -> None:
        """``KeyError`` propagates so the HTTP layer can map it to 400."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        with pytest.raises(KeyError):
            engine.store_kv_bytes_by_tokens("nope", _tokens_for(1), _make_payload(1, 1))
        with pytest.raises(KeyError):
            engine.retrieve_kv_bytes_by_tokens("nope", _tokens_for(1))
        with pytest.raises(KeyError):
            engine.lookup_kv_bytes_by_tokens("nope", _tokens_for(1))

    def test_payload_length_mismatch_rejected(self) -> None:
        """Store rejects payloads that don't match the expected layout."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        tokens = _tokens_for(num_chunks=2)
        payload = _make_payload(num_chunks=2, world_size=1)
        # Truncate by one byte — must raise.
        with pytest.raises(ValueError, match="payload length"):
            engine.store_kv_bytes_by_tokens("m", tokens, payload[:-1])

    def test_no_complete_chunks_returns_zero(self) -> None:
        """Token sequences shorter than one chunk produce empty results."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})

        # Below chunk_size — no whole chunks to hash.
        tokens = list(range(CHUNK_SIZE - 1))
        store_result = engine.store_kv_bytes_by_tokens("m", tokens, b"")
        assert store_result.stored_chunks == 0

        retrieve_result = engine.retrieve_kv_bytes_by_tokens("m", tokens)
        assert retrieve_result.hit_chunks == 0
        assert retrieve_result.payload == b""

        lookup_result = engine.lookup_kv_bytes_by_tokens("m", tokens)
        assert lookup_result.hit_chunks == 0


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
        payload = _make_payload(num_chunks=2, world_size=1, seed=10)

        store = http_client.post(
            "/api/kv/store",
            content=payload,
            headers={
                "X-LMCache-Model-Name": "m",
                "X-LMCache-Tokens": ",".join(str(t) for t in tokens),
                "Content-Type": "application/x-lmcache-kv; v=1",
            },
        )
        assert store.status_code == 200, store.text
        body = store.json()
        assert body["status"] == "ok"
        assert body["stored_chunks"] == 2

        retrieve = http_client.post(
            "/api/kv/retrieve",
            json={"model_name": "m", "tokens": tokens},
        )
        assert retrieve.status_code == 200
        assert retrieve.headers["X-LMCache-Hit-Tokens"] == str(2 * CHUNK_SIZE)
        assert retrieve.headers["X-LMCache-Hit-Chunks"] == "2"
        assert retrieve.headers["X-LMCache-Total-Chunks"] == "2"
        assert retrieve.content == payload

    def test_retrieve_miss_returns_404_with_headers(
        self, http_client: TestClient
    ) -> None:
        # Nothing has been stored; retrieve must miss cleanly.
        tokens = _tokens_for(num_chunks=2, seed=999)
        retrieve = http_client.post(
            "/api/kv/retrieve",
            json={"model_name": "m", "tokens": tokens},
        )
        assert retrieve.status_code == 404
        assert retrieve.headers["X-LMCache-Hit-Tokens"] == "0"
        assert retrieve.headers["X-LMCache-Total-Chunks"] == "2"
        assert retrieve.content == b""

    def test_lookup_returns_hit_metadata(self, http_client: TestClient) -> None:
        tokens = _tokens_for(num_chunks=2, seed=20)
        payload = _make_payload(num_chunks=2, world_size=1, seed=20)
        http_client.post(
            "/api/kv/store",
            content=payload,
            headers={
                "X-LMCache-Model-Name": "m",
                "X-LMCache-Tokens": ",".join(str(t) for t in tokens),
            },
        )
        lookup = http_client.post(
            "/api/kv/lookup",
            json={"model_name": "m", "tokens": tokens},
        )
        assert lookup.status_code == 200
        body = lookup.json()
        assert body["hit_chunks"] == 2
        assert body["total_chunks"] == 2

    def test_store_reports_only_leading_complete_chunks(
        self,
        http_harness: tuple[TestClient, MPCacheEngine],
    ) -> None:
        """A sparse reservation success must not overstate the stored prefix."""
        client, engine = http_harness
        layout = _layout_for(world_size=1)
        tokens = _tokens_for(num_chunks=3)
        obj_keys = _object_keys_for(engine, "m", tokens, world_size=1)

        # Pre-create chunk 1 and keep a read lock on it. An HTTP store of
        # chunks 0..2 can update chunks 0 and 2, but chunk 1 is not writable.
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
            payload = _make_payload(num_chunks=3, world_size=1, seed=6)
            store = client.post(
                "/api/kv/store",
                content=payload,
                headers={
                    "X-LMCache-Model-Name": "m",
                    "X-LMCache-Tokens": ",".join(str(t) for t in tokens),
                },
            )
            assert store.status_code == 200, store.text
            body = store.json()
            assert body["total_chunks"] == 3
            assert body["stored_chunks"] == 1
            assert body["stored_tokens"] == CHUNK_SIZE
        finally:
            engine.storage_manager.finish_read_prefetched([locked_middle_key])

    def test_retrieve_releases_partial_shard_hit_on_miss(
        self,
        http_harness: tuple[TestClient, MPCacheEngine],
    ) -> None:
        """A sub-chunk TP hit that returns 404 must release its read lock."""
        client, engine = http_harness
        layout = _layout_for(world_size=2)
        tokens = _tokens_for(num_chunks=1)
        partial_key = _object_keys_for(engine, "other", tokens, world_size=2)[0]
        _store_empty_object(engine, partial_key, layout)

        retrieve = client.post(
            "/api/kv/retrieve",
            json={"model_name": "other", "tokens": tokens},
        )
        assert retrieve.status_code == 404
        assert retrieve.content == b""

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
        """A partial next chunk must be unlocked even when an earlier chunk hits."""
        client, engine = http_harness
        layout = _layout_for(world_size=2)
        tokens = _tokens_for(num_chunks=2)
        chunk0_tokens = tokens[:CHUNK_SIZE]
        chunk0_payload = _make_payload(num_chunks=1, world_size=2, seed=7)

        store = client.post(
            "/api/kv/store",
            content=chunk0_payload,
            headers={
                "X-LMCache-Model-Name": "other",
                "X-LMCache-Tokens": ",".join(str(t) for t in chunk0_tokens),
            },
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
        assert retrieve.headers["X-LMCache-Hit-Chunks"] == "1"
        assert retrieve.content == chunk0_payload

        reserved = engine.storage_manager.reserve_write(
            [partial_next_chunk_key],
            layout,
            mode="update",
        )
        assert partial_next_chunk_key in reserved
        engine.storage_manager.finish_write([partial_next_chunk_key])

    def test_multi_group_store_rejection_does_not_leave_write_lock(self) -> None:
        """Unsupported multi-group layouts should fail before reserving writes."""
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
        payload = _make_payload(num_chunks=1, world_size=1, seed=8)
        rejected = client.post(
            "/api/kv/store",
            content=payload,
            headers={
                "X-LMCache-Model-Name": "m",
                "X-LMCache-Tokens": ",".join(str(t) for t in tokens),
            },
        )
        assert rejected.status_code == 400
        assert "single KV layer group" in rejected.text

        _register_fake_layouts(engine, {"m": (single_group_layout, 1)})
        stored = client.post(
            "/api/kv/store",
            content=payload,
            headers={
                "X-LMCache-Model-Name": "m",
                "X-LMCache-Tokens": ",".join(str(t) for t in tokens),
            },
        )
        assert stored.status_code == 200, stored.text
        assert stored.json()["stored_chunks"] == 1

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
        """If the engine is missing from app.state, the route returns 503."""
        app = FastAPI()
        app.include_router(kv_router)
        client = TestClient(app)
        r = client.post(
            "/api/kv/retrieve",
            json={"model_name": "m", "tokens": _tokens_for(1)},
        )
        assert r.status_code == 503

    def test_store_missing_headers_returns_400(self, http_client: TestClient) -> None:
        """Missing model_name / tokens headers must cleanly reject."""
        # No headers at all.
        r = http_client.post("/api/kv/store", content=b"")
        assert r.status_code == 400

        # Has model name but no tokens header.
        r = http_client.post(
            "/api/kv/store",
            content=b"",
            headers={"X-LMCache-Model-Name": "m"},
        )
        assert r.status_code == 400

    def test_store_malformed_tokens_header_returns_400(
        self, http_client: TestClient
    ) -> None:
        r = http_client.post(
            "/api/kv/store",
            content=b"",
            headers={
                "X-LMCache-Model-Name": "m",
                "X-LMCache-Tokens": "1,abc,3",
            },
        )
        assert r.status_code == 400
