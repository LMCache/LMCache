# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the bytes-level KV cache HTTP API.

The tests exercise both layers in one file:

- ``TestStoreRetrieveLookupBytes`` drives ``MPCacheEngine.store_bytes /
  retrieve_bytes / lookup_bytes`` directly. ``MPCacheEngine`` is built
  in-process with a small CPU L1 storage manager, and ``_resolve_model``
  is monkey-patched to return a synthetic layout — so the tests do not
  require CUDA or a registered GPU context.
- ``TestKVApiHTTP`` mounts ``kv_api.py`` on a FastAPI test client and
  verifies the HTTP envelope, headers, and error paths.

Wire format invariant exercised by every round-trip test: the payload is
canonical KV_2LTD ``[2, num_layers, num_tokens, hidden_dim]`` with
all-TP-shards aggregated along the hidden dim and all chunks concatenated
along the token dim.
"""

# Standard
import ctypes

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.multiprocess.http_apis.kv_api import router as kv_router
from lmcache.v1.multiprocess.server import MPCacheEngine

CHUNK_SIZE = 16
NUM_LAYERS = 2
HIDDEN_DIM_PER_WORKER = 8
DTYPE = torch.float32  # picked for byte-level comparison ergonomics


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
    """Replace ``engine._resolve_model`` with a static map for tests.

    ``models`` maps model_name → world_size. All models share the same
    per-shard shape so the aggregated full hidden dim equals
    ``HIDDEN_DIM_PER_WORKER * world_size``.
    """

    def resolver(model_name: str) -> tuple[MemoryLayoutDesc, int]:
        if model_name not in models:
            raise KeyError(model_name)
        return _layout_for(models[model_name]), models[model_name]

    engine._resolve_model = resolver  # type: ignore[method-assign]


def _make_payload(num_chunks: int, world_size: int, seed: int = 0) -> bytes:
    """Generate a deterministic random KV_2LTD payload as bytes."""
    torch.manual_seed(seed)
    full_hidden = HIDDEN_DIM_PER_WORKER * world_size
    t = torch.randn(
        (2, NUM_LAYERS, num_chunks * CHUNK_SIZE, full_hidden),
        dtype=DTYPE,
    )
    nbytes = t.numel() * t.element_size()
    return bytes((ctypes.c_ubyte * nbytes).from_address(t.contiguous().data_ptr()))


def _tokens_for(num_chunks: int, seed: int = 0) -> list[int]:
    """Tokens covering ``num_chunks`` whole chunks (plus a stable trailing partial)."""
    return list(range(seed, seed + num_chunks * CHUNK_SIZE + 3))


class TestStoreRetrieveLookupBytes:
    """Direct ``MPCacheEngine`` round-trip tests for the bytes API."""

    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_round_trip_byte_identity(self, world_size: int) -> None:
        """Store then retrieve must return byte-identical payload."""
        engine = _make_engine()
        _install_resolver(engine, {"m": world_size})

        tokens = _tokens_for(num_chunks=3)
        payload = _make_payload(num_chunks=3, world_size=world_size, seed=1)

        store_result = engine.store_bytes("m", tokens, payload)
        assert store_result.total_chunks == 3
        assert store_result.stored_chunks == 3
        assert store_result.stored_tokens == 3 * CHUNK_SIZE

        retrieve_result = engine.retrieve_bytes("m", tokens)
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
        engine.store_bytes("m", store_tokens, store_payload)

        # The query starts with the same 2 chunks of tokens, then extends.
        full_tokens = list(store_tokens[: 2 * CHUNK_SIZE]) + list(range(99_000, 99_032))
        result = engine.retrieve_bytes("m", full_tokens)
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
        engine.store_bytes("m", tokens, payload)

        first = engine.retrieve_bytes("m", tokens)
        second = engine.retrieve_bytes("m", tokens)
        assert first.payload == payload
        assert second.payload == payload
        assert first.hit_chunks == 2
        assert second.hit_chunks == 2

    def test_lookup_matches_retrieve(self) -> None:
        """``lookup_bytes`` reports the same hit count as ``retrieve_bytes``."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})

        tokens = _tokens_for(num_chunks=2)
        payload = _make_payload(num_chunks=2, world_size=1, seed=4)
        engine.store_bytes("m", tokens, payload)

        full_tokens = list(tokens[: 2 * CHUNK_SIZE]) + list(range(80_000, 80_032))
        lookup = engine.lookup_bytes("m", full_tokens)
        assert lookup.total_chunks == 4
        assert lookup.hit_chunks == 2

        retrieve = engine.retrieve_bytes("m", full_tokens)
        assert retrieve.hit_chunks == lookup.hit_chunks

    def test_multi_model_isolation(self) -> None:
        """Storing under model A must not surface under model B."""
        engine = _make_engine()
        _install_resolver(engine, {"a": 1, "b": 1})

        tokens = _tokens_for(num_chunks=2)
        payload = _make_payload(num_chunks=2, world_size=1, seed=5)
        engine.store_bytes("a", tokens, payload)

        # Same tokens, different model — should be a clean miss.
        b_result = engine.retrieve_bytes("b", tokens)
        assert b_result.hit_chunks == 0
        assert b_result.payload == b""

        # Original model still hits.
        a_result = engine.retrieve_bytes("a", tokens)
        assert a_result.payload == payload

    def test_unknown_model_raises(self) -> None:
        """``KeyError`` propagates so the HTTP layer can map it to 400."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        with pytest.raises(KeyError):
            engine.store_bytes("nope", _tokens_for(1), _make_payload(1, 1))
        with pytest.raises(KeyError):
            engine.retrieve_bytes("nope", _tokens_for(1))
        with pytest.raises(KeyError):
            engine.lookup_bytes("nope", _tokens_for(1))

    def test_payload_length_mismatch_rejected(self) -> None:
        """``store_bytes`` rejects payloads that don't match the expected layout."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})
        tokens = _tokens_for(num_chunks=2)
        payload = _make_payload(num_chunks=2, world_size=1)
        # Truncate by one byte — must raise.
        with pytest.raises(ValueError, match="payload length"):
            engine.store_bytes("m", tokens, payload[:-1])

    def test_no_complete_chunks_returns_zero(self) -> None:
        """Token sequences shorter than one chunk produce empty results."""
        engine = _make_engine()
        _install_resolver(engine, {"m": 1})

        # Below chunk_size — no whole chunks to hash.
        tokens = list(range(CHUNK_SIZE - 1))
        store_result = engine.store_bytes("m", tokens, b"")
        assert store_result.stored_chunks == 0

        retrieve_result = engine.retrieve_bytes("m", tokens)
        assert retrieve_result.hit_chunks == 0
        assert retrieve_result.payload == b""

        lookup_result = engine.lookup_bytes("m", tokens)
        assert lookup_result.hit_chunks == 0


@pytest.fixture
def http_client() -> TestClient:
    """FastAPI test client wired to a real CPU engine with stubbed resolver."""
    engine = _make_engine()
    _install_resolver(engine, {"m": 1, "other": 2})

    app = FastAPI()
    app.state.engine = engine
    app.include_router(kv_router)
    return TestClient(app)


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

    def test_unknown_model_returns_400(self, http_client: TestClient) -> None:
        retrieve = http_client.post(
            "/api/kv/retrieve",
            json={"model_name": "ghost", "tokens": _tokens_for(1)},
        )
        assert retrieve.status_code == 400

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
