# SPDX-License-Identifier: Apache-2.0
"""Tests for the /api/kvcache/check endpoint on the MP HTTP server."""

# Standard
from unittest.mock import MagicMock, PropertyMock

# Third Party
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.http_server import app


def _make_kv_tensors(
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Create deterministic CPU KV tensors for testing."""
    torch.manual_seed(42)
    return [
        torch.randn(
            2,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]


@pytest.fixture
def mock_gpu_ctx():
    """Create a mock GPUCacheContext with kv_tensors."""
    ctx = MagicMock()
    type(ctx).kv_tensors = PropertyMock(
        return_value=_make_kv_tensors(),
    )
    return ctx


@pytest.fixture
def mock_engine(mock_gpu_ctx):
    """Create a mock engine with gpu_contexts."""
    engine = MagicMock()
    engine.gpu_contexts = {0: mock_gpu_ctx}
    return engine


@pytest.fixture
def client_with_engine(mock_engine):
    """Create a test client with mocked engine."""
    app.state.engine = mock_engine
    client = TestClient(app)
    yield client
    client.close()
    app.state.engine = None


@pytest.fixture
def client_no_engine():
    """Create a test client without engine."""
    app.state.engine = None
    client = TestClient(app)
    yield client
    client.close()


class TestKVCacheCheckEndpoint:
    """Tests for GET /api/kvcache/check."""

    def test_success_basic(self, client_with_engine):
        """Basic successful checksum request."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["chunk_size"] == 2
        assert data["num_chunks"] == 2
        assert len(data["chunk_checksums"]) == 2
        assert "slot_mapping_ranges" in data

    def test_success_layerwise(self, client_with_engine):
        """Layerwise mode returns per-layer checksums."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1,2,3&chunk_size=4&layerwise=true"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["layerwise"] is True
        cks = data["chunk_checksums"]
        assert "layer_0" in cks
        assert "layer_1" in cks

    def test_deterministic(self, client_with_engine):
        """Same request produces identical checksums."""
        url = "/api/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        d1 = client_with_engine.get(url).json()
        d2 = client_with_engine.get(url).json()
        assert d1["chunk_checksums"] == d2["chunk_checksums"]

    def test_range_slot_mapping(self, client_with_engine):
        """Range format [0,3] is accepted."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=[0,3]&chunk_size=2"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_chunks"] == 2

    def test_mixed_slot_mapping(self, client_with_engine):
        """Mixed format 0,1,[2,3] is accepted."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1,[2,3]&chunk_size=4"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_chunks"] == 1

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_no_engine(self, client_no_engine):
        """503 when engine is not initialized."""
        resp = client_no_engine.get("/api/kvcache/check?slot_mapping=0,1&chunk_size=2")
        assert resp.status_code == 503

    def test_no_gpu_contexts(self, client_with_engine, mock_engine):
        """501 when engine has no gpu_contexts attribute."""
        mock_engine.gpu_contexts = None
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1&chunk_size=2"
        )
        assert resp.status_code == 501

    def test_unknown_instance_id(self, client_with_engine):
        """404 when instance_id is not registered."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1&chunk_size=2&instance_id=99"
        )
        assert resp.status_code == 404

    def test_missing_slot_mapping(self, client_with_engine):
        """400 when slot_mapping is missing."""
        resp = client_with_engine.get("/api/kvcache/check?chunk_size=2")
        assert resp.status_code == 400
        assert "slot_mapping" in resp.json()["error"]

    def test_invalid_slot_mapping(self, client_with_engine):
        """400 when slot_mapping has invalid format."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=abc&chunk_size=2"
        )
        assert resp.status_code == 400
        assert "Invalid" in resp.json()["error"]

    def test_missing_chunk_size(self, client_with_engine):
        """400 when chunk_size is missing."""
        resp = client_with_engine.get("/api/kvcache/check?slot_mapping=0,1")
        assert resp.status_code == 400
        assert "chunk_size" in resp.json()["error"]

    def test_zero_chunk_size(self, client_with_engine):
        """400 when chunk_size is zero."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1&chunk_size=0"
        )
        assert resp.status_code == 400

    def test_negative_chunk_size(self, client_with_engine):
        """400 when chunk_size is negative."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1&chunk_size=-1"
        )
        assert resp.status_code == 400

    def test_empty_kv_caches(self, client_with_engine, mock_gpu_ctx):
        """404 when kv_tensors is empty."""
        type(mock_gpu_ctx).kv_tensors = PropertyMock(
            return_value=[],
        )
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1&chunk_size=2"
        )
        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # Chunk boundary edge cases
    # ------------------------------------------------------------------

    def test_partial_last_chunk(self, client_with_engine):
        """3 slots with chunk_size=2 -> 2 chunks."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1,2&chunk_size=2"
        )
        data = resp.json()
        assert data["num_chunks"] == 2
        assert len(data["chunk_checksums"]) == 2

    def test_single_slot_single_chunk(self, client_with_engine):
        """Single slot produces one chunk."""
        resp = client_with_engine.get("/api/kvcache/check?slot_mapping=0&chunk_size=1")
        data = resp.json()
        assert data["num_chunks"] == 1

    # ------------------------------------------------------------------
    # Checksum validity
    # ------------------------------------------------------------------

    def test_checksums_are_valid_md5(self, client_with_engine):
        """All checksums are 32-char hex strings."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1&chunk_size=1"
        )
        data = resp.json()
        for cksum in data["chunk_checksums"]:
            assert len(cksum) == 32
            int(cksum, 16)  # must be valid hex

    def test_slot_mapping_ranges_in_response(
        self,
        client_with_engine,
    ):
        """Response includes compressed slot_mapping_ranges."""
        resp = client_with_engine.get(
            "/api/kvcache/check?slot_mapping=0,1,2,3&chunk_size=4"
        )
        data = resp.json()
        assert "slot_mapping_ranges" in data
        # 4 consecutive slots -> compressed range
        assert data["slot_mapping_ranges"] == [[0, 3]]


class TestHealthAndMiscEndpoints:
    """Smoke tests for other endpoints on the MP HTTP server."""

    @pytest.fixture(autouse=True)
    def _reset_engine(self):
        yield
        app.state.engine = None

    def test_root(self, client_no_engine):
        """GET / returns ok."""
        resp = client_no_engine.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_healthcheck_no_engine(self, client_no_engine):
        """503 when engine is not set."""
        resp = client_no_engine.get("/api/healthcheck")
        assert resp.status_code == 503

    def test_healthcheck_with_engine(
        self,
        client_with_engine,
    ):
        """200 when engine is available."""
        resp = client_with_engine.get("/api/healthcheck")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_clear_cache_no_engine(self, client_no_engine):
        """503 when engine is not set."""
        resp = client_no_engine.post("/api/clear-cache")
        assert resp.status_code == 503

    def test_clear_cache_success(
        self,
        client_with_engine,
        mock_engine,
    ):
        """200 and engine.clear() called."""
        resp = client_with_engine.post("/api/clear-cache")
        assert resp.status_code == 200
        mock_engine.clear.assert_called_once()

    def test_status_no_engine(self, client_no_engine):
        """503 when engine is not set."""
        resp = client_no_engine.get("/api/status")
        assert resp.status_code == 503

    def test_status_success(
        self,
        client_with_engine,
        mock_engine,
    ):
        """200 and engine.report_status() called."""
        mock_engine.report_status.return_value = {"ok": True}
        resp = client_with_engine.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
