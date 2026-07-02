# SPDX-License-Identifier: Apache-2.0
"""HTTP-level tests for the MP server's token-based pin endpoints.

Covers ``POST /cache/pins`` and ``DELETE /cache/pins`` (``PinService``). The
handlers are thin over the service resolved from the app context, so these
inject a fake engine via ``build_context`` and exercise the HTTP layer without a
real cache engine. Mirrors the prefetch fakes in ``test_cache_api.py``.
"""

# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.multiprocess.http_apis.cache_api import router as cache_router
from lmcache.v1.multiprocess.http_apis.dependencies import build_context
from lmcache.v1.multiprocess.http_apis.error_handlers import register_error_handlers


@dataclass
class _FakeLayoutRegistry:
    layout: Optional[object] = None

    def find(self, model_name: str, world_size: int) -> Optional[object]:
        return self.layout


@dataclass
class _FakeTokenHasher:
    chunk_size: int = 4

    def compute_chunk_hashes(self, token_ids: list[int]) -> list[bytes]:
        n = len(token_ids) // self.chunk_size
        return [i.to_bytes(4, "big") for i in range(n)]


@dataclass
class _FakePinStorageManager:
    pin_calls: list[list] = field(default_factory=list)
    unpin_calls: list[list] = field(default_factory=list)

    def pin_l1_keys(self, keys: list) -> int:
        self.pin_calls.append(list(keys))
        return len(keys)

    def unpin_l1_keys(self, keys: list) -> int:
        self.unpin_calls.append(list(keys))
        return len(keys)


@dataclass
class _FakeContext:
    layout_desc_registry: _FakeLayoutRegistry
    storage_manager: _FakePinStorageManager
    token_hasher: _FakeTokenHasher = field(default_factory=_FakeTokenHasher)


class _PinEngine:
    def __init__(self, ctx: _FakeContext):
        self.context = ctx
        self.storage_manager = ctx.storage_manager


def _make_app(ctx: Optional[_FakeContext]) -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    register_error_handlers(app)
    if ctx is not None:
        app.state.context = build_context(_PinEngine(ctx))
    return app


def _ctx(layout: Optional[object] = object()) -> _FakeContext:
    return _FakeContext(
        layout_desc_registry=_FakeLayoutRegistry(layout=layout),
        storage_manager=_FakePinStorageManager(),
    )


def _body(token_ids: list[int], world_size: int = 2, salt: str = "") -> dict:
    return {
        "model_name": "m",
        "world_size": world_size,
        "token_ids": token_ids,
        "cache_salt": salt,
    }


class TestPinEndpoint:
    def test_pin_returns_counts_and_keys(self):
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.post("/cache/pins", json=_body([1, 2, 3, 4, 5, 6, 7, 8]))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "pinned"
        assert body["requested"] == 2  # chunks
        assert body["pinned"] > 0  # resident L1 keys pinned
        # resolved_keys are returned for the coordinator's L2 pin.
        assert body["pinned"] == len(body["resolved_keys"])
        assert len(ctx.storage_manager.pin_calls) == 1
        assert body["pinned"] == len(ctx.storage_manager.pin_calls[0])

    def test_unpin_returns_counts_and_keys(self):
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.request(
            "DELETE", "/cache/pins", json=_body([1, 2, 3, 4, 5, 6, 7, 8])
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "unpinned"
        assert body["requested"] == 2  # chunks
        assert body["unpinned"] > 0
        assert body["unpinned"] == len(body["resolved_keys"])
        assert len(ctx.storage_manager.unpin_calls) == 1
        assert body["unpinned"] == len(ctx.storage_manager.unpin_calls[0])

    def test_short_sequence_is_noop(self):
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.post("/cache/pins", json=_body([1, 2]))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 0,
            "pinned": 0,
            "resolved_keys": [],
            "status": "noop",
        }
        assert ctx.storage_manager.pin_calls == []

    def test_l2_tier_skips_l1_pin_but_returns_keys(self):
        """tier=l2: L1 is not pinned (pinned=0), but resolved_keys are still
        returned so the coordinator can pin L2."""
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        body = _body([1, 2, 3, 4, 5, 6, 7, 8])
        body["tier"] = "l2"
        resp = client.post("/cache/pins", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["pinned"] == 0
        assert len(data["resolved_keys"]) > 0
        assert ctx.storage_manager.pin_calls == []  # pin_l1_keys never called

    def test_l1_tier_pins_l1(self):
        """tier=l1: L1 is pinned."""
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        body = _body([1, 2, 3, 4, 5, 6, 7, 8])
        body["tier"] = "l1"
        resp = client.post("/cache/pins", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json()["pinned"] > 0
        assert len(ctx.storage_manager.pin_calls) == 1

    def test_422_on_invalid_tier(self):
        client = TestClient(_make_app(_ctx()))
        body = _body([1, 2, 3, 4])
        body["tier"] = "l3"  # not a Tier value -> FastAPI validation rejects it
        assert client.post("/cache/pins", json=body).status_code == 422

    def test_400_on_invalid_cache_salt(self):
        client = TestClient(_make_app(_ctx()))
        resp = client.post("/cache/pins", json=_body([1, 2, 3, 4], salt="bad@salt"))
        assert resp.status_code == 400

    def test_503_when_layout_not_registered(self):
        client = TestClient(_make_app(_ctx(layout=None)))
        resp = client.post("/cache/pins", json=_body([1, 2, 3, 4], world_size=99))
        assert resp.status_code == 503

    def test_503_when_not_initialized(self):
        client = TestClient(_make_app(None))
        assert client.post("/cache/pins", json=_body([1, 2, 3, 4])).status_code == 503
