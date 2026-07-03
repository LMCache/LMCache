# SPDX-License-Identifier: Apache-2.0
"""HTTP-level tests for the MP server's token-based delete endpoint.

Covers ``POST /cache/delete`` (``DeleteService``). The handler is thin over the
service resolved from the app context, so these inject a fake engine via
``build_context`` and exercise the HTTP layer without a real cache engine.
Mirrors ``test_pin_api.py``.
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
class _FakeDeleteStorageManager:
    # Records (keys, force) for each delete_l1_keys call.
    delete_calls: list[tuple[list, bool]] = field(default_factory=list)
    # Number of keys reported as skipped (locked/pinned) by the next call.
    skip: int = 0

    def delete_l1_keys(self, keys: list, force: bool = False) -> tuple[int, int]:
        self.delete_calls.append((list(keys), force))
        deleted = max(0, len(keys) - self.skip)
        return deleted, min(self.skip, len(keys))


@dataclass
class _FakeContext:
    layout_desc_registry: _FakeLayoutRegistry
    storage_manager: _FakeDeleteStorageManager
    token_hasher: _FakeTokenHasher = field(default_factory=_FakeTokenHasher)


class _DeleteEngine:
    def __init__(self, ctx: _FakeContext):
        self.context = ctx
        self.storage_manager = ctx.storage_manager


def _make_app(ctx: Optional[_FakeContext]) -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    register_error_handlers(app)
    if ctx is not None:
        app.state.context = build_context(_DeleteEngine(ctx))
    return app


def _ctx(layout: Optional[object] = object(), skip: int = 0) -> _FakeContext:
    return _FakeContext(
        layout_desc_registry=_FakeLayoutRegistry(layout=layout),
        storage_manager=_FakeDeleteStorageManager(skip=skip),
    )


def _body(
    token_ids: list[int],
    world_size: int = 2,
    salt: str = "",
    tier: Optional[str] = None,
    force: Optional[bool] = None,
) -> dict:
    body: dict = {
        "model_name": "m",
        "world_size": world_size,
        "token_ids": token_ids,
        "cache_salt": salt,
    }
    if tier is not None:
        body["tier"] = tier
    if force is not None:
        body["force"] = force
    return body


class TestDeleteEndpoint:
    def test_delete_returns_counts_and_keys(self):
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.post("/cache/delete", json=_body([1, 2, 3, 4, 5, 6, 7, 8]))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["requested"] == 2  # chunks
        assert body["deleted"] > 0  # resident L1 keys removed
        assert body["skipped"] == 0
        # resolved_keys are returned for the coordinator's L2 delete.
        assert body["deleted"] == len(body["resolved_keys"])
        assert len(ctx.storage_manager.delete_calls) == 1
        keys, force = ctx.storage_manager.delete_calls[0]
        assert body["deleted"] == len(keys)
        assert force is False  # default is non-force

    def test_short_sequence_is_noop(self):
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.post("/cache/delete", json=_body([1, 2]))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 0,
            "deleted": 0,
            "skipped": 0,
            "resolved_keys": [],
            "status": "noop",
        }
        assert ctx.storage_manager.delete_calls == []

    def test_non_force_reports_skipped(self):
        """A locked/pinned key is reported as skipped, not deleted."""
        ctx = _ctx(skip=1)
        client = TestClient(_make_app(ctx))
        resp = client.post("/cache/delete", json=_body([1, 2, 3, 4, 5, 6, 7, 8]))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        n = len(body["resolved_keys"])
        assert n > 1
        assert body["skipped"] == 1
        assert body["deleted"] == n - 1  # one key refused (locked/pinned)

    def test_force_plumbs_through(self):
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.post(
            "/cache/delete", json=_body([1, 2, 3, 4, 5, 6, 7, 8], force=True)
        )
        assert resp.status_code == 200, resp.text
        assert ctx.storage_manager.delete_calls[0][1] is True

    def test_l2_tier_skips_l1_but_returns_keys(self):
        """tier=l2: L1 is not deleted (deleted=0), but resolved_keys are still
        returned so the coordinator can delete L2."""
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.post(
            "/cache/delete", json=_body([1, 2, 3, 4, 5, 6, 7, 8], tier="l2")
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["deleted"] == 0
        assert len(data["resolved_keys"]) > 0
        assert ctx.storage_manager.delete_calls == []  # delete_l1_keys never called

    def test_l1_tier_deletes_l1(self):
        ctx = _ctx()
        client = TestClient(_make_app(ctx))
        resp = client.post(
            "/cache/delete", json=_body([1, 2, 3, 4, 5, 6, 7, 8], tier="l1")
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["deleted"] > 0
        assert len(ctx.storage_manager.delete_calls) == 1

    def test_422_on_invalid_tier(self):
        client = TestClient(_make_app(_ctx()))
        resp = client.post("/cache/delete", json=_body([1, 2, 3, 4], tier="l3"))
        assert resp.status_code == 422

    def test_400_on_invalid_cache_salt(self):
        client = TestClient(_make_app(_ctx()))
        resp = client.post("/cache/delete", json=_body([1, 2, 3, 4], salt="bad@salt"))
        assert resp.status_code == 400

    def test_503_when_layout_not_registered(self):
        client = TestClient(_make_app(_ctx(layout=None)))
        resp = client.post("/cache/delete", json=_body([1, 2, 3, 4], world_size=99))
        assert resp.status_code == 503

    def test_503_when_not_initialized(self):
        client = TestClient(_make_app(None))
        resp = client.post("/cache/delete", json=_body([1, 2, 3, 4]))
        assert resp.status_code == 503
