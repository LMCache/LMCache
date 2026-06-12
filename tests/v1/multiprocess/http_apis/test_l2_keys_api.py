# SPDX-License-Identifier: Apache-2.0
"""
HTTP-level tests for ``l2_keys_api`` — the ``POST /l2/keys:evict`` and
``GET /l2/keys`` endpoints.

The endpoints reach into ``request.app.state.engine.storage_manager``;
these tests inject a fake storage manager that records calls and
serves canned responses, so the HTTP layer can be exercised without
spinning up a real cache engine.
"""

# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2KeyEntry, L2KeyListPage
from lmcache.v1.multiprocess.http_apis.l2_keys_api import router as l2_keys_router


@dataclass
class _FakeStorageManager:
    """Records calls and serves staged responses for the endpoints."""

    evict_calls: list[list[ObjectKey]] = field(default_factory=list)
    evict_response: Optional[dict[str, object]] = None
    evict_raises: Optional[BaseException] = None

    list_page: Optional[L2KeyListPage] = None
    list_raises: Optional[BaseException] = None
    last_list_kwargs: dict[str, object] = field(default_factory=dict)

    def evict_l2_keys(self, keys: list[ObjectKey]) -> dict[str, object]:
        self.evict_calls.append(list(keys))
        if self.evict_raises is not None:
            raise self.evict_raises
        return self.evict_response or {"adapter": "s3", "ok": True}

    def list_l2_keys(
        self,
        cache_salt: Optional[str] = None,
        model_name: Optional[str] = None,
        page_size: int = 500,
        page_token: Optional[str] = None,
    ) -> L2KeyListPage:
        self.last_list_kwargs = {
            "cache_salt": cache_salt,
            "model_name": model_name,
            "page_size": page_size,
            "page_token": page_token,
        }
        if self.list_raises is not None:
            raise self.list_raises
        if self.list_page is None:
            return L2KeyListPage(entries=(), next_page_token=None)
        return self.list_page


class _FakeEngine:
    def __init__(self, sm: _FakeStorageManager):
        self.storage_manager = sm


def _make_app(sm: Optional[_FakeStorageManager]) -> FastAPI:
    """Build a FastAPI app with only the l2_keys router mounted and the
    fake engine bolted onto ``app.state``."""
    app = FastAPI()
    app.include_router(l2_keys_router)
    if sm is not None:
        app.state.engine = _FakeEngine(sm)
    return app


def _hex(n: int, width: int = 4) -> str:
    return n.to_bytes(width, "big").hex()


# =============================================================================
# Evict
# =============================================================================


class TestEvictEndpoint:
    def test_happy_path(self):
        sm = _FakeStorageManager()
        client = TestClient(_make_app(sm))

        resp = client.post(
            "/l2/keys:evict",
            json={
                "keys": [
                    {
                        "chunk_hash_hex": _hex(1),
                        "model_name": "llama",
                        "kv_rank": 0,
                        "cache_salt": "alice",
                    },
                    {
                        "chunk_hash_hex": _hex(2),
                        "model_name": "llama",
                        "kv_rank": 0,
                    },
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"requested": 2, "adapter": "s3", "ok": True}
        forwarded = sm.evict_calls[0]
        assert forwarded[0] == ObjectKey(
            chunk_hash=b"\x00\x00\x00\x01",
            model_name="llama",
            kv_rank=0,
            cache_salt="alice",
        )
        assert forwarded[1].cache_salt == ""  # default for omitted field

    def test_propagates_storage_manager_failure_in_body(self):
        sm = _FakeStorageManager(
            evict_response={"adapter": "s3", "ok": False, "error": "s3 down"}
        )
        client = TestClient(_make_app(sm))

        resp = client.post("/l2/keys:evict", json={"keys": []})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is False
        assert body["error"] == "s3 down"

    def test_503_when_no_adapters_configured(self):
        sm = _FakeStorageManager(evict_raises=ValueError("no L2 adapters configured"))
        client = TestClient(_make_app(sm))
        resp = client.post("/l2/keys:evict", json={"keys": []})
        assert resp.status_code == 503
        assert "no L2 adapters" in resp.json()["error"]

    def test_503_when_engine_not_initialized(self):
        client = TestClient(_make_app(None))
        resp = client.post("/l2/keys:evict", json={"keys": []})
        assert resp.status_code == 503

    @pytest.mark.parametrize(
        "body",
        [
            "not-json-text",  # invalid JSON → 422
            {},  # missing 'keys' → 422
            {"keys": "not-a-list"},  # wrong type → 422
            {"keys": [{"chunk_hash_hex": _hex(1), "kv_rank": 0}]},  # no model → 422
            {
                "keys": [
                    {
                        "chunk_hash_hex": _hex(1),
                        "model_name": "m",
                        "kv_rank": "not-int",
                    }
                ]
            },  # → 422
        ],
    )
    def test_422_on_pydantic_validation_failure(self, body):
        """Pydantic-level body-shape errors surface as 422 (FastAPI's
        default for request validation)."""
        sm = _FakeStorageManager()
        client = TestClient(_make_app(sm))
        if isinstance(body, str):
            resp = client.post(
                "/l2/keys:evict",
                content=body,
                headers={"content-type": "application/json"},
            )
        else:
            resp = client.post("/l2/keys:evict", json=body)
        assert resp.status_code == 422, resp.text
        assert sm.evict_calls == []

    @pytest.mark.parametrize(
        "body",
        [
            # Bad hex — survives Pydantic typing but fails bytes.fromhex.
            {"keys": [{"chunk_hash_hex": "zz", "model_name": "m", "kv_rank": 0}]},
            # @ in model_name — survives Pydantic typing but violates the
            # ObjectKey invariant.
            {
                "keys": [
                    {
                        "chunk_hash_hex": _hex(1),
                        "model_name": "bad@name",
                        "kv_rank": 0,
                    }
                ]
            },
        ],
    )
    def test_400_on_object_key_invariant_violation(self, body):
        """Bodies that type-check but violate ``ObjectKey`` invariants
        surface as 400 from our handler."""
        sm = _FakeStorageManager()
        client = TestClient(_make_app(sm))
        resp = client.post("/l2/keys:evict", json=body)
        assert resp.status_code == 400, resp.text
        assert sm.evict_calls == []


# =============================================================================
# List
# =============================================================================


class TestListEndpoint:
    def test_happy_path(self):
        k1 = ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=2,
            cache_salt="alice",
        )
        sm = _FakeStorageManager(
            list_page=L2KeyListPage(
                entries=(L2KeyEntry(key=k1, size_bytes=4096, adapter_name="s3"),),
                next_page_token="opaque-cursor",
            )
        )
        client = TestClient(_make_app(sm))

        resp = client.get(
            "/l2/keys",
            params={
                "cache_salt": "alice",
                "model_name": "llama",
                "page_size": 100,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["entries"] == [
            {
                "chunk_hash_hex": "deadbeef",
                "model_name": "llama",
                "kv_rank": 2,
                "object_group_id": 0,
                "cache_salt": "alice",
                "size_bytes": 4096,
                "adapter": "s3",
            }
        ]
        assert body["next_page_token"] == "opaque-cursor"
        assert sm.last_list_kwargs == {
            "cache_salt": "alice",
            "model_name": "llama",
            "page_size": 100,
            "page_token": None,
        }

    def test_default_salt_sentinel_translates_to_empty_string(self):
        sm = _FakeStorageManager()
        client = TestClient(_make_app(sm))
        client.get("/l2/keys", params={"cache_salt": "_default"})
        assert sm.last_list_kwargs["cache_salt"] == ""

    def test_no_filters_pass_none_to_storage_manager(self):
        sm = _FakeStorageManager()
        client = TestClient(_make_app(sm))
        client.get("/l2/keys")
        assert sm.last_list_kwargs["cache_salt"] is None
        assert sm.last_list_kwargs["model_name"] is None

    def test_page_token_threads_through(self):
        sm = _FakeStorageManager()
        client = TestClient(_make_app(sm))
        client.get("/l2/keys", params={"page_token": "abc"})
        assert sm.last_list_kwargs["page_token"] == "abc"

    def test_503_when_no_adapters_configured(self):
        sm = _FakeStorageManager(list_raises=ValueError("no L2 adapters configured"))
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys")
        assert resp.status_code == 503

    def test_501_when_primary_adapter_does_not_support_listing(self):
        sm = _FakeStorageManager(
            list_raises=NotImplementedError("FsL2Adapter has no listing")
        )
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys")
        assert resp.status_code == 501
        assert "does not support listing" in resp.json()["error"]

    def test_400_on_invalid_page_size(self):
        sm = _FakeStorageManager()
        client = TestClient(_make_app(sm))
        # Below floor — FastAPI Query ge=1 → 422 from validation layer.
        resp = client.get("/l2/keys", params={"page_size": 0})
        assert resp.status_code in (400, 422)
        # Above ceiling.
        resp = client.get("/l2/keys", params={"page_size": 999_999_999})
        assert resp.status_code in (400, 422)

    def test_503_when_engine_not_initialized(self):
        client = TestClient(_make_app(None))
        resp = client.get("/l2/keys")
        assert resp.status_code == 503

    def test_400_on_malformed_page_token_from_adapter(self):
        # Adapter-level "malformed cursor" ValueError → 400 (the
        # endpoint distinguishes from "no adapters" by message prefix).
        sm = _FakeStorageManager(
            list_raises=ValueError("malformed S3 list cursor: invalid literal")
        )
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys", params={"page_token": "garbage"})
        assert resp.status_code == 400


# =============================================================================
# Auto-discovery
# =============================================================================


class TestAutoDiscovery:
    def test_endpoints_are_registered_via_http_api_registry(self):
        # First Party
        from lmcache.v1.multiprocess.http_api_registry import HTTPAPIRegistry

        app = FastAPI()
        registry = HTTPAPIRegistry(app)
        registry.register_all_apis()
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/l2/keys:evict" in paths
        assert "/l2/keys" in paths
