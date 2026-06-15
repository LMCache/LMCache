# SPDX-License-Identifier: Apache-2.0
"""
HTTP-level tests for ``l2_api`` — the ``DELETE /l2`` and
``GET /l2/keys`` endpoints.

The endpoints reach into ``request.app.state.engine.storage_manager``
and call ``storage_manager.primary_l2()`` to obtain the
``(descriptor, adapter)`` pair, then invoke the adapter's own methods.
These tests inject a fake storage manager that records calls and serves
canned responses, so the HTTP layer can be exercised without spinning
up a real cache engine.
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
from lmcache.v1.distributed.l2_adapters.base import KeyEntry, KeyListPage
from lmcache.v1.multiprocess.http_apis.l2_api import router as l2_keys_router


@dataclass
class _FakeDescriptor:
    """Minimal descriptor — only ``type_name`` is read by the handler."""

    type_name: str = "s3"


@dataclass
class _FakeAdapter:
    """Records calls and serves canned responses for adapter methods."""

    delete_calls: list[list[ObjectKey]] = field(default_factory=list)
    delete_raises: Optional[BaseException] = None

    list_page: Optional[KeyListPage] = None
    list_raises: Optional[BaseException] = None
    last_list_kwargs: dict[str, object] = field(default_factory=dict)

    def delete(self, keys: list[ObjectKey]) -> None:
        self.delete_calls.append(list(keys))
        if self.delete_raises is not None:
            raise self.delete_raises

    def list_l2_keys(
        self,
        model_name: Optional[str] = None,
        page_size: int = 500,
        cursor: Optional[str] = None,
    ) -> KeyListPage:
        self.last_list_kwargs = {
            "model_name": model_name,
            "page_size": page_size,
            "cursor": cursor,
        }
        if self.list_raises is not None:
            raise self.list_raises
        return self.list_page or KeyListPage(entries=(), next_page_token=None)


@dataclass
class _FakeStorageManager:
    """Implements ``primary_l2()``. ``adapter=None`` makes the call
    raise ``ValueError("no L2 adapters configured")`` — the way a real
    SM signals an empty adapter list."""

    adapter: Optional[_FakeAdapter] = None
    descriptor_type_name: str = "s3"

    def primary_l2(self) -> tuple[_FakeDescriptor, _FakeAdapter]:
        if self.adapter is None:
            raise ValueError("no L2 adapters configured")
        return _FakeDescriptor(type_name=self.descriptor_type_name), self.adapter


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
# Delete
# =============================================================================


class TestDeleteEndpoint:
    def test_happy_path(self):
        adapter = _FakeAdapter()
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))

        resp = client.request(
            "DELETE",
            "/l2",
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
        forwarded = adapter.delete_calls[0]
        assert forwarded[0] == ObjectKey(
            chunk_hash=b"\x00\x00\x00\x01",
            model_name="llama",
            kv_rank=0,
            cache_salt="alice",
        )
        assert forwarded[1].cache_salt == ""  # default for omitted field

    def test_propagates_adapter_failure_in_body(self):
        adapter = _FakeAdapter(delete_raises=RuntimeError("s3 down"))
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))

        resp = client.request("DELETE", "/l2", json={"keys": []})
        # Adapter exceptions are surfaced as a 200 body with ok=false +
        # error, NOT as a 500 — operators want a structured failure.
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["adapter"] == "s3"
        assert body["ok"] is False
        assert "s3 down" in body["error"]

    def test_503_when_no_adapters_configured(self):
        sm = _FakeStorageManager(adapter=None)
        client = TestClient(_make_app(sm))
        resp = client.request("DELETE", "/l2", json={"keys": []})
        assert resp.status_code == 503
        assert "no L2 adapters" in resp.json()["detail"]

    def test_503_when_engine_not_initialized(self):
        client = TestClient(_make_app(None))
        resp = client.request("DELETE", "/l2", json={"keys": []})
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
        adapter = _FakeAdapter()
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))
        if isinstance(body, str):
            resp = client.request(
                "DELETE",
                "/l2",
                content=body,
                headers={"content-type": "application/json"},
            )
        else:
            resp = client.request("DELETE", "/l2", json=body)
        assert resp.status_code == 422, resp.text
        assert adapter.delete_calls == []

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
        adapter = _FakeAdapter()
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))
        resp = client.request("DELETE", "/l2", json=body)
        assert resp.status_code == 400, resp.text
        assert adapter.delete_calls == []

    def test_400_when_batch_exceeds_cap(self):
        """The handler enforces the ``_MAX_DELETE_BATCH`` cap (the
        dataclass body type no longer carries a Pydantic Field
        constraint)."""
        # First Party
        from lmcache.v1.multiprocess.http_apis.l2_api import _MAX_DELETE_BATCH

        adapter = _FakeAdapter()
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))
        oversized = [
            {"chunk_hash_hex": _hex(i), "model_name": "m", "kv_rank": 0}
            for i in range(_MAX_DELETE_BATCH + 1)
        ]
        resp = client.request("DELETE", "/l2", json={"keys": oversized})
        assert resp.status_code == 400, resp.text
        assert "too many keys" in resp.json()["detail"]
        assert adapter.delete_calls == []


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
        adapter = _FakeAdapter(
            list_page=KeyListPage(
                entries=(KeyEntry(key=k1.to_encoded_object_key(), size_bytes=4096),),
                next_page_token="opaque-cursor",
            )
        )
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))

        resp = client.get(
            "/l2/keys",
            params={
                "model_name": "llama",
                "page_size": 100,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["adapter"] == "s3"
        assert body["entries"] == [
            {
                "key": {
                    "chunk_hash_hex": "deadbeef",
                    "model_name": "llama",
                    "kv_rank": 2,
                    "object_group_id": 0,
                    "cache_salt": "alice",
                },
                "size_bytes": 4096,
            }
        ]
        assert body["next_page_token"] == "opaque-cursor"
        assert adapter.last_list_kwargs == {
            "model_name": "llama",
            "page_size": 100,
            "cursor": None,
        }

    def test_no_filter_passes_none_to_adapter(self):
        adapter = _FakeAdapter()
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))
        client.get("/l2/keys")
        assert adapter.last_list_kwargs["model_name"] is None

    def test_page_token_threads_through_as_cursor(self):
        adapter = _FakeAdapter()
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))
        client.get("/l2/keys", params={"page_token": "abc"})
        # The HTTP query param ``page_token`` is forwarded to the
        # adapter under its native name ``cursor``.
        assert adapter.last_list_kwargs["cursor"] == "abc"

    def test_503_when_no_adapters_configured(self):
        sm = _FakeStorageManager(adapter=None)
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys")
        assert resp.status_code == 503

    def test_501_when_primary_adapter_does_not_support_listing(self):
        adapter = _FakeAdapter(
            list_raises=NotImplementedError("FsL2Adapter has no listing")
        )
        sm = _FakeStorageManager(adapter=adapter)
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys")
        assert resp.status_code == 501
        assert "does not support listing" in resp.json()["detail"]

    def test_400_on_invalid_page_size(self):
        adapter = _FakeAdapter()
        sm = _FakeStorageManager(adapter=adapter)
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
        # Adapter-level "malformed cursor" ValueError → 400 (distinct
        # path from "no adapters" which the SM owns and maps to 503).
        adapter = _FakeAdapter(
            list_raises=ValueError("malformed S3 list cursor: invalid literal")
        )
        sm = _FakeStorageManager(adapter=adapter)
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
        # ``DELETE /l2`` (the cache-purge action) and ``GET /l2/keys``
        # (the key listing) live on different paths — verify both are
        # registered with the right method.
        methods_by_path: dict[str, set[str]] = {}
        for r in app.routes:
            path = getattr(r, "path", None)
            if path in ("/l2", "/l2/keys"):
                methods_by_path.setdefault(path, set()).update(
                    getattr(r, "methods", set())
                )
        assert "DELETE" in methods_by_path.get("/l2", set())
        assert "GET" in methods_by_path.get("/l2/keys", set())
