# SPDX-License-Identifier: Apache-2.0
"""
HTTP-level tests for ``l2_api`` — the ``DELETE /l2``,
``GET /l2/keys``, and ``GET /l2/adapters`` endpoints.

The endpoints reach into ``request.app.state.engine.storage_manager``
and call ``storage_manager.l2_adapters()`` to obtain the configured
``(descriptor, adapter)`` pairs, then invoke the adapter's own
methods. These tests inject a fake storage manager that returns
``_FakeAdapter`` instances, so the HTTP layer can be exercised without
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
from lmcache.v1.distributed.api import KeyEntry, KeyListPage, ObjectKey
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
    """Implements ``l2_adapters()``. Pass a list of
    ``(type_name, _FakeAdapter)`` tuples; an empty list reproduces the
    "no L2 adapters configured" condition."""

    adapters: list[tuple[str, _FakeAdapter]] = field(default_factory=list)

    def l2_adapters(self) -> list[tuple[_FakeDescriptor, _FakeAdapter]]:
        return [(_FakeDescriptor(type_name=n), a) for n, a in self.adapters]


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


def _sm_with(*entries: tuple[str, _FakeAdapter]) -> _FakeStorageManager:
    return _FakeStorageManager(adapters=list(entries))


# =============================================================================
# Adapter listing
# =============================================================================


class TestAdaptersEndpoint:
    def test_empty_adapter_list_returns_empty_array(self):
        # An engine with no L2 backends is still initialized — return
        # 200 with an empty list, not 503. The empty case is operational
        # signal, not an error.
        sm = _sm_with()
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/adapters")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"adapters": []}

    def test_lists_all_adapters_with_primary_flag(self):
        sm = _sm_with(("s3", _FakeAdapter()), ("fs", _FakeAdapter()))
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/adapters")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "adapters": [
                {"index": 0, "type_name": "s3", "primary": True},
                {"index": 1, "type_name": "fs", "primary": False},
            ]
        }

    def test_503_when_engine_not_initialized(self):
        client = TestClient(_make_app(None))
        resp = client.get("/l2/adapters")
        assert resp.status_code == 503


# =============================================================================
# Delete
# =============================================================================


class TestDeleteEndpoint:
    def test_happy_path_defaults_to_primary(self):
        primary = _FakeAdapter()
        secondary = _FakeAdapter()
        sm = _sm_with(("s3", primary), ("fs", secondary))
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
        # Only the primary adapter saw the call.
        assert len(primary.delete_calls) == 1
        assert secondary.delete_calls == []
        forwarded = primary.delete_calls[0]
        assert forwarded[0] == ObjectKey(
            chunk_hash=b"\x00\x00\x00\x01",
            model_name="llama",
            kv_rank=0,
            cache_salt="alice",
        )
        assert forwarded[1].cache_salt == ""

    def test_adapter_query_param_selects_non_primary(self):
        primary = _FakeAdapter()
        secondary = _FakeAdapter()
        sm = _sm_with(("s3", primary), ("fs", secondary))
        client = TestClient(_make_app(sm))

        resp = client.request(
            "DELETE",
            "/l2?adapter=fs",
            json={
                "keys": [{"chunk_hash_hex": _hex(1), "model_name": "m", "kv_rank": 0}]
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["adapter"] == "fs"
        assert primary.delete_calls == []
        assert len(secondary.delete_calls) == 1

    def test_404_when_adapter_selector_no_match(self):
        sm = _sm_with(("s3", _FakeAdapter()))
        client = TestClient(_make_app(sm))
        resp = client.request(
            "DELETE",
            "/l2?adapter=nope",
            json={"keys": []},
        )
        assert resp.status_code == 404
        assert "nope" in resp.json()["detail"]

    def test_propagates_adapter_failure_in_body(self):
        adapter = _FakeAdapter(delete_raises=RuntimeError("s3 down"))
        sm = _sm_with(("s3", adapter))
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
        sm = _sm_with()
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
        sm = _sm_with(("s3", adapter))
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
        sm = _sm_with(("s3", adapter))
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
        sm = _sm_with(("s3", adapter))
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
    def test_happy_path_defaults_to_primary(self):
        k1 = ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=2,
            cache_salt="alice",
        )
        primary = _FakeAdapter(
            list_page=KeyListPage(
                entries=(KeyEntry(key=k1.to_encoded_object_key(), size_bytes=4096),),
                next_page_token="opaque-cursor",
            )
        )
        secondary = _FakeAdapter()
        sm = _sm_with(("s3", primary), ("fs", secondary))
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
        assert primary.last_list_kwargs == {
            "model_name": "llama",
            "page_size": 100,
            "cursor": None,
        }
        # Secondary not consulted.
        assert secondary.last_list_kwargs == {}

    def test_adapter_query_param_selects_non_primary(self):
        primary = _FakeAdapter()
        secondary = _FakeAdapter(
            list_page=KeyListPage(entries=(), next_page_token="from-fs")
        )
        sm = _sm_with(("s3", primary), ("fs", secondary))
        client = TestClient(_make_app(sm))

        resp = client.get("/l2/keys", params={"adapter": "fs"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["adapter"] == "fs"
        assert body["next_page_token"] == "from-fs"
        assert primary.last_list_kwargs == {}

    def test_404_when_adapter_selector_no_match(self):
        sm = _sm_with(("s3", _FakeAdapter()))
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys", params={"adapter": "nope"})
        assert resp.status_code == 404

    def test_no_filter_passes_none_to_adapter(self):
        adapter = _FakeAdapter()
        sm = _sm_with(("s3", adapter))
        client = TestClient(_make_app(sm))
        client.get("/l2/keys")
        assert adapter.last_list_kwargs["model_name"] is None

    def test_page_token_threads_through_as_cursor(self):
        adapter = _FakeAdapter()
        sm = _sm_with(("s3", adapter))
        client = TestClient(_make_app(sm))
        client.get("/l2/keys", params={"page_token": "abc"})
        # The HTTP query param ``page_token`` is forwarded to the
        # adapter under its native name ``cursor``.
        assert adapter.last_list_kwargs["cursor"] == "abc"

    def test_503_when_no_adapters_configured(self):
        sm = _sm_with()
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys")
        assert resp.status_code == 503

    def test_501_when_selected_adapter_does_not_support_listing(self):
        adapter = _FakeAdapter(
            list_raises=NotImplementedError("FsL2Adapter has no listing")
        )
        sm = _sm_with(("fs", adapter))
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys")
        assert resp.status_code == 501
        assert "does not support listing" in resp.json()["detail"]

    def test_400_on_invalid_page_size(self):
        adapter = _FakeAdapter()
        sm = _sm_with(("s3", adapter))
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
        # path from "no adapters" which the HTTP helper owns and maps
        # to 503).
        adapter = _FakeAdapter(
            list_raises=ValueError("malformed S3 list cursor: invalid literal")
        )
        sm = _sm_with(("s3", adapter))
        client = TestClient(_make_app(sm))
        resp = client.get("/l2/keys", params={"page_token": "garbage"})
        assert resp.status_code == 400
