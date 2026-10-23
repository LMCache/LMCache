# SPDX-License-Identifier: Apache-2.0
"""Token-addressed HTTP controls preserve the identity used by the store path."""

# Standard
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import JsonValue
import httpx
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.http_apis.cache_api import router as node_router
from lmcache.v1.multiprocess.http_apis.dependencies import build_context
from lmcache.v1.multiprocess.http_apis.error_handlers import register_error_handlers
from lmcache.v1.multiprocess.http_apis.schemas import PrefetchRequest
from lmcache.v1.multiprocess.token_hasher import TokenHasher

TOKENS = [1, 2, 3, 4]


def _coordinator() -> FastAPI:
    return create_app(
        MPCoordinatorConfig(
            health_check_interval=0.0, eviction_check_interval=0.0, chunk_size=4
        )
    )


def _body(tags: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": "m",
        "world_size": 1,
        "token_ids": [token for token in TOKENS],
        "cache_salt": "alice",
        "tags": {name: value for name, value in tags.items()},
    }


def _store_key(tags: dict[str, str]) -> ObjectKey:
    """Use the server's store-side IPC expansion, independently of the resolver."""
    ipc_key = IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=0,
        token_ids=tuple(TOKENS),
        start=0,
        end=len(TOKENS),
        request_id="store",
        cache_salt="alice",
        request_configs={f"lmcache.tag.{name}": value for name, value in tags.items()},
    )
    hashes = TokenHasher(chunk_size=4, hash_algorithm="blake3").compute_chunk_hashes(
        TOKENS
    )
    return ipc_key_to_object_keys(ipc_key, hashes, [0])[0][0]


def test_lookup_pin_and_delete_select_only_the_requested_tags() -> None:
    """Identical tokens under two adapters remain separately addressable."""
    keys = [_store_key({"adapter": adapter}) for adapter in ("A", "B")]
    dispatched: list[dict[str, JsonValue]] = []

    def delete_node(request: httpx.Request) -> httpx.Response:
        payload: dict[str, JsonValue] = json.loads(request.content)
        assert isinstance(payload["keys"], list)
        dispatched.append(payload)
        return httpx.Response(
            200, json={"deleted": len(payload["keys"]), "skipped": 0, "ok": True}
        )

    with TestClient(_coordinator()) as client:
        assert (
            client.post(
                "/instances",
                json={"instance_id": "node", "ip": "127.0.0.1", "http_port": 8080},
            ).status_code
            == 200
        )
        client.app.state.outbound_client = httpx.AsyncClient(
            transport=httpx.MockTransport(delete_node)
        )
        response = client.post(
            "/events",
            json={
                "batches": [
                    {
                        "instance_id": "node",
                        "incarnation": 1,
                        "seq": 1,
                        "event_type": "store",
                        "tier": "l2",
                        "backend": "fs",
                        "entries": [
                            {
                                "key": asdict(key.to_encoded_object_key()),
                                "size_bytes": 64,
                            }
                            for key in keys
                        ],
                    }
                ]
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["applied"] == 1

        for key in keys:
            response = client.post(
                "/directory/lookup", json=_body(dict(key.identity.tags))
            )
            assert response.status_code == 200, response.text
            [result] = response.json()["results"]
            assert result["key"] == asdict(key.to_encoded_object_key())
            assert len(result["placements"]) == 1
        response = client.post("/directory/lookup", json=_body({}))
        assert response.json()["results"][0]["placements"] == []

        body_b = _body({"adapter": "B"})
        assert client.post("/cache/pins", json=body_b).status_code == 200
        [pin] = client.get("/cache/pins").json()["pins"]
        assert pin["key"] == asdict(keys[1].to_encoded_object_key())

        def delete(tags: dict[str, str]) -> httpx.Response:
            return client.post(
                "/cache/delete",
                json={**_body(tags), "instance_id": "node", "tier": "l2"},
            )

        response = delete({"adapter": "A"})
        assert response.status_code == 200, response.text
        assert response.json()["affected"] == 1
        assert dispatched[-1]["keys"] == [asdict(keys[0].to_encoded_object_key())]
        response = delete({"adapter": "B"})
        assert response.status_code == 200, response.text
        assert response.json()["affected"] == 0
        assert response.json()["skipped"] == 1
        assert len(dispatched) == 1

        assert client.request("DELETE", "/cache/pins", json=body_b).status_code == 200
        assert client.get("/cache/pins").json()["total"] == 0
        response = delete({"adapter": "B"})
        assert response.status_code == 200, response.text
        assert response.json()["affected"] == 1
        assert dispatched[-1]["keys"] == [asdict(keys[1].to_encoded_object_key())]


@dataclass
class _Handle:
    total_requested_keys: int


@dataclass
class _Storage:
    submitted: list[PrefetchTaskSpec] = field(default_factory=list)

    def submit_prefetch_task(self, spec: PrefetchTaskSpec) -> _Handle:
        self.submitted.append(spec)
        return _Handle(sum(len(row.keys) for row in spec.key_groups))


@dataclass
class _Context:
    layout_desc_registry: LayoutDescRegistry
    token_hasher: TokenHasher


@dataclass
class _Engine:
    context: _Context
    storage_manager: _Storage


def _node(storage: _Storage) -> FastAPI:
    registry = LayoutDescRegistry()
    layout = MemoryLayoutDesc(shapes=[torch.Size([8])], dtypes=[torch.float16])
    registry.register(
        "m", 2, layout, AttnWindowDesc(num_chunks_in_sw=[-1, 1], world_size=2)
    )
    engine = _Engine(
        _Context(registry, TokenHasher(chunk_size=4, hash_algorithm="blake3")), storage
    )
    app = FastAPI()
    app.include_router(node_router)
    register_error_handlers(app)
    app.state.context = build_context(engine)
    return app


@pytest.mark.parametrize("via_coordinator", [False, True])
def test_prefetch_preserves_tags_on_every_rank_and_object_group(
    via_coordinator: bool,
) -> None:
    storage = _Storage()
    node = _node(storage)
    tags = {"adapter": "B", "version": "2"}
    body: dict[str, JsonValue] = {**_body(tags), "world_size": 2}
    app = _coordinator() if via_coordinator else node
    with TestClient(app) as client:
        if via_coordinator:
            assert (
                client.post(
                    "/instances",
                    json={"instance_id": "node", "ip": "127.0.0.1", "http_port": 8080},
                ).status_code
                == 200
            )
            client.app.state.outbound_client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=node)
            )
            body["instance_id"] = "node"
        response = client.post("/cache/prefetches", json=body)
        assert response.status_code == (200 if via_coordinator else 202), response.text
        assert response.json()["chunks"] == 1
        [spec] = storage.submitted
        assert len(spec.key_groups) == 4
        keys = [key for row in spec.key_groups for key in row.keys]
        assert {key.object_group_id for key in keys} == {0, 1}
        assert len({key.kv_rank for key in keys}) == 2
        assert all(dict(key.identity.tags) == tags for key in keys)
        assert all(key.chunk_hash == _store_key(tags).chunk_hash for key in keys)


@pytest.mark.parametrize("tags", [{"": "bad"}, {"adapter": 1}, {"adapter": None}])
@pytest.mark.parametrize(
    "path",
    [
        "/directory/lookup",
        "/directory/blend-lookup",
        "/cache/pins",
        "/cache/delete",
        "/cache/prefetches",
    ],
)
def test_coordinator_rejects_invalid_matching_tags(
    path: str, tags: dict[str, JsonValue]
) -> None:
    body: dict[str, JsonValue] = {**_body(tags), "instance_id": "node"}
    with TestClient(_coordinator()) as client:
        response = client.post(path, json=body)
        assert response.status_code == 422, response.text


@pytest.mark.parametrize("tags", [{"": "bad"}, {"adapter": 1}, {"adapter": None}])
def test_node_rejects_invalid_matching_tags(tags: dict[str, JsonValue]) -> None:
    with TestClient(_node(_Storage())) as client:
        response = client.post("/cache/prefetches", json=_body(tags))
        assert response.status_code == 422, response.text


def test_node_schema_rejects_empty_tag_names() -> None:
    with pytest.raises(ValueError, match="tags require"):
        PrefetchRequest("m", 1, TOKENS, tags={"": "bad"})
