# SPDX-License-Identifier: Apache-2.0
"""
Tests for the basic-info endpoints exposed by info_api.

Covers:
- ``GET /`` static liveness payload.
- ``GET /healthcheck`` 503 before the engine is wired, 200 after.
- ``GET /status`` 503 before the engine is wired, engine status after.
- ``GET /status/memory-pressure`` typed local capacity status.
- The version routes (``/version``, ``/lmc_version``, ``/commit_id``) are
  registered as part of the group.
"""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.distributed.storage_usage import (
    StorageHealth,
    StorageTierUsageSnapshot,
    StorageUsageSnapshot,
)
from lmcache.v1.distributed.tiers import Tier
from lmcache.v1.multiprocess.http_apis.dependencies import build_context
from lmcache.v1.multiprocess.http_apis.info_api import router as info_router


class _FakeStorageManager:
    def get_memory_usage_snapshot(self) -> StorageUsageSnapshot:
        return StorageUsageSnapshot(
            l1=StorageTierUsageSnapshot(
                tier=Tier.L1,
                adapter_id=None,
                backend_type=None,
                used_bytes=80,
                capacity_bytes=100,
                trigger_watermark=0.8,
                eviction_policy="LRU",
                health=StorageHealth.OK,
            ),
            l2=(
                StorageTierUsageSnapshot(
                    tier=Tier.L2,
                    adapter_id=3,
                    backend_type="dax",
                    used_bytes=10,
                    capacity_bytes=100,
                    trigger_watermark=0.9,
                    eviction_policy="LRU",
                    health=StorageHealth.UNKNOWN,
                    collection_errors=("health_unavailable",),
                ),
            ),
        )


class _BrokenStorageManager:
    def get_memory_usage_snapshot(self) -> StorageUsageSnapshot:
        raise RuntimeError("snapshot failed")


class _FakeEngine:
    def __init__(self, storage_manager=None) -> None:
        self.storage_manager = storage_manager or _FakeStorageManager()

    def report_status(self) -> dict[str, str]:
        return {"l1": "ok", "l2": "ok"}


def _make_app(engine=None) -> FastAPI:
    app = FastAPI()
    app.include_router(info_router)
    if engine is not None:
        app.state.engine = engine
        app.state.context = build_context(engine, instance_id="mp-a")
    return app


def test_root_returns_ok():
    client = TestClient(_make_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "LMCache HTTP API"}


def test_healthcheck_503_without_engine():
    client = TestClient(_make_app(engine=None))
    resp = client.get("/healthcheck")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy"


def test_healthcheck_healthy_with_engine():
    client = TestClient(_make_app(engine=_FakeEngine()))
    resp = client.get("/healthcheck")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_status_503_without_engine():
    client = TestClient(_make_app(engine=None))
    resp = client.get("/status")
    assert resp.status_code == 503
    assert resp.json() == {"error": "engine not initialized"}


def test_status_reports_engine_status():
    client = TestClient(_make_app(engine=_FakeEngine()))
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json() == {"l1": "ok", "l2": "ok"}


def test_memory_pressure_503_without_context():
    client = TestClient(_make_app(engine=None))

    resp = client.get("/status/memory-pressure")

    assert resp.status_code == 503
    assert resp.json() == {"detail": "server not initialized"}


def test_memory_pressure_reports_local_per_adapter_snapshot():
    client = TestClient(_make_app(engine=_FakeEngine()))

    resp = client.get("/status/memory-pressure")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["instance_id"] == "mp-a"
    assert body["overall_level"] == "high"
    assert body["complete"] is True
    assert body["tiers"][0] == {
        "tier": "l1",
        "adapter_id": None,
        "backend_type": None,
        "used_bytes": 80,
        "capacity_bytes": 100,
        "used_ratio": 0.8,
        "trigger_watermark": 0.8,
        "eviction_policy": "LRU",
        "health": "ok",
        "level": "high",
        "collection_errors": [],
    }
    assert body["tiers"][1]["adapter_id"] == 3
    assert body["tiers"][1]["backend_type"] == "dax"
    assert body["tiers"][1]["collection_errors"] == ["health_unavailable"]


def test_memory_pressure_503_when_top_level_snapshot_fails():
    engine = _FakeEngine(storage_manager=_BrokenStorageManager())
    client = TestClient(_make_app(engine=engine))

    resp = client.get("/status/memory-pressure")

    assert resp.status_code == 503
    assert resp.json() == {"detail": "memory pressure snapshot unavailable"}


def test_version_routes_registered():
    """The version routes are folded into the basic-info group."""
    app = _make_app()
    schema = app.openapi()
    paths = set(schema["paths"])
    assert {
        "/status/memory-pressure",
        "/version",
        "/lmc_version",
        "/commit_id",
    }.issubset(paths)
    tier_schema = schema["components"]["schemas"]["MemoryTierPressure"]["properties"][
        "tier"
    ]
    assert tier_schema["enum"] == ["l1", "l2"]
