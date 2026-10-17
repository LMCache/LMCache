# SPDX-License-Identifier: Apache-2.0
"""Tests for the new /livez and /readyz Kubernetes health probes.

These tests verify the endpoints added for Issue #5164 — proper
end-to-end liveness and readiness probing for the MP server. They
complement the existing ``/healthcheck`` coverage in
``test_info_api.py`` by exercising the new routes with mocked engines
in every relevant state.
"""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.multiprocess.http_apis.info_api import router as info_router


class _FakeEngine:
    """A mock MP server engine that reports a fully healthy status."""

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "engine_type": "MPCacheServer",
            "chunk_size": 256,
            "hash_algorithm": "blake3",
            "active_sessions": 0,
            "storage_manager": {"is_healthy": True},
            "worker_liveness": {
                "enabled": True,
                "tracked_instances": 3,
                "reap_timeout_seconds": 30,
                "registration_grace_seconds": 5,
            },
        }


class _FakeEngineUnhealthyStorage:
    """Storage manager reports unhealthy — one or more L2 adapters or L1
    managers have failed (e.g. S3 bucket unreachable, filesystem error)."""

    def report_status(self) -> dict:
        return {
            "is_healthy": False,
            "engine_type": "MPCacheServer",
            "storage_manager": {"is_healthy": False},
            "worker_liveness": {
                "enabled": True,
                "tracked_instances": 3,
                "reap_timeout_seconds": 30,
            },
        }


class _FakeEngineNoWorkers:
    """All workers have been reaped or the server never saw any. The
    storage manager itself is fine, but we should not advertise readiness."""

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "engine_type": "MPCacheServer",
            "storage_manager": {"is_healthy": True},
            "worker_liveness": {
                "enabled": True,
                "tracked_instances": 0,
                "reap_timeout_seconds": 30,
            },
        }


class _FakeEngineStandalone:
    """A server running without worker liveness tracking (e.g. purely
    local mode). No ``worker_liveness`` key is present in the status.
    Readiness should not fail on missing workers."""

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "engine_type": "MPCacheServer",
            "storage_manager": {"is_healthy": True},
            # NOTE: no "worker_liveness" key
        }


class _FakeEngineRaisesOnStatus:
    """Simulates the case where ``report_status`` itself throws
    (e.g. a broken storage adapter during status collection)."""

    def report_status(self) -> dict:
        raise RuntimeError("storage manager crashed during status query")


def _make_app(engine=None) -> FastAPI:
    app = FastAPI()
    app.include_router(info_router)
    if engine is not None:
        app.state.engine = engine
    return app


# ---------------------------------------------------------------------------
# /livez tests
# ---------------------------------------------------------------------------


def test_livez_503_without_engine():
    """Engine not wired into app state — HTTP 503 with a machine-readable
    reason, not just a connection refused."""
    client = TestClient(_make_app(engine=None))
    resp = client.get("/livez")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_alive"
    assert "engine not initialized" in body["reason"]


def test_livez_alive_when_engine_present():
    """When the engine is wired in, /livez returns 200 immediately —
    it never touches storage or workers. This is what Kubernetes uses to
    decide whether to restart a stuck process."""
    client = TestClient(_make_app(engine=_FakeEngine()))
    resp = client.get("/livez")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_livez_is_lightweight():
    """/livez must not invoke report_status — it is intended to stay fast
    under load and serve as a pure liveness check."""

    class _SpyEngine:
        def __init__(self) -> None:
            self.report_status_called = False

        def report_status(self) -> dict:
            self.report_status_called = True
            return {}

    spy = _SpyEngine()
    client = TestClient(_make_app(engine=spy))
    resp = client.get("/livez")
    assert resp.status_code == 200
    assert spy.report_status_called is False


# ---------------------------------------------------------------------------
# /readyz tests
# ---------------------------------------------------------------------------


def test_readyz_503_without_engine():
    """Before the engine is wired in, we definitely are not ready."""
    client = TestClient(_make_app(engine=None))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert "engine" in body["failed"]
    assert body["checks"]["engine"] == "missing"


def test_readyz_healthy_returns_200():
    """All components report healthy — readiness is advertised."""
    client = TestClient(_make_app(engine=_FakeEngine()))
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert "failed" not in body
    assert body["checks"]["storage_manager"] is True
    workers = body["checks"]["workers"]
    assert workers["enabled"] is True
    assert workers["tracked_instances"] == 3


def test_readyz_503_when_storage_unhealthy():
    """A single failed L2 adapter or L1 manager should make the entire
    server not ready."""
    client = TestClient(_make_app(engine=_FakeEngineUnhealthyStorage()))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert "storage_manager" in body["failed"]
    assert body["checks"]["storage_manager"] is False


def test_readyz_503_when_no_workers():
    """Worker liveness is enabled but no workers are tracked — either
    startup-in-progress or everything reaped. In either case, not ready."""
    client = TestClient(_make_app(engine=_FakeEngineNoWorkers()))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert "workers" in body["failed"]
    workers = body["checks"]["workers"]
    assert workers["tracked_instances"] == 0


def test_readyz_allows_standalone_no_workers():
    """Standalone servers (no liveness targets, no ``worker_liveness`` key)
    pass the worker check trivially. Only storage manager health matters."""
    client = TestClient(_make_app(engine=_FakeEngineStandalone()))
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert "workers" not in body["checks"]
    assert body["checks"]["storage_manager"] is True


def test_readyz_degraded_storage_but_workers_ok():
    """Both checks present; storage fails but workers are fine. Storage
    alone should trigger 503 and appear in the ``failed`` list."""
    client = TestClient(_make_app(engine=_FakeEngineUnhealthyStorage()))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert "storage_manager" in body["failed"]
    assert "workers" not in body["failed"]


def test_readyz_handles_report_status_exception():
    """If report_status itself raises (defensive coding), we still answer
    503 instead of crashing the probe. Kubernetes should see a consistent
    HTTP-level failure regardless of what went wrong inside the engine."""
    client = TestClient(_make_app(engine=_FakeEngineRaisesOnStatus()))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert "report_status" in body["failed"]
