# SPDX-License-Identifier: Apache-2.0
"""
Tests for the basic-info endpoints exposed by info_api.

Covers:
- ``GET /`` static liveness payload.
- ``GET /healthcheck`` 503 before the engine is wired, 200 after.
- ``GET /readyz`` 503 before the engine is wired or while a component is
  unhealthy, 200 once every component reports healthy.
- ``GET /status`` 503 before the engine is wired, engine status after.
- The version routes (``/version``, ``/lmc_version``, ``/commit_id``) are
  registered as part of the group.
"""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.multiprocess.http_apis.info_api import router as info_router


class _FakeEngine:
    def __init__(self, components: dict[str, bool] | None = None) -> None:
        self._components = (
            {"prefetch_controller": True} if components is None else components
        )

    def report_status(self) -> dict[str, str]:
        return {"l1": "ok", "l2": "ok"}

    def report_readiness(self) -> dict[str, bool]:
        return self._components


def _make_app(engine=None) -> FastAPI:
    app = FastAPI()
    app.include_router(info_router)
    if engine is not None:
        app.state.engine = engine
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


def test_readyz_503_without_engine():
    client = TestClient(_make_app(engine=None))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json() == {
        "status": "not_ready",
        "reason": "engine not initialized",
    }


def test_readyz_ready_when_every_component_is_healthy():
    components = {"prefetch_controller": True, "l2_adapter[0]": True}
    client = TestClient(_make_app(engine=_FakeEngine(components)))
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "components": components}


def test_readyz_503_lists_the_unhealthy_components():
    components = {
        "prefetch_controller": False,
        "store_controller[default]": True,
        "l2_adapter[0]": False,
    }
    client = TestClient(_make_app(engine=_FakeEngine(components)))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["components"] == components
    assert body["unhealthy"] == ["l2_adapter[0]", "prefetch_controller"]


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


def test_version_routes_registered():
    """The version routes are folded into the basic-info group."""
    app = _make_app()
    paths = set(app.openapi()["paths"])
    assert {"/version", "/lmc_version", "/commit_id"}.issubset(paths)
