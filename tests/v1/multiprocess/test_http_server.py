# SPDX-License-Identifier: Apache-2.0

"""Tests for the HTTP server frontend dashboard feature.

Covers:
- GET /           (FileResponse vs JSON fallback)
- GET /api/healthcheck
- POST /api/clear-cache
- GET /api/status
- Static file mounting under /static
"""

# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_server import (
    _FRONTEND_DIR,
    clear_cache,
    healthcheck,
    root,
    status,
)

# ------------------------------------------------------------------
# Helpers – lightweight app that mirrors the real routes but skips
# the heavy lifespan (no CUDA / ZMQ required).
# ------------------------------------------------------------------


def _make_test_app(
    *,
    engine: MagicMock | None = None,
    mount_static: bool = True,
) -> FastAPI:
    """Build a minimal FastAPI app with the same routes."""
    test_app = FastAPI()

    # Register the same route handlers
    test_app.get("/")(root)
    test_app.get("/api/healthcheck")(healthcheck)
    test_app.post("/api/clear-cache")(clear_cache)
    test_app.get("/api/status")(status)

    # Optionally mount static files
    if mount_static and _FRONTEND_DIR.is_dir():
        # Third Party
        from fastapi.staticfiles import StaticFiles

        test_app.mount(
            "/static",
            StaticFiles(directory=str(_FRONTEND_DIR)),
            name="static",
        )

    test_app.state.engine = engine
    return test_app


@pytest.fixture
def mock_engine():
    """Create a mock cache engine."""
    engine = MagicMock()
    engine.report_status.return_value = {
        "engine_type": "default",
        "sessions": {},
    }
    engine.clear.return_value = None
    return engine


@pytest.fixture
def client(mock_engine):
    """TestClient with a mock engine attached."""
    app = _make_test_app(engine=mock_engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_engine():
    """TestClient without an engine (simulates startup
    failure)."""
    app = _make_test_app(engine=None)
    with TestClient(app) as c:
        yield c


# ==================================================================
# GET /
# ==================================================================


class TestRootEndpoint:
    """Tests for the root endpoint serving the dashboard."""

    def test_returns_html_when_index_exists(self, client):
        """When index.html exists, root serves FileResponse."""
        resp = client.get("/")
        assert resp.status_code == 200
        # The real index.html contains the dashboard title
        assert "LMCache" in resp.text

    def test_returns_json_fallback_when_no_index(self, mock_engine):
        """When _FRONTEND_DIR has no index.html, root returns
        a JSON fallback."""
        fake_dir = Path("/nonexistent/dir")
        with patch(
            "lmcache.v1.multiprocess.http_server._FRONTEND_DIR",
            fake_dir,
        ):
            app = _make_test_app(engine=mock_engine, mount_static=False)
            with TestClient(app) as c:
                resp = c.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "LMCache HTTP API"


# ==================================================================
# GET /api/healthcheck
# ==================================================================


class TestHealthcheckEndpoint:
    """Tests for the healthcheck endpoint."""

    def test_healthy_when_engine_present(self, client):
        resp = client.get("/api/healthcheck")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_unhealthy_when_engine_missing(self, client_no_engine):
        resp = client_no_engine.get("/api/healthcheck")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert "engine" in data["reason"]


# ==================================================================
# POST /api/clear-cache
# ==================================================================


class TestClearCacheEndpoint:
    """Tests for the clear-cache endpoint."""

    def test_clear_cache_success(self, client, mock_engine):
        resp = client.post("/api/clear-cache")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_engine.clear.assert_called_once()

    def test_clear_cache_engine_missing(self, client_no_engine):
        resp = client_no_engine.post("/api/clear-cache")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "error"
        assert "engine" in data["reason"]


# ==================================================================
# GET /api/status
# ==================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint."""

    def test_returns_engine_report(self, client, mock_engine):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine_type"] == "default"
        mock_engine.report_status.assert_called_once()

    def test_status_engine_missing(self, client_no_engine):
        resp = client_no_engine.get("/api/status")
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data


# ==================================================================
# Static file serving
# ==================================================================


class TestStaticFileMount:
    """Verify that /static is mounted and serves assets."""

    def test_css_served(self, client):
        resp = client.get("/static/css/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    def test_js_served(self, client):
        resp = client.get("/static/js/mp_app.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]
