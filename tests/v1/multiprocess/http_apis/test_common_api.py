# SPDX-License-Identifier: Apache-2.0
"""
Tests for common_api.py — the aggregation layer that pulls in
``internal_api_server/common`` routers while excluding
vLLM-specific modules.

Covers:
- ``run_script_api`` is NOT registered on the mp HTTP server.
- Other common endpoints (env, loglevel, metrics, …) ARE present.
"""

# Third Party
from fastapi import FastAPI

# First Party
from lmcache.v1.multiprocess.http_api_registry import HTTPAPIRegistry


def _app_with_all_apis() -> FastAPI:
    app = FastAPI()
    registry = HTTPAPIRegistry(app)
    registry.register_all_apis()
    return app


class TestCommonApiAggregation:
    def test_run_script_endpoint_excluded(self):
        """/run_script must NOT be registered on the mp server."""
        app = _app_with_all_apis()
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/run_script" not in paths

    def test_common_env_endpoint_present(self):
        """/env from env_api should be registered."""
        app = _app_with_all_apis()
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/env" in paths

    def test_common_loglevel_endpoint_present(self):
        """/loglevel from loglevel_api should be registered."""
        app = _app_with_all_apis()
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/loglevel" in paths

    def test_conf_endpoint_present(self):
        """/conf from conf_api should be registered."""
        app = _app_with_all_apis()
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/conf" in paths
