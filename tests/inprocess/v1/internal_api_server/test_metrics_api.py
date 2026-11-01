# SPDX-License-Identifier: Apache-2.0
"""
Test the InProcess internal API server's ``POST /metrics/reset`` endpoint.

The shared MP common routers no longer expose this endpoint (the MP path does
not import the single-process observability stack); the internal API server
restores it locally because the engine's PrometheusLogger is live in-process.
"""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.inprocess.v1.internal_api_server.api_server import app


def test_metrics_reset_returns_ok():
    response = TestClient(app).post("/metrics/reset")
    assert response.status_code == 200
    assert response.text == "ok"
