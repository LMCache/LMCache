# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import patch

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.inprocess.v1.internal_api_server.api_server import app


def test_metrics_reset_compatibility_route_is_available_inprocess():
    with patch(
        "lmcache.inprocess.v1.internal_api_server.vllm.metrics_api."
        "reset_observability_metrics"
    ) as reset_metrics:
        response = TestClient(app).post("/metrics/reset")

    assert response.status_code == 200
    assert response.text == "ok"
    reset_metrics.assert_called_once_with()
