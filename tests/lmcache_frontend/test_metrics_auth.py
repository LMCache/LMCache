# SPDX-License-Identifier: Apache-2.0
"""Tests that the frontend's ``/metrics`` aggregation forwards caller auth.

The frontend holds no node credentials: a node that requires an API key is
scraped with the caller's own ``Authorization`` header, or without one.
"""

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.lmcache_frontend import app as app_module


class _FakeResponse:
    text = "lmcache_metric 1\n"

    def raise_for_status(self) -> None:
        pass


class _RecordingAsyncClient:
    """Stands in for ``httpx.AsyncClient``; records each ``get``'s headers."""

    sent_headers: list[dict[str, str]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "_RecordingAsyncClient":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def get(self, url: str, headers: dict[str, str]) -> _FakeResponse:
        type(self).sent_headers.append(headers)
        return _FakeResponse()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A frontend with one registered node and a recording HTTP client."""
    _RecordingAsyncClient.sent_headers = []
    monkeypatch.setattr(
        app_module,
        "target_nodes",
        [
            {
                "name": "proxy",
                "nodes": [{"name": "mp-1", "host": "10.0.0.1", "port": "8080"}],
            }
        ],
    )
    monkeypatch.setattr(app_module.httpx, "AsyncClient", _RecordingAsyncClient)
    return TestClient(app_module.create_app())


def test_metrics_forwards_caller_authorization(client: TestClient):
    response = client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 200
    assert "lmcache_metric 1" in response.text
    assert _RecordingAsyncClient.sent_headers == [{"Authorization": "Bearer s3cret"}]


def test_metrics_sends_no_authorization_when_caller_sends_none(client: TestClient):
    assert client.get("/metrics").status_code == 200
    assert _RecordingAsyncClient.sent_headers == [{}]
