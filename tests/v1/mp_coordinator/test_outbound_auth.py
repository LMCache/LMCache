# SPDX-License-Identifier: Apache-2.0
"""Tests that the coordinator authenticates its calls to MP servers."""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _outbound_headers(mp_server_api_key: str) -> dict[str, str]:
    """Start the coordinator and return its MP-server client's headers."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        mp_server_api_key=mp_server_api_key,
    )
    with TestClient(create_app(config)) as client:
        return dict(client.app.state.outbound_client.headers)


def test_outbound_client_sends_mp_server_api_key():
    """Eviction, prefetch and cache fan-out all share this client."""
    assert _outbound_headers("s3cret")["authorization"] == "Bearer s3cret"


def test_outbound_client_sends_no_authorization_without_key():
    assert "authorization" not in _outbound_headers("")
