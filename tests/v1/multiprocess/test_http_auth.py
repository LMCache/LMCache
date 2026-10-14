# SPDX-License-Identifier: Apache-2.0
"""Tests for MP HTTP server API-key authentication.

Covers the bearer-token middleware on its own and how ``run_http_server``
installs it: fail-closed on a non-loopback host without a key, the explicit
opt-out, and keeping the key out of the configs ``GET /config`` serves.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.multiprocess import http_server as http_server_module
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    HTTPFrontendConfig,
    MPServerConfig,
)
from lmcache.v1.multiprocess.http_auth import APIKeyAuthMiddleware

API_KEY = "s3cret"


def _protected_app() -> FastAPI:
    app = FastAPI()

    @app.get("/healthcheck")
    async def healthcheck() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/env")
    async def env() -> dict[str, str]:
        return {"HOME": "/root"}

    app.add_middleware(APIKeyAuthMiddleware, api_key=API_KEY)
    return app


class TestAPIKeyAuthMiddleware:
    def test_request_without_key_is_rejected(self):
        response = TestClient(_protected_app()).get("/env")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json() == {"detail": "Missing or invalid API key"}

    @pytest.mark.parametrize(
        "authorization",
        ["Bearer wrong", "Bearer s3cret-suffix", "Basic s3cret", API_KEY, "Bearer "],
    )
    def test_request_with_wrong_credentials_is_rejected(self, authorization):
        response = TestClient(_protected_app()).get(
            "/env", headers={"Authorization": authorization}
        )
        assert response.status_code == 401

    def test_request_with_key_is_served(self):
        response = TestClient(_protected_app()).get(
            "/env", headers={"Authorization": "Bearer " + API_KEY}
        )
        assert response.status_code == 200
        assert response.json() == {"HOME": "/root"}

    def test_healthcheck_needs_no_key(self):
        """Kubernetes probes call /healthcheck without credentials."""
        response = TestClient(_protected_app()).get("/healthcheck")
        assert response.status_code == 200

    def test_unknown_path_is_rejected_before_routing(self):
        """Unauthenticated callers cannot probe which routes exist."""
        response = TestClient(_protected_app()).get("/no-such-route")
        assert response.status_code == 401

    def test_empty_key_is_rejected(self):
        with pytest.raises(ValueError):
            APIKeyAuthMiddleware(FastAPI(), api_key="")


class TestRunHTTPServerAuth:
    @pytest.fixture
    def server_app(self, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
        """Swap in a fresh app and a no-op uvicorn so nothing is served.

        Returns:
            The app ``run_http_server`` configures, with one ``/probe`` route.
        """
        app = FastAPI()

        @app.get("/probe")
        async def probe() -> dict[str, str]:
            return {"status": "ok"}

        monkeypatch.setattr(http_server_module, "app", app)
        monkeypatch.setattr(http_server_module, "_configs", {})
        monkeypatch.setattr(http_server_module.uvicorn, "Server", MagicMock())
        return app

    @staticmethod
    def _run(http_config: HTTPFrontendConfig) -> None:
        http_server_module.run_http_server(
            http_config=http_config,
            mp_config=MPServerConfig(),
            storage_manager_config=MagicMock(),
            obs_config=ObservabilityConfig(),
            coordinator_config=CoordinatorConfig(),
        )

    def test_key_installs_authentication(self, server_app):
        self._run(HTTPFrontendConfig(http_host="0.0.0.0", http_api_key=API_KEY))
        client = TestClient(server_app)
        assert client.get("/probe").status_code == 401
        authorized = client.get("/probe", headers={"Authorization": "Bearer s3cret"})
        assert authorized.status_code == 200

    def test_served_configs_do_not_contain_key(self, server_app):
        """``GET /config`` and runtime plugins read these configs."""
        self._run(HTTPFrontendConfig(http_host="0.0.0.0", http_api_key=API_KEY))
        served = server_app.state.configs["http"]
        assert served.http_api_key == ""
        assert served.http_host == "0.0.0.0"

    def test_non_loopback_without_key_refuses_to_start(self, server_app):
        with pytest.raises(ValueError, match="LMCACHE_HTTP_API_KEY"):
            self._run(HTTPFrontendConfig(http_host="0.0.0.0"))
        http_server_module.uvicorn.Server.assert_not_called()

    def test_opt_out_serves_without_authentication(self, server_app):
        self._run(
            HTTPFrontendConfig(http_host="0.0.0.0", http_allow_unauthenticated=True)
        )
        assert TestClient(server_app).get("/probe").status_code == 200
        http_server_module.uvicorn.Server.return_value.run.assert_called_once()

    def test_loopback_without_key_serves_without_authentication(self, server_app):
        self._run(HTTPFrontendConfig())
        assert TestClient(server_app).get("/probe").status_code == 200
