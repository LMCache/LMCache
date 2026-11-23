# SPDX-License-Identifier: Apache-2.0
"""Configurable credential-free paths on the MP coordinator.

`/healthz` is public by default so the liveness probe keeps working without
credentials; `/metrics` is not. Operators can replace the default list, which is
how a deployment publishes other paths or requires a key for everything.
"""

# Standard
import argparse

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.cli.commands.coordinator import _resolve_public_endpoints
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.http_auth import (
    PUBLIC_COORDINATOR_PATHS,
    CoordinatorAuthenticationConfig,
    CoordinatorAuthenticationMiddleware,
)


def _make_app(
    public_paths: list[str] | None,
    *,
    keys: tuple[str, ...] = ("secret",),
) -> FastAPI:
    """Build a minimal app with only the endpoints this behaviour concerns."""
    app = FastAPI()
    app.state.http_authentication = CoordinatorAuthenticationConfig(list(keys))
    app.add_middleware(CoordinatorAuthenticationMiddleware, public_paths=public_paths)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/metrics")
    def metrics() -> dict[str, str]:
        return {"ok": "metrics"}

    @app.get("/instances")
    def instances() -> dict[str, str]:
        return {"ok": "instances"}

    return app


def _config(**kwargs: object) -> MPCoordinatorConfig:
    return MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        **kwargs,  # type: ignore[arg-type]
    )


class TestOmittedKeepsTheDefault:
    def test_default_list_is_unchanged(self) -> None:
        assert _config().public_endpoints is None

    def test_only_the_probe_is_open_without_the_option(self) -> None:
        with TestClient(_make_app(None)) as client:
            assert client.get("/healthz").status_code == 200
            # Metrics describe fleet placement, so they are not public.
            assert client.get("/metrics").status_code == 401
            assert client.get("/instances").status_code == 401

    def test_the_middleware_default_matches_the_constant(self) -> None:
        with TestClient(_make_app(None)) as client:
            for path in PUBLIC_COORDINATOR_PATHS:
                assert client.get(path).status_code == 200


class TestCustomListReplacesTheDefault:
    def test_an_explicit_healthz_only_list_is_the_default(self) -> None:
        with TestClient(_make_app(["/healthz"])) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/metrics").status_code == 401
            assert (
                client.get(
                    "/metrics", headers={"Authorization": "Bearer secret"}
                ).status_code
                == 200
            )

    def test_a_valid_key_reaches_a_protected_path(self) -> None:
        with TestClient(_make_app(["/healthz"])) as client:
            response = client.get("/instances", headers={"X-LMCache-API-Key": "secret"})

        assert response.status_code == 200

    def test_an_invalid_key_is_rejected(self) -> None:
        with TestClient(_make_app(["/healthz"])) as client:
            response = client.get(
                "/instances", headers={"Authorization": "Bearer wrong"}
            )

        assert response.status_code == 401

    def test_paths_match_exactly_not_by_prefix(self) -> None:
        with TestClient(_make_app(["/healthz"])) as client:
            # '/healthz/detail' is not '/healthz'.
            assert client.get("/healthz/detail").status_code == 401

    def test_metrics_can_be_published_explicitly(self) -> None:
        with TestClient(_make_app(["/healthz", "/metrics"])) as client:
            assert client.get("/metrics").status_code == 200


class TestEmptyListProtectsEverything:
    def test_empty_list_requires_a_key_on_the_probe(self) -> None:
        with TestClient(_make_app([])) as client:
            assert client.get("/healthz").status_code == 401
            assert (
                client.get(
                    "/healthz", headers={"Authorization": "Bearer secret"}
                ).status_code
                == 200
            )

    def test_empty_list_still_opens_when_auth_is_disabled(self) -> None:
        with TestClient(_make_app([], keys=())) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/metrics").status_code == 200


class TestConfigValidation:
    def test_relative_paths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="absolute path"):
            _config(public_endpoints=("healthz",))

    def test_absolute_paths_are_accepted(self) -> None:
        assert _config(public_endpoints=("/healthz",)).public_endpoints == ("/healthz",)


class TestCliResolution:
    """The two flags have to be able to express all three states."""

    @staticmethod
    def _args(**kwargs: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "public_endpoint": None,
            "no_public_endpoints": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_unset_flags_leave_the_default(self) -> None:
        assert _resolve_public_endpoints(self._args()) is None

    def test_repeated_paths_become_a_tuple(self) -> None:
        args = self._args(public_endpoint=["/healthz", "/metrics"])
        assert _resolve_public_endpoints(args) == ("/healthz", "/metrics")

    def test_no_public_endpoints_is_the_empty_list(self) -> None:
        args = self._args(no_public_endpoints=True)
        assert _resolve_public_endpoints(args) == ()

    def test_the_two_flags_are_mutually_exclusive(self) -> None:
        args = self._args(no_public_endpoints=True, public_endpoint=["/healthz"])
        with pytest.raises(ValueError, match="cannot be combined"):
            _resolve_public_endpoints(args)
