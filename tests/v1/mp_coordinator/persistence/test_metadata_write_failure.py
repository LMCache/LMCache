# SPDX-License-Identifier: Apache-2.0
"""When the metadata document cannot be written, the endpoint that changed
a pin or a quota undoes the change and answers 503, instead of a 200 for
intent the next restart would lose."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys

_GB = 1024**3
_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
_PIN_BODY = {
    "model_name": "m",
    "world_size": 1,
    "token_ids": _TOKENS,
    "cache_salt": "alice",
}


def _config(metadata: Path) -> MPCoordinatorConfig:
    return MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        chunk_size=4,
        metadata_path=str(metadata),
    )


@contextmanager
def _unwritable(tmp_path: Path) -> Iterator[TestClient]:
    """A coordinator whose metadata directory is a file, so every save fails
    the way a missing, read-only or full volume does."""
    (tmp_path / "blocker").write_text("")
    with TestClient(create_app(_config(tmp_path / "blocker" / "m.json"))) as c:
        yield c


def _controller(client: TestClient) -> FleetEvictionController:
    return client.app.state.ctx.controllers.get(FleetEvictionController)


def _pinned_keys(client: TestClient):
    keys, _ = resolve_object_keys(
        client.app.state.ctx.token_hasher, "m", 1, _TOKENS, "alice"
    )
    return keys


class TestQuotaWrites:
    def test_setting_a_quota_is_undone(self, tmp_path: Path):
        with _unwritable(tmp_path) as client:
            response = client.put("/quota/tenant-a", json={"limit_gb": 2})

            assert response.status_code == 503
            assert "blocker" in response.json()["detail"]
            assert client.get("/quota/tenant-a").json()["quota_exists"] is False

    def test_changing_a_quota_keeps_the_old_limit(self, tmp_path: Path):
        with _unwritable(tmp_path) as client:
            _controller(client).quota.set_quota("tenant-a", 1 * _GB)
            response = client.put("/quota/tenant-a", json={"limit_gb": 5})

            assert response.status_code == 503
            assert client.get("/quota/tenant-a").json()["quota_limit_gb"] == 1.0

    def test_removing_a_quota_is_undone(self, tmp_path: Path):
        with _unwritable(tmp_path) as client:
            _controller(client).quota.set_quota("tenant-a", 1 * _GB)

            assert client.delete("/quota/tenant-a").status_code == 503
            assert client.get("/quota/tenant-a").json()["quota_exists"] is True

    def test_setting_the_default_limit_is_undone(self, tmp_path: Path):
        """Undone matters most here: a default of 0 that silently reverts on
        restart would stop eviction for every tenant without a quota."""
        with _unwritable(tmp_path) as client:
            response = client.put("/quota/config", json={"default_limit_gb": 0})

            assert response.status_code == 503
            assert client.get("/quota/config").json()["default_limit_gb"] is None


class TestPins:
    def test_a_pin_is_undone(self, tmp_path: Path):
        with _unwritable(tmp_path) as client:
            assert client.post("/cache/pins", json=_PIN_BODY).status_code == 503
            assert client.get("/cache/pins").json()["total"] == 0

    def test_an_unpin_keeps_the_pin(self, tmp_path: Path):
        with _unwritable(tmp_path) as client:
            _controller(client).pin(_pinned_keys(client))
            response = client.request("DELETE", "/cache/pins", json=_PIN_BODY)

            assert response.status_code == 503
            pins = client.get("/cache/pins").json()
            assert pins["total"] == len(_pinned_keys(client))
            assert all(p["pin_count"] == 1 for p in pins["pins"])


class TestWritableStorage:
    def test_a_saved_change_still_answers_200(self, tmp_path: Path):
        """The failure path must not change the normal one."""
        with TestClient(create_app(_config(tmp_path / "metadata.json"))) as client:
            quota = client.put("/quota/tenant-a", json={"limit_gb": 2})
            pin = client.post("/cache/pins", json=_PIN_BODY)

            assert quota.status_code == 200
            assert pin.status_code == 200
        assert (tmp_path / "metadata.json").is_file()
