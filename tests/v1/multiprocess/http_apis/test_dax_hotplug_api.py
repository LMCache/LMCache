# SPDX-License-Identifier: Apache-2.0
"""Tests for MP DAX hotplug HTTP endpoints."""

# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.reconfiguration import L2ReconfigureError
from lmcache.v1.multiprocess.http_apis.dax_hotplug_api import router


@dataclass
class _FakeStorageManager:
    calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    raise_error: Optional[L2ReconfigureError] = None
    status: Optional[dict] = None

    def get_l2_adapter_reconfigure_status(self) -> dict:
        self.calls.append(("status", ()))
        if self.status is not None:
            return self.status
        return {
            "enabled": True,
            "num_adapters": 1,
            "adapters": [
                {
                    "type": "dax",
                    "hotplug_enabled": True,
                    "devices": [],
                    "adapter_index": 0,
                }
            ],
        }

    def reconfigure_l2_adapter(
        self,
        adapter_index: int,
        operation: str,
        payload: dict[str, object],
    ) -> dict:
        self.calls.append(("reconfigure", (adapter_index, operation, payload)))
        if self.raise_error is not None:
            raise self.raise_error
        return {"status": "ok", "operation": operation}


@dataclass
class _FakeEngine:
    storage_manager: _FakeStorageManager


def _client(sm: _FakeStorageManager) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.engine = _FakeEngine(storage_manager=sm)
    return TestClient(app)


def test_calls_storage_manager_without_timeout_and_without_accepted_response():
    sm = _FakeStorageManager()
    client = _client(sm)

    status_resp = client.get("/dax/status")
    add_resp = client.post(
        "/dax/add",
        json={
            "adapter_index": 0,
            "device_path": "/dev/daxX.X",
            "size": "2GiB",
        },
    )
    remove_resp = client.post(
        "/dax/remove",
        json={
            "adapter_index": 0,
            "device_path": "/dev/daxX.X",
            "mode": "drain",
            "force": True,
        },
    )
    resize_resp = client.post(
        "/dax/resize",
        json={
            "adapter_index": 0,
            "device_path": "/dev/daxX.X",
            "size": "1536MiB",
            "mode": "migrate",
            "force": False,
        },
    )

    assert status_resp.status_code == 200
    assert add_resp.status_code == 200
    assert remove_resp.status_code == 200
    assert resize_resp.status_code == 200
    assert status_resp.json()["num_dax_adapters"] == 1
    assert sm.calls == [
        ("status", ()),
        ("status", ()),
        (
            "reconfigure",
            (0, "add", {"device_path": "/dev/daxX.X", "size_bytes": 2 * 1024**3}),
        ),
        ("status", ()),
        (
            "reconfigure",
            (
                0,
                "remove",
                {"device_path": "/dev/daxX.X", "mode": "drain", "force": True},
            ),
        ),
        ("status", ()),
        (
            "reconfigure",
            (
                0,
                "resize",
                {
                    "device_path": "/dev/daxX.X",
                    "size_bytes": int(1.5 * 1024**3),
                    "mode": "migrate",
                    "force": False,
                },
            ),
        ),
    ]


def test_status_filters_non_dax_reconfigurable_adapters():
    sm = _FakeStorageManager(
        status={
            "enabled": True,
            "num_adapters": 2,
            "adapters": [
                {"type": "fake", "ready": True, "adapter_index": 0},
                {
                    "type": "dax",
                    "hotplug_enabled": True,
                    "devices": [],
                    "adapter_index": 1,
                },
            ],
        }
    )

    resp = _client(sm).get("/dax/status")

    assert resp.status_code == 200
    assert resp.json() == {
        "enabled": True,
        "hotplug_enabled": True,
        "num_dax_adapters": 1,
        "adapters": [
            {
                "type": "dax",
                "hotplug_enabled": True,
                "devices": [],
                "adapter_index": 0,
            }
        ],
    }


def test_add_resolves_public_dax_index_to_generic_reconfigure_index():
    sm = _FakeStorageManager(
        status={
            "enabled": True,
            "num_adapters": 2,
            "adapters": [
                {"type": "fake", "ready": True, "adapter_index": 0},
                {
                    "type": "dax",
                    "hotplug_enabled": True,
                    "devices": [],
                    "adapter_index": 1,
                },
            ],
        }
    )

    resp = _client(sm).post(
        "/dax/add",
        json={
            "adapter_index": 0,
            "device_path": "/dev/daxX.X",
            "size": 1024,
        },
    )

    assert resp.status_code == 200
    assert sm.calls == [
        ("status", ()),
        ("reconfigure", (1, "add", {"device_path": "/dev/daxX.X", "size_bytes": 1024})),
    ]


@pytest.mark.parametrize(
    ("payload", "status_code"),
    [
        ({"device_path": "/dev/daxX.X", "size_bytes": 1024}, 422),
        ({"device_path": "/dev/daxX.X", "size": "many"}, 400),
    ],
)
def test_add_rejects_invalid_size_payloads(
    payload: dict[str, object],
    status_code: int,
):
    resp = _client(_FakeStorageManager()).post("/dax/add", json=payload)
    assert resp.status_code == status_code


def test_add_rejects_pathological_size_string_without_echoing_input():
    sm = _FakeStorageManager()
    bad_size = "9" + " " * 5000 + "x"

    resp = _client(sm).post(
        "/dax/add",
        json={"device_path": "/dev/daxX.X", "size": bad_size},
    )

    assert resp.status_code == 400
    assert bad_size not in resp.text
    assert sm.calls == []


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/dax/remove", {"device_path": "/dev/daxX.X", "timeout_s": 1}),
        (
            "/dax/resize",
            {"device_path": "/dev/daxX.X", "size": 1024, "timeout_s": 1},
        ),
        (
            "/dax/resize",
            {"device_path": "/dev/daxX.X", "size": 1024, "mode": "drain"},
        ),
    ],
)
def test_rejects_removed_fields_and_invalid_resize_mode(
    path: str,
    payload: dict[str, object],
):
    resp = _client(_FakeStorageManager()).post(path, json=payload)
    assert resp.status_code == 422


def test_hotplug_error_status_code_is_preserved():
    sm = _FakeStorageManager(
        raise_error=L2ReconfigureError(
            507,
            "no active destination DAX capacity",
        )
    )
    resp = _client(sm).post(
        "/dax/add",
        json={
            "device_path": "/dev/daxX.X",
            "size": 1024,
        },
    )
    assert resp.status_code == 507
    assert resp.json() == {"error": "no active destination DAX capacity"}
