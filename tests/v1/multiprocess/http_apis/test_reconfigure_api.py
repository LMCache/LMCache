# SPDX-License-Identifier: Apache-2.0
"""Tests for MP runtime reconfiguration HTTP endpoints."""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional
import asyncio
import contextlib
import threading

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.reconfiguration import L2ReconfigureError
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
)
from lmcache.v1.mp_coordinator.registrar import keep_registered
from lmcache.v1.multiprocess.http_apis.l1_reconfigure_api import router as l1_router
from lmcache.v1.multiprocess.http_apis.reconfigure_api import router

_DAX_OPS = ["status", "add", "remove", "resize"]


def _adapter_status(
    backend: str,
    adapter_index: int,
    status: Optional[dict[str, object]] = None,
    supported_operations: Optional[list[str]] = None,
) -> dict[str, object]:
    return {
        "backend": backend,
        "supported_operations": supported_operations or [],
        "status": status or {},
        "adapter_index": adapter_index,
    }


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
                _adapter_status(
                    "dax",
                    0,
                    {"hotplug_enabled": True, "devices": []},
                    _DAX_OPS,
                )
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

    status_resp = client.get("/reconfigure/dax/l2/status")
    add_resp = client.post(
        "/reconfigure/dax/l2/add",
        json={
            "adapter_index": 0,
            "device_path": "/dev/daxX.X",
            "size": "2GiB",
        },
    )
    remove_resp = client.post(
        "/reconfigure/dax/l2/remove",
        json={
            "adapter_index": 0,
            "device_path": "/dev/daxX.X",
            "mode": "drain",
            "force": True,
        },
    )
    resize_resp = client.post(
        "/reconfigure/dax/l2/resize",
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
    assert status_resp.json()["backend"] == "dax"
    assert status_resp.json()["num_adapters"] == 1
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
                _adapter_status("fake", 0, {"ready": True}, ["flip"]),
                _adapter_status(
                    "dax",
                    1,
                    {"hotplug_enabled": True, "devices": []},
                    _DAX_OPS,
                ),
            ],
        }
    )

    resp = _client(sm).get("/reconfigure/dax/l2/status")

    assert resp.status_code == 200
    assert resp.json() == {
        "enabled": True,
        "backend": "dax",
        "num_adapters": 1,
        "adapters": [
            {
                "backend": "dax",
                "supported_operations": _DAX_OPS,
                "status": {"hotplug_enabled": True, "devices": []},
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
                _adapter_status("fake", 0, {"ready": True}, ["flip"]),
                _adapter_status(
                    "dax",
                    1,
                    {"hotplug_enabled": True, "devices": []},
                    _DAX_OPS,
                ),
            ],
        }
    )

    resp = _client(sm).post(
        "/reconfigure/dax/l2/add",
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
    resp = _client(_FakeStorageManager()).post("/reconfigure/dax/l2/add", json=payload)
    assert resp.status_code == status_code


def test_size_rejects_boolean_and_float_payloads() -> None:
    resp = _client(_FakeStorageManager()).post(
        "/reconfigure/dax/l2/add", json={"device_path": "/dev/daxX.X", "size": True}
    )
    assert resp.status_code == 422
    resp = _client(_FakeStorageManager()).post(
        "/reconfigure/dax/l2/resize",
        json={"device_path": "/dev/daxX.X", "size": 4096.0},
    )
    assert resp.status_code == 422


def test_add_rejects_pathological_size_string_without_echoing_input():
    sm = _FakeStorageManager()
    bad_size = "9" + " " * 5000 + "x"

    resp = _client(sm).post(
        "/reconfigure/dax/l2/add",
        json={"device_path": "/dev/daxX.X", "size": bad_size},
    )

    assert resp.status_code == 400
    assert bad_size not in resp.text
    assert sm.calls == []


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/reconfigure/dax/l2/remove",
            {"device_path": "/dev/daxX.X", "timeout_s": 1},
        ),
        (
            "/reconfigure/dax/l2/resize",
            {"device_path": "/dev/daxX.X", "size": 1024, "timeout_s": 1},
        ),
        (
            "/reconfigure/dax/l2/resize",
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
        "/reconfigure/dax/l2/add",
        json={
            "device_path": "/dev/daxX.X",
            "size": 1024,
        },
    )
    assert resp.status_code == 507
    assert resp.json() == {"error": "no active destination DAX capacity"}


def test_generic_backend_routes_payload_to_matching_reconfigurable_adapter():
    sm = _FakeStorageManager(
        status={
            "enabled": True,
            "num_adapters": 2,
            "adapters": [
                _adapter_status("fake", 0, {"ready": True}, ["flip"]),
                _adapter_status("dax", 1, {"hotplug_enabled": True}, _DAX_OPS),
            ],
        }
    )

    resp = _client(sm).post(
        "/reconfigure/fake/l2/flip",
        json={"adapter_index": 0, "enabled": True},
    )

    assert resp.status_code == 200
    assert sm.calls == [
        ("status", ()),
        ("reconfigure", (0, "flip", {"enabled": True})),
    ]


def test_reconfigure_post_rejects_missing_backend_adapter():
    sm = _FakeStorageManager(
        status={
            "enabled": True,
            "num_adapters": 1,
            "adapters": [
                _adapter_status("dax", 0, {"hotplug_enabled": True}, _DAX_OPS)
            ],
        }
    )

    resp = _client(sm).post("/reconfigure/fake/l2/flip", json={"enabled": True})

    assert resp.status_code == 404
    assert resp.json() == {"error": "fake adapter not found"}


def test_old_dax_routes_are_not_registered() -> None:
    resp = _client(_FakeStorageManager()).get("/reconfigure/dax/status")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    ("tier", "operation"),
    [("l1", "add"), ("l1", "remove"), ("l2", "add")],
)
def test_blocked_reconfigure_and_status_allow_http_and_heartbeats(
    tier: str, operation: str
) -> None:
    """Exercise actual ASGI routes and registrar on the same asyncio loop."""
    loop_thread = threading.get_ident()
    lock = threading.Lock()
    entered = threading.Event()
    status_entered = threading.Event()
    released = threading.Event()
    lock.acquire()
    release_lock = threading.Lock()

    def release() -> None:
        with release_lock:
            if not released.is_set():
                released.set()
                lock.release()

    # A regressed loop-blocking route cannot prevent test cleanup.
    watchdog = threading.Timer(5, release)
    watchdog.start()

    def blocked(*args: object, **kwargs: object) -> object:
        entered.set()
        with lock:
            arena = DevDaxArenaStatus(
                device_path="/dev/dax-test",
                size_in_bytes=4096,
                used_bytes=4096,
                free_bytes=0,
                active_allocations=1,
                state=DevDaxArenaState.DRAINING,
                is_primary=False,
            )
            if tier == "l1":
                return arena
            return {"status": "ok"}

    def status() -> object:
        assert threading.get_ident() != loop_thread, "status probe ran on the loop"
        l2_status = {"adapters": [{"backend": "dax", "adapter_index": 0}]}
        if tier == "l2" and not entered.is_set():
            # Let backend resolution finish so the mutation can hold the lock.
            return l2_status
        status_entered.set()
        with lock:
            return [] if tier == "l1" else l2_status

    async def run() -> None:
        heartbeat = asyncio.Event()

        def coordinator(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200, json={"instance_id": "test", "re_registered": False}
                )
            if request.method == "PUT":
                heartbeat.set()
            return httpx.Response(200)

        app = FastAPI()
        app.include_router(l1_router)
        app.include_router(router)
        app.state.engine = SimpleNamespace(
            storage_manager=SimpleNamespace(
                get_l1_devdax_arena_statuses=status,
                add_l1_devdax_device=blocked,
                remove_l1_devdax_device=blocked,
                get_l2_adapter_reconfigure_status=status,
                reconfigure_l2_adapter=blocked,
            )
        )

        @app.get("/ping")
        async def ping() -> dict:
            return {"ok": True}

        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(coordinator)) as coord,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            registrar = asyncio.create_task(
                keep_registered(
                    coord,
                    "http://coord",
                    http_port=8080,
                    advertise_ip="127.0.0.1",
                    heartbeat_interval=0.01,
                    on_registered=lambda: None,
                )
            )
            url = f"/reconfigure/dax/{tier}/{operation}"
            payload: dict[str, object] = {"device_path": "/dev/dax-test"}
            if operation == "add":
                payload["size"] = 4096
            request = asyncio.create_task(client.post(url, json=payload))
            status_request: asyncio.Task[httpx.Response] | None = None

            async def check_progress() -> None:
                nonlocal status_request
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                status_request = asyncio.create_task(
                    client.get(f"/reconfigure/dax/{tier}/status")
                )
                while not status_entered.is_set():
                    await asyncio.sleep(0.001)
                heartbeat.clear()
                assert (await client.get("/ping")).json() == {"ok": True}
                await heartbeat.wait()
                assert not released.is_set(), "blocking call stalled the event loop"
                assert not request.done(), "response preceded operation completion"
                if status_request is not None:
                    assert not status_request.done()

            try:
                await asyncio.wait_for(check_progress(), timeout=2)
                release()
                response = await request
                assert response.status_code == 200
                if tier == "l1" and operation == "remove":
                    assert (
                        response.json()["removed"]["arenas"][0]["state"] == "draining"
                    )
                if status_request is not None:
                    assert (await status_request).status_code == 200
            finally:
                release()
                await asyncio.gather(
                    request,
                    *([status_request] if status_request else []),
                    return_exceptions=True,
                )
                registrar.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await registrar

    try:
        asyncio.run(run())
    finally:
        release()
        watchdog.cancel()
        watchdog.join()
