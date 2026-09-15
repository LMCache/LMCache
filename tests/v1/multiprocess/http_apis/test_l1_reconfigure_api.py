# SPDX-License-Identifier: Apache-2.0
"""Tests for MP runtime Device-DAX L1 reconfiguration HTTP endpoints."""

# Standard
from dataclasses import dataclass, field
import os

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.distributed.memory_manager.reconfiguration import (
    L1ReconfigureError,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
    DevDaxRemoveMode,
)
from lmcache.v1.multiprocess.http_apis.l1_reconfigure_api import router

_PRIMARY = "/dev/dax0.0"
_EXTRA = "/dev/dax0.1"
_MAPPED_BYTES = 4096


def _arena_status(
    device_path: str,
    *,
    state: DevDaxArenaState = DevDaxArenaState.ACTIVE,
    is_primary: bool = False,
) -> DevDaxArenaStatus:
    """Build an arena status used by the fake storage manager.

    Args:
        device_path: Path represented by the arena.
        state: Lifecycle state to report.
        is_primary: Whether the arena is the non-removable primary mapping.

    Returns:
        A fixed-size arena status suitable for HTTP response assertions.
    """
    return DevDaxArenaStatus(
        device_path=device_path,
        size_in_bytes=_MAPPED_BYTES,
        used_bytes=0,
        free_bytes=_MAPPED_BYTES if state == DevDaxArenaState.ACTIVE else 0,
        active_allocations=0,
        state=state,
        is_primary=is_primary,
    )


@dataclass
class _FakeStorageManager:
    calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    raise_error: L1ReconfigureError | None = None

    def _raise_if_configured(self) -> None:
        if self.raise_error is not None:
            raise self.raise_error

    def get_l1_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
        self.calls.append(("status", ()))
        self._raise_if_configured()
        return [_arena_status(_PRIMARY, is_primary=True)]

    def add_l1_devdax_device(
        self,
        device_path: str,
        size_in_bytes: int,
    ) -> DevDaxArenaStatus:
        self.calls.append(("add", (device_path, size_in_bytes)))
        self._raise_if_configured()
        return _arena_status(device_path)

    def remove_l1_devdax_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode,
    ) -> DevDaxArenaStatus:
        self.calls.append(("remove", (device_path, mode)))
        self._raise_if_configured()
        return _arena_status(
            os.path.normpath(device_path), state=DevDaxArenaState.REMOVED
        )


@dataclass
class _FakeEngine:
    storage_manager: _FakeStorageManager


def _client(sm: _FakeStorageManager) -> TestClient:
    """Create a test client whose engine exposes ``sm``.

    Args:
        sm: Fake storage manager receiving route delegation calls.

    Returns:
        A client with only the L1 reconfigure router registered.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.engine = _FakeEngine(storage_manager=sm)
    return TestClient(app)


def test_l1_dax_routes_requests_to_storage_manager() -> None:
    """The HTTP surface translates requests into StorageManager calls."""
    storage_manager = _FakeStorageManager()
    client = _client(storage_manager)

    response = client.get("/reconfigure/dax/l1/status")
    assert response.status_code == 200
    (arena,) = response.json()["arenas"]
    assert arena["device_path"] == _PRIMARY
    assert arena["state"] == "active"
    assert arena["is_primary"] is True

    response = client.post(
        "/reconfigure/dax/l1/add",
        json={"device_path": _EXTRA, "size": "4KiB"},
    )
    assert response.status_code == 200
    added = response.json()["added"]
    assert (added["device_path"], added["state"]) == (_EXTRA, "active")

    remove_alias = "/dev//dax0.1"
    response = client.post(
        "/reconfigure/dax/l1/remove",
        json={"device_path": remove_alias},
    )
    assert response.status_code == 200
    removed = response.json()["removed"]
    assert removed["device_path"] == _EXTRA
    assert removed["arenas"][0]["state"] == "removed"

    assert storage_manager.calls == [
        ("status", ()),
        ("add", (_EXTRA, _MAPPED_BYTES)),
        ("remove", (remove_alias, DevDaxRemoveMode.DRAIN)),
    ]


def test_l1_dax_reconfigure_error_status_is_preserved() -> None:
    storage_manager = _FakeStorageManager(
        raise_error=L1ReconfigureError(409, "mapping conflict")
    )
    response = _client(storage_manager).post(
        "/reconfigure/dax/l1/add",
        json={"device_path": _EXTRA, "size": _MAPPED_BYTES},
    )

    assert response.status_code == 409
    assert response.json() == {"error": "mapping conflict"}
    assert storage_manager.calls == [("add", (_EXTRA, _MAPPED_BYTES))]


def test_l1_dax_wire_validation() -> None:
    """Representative wire rejections: strict 422s and a size-value 400."""
    storage_manager = _FakeStorageManager()
    client = _client(storage_manager)

    # The wire type is strict: a JSON boolean is a schema violation, never
    # silently coerced into a byte count.
    response = client.post(
        "/reconfigure/dax/l1/add", json={"device_path": _EXTRA, "size": True}
    )
    assert response.status_code == 422
    assert "detail" in response.json()
    # A well-formed size string with a nonsense value is a 400 value error.
    response = client.post(
        "/reconfigure/dax/l1/add", json={"device_path": _EXTRA, "size": "4Zi"}
    )
    assert response.status_code == 400
    assert "error" in response.json()
    # The field is reserved for future remove strategies, but only the
    # currently implemented drain mode is accepted.
    response = client.post(
        "/reconfigure/dax/l1/remove",
        json={"device_path": _EXTRA, "mode": "evict"},
    )
    assert response.status_code == 422
    assert storage_manager.calls == []


def test_l1_dax_missing_engine_returns_503() -> None:
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/reconfigure/dax/l1/status")
    assert response.status_code == 503
    assert "engine" in response.json()["error"]
