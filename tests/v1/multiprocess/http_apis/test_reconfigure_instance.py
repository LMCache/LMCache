# SPDX-License-Identifier: Apache-2.0
"""A replacement worker cannot authorize release of its predecessor's memory."""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_auth import (
    HTTPAuthenticationConfig,
    HTTPAuthenticationMiddleware,
)
from lmcache.v1.multiprocess.http_reconfigure import ReconfigureInstanceMiddleware


def test_worker_restart_rejects_old_incarnation_before_hotplug() -> None:
    app = FastAPI()
    app.add_middleware(ReconfigureInstanceMiddleware)
    mutations = []

    @app.get("/reconfigure/dax/l1/status")
    def status() -> dict[str, object]:
        return {"arenas": []}

    @app.post("/reconfigure/dax/l1/add")
    def add() -> dict[str, object]:
        mutations.append("add")
        return {"status": "ok"}

    with TestClient(app) as client:
        original = client.get("/reconfigure/dax/l1/status").headers[
            "X-LMCache-Instance"
        ]
        headers = {"X-LMCache-Expected-Instance": original}
        assert (
            client.post("/reconfigure/dax/l1/add", headers=headers).status_code == 200
        )
        assert mutations == ["add"]
    with TestClient(app) as client:
        rejected = client.post("/reconfigure/dax/l1/add", headers=headers)
        assert rejected.status_code == 409
        assert rejected.headers["X-LMCache-Instance"] != original
        assert mutations == ["add"]
        assert client.post("/reconfigure/dax/l1/add").status_code == 200


@pytest.mark.parametrize("error", [BufferError, OSError])
def test_cleanup_errors_keep_identity_without_bypassing_auth(
    error: type[Exception],
) -> None:
    app = FastAPI()
    app.state.http_authentication = HTTPAuthenticationConfig(["secret"])
    app.add_middleware(ReconfigureInstanceMiddleware)
    app.add_middleware(HTTPAuthenticationMiddleware)
    mutations: list[str] = []

    @app.get("/reconfigure/dax/l1/status")
    def status() -> dict[str, object]:
        return {"arenas": []}

    @app.post("/reconfigure/dax/l1/remove")
    def remove() -> None:
        mutations.append("remove")
        raise error("private cleanup details")

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/reconfigure/dax/l1/remove").status_code == 401
        assert mutations == []
        headers = {"Authorization": "Bearer secret"}
        instance = client.get("/reconfigure/dax/l1/status", headers=headers).headers[
            "X-LMCache-Instance"
        ]
        headers["X-LMCache-Expected-Instance"] = instance
        response = client.post("/reconfigure/dax/l1/remove", headers=headers)
        assert response.status_code == 500
        assert response.headers.get("X-LMCache-Instance") == instance
        assert "private cleanup details" not in response.text
        assert mutations == ["remove"]
