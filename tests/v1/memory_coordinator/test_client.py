# SPDX-License-Identifier: Apache-2.0
"""End-to-end client tests over a live HTTP server, epoch latching included."""

# Standard
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import replace
import socket
import threading
import time

# Third Party
import httpx
import pytest
import uvicorn

# First Party
from lmcache.v1.memory_coordinator.api import MemoryCoordinatorError, StaleEpochError
from lmcache.v1.memory_coordinator.app import create_app
from lmcache.v1.memory_coordinator.client import MemoryCoordinatorHttpClient
from lmcache.v1.memory_coordinator.config import MemoryCoordinatorConfig

# Local
from .conftest import item as _item
from .conftest import key as _key
from .conftest import ref as _ref


@contextmanager
def _server(config: MemoryCoordinatorConfig, port: int) -> Iterator[None]:
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config), host="127.0.0.1", port=port, log_level="error"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("coordinator server did not start")
            time.sleep(0.01)
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_client_roundtrip_and_usage(
    coordinator_config: MemoryCoordinatorConfig,
    port: int,
) -> None:
    with (
        _server(coordinator_config, port),
        closing(
            MemoryCoordinatorHttpClient(
                f"http://127.0.0.1:{port}",
                coordinator_config.token_file,
            )
        ) as client,
    ):
        contract = client.region_contract()
        assert contract.region_id == "region"

        grants = client.reserve_writes([_item(1), _item(2)])
        granted = [grant for grant in grants if grant is not None]
        assert len(granted) == 2
        client.finish_writes([_ref(grant) for grant in granted])

        hits = client.lookup([_key(1), _key(3)])
        assert hits[0] is not None and hits[1] is None

        used, capacity = client.get_memory_usage()
        assert capacity == coordinator_config.capacity_bytes
        assert used >= granted[0].handle.length + granted[1].handle.length
        # Duplicate stores do not allocate a second physical copy.
        assert client.reserve_writes([_item(1)]) == [None]
        assert client.get_memory_usage()[0] == used
        client.close()
        with pytest.raises(MemoryCoordinatorError, match="closed"):
            client.status()


def test_client_latches_epoch_and_fails_closed_after_restart(
    coordinator_config: MemoryCoordinatorConfig,
    port: int,
) -> None:
    endpoint = f"http://127.0.0.1:{port}"
    with ExitStack() as clients:
        with _server(coordinator_config, port):
            client = clients.enter_context(
                closing(
                    MemoryCoordinatorHttpClient(endpoint, coordinator_config.token_file)
                )
            )
            assert client.lookup([_key(1)]) == [None]
        # A fresh latch models coordinated reset; no DAX views exist in this test.
        reset = replace(
            coordinator_config, state_file=coordinator_config.state_file + ".reset"
        )
        with _server(reset, port):
            for operation in (
                lambda: client.finish_writes([]),
                lambda: client.lookup([_key(1)]),
                lambda: client.reserve_writes([_item(2)]),
                client.status,
            ):
                with pytest.raises(StaleEpochError):
                    operation()
            with closing(
                MemoryCoordinatorHttpClient(endpoint, reset.token_file)
            ) as replacement:
                assert replacement.lookup([_key(1)]) == [None]


@pytest.mark.parametrize(
    ("operation", "failure"),
    [
        (operation, failure)
        for operation in ("lookup", "reserve_writes", "finish_writes", "abort_writes")
        for failure in ("timeout", "server")
    ]
    + [("lookup", "stale_epoch"), ("lookup", "changed_epoch")],
)
def test_client_fences_ambiguous_writes_but_not_failed_lookups(
    coordinator_config: MemoryCoordinatorConfig,
    port: int,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: str,
) -> None:
    with (
        _server(coordinator_config, port),
        closing(
            MemoryCoordinatorHttpClient(
                f"http://127.0.0.1:{port}", coordinator_config.token_file
            )
        ) as client,
    ):
        requests = []

        def fail_request(
            _client: httpx.Client, request: httpx.Request, **_kwargs: object
        ) -> httpx.Response:
            requests.append(request)
            if failure == "timeout":
                raise httpx.ReadTimeout(
                    "injected lookup/write timeout", request=request
                )
            if failure == "stale_epoch":
                return httpx.Response(409, json={"error": "stale_epoch"})
            if failure == "changed_epoch":
                return httpx.Response(200, json={"region_epoch": "other", "hits": []})
            return httpx.Response(503, text="unavailable")

        fenced = operation != "lookup" or failure in ("stale_epoch", "changed_epoch")
        with monkeypatch.context() as patch:
            patch.setattr(httpx.Client, "send", fail_request)
            error = (
                httpx.ReadTimeout if failure == "timeout" else MemoryCoordinatorError
            )
            with pytest.raises(error):
                getattr(client, operation)([])
            if fenced:
                with pytest.raises(StaleEpochError, match="fenced"):
                    client.lookup([])
            assert len(requests) == 1

        if not fenced:
            assert client.lookup([_key(1)]) == [None]
            grant = client.reserve_writes([_item(1)])[0]
            assert grant is not None
            client.finish_writes([_ref(grant)])
            assert client.lookup([_key(1)])[0] is not None
