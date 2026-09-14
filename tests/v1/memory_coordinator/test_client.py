# SPDX-License-Identifier: Apache-2.0
"""End-to-end client tests over a live HTTP server, epoch latching included."""

# Standard
from dataclasses import replace
from pathlib import Path
import socket
import threading
import time

# Third Party
import pytest
import uvicorn

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey
from lmcache.v1.memory_coordinator.api import (
    ReservationRef,
    StaleEpochError,
    WireLayout,
    WriteGrant,
    WriteReserveItem,
)
from lmcache.v1.memory_coordinator.app import create_app
from lmcache.v1.memory_coordinator.client import MemoryCoordinatorHttpClient
from lmcache.v1.memory_coordinator.config import MemoryCoordinatorConfig

_TOKEN = "test-memory-coordinator-token"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _ServerThread:
    """Run one coordinator app in a background uvicorn thread."""

    def __init__(self, config: MemoryCoordinatorConfig, port: int) -> None:
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(config),
                host="127.0.0.1",
                port=port,
                log_level="error",
            )
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("coordinator server did not start")
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture
def coordinator_config(tmp_path: Path) -> MemoryCoordinatorConfig:
    token_file = tmp_path / "token"
    token_file.write_text(_TOKEN)
    return MemoryCoordinatorConfig(
        token_file=str(token_file),
        state_file=str(tmp_path / "coordinator.state"),
        region_id="region",
        capacity_bytes=64 * 1024,
        alignment_bytes=4096,
        layout_id="layout",
    )


def _key(seed: int) -> EncodedObjectKey:
    return EncodedObjectKey(chunk_hash_hex=f"{seed:08x}", model_name="model", kv_rank=0)


def _item(seed: int) -> WriteReserveItem:
    return WriteReserveItem(
        key=_key(seed),
        layout=WireLayout(shapes=[[64]], dtypes=["float16"]),
    )


def _ref(grant: WriteGrant) -> ReservationRef:
    return ReservationRef(key=grant.key, token=grant.token)


def test_client_roundtrip_and_usage(
    coordinator_config: MemoryCoordinatorConfig,
) -> None:
    port = _free_port()
    server = _ServerThread(coordinator_config, port)
    server.start()
    client = None
    try:
        client = MemoryCoordinatorHttpClient(
            f"http://127.0.0.1:{port}",
            coordinator_config.token_file,
        )
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
    finally:
        if client is not None:
            client.close()
        server.stop()


def test_client_latches_epoch_and_fails_closed_after_restart(
    coordinator_config: MemoryCoordinatorConfig,
) -> None:
    port = _free_port()
    first = _ServerThread(coordinator_config, port)
    first.start()
    client = None
    second = None
    try:
        client = MemoryCoordinatorHttpClient(
            f"http://127.0.0.1:{port}",
            coordinator_config.token_file,
        )
        assert client.lookup([_key(1)]) == [None]

        # Simulate a coordinated reset using a fresh latch. This test has no
        # mappings or transfers that could retain access to old extents.
        first.stop()
        second = _ServerThread(
            replace(
                coordinator_config,
                state_file=coordinator_config.state_file + ".reset",
            ),
            port,
        )
        second.start()

        # Even an empty release batch observes the epoch check: the first
        # post-restart operation fences the client no matter its shape.
        with pytest.raises(StaleEpochError):
            client.finish_writes([])
        # The client is fenced: subsequent operations fail without a
        # coordinated reset/restart, even though the coordinator is healthy.
        with pytest.raises(StaleEpochError):
            client.lookup([_key(1)])
        with pytest.raises(StaleEpochError):
            client.reserve_writes([_item(2)])
        with pytest.raises(StaleEpochError):
            client.status()

        # Only an explicit new client (the coordinated restart) may adopt
        # the new epoch.
        replacement = MemoryCoordinatorHttpClient(
            f"http://127.0.0.1:{port}",
            coordinator_config.token_file,
        )
        assert replacement.lookup([_key(1)]) == [None]
        replacement.close()
    finally:
        if client is not None:
            client.close()
        if second is not None:
            second.stop()
        else:
            first.stop()
