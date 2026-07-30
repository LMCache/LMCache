# SPDX-License-Identifier: Apache-2.0
"""Coordinator child lifecycle and singleton-pool tests."""

# Standard
from multiprocessing.context import AuthenticationError
from pathlib import Path
import socket

# Third Party
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.shared_l1_service import (
    connect_shared_l1_manager,
    read_shared_l1_authkey,
    start_shared_l1_manager,
)

_AUTHKEY = b"test-shared-l1-key"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _start(port: int):
    return start_shared_l1_manager(
        host="127.0.0.1",
        port=port,
        authkey=_AUTHKEY,
        region_id="region",
        capacity=64 * 1024,
        alignment=4096,
        layout_id="layout",
    )


def test_two_clients_share_the_childs_single_pool() -> None:
    port = _free_port()
    server = _start(port)
    try:
        pool_a = connect_shared_l1_manager("127.0.0.1", port, _AUTHKEY).get_pool()
        pool_b = connect_shared_l1_manager("127.0.0.1", port, _AUTHKEY).get_pool()
        layout = MemoryLayoutDesc([torch.Size([4, 4])], [torch.float16])
        write = pool_a.reserve_writes([("shared-key", layout)])[0]
        assert write is not None
        pool_a.finish_writes([write])
        read = pool_b.reserve_reads(["shared-key"])[0]
        assert read is not None and read.handle == write.handle
        pool_b.finish_reads([read])
        assert pool_a.snapshot()["objects"]["shared-key"] == {
            "handle": write.handle,
            "state": "VALID",
            "active_readers": 0,
        }
    finally:
        server.shutdown()


def test_manager_rejects_wrong_authkey() -> None:
    port = _free_port()
    server = _start(port)
    try:
        with pytest.raises(AuthenticationError):
            connect_shared_l1_manager("127.0.0.1", port, b"wrong")
    finally:
        server.shutdown()


def test_app_lifespan_starts_and_stops_child(tmp_path: Path) -> None:
    port = _free_port()
    authkey_file = tmp_path / "authkey"
    authkey_file.write_bytes(_AUTHKEY)
    app = create_app(
        MPCoordinatorConfig(
            health_check_interval=0.0,
            eviction_check_interval=0.0,
            enable_startup_resync=False,
            shared_l1_host="127.0.0.1",
            shared_l1_port=port,
            shared_l1_authkey_file=str(authkey_file),
            shared_l1_region_id="region",
            shared_l1_capacity_bytes=64 * 1024,
            shared_l1_alignment_bytes=4096,
            shared_l1_layout_id="layout",
        )
    )
    with TestClient(app):
        contract = (
            connect_shared_l1_manager(
                "127.0.0.1",
                port,
                _AUTHKEY,
            )
            .get_pool()
            .region_contract()
        )
        assert contract.region_id == "region"

    with pytest.raises(ConnectionRefusedError):
        connect_shared_l1_manager("127.0.0.1", port, _AUTHKEY)


def test_authkey_file_must_be_nonempty_and_regular(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_shared_l1_authkey(str(empty))
    with pytest.raises(ValueError, match="regular file"):
        read_shared_l1_authkey(str(tmp_path))
