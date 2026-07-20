# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import argparse

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    MPServerConfig,
    P2PConfig,
    add_p2p_args,
    parse_args_to_p2p_config,
)
from lmcache.v1.multiprocess.http_server import run_http_server
from lmcache.v1.multiprocess.modules import p2p_controller


def _parse(argv: list[str]) -> P2PConfig:
    parser = argparse.ArgumentParser()
    add_p2p_args(parser)
    return parse_args_to_p2p_config(parser.parse_args(argv))


def test_native_verbs_cli_round_trip():
    config = _parse(
        [
            "--p2p-advertise-url",
            "10.0.0.1:7600",
            "--p2p-transfer-engine",
            "verbs",
            "--p2p-rdma-device",
            "mlx5_0,mlx5_1",
            "--p2p-rdma-port",
            "2",
            "--p2p-rdma-gid-indices",
            "3,5",
            "--p2p-rdma-queue-depth",
            "2048",
            "--p2p-rdma-handshake-timeout-ms",
            "5000",
        ]
    )

    assert config.transfer_engine == "verbs"
    assert config.rdma_device == "mlx5_0,mlx5_1"
    assert config.rdma_port == 2
    assert config.rdma_gid_indices == "3,5"
    assert config.rdma_queue_depth == 2048
    assert config.rdma_handshake_timeout_ms == 5000


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--p2p-transfer-engine", "verbs"], "device is required"),
        (
            [
                "--p2p-transfer-engine",
                "verbs",
                "--p2p-rdma-device",
                "mlx5_0,mlx5_1",
                "--p2p-rdma-gid-indices",
                "3",
            ],
            "count must match",
        ),
        (
            [
                "--p2p-transfer-engine",
                "verbs",
                "--p2p-rdma-device",
                "mlx5_0",
                "--p2p-rdma-queue-depth",
                "0",
            ],
            "queue depth",
        ),
        (
            [
                "--p2p-transfer-engine",
                "verbs",
                "--p2p-rdma-device",
                "mlx5_0",
                "--p2p-rdma-gid-index",
                "256",
            ],
            "gid index",
        ),
        (
            [
                "--p2p-transfer-engine",
                "verbs",
                "--p2p-rdma-device",
                "mlx5_0",
                "--p2p-rdma-handshake-timeout-ms",
                str(2**31),
            ],
            "handshake timeout",
        ),
    ],
)
def test_native_verbs_cli_rejects_invalid_options(argv, match):
    with pytest.raises(ValueError, match=match):
        _parse(argv)


def test_non_verbs_engine_ignores_verbs_specific_options():
    config = _parse(["--p2p-rdma-queue-depth", "0"])

    assert config.transfer_engine == "nixl"
    assert config.rdma_queue_depth == 0


def test_controller_forwards_verbs_options_only(monkeypatch):
    initialize = MagicMock()
    periodic = MagicMock()
    monkeypatch.setattr(
        p2p_controller,
        "initialize_transfer_channel_context",
        initialize,
    )
    monkeypatch.setattr(p2p_controller.httpx, "Client", MagicMock())
    monkeypatch.setattr(
        p2p_controller,
        "create_periodic_thread",
        MagicMock(return_value=periodic),
    )
    controller = object.__new__(p2p_controller.P2PController)
    controller._p2p_config = P2PConfig(
        advertise_url="10.0.0.1:7600",
        transfer_engine="verbs",
        rdma_device="mlx5_0,mlx5_1",
        rdma_port=2,
        rdma_gid_indices="3,5",
        rdma_queue_depth=2048,
        rdma_handshake_timeout_ms=5000,
    )
    controller._ctx = MagicMock()

    controller._start_orchestration(CoordinatorConfig(url="http://coordinator:9300"))

    initialize.assert_called_once_with(
        "verbs",
        controller._ctx.storage_manager.l1_memory_desc,
        listen_url="10.0.0.1:7600",
        advertise_url="10.0.0.1:7600",
        device_name="mlx5_0,mlx5_1",
        port_num=2,
        gid_index=-1,
        gid_indices="3,5",
        queue_depth=2048,
        handshake_timeout_ms=5000,
    )
    periodic.start.assert_called_once_with()


def test_controller_keeps_nixl_call_shape(monkeypatch):
    initialize = MagicMock()
    periodic = MagicMock()
    monkeypatch.setattr(
        p2p_controller,
        "initialize_transfer_channel_context",
        initialize,
    )
    monkeypatch.setattr(p2p_controller.httpx, "Client", MagicMock())
    monkeypatch.setattr(
        p2p_controller,
        "create_periodic_thread",
        MagicMock(return_value=periodic),
    )
    controller = object.__new__(p2p_controller.P2PController)
    controller._p2p_config = P2PConfig(
        advertise_url="10.0.0.1:7600",
        transfer_engine="nixl",
        rdma_queue_depth=0,
    )
    controller._ctx = MagicMock()

    controller._start_orchestration(CoordinatorConfig(url="http://coordinator:9300"))

    initialize.assert_called_once_with(
        "nixl",
        controller._ctx.storage_manager.l1_memory_desc,
        listen_url="10.0.0.1:7600",
        advertise_url="10.0.0.1:7600",
    )


def test_http_server_rejects_lazy_l1_for_native_verbs():
    storage_config = MagicMock()
    storage_config.l1_manager_config.gds_l1_config = None
    storage_config.l1_manager_config.memory_config.devdax_path = None
    storage_config.l1_manager_config.memory_config.use_lazy = True
    mp_config = MPServerConfig(
        p2p_config=P2PConfig(
            advertise_url="10.0.0.1:7600",
            transfer_engine="verbs",
            rdma_device="mlx5_0",
        )
    )

    with pytest.raises(ValueError, match="--no-l1-use-lazy"):
        run_http_server(
            http_config=MagicMock(),
            mp_config=mp_config,
            storage_manager_config=storage_config,
            obs_config=MagicMock(),
            coordinator_config=CoordinatorConfig(url="http://coordinator:9300"),
        )
