# SPDX-License-Identifier: Apache-2.0
"""CXL peer eligibility and incarnation changes through discovery's public seams."""

# Standard
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.config import get_arg_parser, parse_args_to_config
from lmcache.v1.distributed.internal_api import CXL_METADATA_KEY, CxlArenaDescriptor
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    P2PConfig,
    add_mp_server_args,
)
from lmcache.v1.multiprocess.modules import p2p_controller
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController


def test_discovery_filters_pools_and_retries_session_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CXL discovery uses metadata without RDMA URLs and preserves failed drains."""
    local = CxlArenaDescriptor("pool", 0, 16384, 4096, "a")
    remote = replace(local, offset=32768, session_id="b")

    def instance(name: str, arena: CxlArenaDescriptor) -> dict:
        return {
            "instance_id": name,
            "ip": "127.0.0.1",
            "mq_port": 5555,
            "metadata": {CXL_METADATA_KEY: arena.to_json()},
        }

    payload = {
        "instances": [
            instance("self", local),
            instance("peer", remote),
            instance("other-pool", replace(remote, pool_id="other")),
            {
                "instance_id": "rdma",
                "ip": "127.0.0.2",
                "mq_port": 5555,
                "p2p_advertised_url": "127.0.0.2:9000",
            },
        ]
    }
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload),
        )
    )
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    timer_args: dict[str, Any] = {}

    def timer(**kwargs: Any) -> MagicMock:
        timer_args.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(p2p_controller, "create_periodic_thread", timer)
    context = MagicMock()
    context.storage_manager.cxl_arena = local
    context.storage_manager.add_cxl_peer.return_value = 0
    controller = P2PController(
        context,
        P2PConfig(transfer_engine="cxl"),
        CoordinatorConfig(url="http://coordinator"),
        "self",
    )
    poll = timer_args["execute_fn"]
    try:
        poll()
        assert controller.report_status()["p2p_peers"] == ["peer"]
        context.storage_manager.add_cxl_peer.assert_called_once_with(
            remote,
            "tcp://127.0.0.1:5555",
            30.0,
        )
        context.storage_manager.add_l2_adapter.assert_not_called()
        new_arena = replace(remote, session_id="restarted")
        payload["instances"][1] = instance("peer", new_arena)
        context.storage_manager.delete_l2_adapter.side_effect = [
            RuntimeError("still draining"),
            None,
        ]
        poll()
        assert context.storage_manager.add_cxl_peer.call_count == 1
        assert controller.report_status()["p2p_peer_count"] == 1
        poll()
        assert context.storage_manager.add_cxl_peer.call_count == 2
        assert context.storage_manager.add_cxl_peer.call_args.args[0] == new_arena
        context.storage_manager.delete_l2_adapter.side_effect = None
    finally:
        controller.close()


def test_cli_shared_cxl_slab_configuration() -> None:
    """Slab offset is bytes, and CXL mode enables P2P without an RDMA endpoint."""
    parser = add_mp_server_args(get_arg_parser())
    args = parser.parse_args(
        [
            "--l1-size-gb",
            "2",
            "--eviction-policy",
            "noop",
            "--l1-devdax-path",
            "/dev/dax0.0",
            "--no-l1-use-lazy",
            "--shm-name",
            "",
            "--l1-align-bytes",
            "2097152",
            "--cxl-pool-id",
            "pool",
            "--cxl-pool-offset",
            "2149580800",
        ]
    )
    config = parse_args_to_config(args).l1_manager_config.memory_config
    assert config.cxl_pool_id == "pool"
    assert config.cxl_pool_offset == 2149580800
    assert P2PConfig(transfer_engine="cxl").enabled
    assert not P2PConfig().enabled
