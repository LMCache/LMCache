# SPDX-License-Identifier: Apache-2.0
"""Independent peer control RPCs, serialization, authentication and lifecycle."""

# Standard
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import MagicMock
import argparse

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.client.transport.errors import MPRequestError
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.common.transfer_channel.api import TransferChannelAddress
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.p2p import P2PClient
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    MPServerConfig,
    P2PConfig,
    add_p2p_args,
    parse_args_to_p2p_config,
)
from lmcache.v1.multiprocess.engine_module import modules_for_service
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.server import MPCacheServer
from lmcache.v1.multiprocess.transport.grpc_impl.server_builder import (
    build_grpc_request_server,
)
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from tests.v1.multiprocess.transport_test_utils import requires_pygrpc_native


@pytest.mark.parametrize(
    ("worker_transport", "peer_mode"),
    [
        ("zmq", "sync"),
        ("zmq", "aio"),
        pytest.param("zmq", "native", marks=requires_pygrpc_native),
        ("grpc", "sync"),
    ],
)
def test_independent_peer_channel(
    worker_transport: Literal["zmq", "grpc"],
    peer_mode: Literal["sync", "aio", "native"],
    unused_tcp_port_factory: Callable[[], int],
    tmp_path: Path,
) -> None:
    """P2P has its own port and lifetime with either worker transport."""
    keys = [ObjectKey(b"chunk", "model", 0)]
    layouts = {0: MemoryLayoutDesc([torch.Size([2, 4])], [torch.float16])}
    addresses = [TransferChannelAddress(offset=8, size=16)]
    unlocked = []

    class Peer(P2PController):
        @request_handler(HandlerType.BLOCKING)
        def p2p_lookup_and_lock(
            self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
        ) -> int:
            assert keys == [ObjectKey(b"chunk", "model", 0)]
            assert group_layout_descs == layouts
            return 41

        @request_handler(HandlerType.BLOCKING)
        def p2p_query_lookup_results(
            self, task_id: int
        ) -> list[TransferChannelAddress] | None:
            return None if task_id == 0 else addresses

        @request_handler(HandlerType.BLOCKING)
        def p2p_unlock_objects(self, keys: list[ObjectKey]) -> None:
            unlocked.extend(keys)

    class Worker:
        def close(self) -> None:
            pass

        @request_handler()
        def noop(self) -> str:
            return "worker"

    key_file = tmp_path / "key"
    key_file.write_text("secret\n")
    parser = add_p2p_args(argparse.ArgumentParser())
    peer_port = unused_tcp_port_factory() if worker_transport == "grpc" else 0
    args = ["--p2p-rpc-port", str(peer_port)] if peer_port else []
    p2p_config = parse_args_to_p2p_config(parser.parse_args(args))
    assert p2p_config.rpc_port == peer_port
    config = MPServerConfig(
        transport=worker_transport,
        grpc_server_mode=peer_mode,
        host="127.0.0.1",
        port=unused_tcp_port_factory(),
        api_key_files=[str(key_file)],
        p2p_config=replace(p2p_config, advertise_url="localhost:7600"),
    )
    peer = Peer(MagicMock(), P2PConfig(), CoordinatorConfig(), instance_id="peer")
    modules = [cast(Any, Worker()), peer]
    assert modules_for_service(modules, "p2p") == [peer]
    peer_server = build_grpc_request_server(
        modules_for_service(modules, "p2p"),
        replace(config, port=config.p2p_config.rpc_port),
        rpc_client=P2PClient,
    )
    engine = MPCacheServer(peer.context, modules, peer_server)
    peer_server.start()
    if peer_port:
        assert engine.mq_port == peer_port
    peer_port = engine.mq_port
    assert peer_port > 0 and peer_port != config.port
    peer_url = f"grpc://127.0.0.1:{peer_port}"
    worker = create_request_server(modules_for_service(modules, "worker"), config)
    worker.start()
    scheme = "tcp" if worker_transport == "zmq" else "grpc"
    worker_client = RequestClientFactory.create(
        f"{scheme}://127.0.0.1:{config.port}", api_key="secret"
    )
    client = P2PClient(peer_url, api_key="secret")
    unauthorized = P2PClient(peer_url, api_key="wrong")
    try:
        assert worker_client.noop().result(5) == "worker"
        assert client.p2p_lookup_and_lock(keys, layouts).result(5) == 41
        assert client.p2p_query_lookup_results(0).result(5) is None
        assert client.p2p_query_lookup_results(41).result(5) == addresses
        assert client.p2p_unlock_objects(keys).result(5) is None
        assert unlocked == keys
        with pytest.raises(MPRequestError, match="UNAUTHENTICATED"):
            unauthorized.p2p_query_lookup_results(41).result(5)
        peer_server.close(wait=True)
        # Closing P2P leaves worker RPCs alive and releases the peer listener.
        assert worker_client.noop().result(5) == "worker"
        replacement = build_grpc_request_server(
            [peer], replace(config, port=peer_port), rpc_client=P2PClient
        )
        replacement.close(wait=True)
    finally:
        unauthorized.close()
        client.close()
        worker_client.close()
        worker.close()
        engine.close()
