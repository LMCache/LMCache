# SPDX-License-Identifier: Apache-2.0
"""Exercise offload request recovery with real ZMQ REQ/REP sockets.

In-process transport isolates the protocol from filesystem IPC naming. The
cache engine is mocked, so these tests need neither vLLM nor a GPU.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer


@pytest.fixture
def offload_rpc() -> Iterator[tuple[ZMQOffloadServer, zmq.Socket, MagicMock]]:
    """Provide real REQ/REP sockets and an engine for injecting store errors."""
    context = zmq.Context()
    endpoint = "inproc://offload-recovery-test"
    reply_socket = context.socket(zmq.REP)
    reply_socket.bind(endpoint)
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 2000)
    client.setsockopt(zmq.SNDTIMEO, 2000)
    client.connect(endpoint)
    engine = MagicMock(spec=LMCacheEngine)
    engine.metadata = SimpleNamespace(engine_id="offload-recovery-test")

    with (
        patch(
            "lmcache.v1.offload_server.zmq_server.get_zmq_context",
            return_value=context,
        ),
        patch(
            "lmcache.v1.offload_server.zmq_server.get_zmq_rpc_path_lmcache",
            return_value="unused",
        ),
        patch(
            "lmcache.v1.offload_server.zmq_server.get_zmq_socket",
            return_value=reply_socket,
        ),
    ):
        server = ZMQOffloadServer(engine, 0)

    try:
        yield server, client, engine
    finally:
        # Wake the receiver before closing its socket from this thread.
        server.running = False
        engine.store.side_effect = None
        wake_client = context.socket(zmq.REQ)
        wake_client.setsockopt(zmq.SNDTIMEO, 2000)
        wake_client.connect(endpoint)
        try:
            if server.thread.is_alive():
                wake_client.send(
                    msgspec.msgpack.encode(
                        OffloadMsg(hashes=[], slot_mapping=[], offsets=[])
                    )
                )
                server.thread.join(timeout=5)
            assert not server.thread.is_alive()
        finally:
            client.close(linger=0)
            wake_client.close(linger=0)
            server.close()
            context.term()


@pytest.mark.parametrize(
    "failure", ["decode", "validation", "multipart", "store", "store_zmq"]
)
def test_failed_request_is_replied_to_and_next_request_succeeds(
    offload_rpc: tuple[ZMQOffloadServer, zmq.Socket, MagicMock], failure: str
) -> None:
    """A failed request must not prevent the same client from sending another."""
    server, client, engine = offload_rpc
    request = OffloadMsg(hashes=[123], slot_mapping=[0], offsets=[1])
    valid_frame = msgspec.msgpack.encode(request)
    if failure == "decode":
        first_frames = [b"\xc1"]  # Reserved MessagePack byte: decoding must fail.
    elif failure == "validation":
        first_frames = [
            msgspec.msgpack.encode(
                {"hashes": "invalid", "slot_mapping": [0], "offsets": [1]}
            )
        ]
    elif failure == "multipart":
        first_frames = [valid_frame, valid_frame]
    else:
        first_frames = [valid_frame]
        error = (
            zmq.ZMQError(zmq.EFSM)
            if failure == "store_zmq"
            else ValueError("store failed")
        )
        engine.store.side_effect = [error, None]

    client.send_multipart(first_frames)
    failed_reply = msgspec.msgpack.decode(client.recv(), type=OffloadRetMsg)
    assert failed_reply.success is False

    client.send(valid_frame)
    successful_reply = msgspec.msgpack.decode(client.recv(), type=OffloadRetMsg)
    assert successful_reply.success is True
    engine.store.assert_called_with(hashes=[123], slot_mapping=[0], offsets=[1])
    assert engine.store.call_count == (2 if failure in {"store", "store_zmq"} else 1)
    assert server.running
    assert server.thread.is_alive()


@pytest.mark.parametrize("operation", ["recv_multipart", "send"])
def test_transport_failure_marks_server_stopped(operation: str) -> None:
    """A transport error must stop the thread and clear its running flag."""
    socket = MagicMock(spec=zmq.Socket)
    frame = msgspec.msgpack.encode(
        OffloadMsg(hashes=[123], slot_mapping=[0], offsets=[1])
    )
    socket.recv_multipart.return_value = [frame]
    getattr(socket, operation).side_effect = zmq.ZMQError(zmq.ETERM)
    engine = MagicMock(spec=LMCacheEngine)
    engine.metadata = SimpleNamespace(engine_id="offload-transport-failure")

    with (
        patch("lmcache.v1.offload_server.zmq_server.get_zmq_context"),
        patch(
            "lmcache.v1.offload_server.zmq_server.get_zmq_rpc_path_lmcache",
            return_value="unused",
        ),
        patch(
            "lmcache.v1.offload_server.zmq_server.get_zmq_socket",
            return_value=socket,
        ),
    ):
        server = ZMQOffloadServer(engine, 0)

    try:
        server.thread.join(timeout=5)
        assert not server.thread.is_alive()
        assert server.running is False
        assert getattr(socket, operation).call_count == 1
    finally:
        server.close()
