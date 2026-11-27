# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, cast
import socket
import threading

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.common.custom_types import IPCCacheServerKey, KVEventBatch
from lmcache.v1.distributed.api import get_request_tags
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    BlockingRequestHandler,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.transport.zmq_impl.server import (
    add_handler_helper,
    build_zmq_request_server,
)
from lmcache.v1.transport.futures import MessagingStream
from lmcache.v1.transport.zmq.codec import msgspec_encode
from lmcache.v1.transport.zmq.wire import (
    ZMQ_ERROR_OPERATION,
    decode_error_response,
    decode_operation,
    encode_operation,
)
from tests.v1.multiprocess import test_mq_handler_helpers


def _connect_peer(server: MessageQueueServer) -> zmq.Socket:
    """Connect a wire-level peer to the legacy listener for server tests."""
    peer = server.socket.context.socket(zmq.DEALER)
    peer.setsockopt(zmq.RCVTIMEO, 5000)
    peer.setsockopt(zmq.LINGER, 0)
    peer.connect(server.socket.getsockopt_string(zmq.LAST_ENDPOINT))
    return peer


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_zmq_server_registers_out_of_tree_services() -> None:
    """Server modules may attach package-owned ZMQ services before start."""
    registered_servers: list[MessageQueueServer] = []

    class ServiceModule:
        def register_zmq_services(self, server: MessageQueueServer) -> None:
            registered_servers.append(server)

    server = build_zmq_request_server(
        [cast(Any, ServiceModule())],
        MPServerConfig(port=_unused_tcp_port()),
    )
    try:
        assert registered_servers == [server]
    finally:
        server.close()


def test_legacy_zmq_stream_waits_for_ack_and_closes_on_cancel() -> None:
    """A legacy subscriber grants one batch per ACK and can cancel a live stream."""
    closed = threading.Event()
    batches = [
        KVEventBatch(enabled=True, incarnation=1, next_cursor=i, lost=False, events=[])
        for i in (1, 2, 3)
    ]

    def subscribe(
        instance_id: int, model_name: str, cursor: int, max_events: int
    ) -> MessagingStream[KVEventBatch]:
        ready = iter(batches)
        return MessagingStream(lambda _: next(ready), closed.set)

    server = MessageQueueServer("tcp://127.0.0.1:0", zmq.Context.instance())
    server.add_handler("subscribe_kv_events", HandlerType.SERIALIZED, subscribe)
    server.start()
    try:
        with _connect_peer(server) as peer:
            uid = msgspec.msgpack.encode(1)
            header = [uid, encode_operation("subscribe_kv_events")]
            args = [msgspec.msgpack.encode(v) for v in (1, "model", 0, 1)]
            peer.send_multipart(header + args)
            assert peer.recv_multipart() == header + [
                msgspec_encode(batches[0], KVEventBatch)
            ]
            assert peer.poll(100, zmq.POLLIN) == 0

            peer.send_multipart([uid, b"", b"ack"])
            assert peer.recv_multipart() == header + [
                msgspec_encode(batches[1], KVEventBatch)
            ]

            peer.send_multipart([uid, b"", b"cancel"])
            assert peer.recv_multipart() == header
            assert closed.is_set()
    finally:
        server.close()


def test_server_survives_malformed_request_headers() -> None:
    """Malformed peers must not terminate the shared server polling loop."""
    # Standard
    from unittest.mock import patch

    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl import mq as mq_module

    server_url = "tcp://127.0.0.1:16021"
    context = zmq.Context.instance()

    server = MessageQueueServer(server_url, context)
    add_handler_helper(server, "noop", test_mq_handler_helpers.noop_handler)
    server.start()

    malformed = _connect_peer(server)
    short_header_seen = threading.Event()
    undecodable_type_seen = threading.Event()
    try:
        with (
            patch.object(
                mq_module.logger,
                "error",
                side_effect=lambda *_args, **_kwargs: short_header_seen.set(),
            ),
            patch.object(
                mq_module.logger,
                "exception",
                side_effect=lambda *_args, **_kwargs: undecodable_type_seen.set(),
            ),
        ):
            # ROUTER prepends the peer identity. This first message therefore
            # has only two frames at the server and lacks a request type.
            malformed.send_multipart([msgspec.msgpack.encode(1)])
            assert short_header_seen.wait(timeout=5)

            # This message has enough frames, but its operation has an invalid type
            # and must fail decoding without killing the loop.
            malformed.send_multipart(
                [msgspec.msgpack.encode(2), msgspec.msgpack.encode([1])]
            )
            assert undecodable_type_seen.wait(timeout=5)

        with _connect_peer(server) as healthy:
            frames = [msgspec.msgpack.encode(3), encode_operation("noop")]
            healthy.send_multipart(frames)
            assert healthy.recv_multipart() == [
                *frames,
                msgspec.msgpack.encode("NOOP_OK"),
            ]
        assert server.worker_thread.is_alive()
    finally:
        malformed.close()
        server.close()


def test_unregistered_operation_returns_remote_error() -> None:
    """An unsupported RPC must fail promptly without poisoning later requests."""
    server_url = f"tcp://127.0.0.1:{_unused_tcp_port()}"
    context = zmq.Context.instance()
    server = MessageQueueServer(server_url, context)
    add_handler_helper(server, "noop", test_mq_handler_helpers.noop_handler)
    server.start()

    try:
        with _connect_peer(server) as peer:
            uid = msgspec.msgpack.encode(1)
            peer.send_multipart([uid, encode_operation("get_chunk_size")])
            _, operation, payload = peer.recv_multipart()
            assert decode_operation(operation) == ZMQ_ERROR_OPERATION
            error = decode_error_response(payload)
            assert error.code == "operation_not_enabled"
            assert "--supported-transfer-mode" in error.message

            frames = [msgspec.msgpack.encode(2), encode_operation("noop")]
            peer.send_multipart(frames)
            assert peer.recv_multipart() == [*frames, msgspec.msgpack.encode("NOOP_OK")]
    finally:
        server.close()


def test_blocking_handler_value_error_returns_remote_error() -> None:
    """A rejected request must fail promptly without poisoning later requests."""

    def lookup_with_tag_validation(key: IPCCacheServerKey, tp_size: int) -> None:
        """Validate request tags before starting lookup work."""
        del tp_size
        get_request_tags(key.request_configs)

    server_url = f"tcp://127.0.0.1:{_unused_tcp_port()}"
    context = zmq.Context.instance()
    server = MessageQueueServer(server_url, context)
    server.add_handler("lookup", HandlerType.BLOCKING, lookup_with_tag_validation)
    server.add_normal_thread_pool(["lookup"], max_workers=1)
    server.start()

    try:
        with _connect_peer(server) as peer:
            for value in (2, "2"):
                key = IPCCacheServerKey.from_token_ids(
                    "model",
                    1,
                    None,
                    [1, 2],
                    request_configs={"lmcache.tag.version": value},
                )
                header = [msgspec.msgpack.encode(1), encode_operation("lookup")]
                peer.send_multipart(
                    [
                        *header,
                        msgspec_encode(key, IPCCacheServerKey),
                        msgspec.msgpack.encode(1),
                    ]
                )
                reply = peer.recv_multipart()
                if value == 2:
                    assert decode_operation(reply[1]) == ZMQ_ERROR_OPERATION
                    error = decode_error_response(reply[2])
                    assert error.code == "invalid_argument"
                    assert "string values" in error.message
                else:
                    assert reply == header
    finally:
        server.close()


def test_server_answers_a_burst_larger_than_one_poll() -> None:
    server_url = "tcp://127.0.0.1:16031"
    context = zmq.Context.instance()
    server = MessageQueueServer(server_url, context)
    add_handler_helper(server, "noop", test_mq_handler_helpers.noop_handler)
    server.start()

    try:
        with _connect_peer(server) as peer:
            count = 3 * MessageQueueServer.MAX_REQUESTS_PER_POLL + 1
            noop = encode_operation("noop")
            for uid in range(count):
                peer.send_multipart([msgspec.msgpack.encode(uid), noop])
            replies = [peer.recv_multipart() for _ in range(count)]
            assert {msgspec.msgpack.decode(uid) for uid, _, _ in replies} == set(
                range(count)
            )
            assert all(
                msgspec.msgpack.decode(payload) == "NOOP_OK"
                for _, _, payload in replies
            )
    finally:
        server.close()


def test_add_normal_thread_pool():
    """
    Test that add_normal_thread_pool assigns handler executors.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15700", context)

    add_handler_helper(
        server,
        "lookup",
        test_mq_handler_helpers.lookup_handler,
        HandlerType.BLOCKING,
    )
    add_handler_helper(server, "noop", test_mq_handler_helpers.noop_handler)

    lookup_handler = server.handlers["lookup"]
    assert isinstance(lookup_handler, BlockingRequestHandler)
    assert lookup_handler.executor is None

    server.add_normal_thread_pool(["lookup"], max_workers=4)

    assert lookup_handler.executor is not None
    assert len(server.extra_pools) == 1

    server.close()


def test_add_affinity_thread_pool():
    """
    Test that add_affinity_thread_pool assigns AffinityThreadPool executors.
    """
    # First Party
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15700", context)

    add_handler_helper(
        server,
        "store",
        test_mq_handler_helpers.store_handler,
        HandlerType.BLOCKING,
    )
    add_handler_helper(
        server,
        "retrieve",
        test_mq_handler_helpers.retrieve_handler,
        HandlerType.BLOCKING,
    )

    store_handler = server.handlers["store"]
    retrieve_handler = server.handlers["retrieve"]
    assert isinstance(store_handler, BlockingRequestHandler)
    assert isinstance(retrieve_handler, BlockingRequestHandler)
    assert store_handler.executor is None

    server.add_affinity_thread_pool(["store", "retrieve"], max_workers=2)

    assert isinstance(store_handler.executor, AffinityThreadPool)
    assert store_handler.executor is retrieve_handler.executor
    assert len(server.extra_pools) == 1

    server.close()


def test_normal_pool_error_on_serialized_handler():
    """
    Test that add_normal_thread_pool raises TypeError for SERIALIZED handlers.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15701", context)

    add_handler_helper(server, "noop", test_mq_handler_helpers.noop_handler)

    with pytest.raises(TypeError, match="not BlockingRequestHandler"):
        server.add_normal_thread_pool(["noop"], max_workers=1)

    server.close()


def test_affinity_pool_error_on_serialized_handler():
    """
    Test that add_affinity_thread_pool raises TypeError for SERIALIZED handlers.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15701", context)

    add_handler_helper(server, "noop", test_mq_handler_helpers.noop_handler)

    with pytest.raises(TypeError, match="not BlockingRequestHandler"):
        server.add_affinity_thread_pool(["noop"], max_workers=1)

    server.close()


def test_pool_error_on_unregistered():
    """
    Test that pool methods raise ValueError for unregistered request types.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15702", context)

    with pytest.raises(ValueError, match="No handler registered"):
        server.add_normal_thread_pool(["store"], max_workers=1)

    with pytest.raises(ValueError, match="No handler registered"):
        server.add_affinity_thread_pool(["store"], max_workers=1)

    server.close()


def test_multiple_pools():
    """
    Test that normal and affinity pools can coexist.
    """
    # First Party
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15703", context)

    add_handler_helper(
        server,
        "store",
        test_mq_handler_helpers.store_handler,
        HandlerType.BLOCKING,
    )
    add_handler_helper(
        server,
        "retrieve",
        test_mq_handler_helpers.retrieve_handler,
        HandlerType.BLOCKING,
    )
    add_handler_helper(
        server,
        "lookup",
        test_mq_handler_helpers.lookup_handler,
        HandlerType.BLOCKING,
    )

    server.add_affinity_thread_pool(["store", "retrieve"], max_workers=2)
    server.add_normal_thread_pool(["lookup"], max_workers=3)

    store_handler = server.handlers["store"]
    retrieve_handler = server.handlers["retrieve"]
    lookup_handler = server.handlers["lookup"]
    assert isinstance(store_handler, BlockingRequestHandler)
    assert isinstance(retrieve_handler, BlockingRequestHandler)
    assert isinstance(lookup_handler, BlockingRequestHandler)

    # STORE/RETRIEVE share affinity pool
    assert isinstance(store_handler.executor, AffinityThreadPool)
    assert store_handler.executor is retrieve_handler.executor
    # LOOKUP uses normal pool
    assert store_handler.executor is not lookup_handler.executor
    assert not isinstance(lookup_handler.executor, AffinityThreadPool)
    assert len(server.extra_pools) == 2

    server.close()


def test_start_fails_without_pool_assignment():
    """
    Test that start() raises RuntimeError if a blocking handler
    has no executor assigned.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15704", context)

    add_handler_helper(
        server,
        "store",
        test_mq_handler_helpers.store_handler,
        HandlerType.BLOCKING,
    )
    # Don't assign any pool

    with pytest.raises(RuntimeError, match="no thread pool assigned"):
        server.start()

    server.close()


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires available CUDA runtime",
)
def test_q_ring_info_list_zmq_serialization():
    """The ZMQ transport encodes GET_Q_RING replies, which carry IPC wrappers."""
    # First Party
    from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper
    from lmcache.v1.transport.qstore import QRingInfo
    from lmcache.v1.transport.zmq.codec import msgspec_decode, msgspec_encode

    wrapper = CudaIPCWrapper(torch.randn(4, 2, 3, device=torch_device_type))
    rings = [
        QRingInfo(
            instance_id=1,
            worker_id=0,
            q_caches=[wrapper],
            used_blocks=[1, 2],
            num_alloc=3,
        )
    ]

    decoded = msgspec_decode(msgspec_encode(rings, list[QRingInfo]), list[QRingInfo])

    assert decoded == rings
