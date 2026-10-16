# SPDX-License-Identifier: Apache-2.0
"""Handler failures reach callers without poisoning later RPCs."""

# Standard
from typing import Any, cast

# Third Party
import grpc
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import KVEventBatch
from lmcache.v1.multiprocess.futures import MessagingFuture, MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.grpc_impl.client import GrpcMultiprocessClient
from lmcache.v1.multiprocess.transport.grpc_impl.server import GrpcMultiprocessServer
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    MessageQueueClient,
    MessageQueueServer,
    encode_request_uid,
)
from lmcache.v1.multiprocess.transport.zmq_impl.wire import encode_operation


@pytest.mark.parametrize("handler_type", [HandlerType.SYNC, HandlerType.BLOCKING])
@pytest.mark.parametrize("error_type", [ValueError, ConnectionError])
def test_zmq_handler_error_and_omitted_default(tmp_path, handler_type, error_type):
    received = []

    def clear(force: bool = False) -> None:
        received.append(force)
        if force:
            raise error_type("rejected request")

    url = f"ipc://{tmp_path}/rpc.sock"
    context = zmq.Context.instance()
    server = MessageQueueServer(url, context)
    server.add_handler("clear", handler_type, clear)
    if handler_type == HandlerType.BLOCKING:
        server.add_normal_thread_pool(["clear"], max_workers=1)
    server.start()
    client = MessageQueueClient(url, context)
    try:
        failed = client.submit_request("clear", [True])
        assert isinstance(failed, MessagingFuture)
        with pytest.raises(error_type, match="rejected request"):
            failed.result(timeout=5)
        healthy = client.submit_request("clear", [])
        assert isinstance(healthy, MessagingFuture)
        assert healthy.result(timeout=5) is None
        # A peer predating the trailing default must also receive a response.
        with context.socket(zmq.DEALER) as peer:
            peer.setsockopt(zmq.RCVTIMEO, 5000)
            peer.connect(url)
            peer.send_multipart([encode_request_uid(1), encode_operation("clear")])
            assert len(peer.recv_multipart()) == 2
        assert received == [True, False, False]
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize("error_type", [ValueError, ConnectionError])
def test_subscription_setup_failure_reaches_reader(tmp_path, transport, error_type):
    class Module:
        @request_handler()
        def subscribe_kv_events(
            self, instance_id: int, model_name: str, cursor: int, max_events: int
        ) -> MessagingStream[KVEventBatch]:
            raise error_type("rejected subscription")

    if transport == "grpc":
        server = GrpcMultiprocessServer(
            "grpc://127.0.0.1:0",
            max_cpu_workers=2,
            max_gpu_workers=1,
            grpc_server_workers=4,
        )
        server.add_modules([cast(Any, Module())])
        server.start()
        client = GrpcMultiprocessClient(f"grpc://127.0.0.1:{server.bound_port}")
        stream = client.subscribe_kv_events(1, "model", 0, 1)
    else:
        url = f"ipc://{tmp_path}/stream.sock"
        context = zmq.Context.instance()
        server = MessageQueueServer(url, context)
        server.add_handler(
            "subscribe_kv_events", HandlerType.SYNC, Module().subscribe_kv_events
        )
        server.start()
        client = MessageQueueClient(url, context)
        stream = client.submit_request("subscribe_kv_events", [1, "model", 0, 1])
    try:
        assert isinstance(stream, MessagingStream)
        if transport == "grpc" and error_type is ValueError:
            with pytest.raises(grpc.RpcError) as caught:
                next(stream)
            assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        else:
            with pytest.raises(error_type):
                next(stream)
    finally:
        stream.close()
        client.close()
        server.close()
