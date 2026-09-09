# SPDX-License-Identifier: Apache-2.0
"""Runtime tests for the layer-wise streaming message queue path.

These exercise the real wire: a :class:`StreamingMessageQueueServer` bound to
loopback, a real :class:`MessageQueueClient`, and msgspec encoding of the
actual ``RETRIEVE_LAYERWISE`` response class.  The other layer-wise tests only
introspect signatures, so without these the streaming dispatch, the
``response_channel`` keyword forwarding and the multi-frame framing never
actually execute.
"""

# Standard
from typing import Any, Callable, Optional
import queue
import socket
import threading

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures_layerwise import LayerwiseRawFuture
from lmcache.v1.multiprocess.mq import (
    BlockingRequestHandler,
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.mq_streaming import (
    StreamingMessageQueueServer,
    StreamingRequestHandler,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_payload_classes,
)

REQUEST_TYPE = RequestType.RETRIEVE_LAYERWISE


def _free_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


def _cache_key() -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "testmodel", 1, 0, [7] * 256, start=0, end=256, request_id="req-0"
    )


def _payloads() -> list[Any]:
    return [_cache_key(), 1234, [[0, 1]], b"", 0]


class _Recorder:
    """Handler that streams a fixed number of partial frames."""

    def __init__(self, num_partials: int = 3) -> None:
        self.num_partials = num_partials
        self.saw_response_channel: Optional[bool] = None
        self.received_payloads: Optional[list[Any]] = None
        self.release = threading.Event()
        self.entered = threading.Event()

    def handle(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
        *,
        response_channel: Optional[Callable[[Any], None]] = None,
    ) -> tuple[bytes, bool, bool]:
        self.received_payloads = [
            key,
            instance_id,
            gpu_block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        ]
        self.saw_response_channel = callable(response_channel)
        assert response_channel is not None
        for i in range(self.num_partials):
            response_channel((f"partial-{i}".encode(), False, True))
        self.entered.set()
        # Held closed by default; tests that care about the pre-final state
        # clear it so the closing frame cannot race the assertions.
        self.release.wait(timeout=5)
        return (b"final", True, True)


@pytest.fixture
def streaming_pair():
    """Yield a live (server, client, url) triple on loopback."""
    created: list[Any] = []

    def _build(handler) -> tuple[StreamingMessageQueueServer, MessageQueueClient]:
        url = _free_url()
        context = zmq.Context.instance()
        server = StreamingMessageQueueServer(url, context)
        server.add_handler(
            REQUEST_TYPE,
            get_payload_classes(REQUEST_TYPE),
            HandlerType.STREAMING,
            handler,
        )
        server.add_affinity_thread_pool([REQUEST_TYPE], max_workers=2)
        server.start()
        client = MessageQueueClient(url, context)
        created.append((server, client))
        return server, client

    yield _build

    for server, client in created:
        client.close()
        server.close()


def _submit(client: MessageQueueClient) -> tuple[LayerwiseRawFuture, queue.Queue]:
    partials: queue.Queue = queue.Queue()
    raw: LayerwiseRawFuture = LayerwiseRawFuture(partials)
    client.submit_streaming_request(REQUEST_TYPE, _payloads(), raw)
    return raw, partials


def test_partial_frames_arrive_in_order_and_final_completes(streaming_pair):
    """Every partial reaches the client, in order, before the closing frame."""
    handler = _Recorder(num_partials=3)
    handler.release.set()
    _, client = streaming_pair(handler.handle)

    raw, partials = _submit(client)

    assert raw.wait(timeout=10) is True
    assert raw.result(timeout=10) == (b"final", True, True)

    drained = [partials.get(timeout=5) for _ in range(4)]
    assert drained == [b"partial-0", b"partial-1", b"partial-2", None]


def test_handler_is_given_a_response_channel_and_the_decoded_payloads(streaming_pair):
    """The streaming handler receives response_channel plus its real payloads."""
    handler = _Recorder(num_partials=1)
    handler.release.set()
    _, client = streaming_pair(handler.handle)

    raw, _ = _submit(client)
    assert raw.wait(timeout=10) is True

    assert handler.saw_response_channel is True
    assert handler.received_payloads is not None
    key, instance_id, block_ids, ipc_handle, skip = handler.received_payloads
    assert key == _cache_key()
    assert instance_id == 1234
    assert block_ids == [[0, 1]]
    assert ipc_handle == b""
    assert skip == 0


def test_request_stays_registered_while_only_partials_have_arrived(streaming_pair):
    """Partials must not complete the future or drop it from pending_futures."""
    handler = _Recorder(num_partials=2)
    _, client = streaming_pair(handler.handle)

    raw, partials = _submit(client)

    # Both partials are on the wire; the handler is parked before returning.
    assert partials.get(timeout=5) == b"partial-0"
    assert partials.get(timeout=5) == b"partial-1"

    assert raw.query() is False
    assert len(client.pending_futures) == 1
    assert next(iter(client.pending_futures.values())) is raw

    handler.release.set()
    assert raw.wait(timeout=10) is True
    assert raw.query() is True
    assert client.pending_futures == {}


def test_streaming_handler_is_registered_and_pool_assignment_still_works():
    """add_handler routes STREAMING to a StreamingRequestHandler with a pool."""
    handler = _Recorder()
    server = StreamingMessageQueueServer(_free_url(), zmq.Context.instance())
    try:
        server.add_handler(
            REQUEST_TYPE,
            get_payload_classes(REQUEST_TYPE),
            HandlerType.STREAMING,
            handler.handle,
        )
        entry = server.handlers[REQUEST_TYPE]
        assert isinstance(entry, StreamingRequestHandler)
        assert isinstance(entry, BlockingRequestHandler)
        assert entry.get_handler_type() is HandlerType.STREAMING
        assert entry.executor is None

        server.add_affinity_thread_pool([REQUEST_TYPE], max_workers=1)
        assert entry.executor is not None
    finally:
        server.close()


def test_plain_server_rejects_a_streaming_handler():
    """A misconfigured deployment fails loudly instead of silently degrading."""
    server = MessageQueueServer(_free_url(), zmq.Context.instance())
    try:
        with pytest.raises(ValueError, match="Unknown handler type"):
            server.add_handler(
                REQUEST_TYPE,
                get_payload_classes(REQUEST_TYPE),
                HandlerType.STREAMING,
                _Recorder().handle,
            )
    finally:
        server.close()
