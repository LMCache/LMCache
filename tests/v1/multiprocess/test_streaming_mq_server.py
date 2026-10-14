# SPDX-License-Identifier: Apache-2.0
"""Runtime tests for the layer-wise streaming message queue path.

These exercise the real wire: a :class:`StreamingMessageQueueServer` bound to
loopback, a real :class:`StreamingMessageQueueClient`, and msgspec encoding of the
actual ``retrieve_layerwise`` response type.  The other layer-wise tests only
introspect signatures, so without these the streaming dispatch, the
``response_channel`` keyword forwarding and the multi-frame framing never
actually execute.
"""

# Standard
from typing import Any, Callable, Optional
import queue
import shutil
import tempfile
import threading
import uuid

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures_layerwise import LayerwiseRawFuture
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    BlockingRequestHandler,
    MessageQueueServer,
    SyncRequestHandler,
)
from lmcache.v1.multiprocess.transport.zmq_impl.mq_streaming import (
    StreamingMessageQueueClient,
    StreamingMessageQueueServer,
    StreamingRequestHandler,
)

pytestmark = pytest.mark.layerwise


OPERATION = "retrieve_layerwise"


_SOCKET_DIR = tempfile.mkdtemp(prefix="lmcache-streaming-mq-")


@pytest.fixture(scope="module", autouse=True)
def _remove_the_socket_dir():
    yield
    shutil.rmtree(_SOCKET_DIR, ignore_errors=True)


def _free_url() -> str:
    """A private ipc endpoint for one server.

    A tcp port picked by bind-then-close can be taken by another process in
    the window before the server binds it, which surfaces as a rare CI
    failure; a path under a private directory cannot collide.
    """
    return f"ipc://{_SOCKET_DIR}/{uuid.uuid4().hex}.sock"


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

    def _build(
        handler,
    ) -> tuple[StreamingMessageQueueServer, StreamingMessageQueueClient]:
        url = _free_url()
        context = zmq.Context.instance()
        server = StreamingMessageQueueServer(url, context)
        server.add_handler(OPERATION, HandlerType.BLOCKING, handler)
        server.add_handler(
            "register_layerwise_ipc_event_pool",
            HandlerType.SYNC,
            _register_event_pool,
        )
        server.add_affinity_thread_pool([OPERATION], max_workers=2)
        server.start()
        client = StreamingMessageQueueClient(url, context)
        created.append((server, client))
        return server, client

    yield _build

    for server, client in created:
        client.close()
        server.close()


def _register_event_pool(instance_id: int) -> tuple[int, list[bytes]]:
    return 8, [f"handle-{instance_id}".encode()]


def _submit(
    client: StreamingMessageQueueClient,
) -> tuple[LayerwiseRawFuture, queue.Queue]:
    partials: queue.Queue = queue.Queue()
    raw: LayerwiseRawFuture = LayerwiseRawFuture(partials)
    client.submit_streaming_request(OPERATION, _payloads(), raw)
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
    pending_future, pending_spec = next(iter(client.pending_futures.values()))
    assert pending_future is raw
    assert pending_spec.operation == OPERATION

    handler.release.set()
    assert raw.wait(timeout=10) is True
    assert raw.query() is True
    assert client.pending_futures == {}


def test_single_reply_layerwise_request_round_trips(streaming_pair):
    """The non-streaming layer-wise operation answers once, like any RPC."""
    _, client = streaming_pair(_Recorder().handle)

    future = client.submit_layerwise_request("register_layerwise_ipc_event_pool", [42])

    assert future.result(timeout=10) == (8, [b"handle-42"])
    assert client.pending_futures == {}


def test_streaming_handler_is_registered_and_pool_assignment_still_works():
    """The streaming operation gets a StreamingRequestHandler with a pool."""
    handler = _Recorder()
    server = StreamingMessageQueueServer(_free_url(), zmq.Context.instance())
    try:
        server.add_handler(OPERATION, HandlerType.BLOCKING, handler.handle)
        entry = server.handlers[OPERATION]
        assert isinstance(entry, StreamingRequestHandler)
        # Still a blocking handler, so the shared pool wiring accepts it.
        assert isinstance(entry, BlockingRequestHandler)
        assert entry.get_handler_type() is HandlerType.BLOCKING
        assert entry.executor is None

        server.add_affinity_thread_pool([OPERATION], max_workers=1)
        assert entry.executor is not None

        server.add_handler(
            "register_layerwise_ipc_event_pool",
            HandlerType.SYNC,
            _register_event_pool,
        )
        assert isinstance(
            server.handlers["register_layerwise_ipc_event_pool"],
            SyncRequestHandler,
        )
    finally:
        server.close()


def test_streaming_operation_must_be_registered_as_blocking():
    """A streaming reply needs a worker thread; SYNC cannot provide one."""
    server = StreamingMessageQueueServer(_free_url(), zmq.Context.instance())
    try:
        with pytest.raises(ValueError, match="Unsupported handler type"):
            server.add_handler(OPERATION, HandlerType.SYNC, _Recorder().handle)
    finally:
        server.close()


def test_mismatched_layerwise_handler_is_rejected():
    """Layer-wise handlers get the same signature check as shared ones."""

    def wrong(instance_id: str) -> tuple[int, list[bytes]]:
        raise AssertionError("never called")

    server = StreamingMessageQueueServer(_free_url(), zmq.Context.instance())
    try:
        with pytest.raises(ValueError, match="Handler signature does not match"):
            server.add_handler(
                "register_layerwise_ipc_event_pool", HandlerType.SYNC, wrong
            )
    finally:
        server.close()


def test_plain_server_rejects_a_layerwise_operation():
    """A misconfigured deployment fails loudly instead of silently degrading."""
    server = MessageQueueServer(_free_url(), zmq.Context.instance())
    try:
        with pytest.raises(KeyError, match=OPERATION):
            server.add_handler(OPERATION, HandlerType.BLOCKING, _Recorder().handle)
    finally:
        server.close()
