# SPDX-License-Identifier: Apache-2.0
"""The layer-wise retrieve future must be streaming-aware before it is sent.

``submit_streaming_request`` publishes the request to the polling loop before
it returns, so the future it is handed already has to know how to interpret a
multi-frame answer.  That knowledge lives in the future's *type*
(:class:`LayerwiseRawFuture`) rather than in state attached after
construction, so there is no window in which an early frame could be mistaken
for the final one.  These tests pin that the streaming future is what reaches
the wire, that it is armed to re-register itself before it can receive
anything, and that it reports itself incomplete until the closing frame.
"""

# Standard
from dataclasses import fields
from typing import Any, cast
import itertools
import queue
import struct

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.futures_layerwise import LayerwiseRawFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.transfer_context import worker_transfer_layerwise
from lmcache.v1.multiprocess.transfer_context.worker_transfer_layerwise import (
    LMCacheLayerwiseTransferContext,
)
from lmcache.v1.multiprocess.transport.zmq_impl import ZmqMultiprocessClient


class _RecordingClient:
    """Stand-in for ``MessageQueueClient`` exposing what the helper touches.

    ``submit_streaming_request`` writes straight to the client's request
    plumbing, so the stub mirrors it: a uid counter, the pending-request
    table, the outbound queue and the polling loop.  The loop stub doubles as
    the recorder, which is what makes the snapshot below meaningful -- it runs
    at the exact moment the request becomes visible to the transport.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._request_counter = itertools.count()
        self.pending_futures: dict[int, Any] = {}
        self.input_queue: queue.Queue = queue.Queue()
        self._polling_loop = self._Loop(self)

    def submit_streaming_request(self, request_type, request_payloads, future):
        """Real streaming submit, driven against this stub's plumbing."""
        return MessageQueueClient.submit_streaming_request(
            self, request_type, request_payloads, future
        )

    def submit_request(self, *args: Any, **kwargs: Any):
        pytest.fail(
            "submit_retrieve must not route through submit_request: the "
            "per-chunk path cannot carry a pre-built, self-re-arming future"
        )

    class _Loop:
        def __init__(self, client: "_RecordingClient") -> None:
            self._client = client

        def notify(self) -> None:
            request = self._client.input_queue.get_nowait()
            self._client.calls.append(
                {
                    "request_type": request.request_type,
                    "payloads": request.request_payloads,
                    "future": request.future,
                    "request_uid": request.request_uid,
                    # Snapshot *at submit time*: a future that already reported
                    # itself done could never receive a second frame.
                    "done_at_submit": request.future.query(),
                    # Likewise, a future not yet bound to the pending table
                    # could not put itself back after a partial frame.
                    "bound_at_submit": (
                        request.future._registry is self._client.pending_futures
                    ),
                }
            )


class _BareClient:
    """A ``MessageQueueClient`` reduced to the state the submit paths use.

    Building the real client opens sockets and joins the shared polling
    loop; the submit paths only read these four attributes, so both can be
    driven against this without any transport.
    """

    def __init__(self) -> None:
        self._request_counter = itertools.count()
        self.pending_futures: dict[int, Any] = {}
        self.input_queue: queue.Queue = queue.Queue()
        self._polling_loop = self
        self.notified = 0

    def notify(self) -> None:
        self.notified += 1


def _make_bare_client() -> Any:
    return _BareClient()


class _StubEventBackend:
    def export_event(self, event, device):
        del event, device
        return b"event-handle"

    def check_event_support(self, device):
        del device

    def wait_event(self, event, stream):
        del event, stream


class _StubEventPool:
    size = 4

    def event_at(self, idx):
        return ("event", idx)


def _make_context(client: _RecordingClient) -> LMCacheLayerwiseTransferContext:
    """Build a registered-looking context without touching a real device."""
    # ``Any`` so the stub collaborators below can stand in for the concrete
    # client/backend/pool types the context annotates.
    ctx: Any = object.__new__(LMCacheLayerwiseTransferContext)
    ctx._req_client = ZmqMultiprocessClient(cast(MessageQueueClient, client))
    ctx._device = 0
    ctx._event_backend = _StubEventBackend()
    ctx._event_pool = _StubEventPool()
    ctx._layerwise_batch = 8
    return ctx


@pytest.fixture
def patched_future(monkeypatch):
    """Replace the device-side wrapper with a stub owning a real raw future."""
    created: list[LayerwiseRawFuture] = []

    class _StubLayerwiseFuture:
        def __init__(self, device=None, event_pool=None):
            del device, event_pool
            self._partial_queue: queue.Queue = queue.Queue()
            self.raw_future_ = LayerwiseRawFuture(self._partial_queue)
            created.append(self.raw_future_)

    monkeypatch.setattr(
        worker_transfer_layerwise,
        "LayerwiseDeviceMessagingFuture",
        _StubLayerwiseFuture,
    )
    return created


def test_streaming_future_is_what_gets_submitted(patched_future):
    client = _RecordingClient()
    ctx = _make_context(client)

    result = ctx.submit_retrieve(
        "req-0",
        "cache-key",
        7,
        {},
        [[0, 1]],
        object(),
        2,
        skip_first_n_tokens=0,
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["request_type"] is RequestType.RETRIEVE_LAYERWISE
    # The polling loop must receive the streaming future, still incomplete,
    # so that a first layer batch arriving immediately is buffered rather
    # than mistaken for the final answer.
    assert isinstance(call["future"], LayerwiseRawFuture)
    assert call["done_at_submit"] is False
    # ...and already able to re-register itself, since the first frame may
    # arrive before this assertion would have had a chance to run.
    assert call["bound_at_submit"] is True
    assert call["future"]._request_uid == call["request_uid"]
    # The future handed to the client is the one the wrapper owns.
    assert call["future"] is result.raw_future_


def test_submitted_future_is_the_wrapper_owned_one(patched_future):
    client = _RecordingClient()
    ctx = _make_context(client)

    ctx.submit_retrieve(
        "req-1", "cache-key", 7, {}, [[0]], object(), 1, skip_first_n_tokens=3
    )

    assert client.calls[0]["future"] is patched_future[0]
    # skip_first_n_tokens must still reach the wire unchanged.
    assert client.calls[0]["payloads"][-1] == 3


def test_partial_frame_leaves_the_future_incomplete():
    """A non-final frame is buffered and the request stays open."""
    partial_queue: "queue.Queue[bytes | None]" = queue.Queue()
    future: LayerwiseRawFuture = LayerwiseRawFuture(partial_queue)

    partial = (struct.pack("<3i", 0, 8, 0), False, None)
    future.set_result(partial)
    assert future.query() is False
    assert partial_queue.get_nowait() == partial[0]

    final = (b"", True, True)
    future.set_result(final)
    assert future.query() is True
    assert future.result() == final
    # Sentinel that unblocks a reader waiting on a layer that never arrives.
    assert partial_queue.get_nowait() is None


def test_plain_future_completes_on_the_first_frame():
    """Contrast: the base future has no notion of a partial answer."""
    future: MessagingFuture = MessagingFuture()

    partial = (struct.pack("<3i", 0, 8, 0), False, None)
    future.set_result(partial)
    assert future.query() is True
    assert future.result() == partial


def test_partial_frame_re_registers_the_future():
    """A bound future puts itself back so later frames still reach it."""
    registry: dict[int, Any] = {}
    partial_queue: "queue.Queue[bytes | None]" = queue.Queue()
    future: LayerwiseRawFuture = LayerwiseRawFuture(partial_queue)
    future.bind_registry(registry, 42)

    # The client pops a future before completing it, so a partial frame has
    # to leave the table repopulated or the next frame is dropped.
    future.set_result((struct.pack("<3i", 0, 8, 0), False, None))
    assert registry == {42: future}

    registry.pop(42)
    future.set_result((b"", True, True))
    # The closing frame must *not* re-register: nothing more is coming.
    assert registry == {}


def test_streaming_submit_matches_the_base_client():
    """The streaming helper must build the same request as ``submit_request``.

    ``MessageQueueClient.submit_streaming_request`` duplicates the
    request-building half of ``submit_request`` so that the per-chunk path
    carries nothing about streaming.  This pins the two together: the comparison is
    driven by ``dataclasses.fields``, so a field added to ``WrappedRequest``
    and populated by the base path fails here until the copy catches up.
    """
    base_client = _make_bare_client()
    MessageQueueClient.submit_request(
        base_client, RequestType.RETRIEVE_LAYERWISE, ["payload", 7]
    )
    base_request = base_client.input_queue.get_nowait()

    stream_client = _make_bare_client()
    MessageQueueClient.submit_streaming_request(
        stream_client,
        RequestType.RETRIEVE_LAYERWISE,
        ["payload", 7],
        LayerwiseRawFuture(queue.Queue()),
    )
    stream_request = stream_client.input_queue.get_nowait()

    # A uid comes from a counter and the future is the point of the helper,
    # so those two are expected to differ; everything else must not.
    varies = {"request_uid", "future"}
    names = [f.name for f in fields(MessageQueueClient.WrappedRequest)]
    assert set(names) - varies, "nothing left to compare: update `varies`"
    assert {n: getattr(base_request, n) for n in names if n not in varies} == {
        n: getattr(stream_request, n) for n in names if n not in varies
    }

    # Both must reach the polling loop, and under a fresh uid.
    assert base_client.notified == stream_client.notified == 1
    assert base_request.request_uid == stream_request.request_uid == 0
