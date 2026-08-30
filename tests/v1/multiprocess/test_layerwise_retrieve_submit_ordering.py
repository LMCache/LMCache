# SPDX-License-Identifier: Apache-2.0
"""The layer-wise retrieve future must be armed before the request is sent.

``MessageQueueClient.submit_request`` publishes the request to the polling
loop before it returns, so any state the future needs in order to interpret
a response has to be installed beforehand.  For the layer-wise path that
state is the delivery sink: without it the first per-layer message takes the
default single-frame path and completes the raw future with a non-final
payload, silently discarding every later batch.
"""

# Standard
from typing import Any
import struct

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.transfer_context import worker_transfer_layerwise
from lmcache.v1.multiprocess.transfer_context.worker_transfer_layerwise import (
    LMCacheLayerwiseTransferContext,
)


class _RecordingClient:
    """Stand-in for ``MessageQueueClient`` that snapshots the future state."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def submit_request(self, request_type, payloads, response_cls, future=None):
        self.calls.append(
            {
                "request_type": request_type,
                "payloads": payloads,
                "future": future,
                # Snapshot *at submit time* -- checking after the call would
                # pass even if the sink were installed too late.
                "sink_installed": getattr(future, "_delivery_sink", None) is not None,
            }
        )
        return future


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
    ctx._mq_client = client
    ctx._send_request = lambda *a, **kw: pytest.fail(
        "submit_retrieve must not route through _send_request: it cannot "
        "carry a pre-built future"
    )
    ctx._device = 0
    ctx._event_backend = _StubEventBackend()
    ctx._event_pool = _StubEventPool()
    ctx._layerwise_batch = 8
    return ctx


@pytest.fixture
def patched_future(monkeypatch):
    """Replace the layer-wise future with a sink-installing stub."""
    created: list[MessagingFuture] = []

    class _StubLayerwiseFuture:
        def __init__(self, raw_future, device=None, event_pool=None):
            self.raw_future_ = raw_future
            created.append(raw_future)
            raw_future.set_delivery_sink(lambda response: True)

    monkeypatch.setattr(
        worker_transfer_layerwise,
        "LayerwiseDeviceMessagingFuture",
        _StubLayerwiseFuture,
    )
    return created


def test_sink_is_installed_before_the_request_is_submitted(patched_future):
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
    # The regression: the sink must already be there when the request is
    # handed to the polling loop, not attached afterwards.
    assert call["sink_installed"], (
        "the delivery sink was not installed before submit_request; a first "
        "layer batch arriving immediately would be treated as the final one"
    )
    # The future handed to the client is the one the wrapper wraps.
    assert call["future"] is result.raw_future_


def test_submitted_future_is_the_caller_supplied_one(patched_future):
    client = _RecordingClient()
    ctx = _make_context(client)

    ctx.submit_retrieve(
        "req-1", "cache-key", 7, {}, [[0]], object(), 1, skip_first_n_tokens=3
    )

    assert client.calls[0]["future"] is patched_future[0]
    # skip_first_n_tokens must still reach the wire unchanged.
    assert client.calls[0]["payloads"][-1] == 3


def test_early_frame_without_a_sink_would_be_treated_as_final():
    """Pin the base behaviour the ordering fix protects against."""
    future: MessagingFuture = MessagingFuture()

    # No sink installed yet: a non-final partial frame completes the future.
    partial = (struct.pack("<3i", 0, 8, 0), False, None)
    assert future.deliver(partial) is True
    assert future.query() is True
    assert future.result() == partial
