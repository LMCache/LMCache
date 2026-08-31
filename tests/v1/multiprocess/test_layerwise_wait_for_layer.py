# SPDX-License-Identifier: Apache-2.0
"""Tests for the missing-layer guard in :meth:`wait_for_layer`.

A layer-wise retrieve announces one event per layer batch.  If the stream
closes without ever announcing the layer the consumer asks for, inserting no
stream wait would let attention read KV that was never transferred.  The
guard has to tell that apart from a failed retrieve, where nothing landed on
purpose and the vLLM adapter still calls this for every pending request.
"""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.futures_layerwise import LayerwiseDeviceMessagingFuture


class _StubEventBackend:
    """Records the events a stream was told to wait on."""

    def __init__(self) -> None:
        self.waited: list[Any] = []

    def wait_event(self, evt: Any, stream: Any) -> None:
        self.waited.append(evt)

    def synchronize_event(self, evt: Any, device: Any) -> None:
        pass

    def query_event(self, evt: Any) -> bool:
        return True


@pytest.fixture
def future() -> LayerwiseDeviceMessagingFuture:
    fut: LayerwiseDeviceMessagingFuture = LayerwiseDeviceMessagingFuture()
    fut._event_backend = _StubEventBackend()  # type: ignore[assignment]
    return fut


def test_raises_when_a_successful_retrieve_never_delivered_the_layer(future):
    """Silently skipping the wait here would corrupt the forward pass."""
    # Closing frame with no batch ever announced, reporting success.
    future.raw_future_.set_result((b"", True, True))

    with pytest.raises(RuntimeError, match="never delivered layer 3"):
        future.wait_for_layer(3)


def test_returns_quietly_when_the_retrieve_failed(future):
    """No KV landed, the caller recomputes, so there is nothing to order."""
    future.raw_future_.set_result((b"", True, False))

    future.wait_for_layer(3)

    assert future._event_backend.waited == []


def test_still_waits_on_a_layer_that_did_arrive(future):
    """Positive control: the guard must not swallow the normal path."""
    sentinel = object()
    future._layer_event_map[3] = sentinel
    future.raw_future_.set_result((b"", True, True))

    future.wait_for_layer(3)

    assert future._event_backend.waited == [sentinel]

    # Re-waiting on the same event is still skipped.
    future.wait_for_layer(3)
    assert future._event_backend.waited == [sentinel]
