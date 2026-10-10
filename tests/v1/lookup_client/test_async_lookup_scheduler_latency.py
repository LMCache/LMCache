# SPDX-License-Identifier: Apache-2.0
"""CPU-only regressions for scheduler-side asynchronous lookup latency."""

# Standard
from collections.abc import Iterator
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock
import threading

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client import lmcache_async_lookup_client as async_lookup
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupCleanupMsg,
    LookupRequestMsg,
    LookupResponseMsg,
)
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)
from lmcache.v1.metadata import LMCacheMetadata

pytestmark = pytest.mark.no_shared_allocator


class FakeClock:
    """Advance time explicitly so latency assertions do not depend on CPU load."""

    def __init__(self) -> None:
        self.now = 100.0
        self.waits: list[float] = []

    def time(self) -> float:
        """Return the simulated wall-clock time in seconds."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record a requested delay in seconds and advance the simulated clock."""
        self.waits.append(seconds)
        self.now += seconds


class ResponseInbox:
    """Feed serialized responses through the client's real response thread."""

    def __init__(self) -> None:
        self.messages: Queue[bytes] = Queue()
        self.waiting = threading.Event()

    def recv(self, copy: bool = False) -> bytes:
        """Acknowledge the previous response and wait for the next message.

        Args:
            copy: Unused ZMQ-compatible argument.

        Returns:
            The next serialized worker response.
        """
        self.waiting.set()
        return self.messages.get()

    def respond(self, lookup_id: str, hit_tokens: int) -> None:
        """Publish a worker result and wait until the client has processed it.

        Args:
            lookup_id: Request identifier to complete.
            hit_tokens: Number of tokens found by one worker.

        Raises:
            AssertionError: If the response thread fails to make progress.
        """
        self.waiting.clear()
        self.messages.put(
            msgspec.msgpack.encode(
                LookupResponseMsg(lookup_id=lookup_id, num_hit_tokens=hit_tokens)
            )
        )
        assert self.waiting.wait(timeout=5), (
            "Response thread did not publish the result"
        )

    def close(self, linger: int = 0) -> None:
        """Accept the client's close call; no operating-system socket is owned.

        Args:
            linger: Unused ZMQ-compatible argument.
        """


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Replace only the lookup module's clock, leaving thread waits real."""
    result = FakeClock()
    monkeypatch.setattr(async_lookup, "time", result)
    return result


@pytest.fixture
def lookup_client(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    clock: FakeClock,
) -> Iterator[tuple[LMCacheAsyncLookupClient, list[Mock], ResponseInbox]]:
    """Construct a two-worker client with in-memory transport and token hashing."""
    inbox = ResponseInbox()
    outgoing = [Mock(), Mock()]
    monkeypatch.setattr(async_lookup, "get_zmq_context", Mock())
    monkeypatch.setattr(
        async_lookup, "get_zmq_rpc_path_lmcache", Mock(return_value="unused")
    )
    monkeypatch.setattr(
        async_lookup, "get_zmq_socket", Mock(side_effect=[*outgoing, inbox])
    )
    monkeypatch.setattr(
        "lmcache.v1.token_database.ChunkedTokenDatabase",
        Mock(
            return_value=SimpleNamespace(
                process_tokens=Mock(return_value=[(0, 4, 11), (4, 8, 22)])
            )
        ),
    )
    backoff = getattr(request, "param", None)
    config = LMCacheEngineConfig.from_defaults(
        extra_config={} if backoff is None else {"lookup_backoff_time": backoff},
        lookup_timeout_ms=3000,
    )
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=2,
        local_world_size=2,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 4, 1, 8),
        engine_id="scheduler-latency-test",
    )
    client = LMCacheAsyncLookupClient(config, metadata)
    assert inbox.waiting.wait(timeout=5)
    try:
        yield client, outgoing, inbox
    finally:
        # Wake recv so close can join the response thread without a timeout.
        client.running = False
        inbox.messages.put(
            msgspec.msgpack.encode(
                LookupResponseMsg(lookup_id="shutdown", num_hit_tokens=0)
            )
        )
        client.close()


@pytest.mark.parametrize("lookup_client", [None, 0, 0.001, 0.01], indirect=True)
@pytest.mark.parametrize("num_requests", [1, 8, 32, 128])
def test_submission_does_not_back_off_per_request(
    lookup_client: tuple[LMCacheAsyncLookupClient, list[Mock], ResponseInbox],
    clock: FakeClock,
    num_requests: int,
) -> None:
    """Submit each request to every worker without waiting for a response."""
    client, outgoing, _ = lookup_client
    configs = {"lmcache.tag.user": "test-user"}
    for index in range(num_requests):
        lookup_id = f"request-{index}"
        assert client.lookup_cache(lookup_id) == -1
        assert client.lookup(list(range(8)), lookup_id, configs) is None

    assert clock.waits == []
    for socket in outgoing:
        assert socket.send.call_count == num_requests
        for index, call in enumerate(socket.send.call_args_list):
            message = msgspec.msgpack.decode(call.args[0], type=LookupRequestMsg)
            assert message.lookup_id == f"request-{index}"
            assert message.hashes == [11, 22]
            assert message.offsets == [4, 4]
            assert message.request_configs == configs


@pytest.mark.parametrize("lookup_client", [None, 0, 0.001, 0.01], indirect=True)
@pytest.mark.parametrize("num_requests", [1, 8, 32, 128])
def test_pending_scan_does_not_back_off_per_request(
    lookup_client: tuple[LMCacheAsyncLookupClient, list[Mock], ResponseInbox],
    clock: FakeClock,
    num_requests: int,
) -> None:
    """Polling pending requests must not accumulate a delay or resend work."""
    client, outgoing, _ = lookup_client
    for index in range(num_requests):
        assert client.lookup_cache(f"request-{index}") == -1

    for _ in range(2):
        for index in range(num_requests):
            assert client.lookup_cache(f"request-{index}") is None

    assert clock.waits == []
    for socket in outgoing:
        socket.send.assert_not_called()


def test_responses_are_published_between_pending_polls(
    lookup_client: tuple[LMCacheAsyncLookupClient, list[Mock], ResponseInbox],
    clock: FakeClock,
) -> None:
    """Return pending until all workers respond, then cache their minimum hit count."""
    client, _, inbox = lookup_client
    assert client.lookup_cache("request") == -1
    assert client.lookup(list(range(8)), "request") is None
    assert client.lookup_cache("request") is None
    inbox.respond("request", 8)
    assert client.lookup_cache("request") is None
    inbox.respond("request", 4)
    assert client.lookup_cache("request") == 4
    assert client.lookup_cache("request") == 4
    assert clock.waits == []
    client.clear_lookup_status("request")
    assert client.lookup_cache("request") == -1


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
def test_aborted_lookup_waits_for_all_workers_before_cleanup(
    lookup_client: tuple[LMCacheAsyncLookupClient, list[Mock], ResponseInbox],
    clock: FakeClock,
    cancel: bool,
) -> None:
    """Preserve timeout boundaries and deferred cleanup for late worker responses."""
    client, outgoing, inbox = lookup_client
    assert client.lookup_cache("request") == -1
    if cancel:
        client.cancel_lookup("request")
    else:
        clock.now += 3
        assert client.lookup_cache("request") is None
        clock.now += 0.001
        assert client.lookup_cache("request") == 0

    inbox.respond("request", 8)
    client.lookup_cache("another-request")
    for socket in outgoing:
        socket.send.assert_not_called()

    inbox.respond("request", 4)
    assert client.lookup_cache("another-request") is None
    for socket in outgoing:
        socket.send.assert_called_once()
        message = msgspec.msgpack.decode(
            socket.send.call_args.args[0], type=LookupCleanupMsg
        )
        assert message.lookup_id == "request"

    assert client.lookup_cache("request") == -1
    for socket in outgoing:
        assert socket.send.call_count == 1
