# SPDX-License-Identifier: Apache-2.0
"""Small real-client regression coverage for cached hit limiting."""

# Standard
from pathlib import Path
import json
import threading
import time

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupRequestMsg,
    LookupResponseMsg,
)
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)
from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc.transport import RpcClientTransport
import lmcache.v1.lookup_client.lmcache_async_lookup_client as async_client_module

pytestmark = pytest.mark.no_shared_allocator


class RecordedRpcClientTransport(RpcClientTransport):
    """Deterministic RPC boundary that records actual client messages."""

    def __init__(self, responses: list[int]) -> None:
        """Initialize the transport with one hit count per RPC request."""
        self._responses = iter(responses)
        self.messages: list[list[object]] = []
        self.closed = False

    @property
    def world_size(self) -> int:
        """Return the single deterministic worker represented by this transport."""
        return 1

    def send_and_recv_all(self, msg: list[object]) -> list[bytes]:
        """Record one client message and return its configured hit count."""
        self.messages.append(msg)
        return [next(self._responses).to_bytes(4, "big")]

    def close(self) -> None:
        """Mark this deterministic transport as closed."""
        self.closed = True


def _make_client(
    hit_miss_ratio: float, responses: list[int], chunk_size: int = 4
) -> tuple[HitLimitLookupClient, LMCacheEngineConfig, RecordedRpcClientTransport]:
    """Build a real synchronous lookup client with deterministic RPC replies."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        hit_miss_ratio=hit_miss_ratio,
        lmcache_instance_id="lookup-hit-limit-test",
    )
    metadata = LMCacheMetadata(
        model_name="lookup-hit-limit-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(1, 2, chunk_size, 1, 1),
        chunk_size=chunk_size,
    )
    transport = RecordedRpcClientTransport(responses)
    actual_client = LMCacheLookupClient(config, metadata, transport)
    return HitLimitLookupClient(actual_client, config), config, transport


@pytest.mark.parametrize(
    "case,hit_miss_ratio,expected",
    [
        ("no_limit", 0.0, 8),
        ("half_limit", 0.5, 4),
        ("zero_limit", 1.0, 0),
    ],
)
def test_lookup_cache_keeps_the_lookup_hit_limit(
    case: str, hit_miss_ratio: float, expected: int
) -> None:
    """A real completed lookup and its cached re-entry share one public limit."""
    client, _config, transport = _make_client(hit_miss_ratio, responses=[8])
    tokens = list(range(8))
    lookup_id = f"cached-hit-limit-{case}"
    request_configs = {"lmcache.tag.tenant": "cached-hit-limit"}
    try:
        assert client.lookup_cache(lookup_id) == -1
        assert client.lookup(tokens, lookup_id, request_configs) == expected
        assert client.lookup_cache(lookup_id) == expected
        request_config_frame = transport.messages[0][-1]
        assert isinstance(request_config_frame, str)
        assert json.loads(request_config_frame) == request_configs
        client.clear_lookup_status(lookup_id)
        assert client.lookup_cache(lookup_id) == -1
    finally:
        client.close()
    assert transport.closed is True


def test_lookup_cache_uses_the_current_dynamic_ratio_and_chunk_size() -> None:
    """A completed request reads both live limit settings on re-entry."""
    client, config, transport = _make_client(0.0, responses=[10])
    try:
        assert client.lookup(list(range(10)), "dynamic-ratio-and-chunk") == 10
        config.hit_miss_ratio = 0.1
        config.chunk_size = 8
        assert client.lookup_cache("dynamic-ratio-and-chunk") == 8
    finally:
        client.close()
    assert transport.closed is True


def test_lookup_cache_accepts_a_live_none_ratio_then_a_numeric_ratio() -> None:
    """A disabled live quota preserves raw hits until a number is restored."""
    client, config, transport = _make_client(0.0, responses=[8])
    try:
        assert client.lookup(list(range(8)), "dynamic-none-ratio") == 8
        config.hit_miss_ratio = None
        assert client.lookup_cache("dynamic-none-ratio") == 8
        config.hit_miss_ratio = 0.5
        assert client.lookup_cache("dynamic-none-ratio") == 4
    finally:
        client.close()
    assert transport.closed is True


def test_lookup_cache_keeps_a_partial_hit_below_the_limit() -> None:
    """The established wrapper keeps raw partial hits below its ratio boundary."""
    client, _config, transport = _make_client(0.1, responses=[12], chunk_size=8)
    try:
        tokens = list(range(17))
        assert client.lookup(tokens, "partial-below-limit") == 12
        assert client.lookup_cache("partial-below-limit") == 12
    finally:
        client.close()
    assert transport.closed is True


def test_clear_permits_lookup_id_reuse_at_a_different_token_length() -> None:
    """Clear removes the cached token count alongside the inner status."""
    client, _config, transport = _make_client(0.0, responses=[8, 10])
    lookup_id = "reused-lookup-id"
    try:
        assert client.lookup(list(range(8)), lookup_id) == 8
        assert client.lookup_cache(lookup_id) == 8
        client.clear_lookup_status(lookup_id)
        assert client.lookup_cache(lookup_id) == -1
        assert client.lookup(list(range(10)), lookup_id) == 10
        assert client.lookup_cache(lookup_id) == 10
        assert len(transport.messages) == 2
    finally:
        client.close()
    assert transport.closed is True


def test_lookup_cache_preserves_zero_hit_and_empty_request_contracts() -> None:
    """Zero is cacheable; an empty request keeps the inner not-found sentinel."""
    client, _config, transport = _make_client(0.5, responses=[0])
    try:
        assert client.lookup(list(range(8)), "zero-hit") == 0
        assert client.lookup_cache("zero-hit") == 0
        assert client.lookup([], "empty-request") == 0
        assert client.lookup_cache("empty-request") == -1
    finally:
        client.close()
    assert transport.closed is True


class SingleResponseLookupPeer:
    """One real ZMQ/msgspec worker response, with test-controlled release."""

    def __init__(self, worker_path: str, scheduler_path: str, hit_tokens: int) -> None:
        """Initialize a peer that withholds one real response until released."""
        self.worker_path = worker_path
        self.scheduler_path = scheduler_path
        self.hit_tokens = hit_tokens
        self.ready = threading.Event()
        self.request_seen = threading.Event()
        self.release_response = threading.Event()
        self.response_sent = threading.Event()
        self.stop = threading.Event()
        self.request: LookupRequestMsg | None = None
        self.thread = threading.Thread(
            target=self._run, name="lookup-hit-limit-zmq-peer"
        )

    def start(self) -> None:
        """Start the peer and wait until both IPC endpoints are ready."""
        self.thread.start()
        assert self.ready.wait(timeout=10)

    def _run(self) -> None:
        """Receive one request and publish its configured response."""
        context = zmq.Context()
        pull_socket = context.socket(zmq.PULL)
        push_socket = context.socket(zmq.PUSH)
        pull_socket.setsockopt(zmq.RCVTIMEO, 100)
        pull_socket.bind(f"ipc://{self.worker_path}")
        push_socket.connect(f"ipc://{self.scheduler_path}")
        self.ready.set()
        try:
            while not self.stop.is_set():
                try:
                    message = pull_socket.recv(copy=False)
                except zmq.Again:
                    continue
                decoded = msgspec.msgpack.decode(message, type=LookupRequestMsg)
                self.request = decoded
                self.request_seen.set()
                self.release_response.wait(timeout=10)
                if self.stop.is_set():
                    return
                response = LookupResponseMsg(
                    lookup_id=decoded.lookup_id, num_hit_tokens=self.hit_tokens
                )
                push_socket.send(msgspec.msgpack.encode(response), copy=False)
                self.response_sent.set()
                self.stop.wait(timeout=10)
                return
        finally:
            pull_socket.close(linger=0)
            push_socket.close(linger=0)
            context.term()

    def close(self) -> None:
        """Release a blocked response and join the peer thread."""
        self.stop.set()
        self.release_response.set()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()


def test_async_cached_result_keeps_the_hit_limit_over_real_zmq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async completion reaches the real client state machine before capping."""
    worker_path = str(tmp_path / "lookup-worker.sock")
    scheduler_path = str(tmp_path / "lookup-scheduler.sock")

    def private_rpc_path(
        _engine_id: str, service_name: str, _rpc_port: int, _rank: int
    ) -> str:
        """Map the two test services onto isolated IPC paths."""
        return worker_path if service_name == "lookup_worker" else scheduler_path

    client_context = zmq.Context()
    monkeypatch.setattr(
        async_client_module, "get_zmq_context", lambda use_asyncio: client_context
    )
    monkeypatch.setattr(
        async_client_module, "get_zmq_rpc_path_lmcache", private_rpc_path
    )

    real_get_zmq_socket = async_client_module.get_zmq_socket

    def get_private_zmq_socket(
        context: zmq.Context,
        socket_path: str,
        protocol: str,
        role: zmq.SocketType,
        bind_or_connect: str,
    ) -> zmq.Socket:
        """Create a test socket with a bounded pull wait."""
        socket = real_get_zmq_socket(
            context, socket_path, protocol, role, bind_or_connect
        )
        if role == zmq.PULL:
            socket.setsockopt(zmq.RCVTIMEO, 100)
        return socket

    monkeypatch.setattr(async_client_module, "get_zmq_socket", get_private_zmq_socket)

    peer = SingleResponseLookupPeer(worker_path, scheduler_path, hit_tokens=8)
    client = None
    actual_client = None
    lookup_id = "async-cached-hit-limit"
    request_configs = {"lmcache.tag.tenant": "async-hit-limit"}
    try:
        peer.start()
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=4,
            hit_miss_ratio=1.0,
            enable_async_loading=True,
            lmcache_instance_id="async-lookup-hit-limit",
            lookup_timeout_ms=60_000,
            extra_config={"lookup_backoff_time": 0.001},
        )
        metadata = LMCacheMetadata(
            model_name="async-lookup-hit-limit",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float32,
            kv_shape=(1, 2, 4, 1, 1),
            chunk_size=4,
            engine_id="async-hit-limit-engine",
            kv_connector_extra_config={"lmcache_rpc_port": 0},
        )
        client = LookupClientFactory.create_lookup_client(config, metadata)
        assert isinstance(client, HitLimitLookupClient)
        assert isinstance(client.actual_lookup_client, LMCacheAsyncLookupClient)
        actual_client = client.actual_lookup_client

        assert client.lookup_cache(lookup_id) == -1
        assert client.lookup(list(range(8)), lookup_id, request_configs) is None
        assert peer.request_seen.wait(timeout=10)
        assert peer.request is not None
        assert peer.request.lookup_id == lookup_id
        assert peer.request.offsets == [4, 4]
        assert peer.request.request_configs == request_configs
        assert client.lookup_cache(lookup_id) is None

        peer.release_response.set()
        assert peer.response_sent.wait(timeout=10)
        deadline = time.monotonic() + 10
        raw_result = None
        while time.monotonic() < deadline:
            raw_result = actual_client.lookup_cache(lookup_id)
            if raw_result == 8:
                break
            time.sleep(0.001)
        assert raw_result == 8
        assert client.lookup_cache(lookup_id) == 0
    finally:
        if client is not None:
            client.close()
        else:
            client_context.term()
        peer.close()
        Path(worker_path).unlink(missing_ok=True)
        Path(scheduler_path).unlink(missing_ok=True)
    assert actual_client is not None
    assert not actual_client.thread.is_alive()
