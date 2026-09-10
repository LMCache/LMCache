# SPDX-License-Identifier: Apache-2.0
"""Public adapter contracts for nonblocking status polling."""

# Standard
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any
import threading

# Third Party
import pytest
import zmq

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_module
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.futures import MessagingFuture


class CheckedFuture(MessagingFuture[Any]):
    """Use real completion state while rejecting a blocking scheduler read."""

    def __init__(self) -> None:
        super().__init__()
        self.observed_ready = False

    def query(self) -> bool:
        """Record when the real future becomes observable."""
        ready = super().query()
        self.observed_ready |= ready
        return ready

    def result(self, timeout: float | None = None) -> Any:
        """Enforce nonblocking reads while testing scheduler callbacks."""
        assert timeout == 0, "lookup callback attempted a blocking future read"
        assert self.observed_ready, "result read preceded ready query"
        return super().result(timeout)


def ready(value: Any) -> MessagingFuture[Any]:
    """Return a completed real messaging future."""
    future: MessagingFuture[Any] = MessagingFuture()
    future.set_result(value)
    return future


class Client:
    """Old-peer RPC surface only; no L0 capabilities or GPU API are supplied."""

    def __init__(self) -> None:
        self.ack = CheckedFuture()
        self.status = CheckedFuture()
        self.replies: deque[MessagingFuture[Any]] = deque([self.status])
        self.lookups: list[Any] = []
        self.queries: list[str] = []
        self.frees: list[Any] = []
        self.ends: list[str] = []
        self.free_ack = ready(None)
        self.end_ack: MessagingFuture[Any] = MessagingFuture()

    def get_chunk_size(self) -> MessagingFuture[Any]:
        """Advertise the fixed chunk64 test geometry."""
        return ready(64)

    def lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Record submission and return its controllable acknowledgement."""
        assert key.world_size == 1 and tp_size == 1
        self.lookups.append(key)
        return self.ack

    def query_prefetch_status(self, request_id: str) -> MessagingFuture[Any]:
        """Return the next explicitly prepared status response."""
        self.queries.append(request_id)
        assert self.replies, "duplicate status RPC while previous one is pending"
        return self.replies.popleft()

    def free_lookup_locks(self, key: Any, tp_size: int) -> MessagingFuture[Any]:
        """Record released ranges and expose their acknowledgement."""
        self.frees.append(key)
        return self.free_ack

    def end_session(self, request_id: str) -> MessagingFuture[Any]:
        """Record session removal after its preceding obligations."""
        self.ends.append(request_id)
        return self.end_ack

    def close(self) -> None:
        """Close the stub, which owns no transport resources."""
        pass


AdapterFactory = Callable[..., tuple[LMCacheMPSchedulerAdapter, list[Client]]]


@pytest.fixture
def make_adapter(monkeypatch: pytest.MonkeyPatch) -> Iterator[AdapterFactory]:
    """Construct public adapters against controllable old-peer clients."""
    adapters: list[LMCacheMPSchedulerAdapter] = []
    context = zmq.Context()

    def create(
        count: int = 1, timeout: float = 5
    ) -> tuple[LMCacheMPSchedulerAdapter, list[Client]]:
        """Return a scheduler and its per-server RPC stubs."""
        clients = [Client() for _ in range(count)]
        urls = {f"tcp://test-{index}": client for index, client in enumerate(clients)}
        monkeypatch.setattr(
            adapter_module.RequestClientFactory,
            "create",
            lambda url, **kwargs: urls[url],
        )
        adapter = LMCacheMPSchedulerAdapter(
            list(urls),
            context,
            "model",
            16,
            ParallelStrategy(False, count, 0, count, 1, count),
            mq_timeout=timeout,
        )
        adapters.append(adapter)
        return adapter, clients

    yield create
    for adapter in adapters:
        adapter.shutdown()
    context.term()


def submit(adapter: LMCacheMPSchedulerAdapter, clients: list[Client]) -> None:
    """Submit one salted request and complete its LOOKUP acknowledgements."""
    adapter.maybe_submit_lookup_request(
        "r", list(range(256)), cache_salt="tenant", request_configs={"tag": "value"}
    )
    for client in clients:
        client.ack.set_result(None)


def resolved(adapter: LMCacheMPSchedulerAdapter) -> int:
    """Observe a completed response across bounded scheduler ticks."""
    for _ in range(5):
        result = adapter.check_lookup_result("r")
        if result is not None:
            return result
    raise AssertionError("completed status did not become observable")


def test_ack_ordering_and_one_status_in_flight_per_server(
    make_adapter: AdapterFactory,
) -> None:
    adapter, clients = make_adapter(2)
    adapter.maybe_submit_lookup_request("r", list(range(256)))
    clients[0].ack.set_result(None)
    for _ in range(10):
        adapter.maybe_submit_lookup_request("r", list(range(256)))
        assert adapter.check_lookup_result("r") is None
    assert all(len(client.lookups) == 1 and not client.queries for client in clients)
    clients[1].ack.set_result(None)
    for _ in range(10):
        assert adapter.check_lookup_result("r") is None
    assert all(client.queries == ["r"] for client in clients)
    clients[0].status.set_result(2)
    for _ in range(10):
        assert adapter.check_lookup_result("r") is None
    clients[1].status.set_result(2)
    assert resolved(adapter) == 128
    assert [adapter.check_lookup_result("r") for _ in range(5)] == [128] * 5
    assert all(client.queries == ["r"] for client in clients)


@pytest.mark.parametrize("chunks", [0, 2])
def test_none_is_unresolved_and_terminal_zero_is_cached(
    make_adapter: AdapterFactory, chunks: int
) -> None:
    adapter, (client,) = make_adapter()
    submit(adapter, [client])
    client.status.set_result(None)
    final = CheckedFuture()
    client.replies.append(final)
    for _ in range(5):
        assert adapter.check_lookup_result("r") is None
    assert client.queries == ["r", "r"]
    final.set_result(chunks)
    assert resolved(adapter) == chunks * 64
    assert adapter.check_lookup_result("r") == chunks * 64
    assert client.queries == ["r", "r"]


def test_inflight_status_is_request_scoped(make_adapter: AdapterFactory) -> None:
    adapter, (client,) = make_adapter()
    other = CheckedFuture()
    client.replies.append(other)
    for request_id in ("a", "b"):
        adapter.maybe_submit_lookup_request(request_id, list(range(256)))
    client.ack.set_result(None)
    for _ in range(3):
        assert adapter.check_lookup_result("a") is None
        assert adapter.check_lookup_result("b") is None
    other.set_result(2)
    assert adapter.check_lookup_result("b") == 128
    assert adapter.check_lookup_result("a") is None
    client.status.set_result(0)
    assert adapter.check_lookup_result("a") == 0
    assert client.queries == ["a", "b"]


def test_status_deadline_does_not_restart_on_poll(
    make_adapter: AdapterFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [10.0]
    monkeypatch.setattr(adapter_module.time, "monotonic", lambda: now[0])
    adapter, (client,) = make_adapter(timeout=5)
    submit(adapter, [client])
    assert adapter.check_lookup_result("r") is None
    for value in (11.0, 13.0, 14.9):
        now[0] = value
        assert adapter.check_lookup_result("r") is None
    now[0] = 15.1
    assert adapter.check_lookup_result("r") == 0
    assert not adapter.is_healthy
    assert client.queries == ["r"]


def test_status_exception_allows_a_fresh_poll(make_adapter: AdapterFactory) -> None:
    adapter, (client,) = make_adapter()
    submit(adapter, [client])
    client.status.set_exception(RuntimeError("status failed"))
    with pytest.raises(RuntimeError, match="status failed"):
        adapter.check_lookup_result("r")
    client.replies.append(ready(2))
    assert resolved(adapter) == 128
    assert client.queries == ["r", "r"]


def test_unequal_hits_release_only_excess_tail_once(
    make_adapter: AdapterFactory,
) -> None:
    adapter, clients = make_adapter(2)
    submit(adapter, clients)
    for client, chunks in zip(clients, (4, 2), strict=True):
        client.status.set_result(chunks)
    assert resolved(adapter) == 128
    assert [adapter.check_lookup_result("r") for _ in range(5)] == [128] * 5
    assert [(key.start, key.end) for key in clients[0].frees] == [(128, 256)]
    assert not clients[1].frees
    assert all(client.queries == ["r"] for client in clients)


def test_cleanup_discards_old_status_before_request_id_reuse(
    make_adapter: AdapterFactory,
) -> None:
    adapter, (client,) = make_adapter()
    submit(adapter, [client])
    assert adapter.check_lookup_result("r") is None
    adapter.cleanup_lookup_result("r")
    adapter.cleanup_lookup_result("r")
    assert adapter.check_lookup_result("r") == 0
    adapter.maybe_submit_lookup_request("r", list(range(256)))
    client.replies.append(ready(0))
    client.status.set_result(4)
    assert resolved(adapter) == 0
    assert len(client.lookups) == 2 and client.queries == ["r", "r"]
    assert not client.frees


@pytest.mark.parametrize("reply", [None, 2])
def test_end_waits_for_already_sent_status_only(
    make_adapter: AdapterFactory, reply: int | None
) -> None:
    adapter, (client,) = make_adapter()
    submit(adapter, [client])
    assert adapter.check_lookup_result("r") is None
    done = threading.Event()
    errors: list[BaseException] = []

    def finish() -> None:
        try:
            adapter.end_session("r")
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=finish)
    thread.start()
    try:
        assert not done.wait(0.05)
        assert not client.ends
        client.status.set_result(reply)
        assert done.wait(1)
        assert not errors
        assert client.queries == ["r"] and client.ends == ["r"]
        assert not client.frees
        assert not client.end_ack.query()
    finally:
        client.status.set_result(reply)
        thread.join(timeout=6)
        assert not thread.is_alive()
