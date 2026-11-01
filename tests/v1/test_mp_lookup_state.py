# SPDX-License-Identifier: Apache-2.0
"""Contracts of the per-request MP lookup state."""

# Standard
from collections import deque
from collections.abc import Callable
from typing import Any, cast
import random
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.mp_lookup_state import LookupState
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient

TOKENS_PER_CHUNK = 64


class SilentFuture(MessagingFuture[Any]):
    """A future whose done-callbacks only run when the test fires them."""

    def __init__(self) -> None:
        super().__init__()
        self.held: list[Callable[[MessagingFuture[Any]], None]] = []

    def add_done_callback(self, fn: Callable[[MessagingFuture[Any]], None]) -> None:
        """Hold the callback instead of arming it."""
        self.held.append(fn)

    def fire(self) -> None:
        """Run the held callbacks now."""
        for fn in self.held:
            fn(self)


class WaitRecorder(MessagingFuture[Any]):
    """A future that records the timeout of every wait on it."""

    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float | None] = []
        self.waiting = threading.Event()

    def wait(self, timeout: float | None = None) -> bool:
        """Record the wait, then wait for real."""
        self.waits.append(timeout)
        self.waiting.set()
        return super().wait(timeout)


class FakeClient:
    """RequestClient stub that hands out futures the test completes."""

    def __init__(self) -> None:
        self.lookups: list[MessagingFuture[Any]] = []
        self.queries: list[MessagingFuture[Any]] = []
        self.frees: list[IPCCacheServerKey] = []
        self.next_lookups: deque[MessagingFuture[Any]] = deque()
        self.next_queries: deque[MessagingFuture[Any]] = deque()
        self.violations: list[str] = []
        self.lock = threading.Lock()

    def lookup(self, key: IPCCacheServerKey, tp_size: int) -> MessagingFuture[Any]:
        """Return the next prepared LOOKUP future, or a pending one."""
        assert tp_size == 1
        future = self.next_lookups.popleft() if self.next_lookups else MessagingFuture()
        self.lookups.append(future)
        return future

    def query_prefetch_status(self, request_id: str) -> MessagingFuture[Any]:
        """Return the next prepared status future, or a pending one."""
        assert request_id == "r"
        with self.lock:
            # Recorded as well as raised: a raise inside a done-callback is
            # only logged.
            if not (self.lookups and self.lookups[-1].query()):
                self.violations.append("status before LOOKUP ack")
            if self.queries and not self.queries[-1].query():
                self.violations.append("two status queries in flight")
            assert not self.violations, self.violations
            future = (
                self.next_queries.popleft() if self.next_queries else MessagingFuture()
            )
            self.queries.append(future)
        return future

    def free_lookup_locks(
        self, key: IPCCacheServerKey, tp_size: int
    ) -> MessagingFuture[Any]:
        """Record a released lock range."""
        self.frees.append(key)
        return ready(None)

    def pending_query(self) -> MessagingFuture[Any] | None:
        """Return the status query in flight, if any."""
        with self.lock:
            if self.queries and not self.queries[-1].query():
                return self.queries[-1]
        return None


def ready(value: Any) -> MessagingFuture[Any]:
    """Return a completed future."""
    future: MessagingFuture[Any] = MessagingFuture()
    future.set_result(value)
    return future


def make_key(num_tokens: int = 256) -> IPCCacheServerKey:
    """Return a LOOKUP key for request ``r``."""
    return IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=1,
        worker_id=None,
        token_ids=range(num_tokens),
        start=0,
        end=num_tokens,
        request_id="r",
        num_kv_readers=1,
    )


def make_state(
    count: int = 1, timeout: float = 5.0, blocking_lookup: bool = False
) -> tuple[LookupState, list[FakeClient]]:
    """Return a lookup state over ``count`` fake servers."""
    clients = [FakeClient() for _ in range(count)]
    state = LookupState(
        make_key(),
        {f"s{i}": cast(RequestClient, client) for i, client in enumerate(clients)},
        tp_size=1,
        tokens_per_chunk=TOKENS_PER_CHUNK,
        timeout=timeout,
        blocking_lookup=blocking_lookup,
    )
    return state, clients


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the monotonic clock with a settable one."""
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def run_bounded(fn: Callable[[], Any], timeout: float = 5.0) -> Any:
    """Run ``fn`` in a thread and fail if it does not return in time."""
    results: list[Any] = []
    thread = threading.Thread(target=lambda: results.append(fn()), daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "call did not return"
    return results[0]


def test_each_server_is_queried_after_its_own_ack() -> None:
    state, (first, second) = make_state(2)
    state.submit()
    assert len(first.lookups) == len(second.lookups) == 1
    assert state.on_scheduler_check() == []
    assert not first.queries and not second.queries

    first.lookups[0].set_result(None)
    assert len(first.queries) == 1 and not second.queries

    second.lookups[0].set_result(None)
    assert len(second.queries) == 1
    assert state.get_result() is None


def test_none_before_any_check_is_deferred_to_the_next_check() -> None:
    state, (client,) = make_state()
    state.submit()
    client.lookups[0].set_result(None)
    client.queries[0].set_result(None)
    assert len(client.queries) == 1

    assert state.on_scheduler_check() == []
    assert len(client.queries) == 2
    client.queries[1].set_result(3)
    assert state.get_result() == 3 * TOKENS_PER_CHUNK


def test_none_after_a_check_is_requeried_from_the_callback() -> None:
    state, (client,) = make_state()
    state.submit()
    client.lookups[0].set_result(None)
    state.on_scheduler_check()

    client.queries[0].set_result(None)
    assert len(client.queries) == 2
    client.queries[1].set_result(None)
    assert len(client.queries) == 2, "re-sent twice without a check in between"


def test_at_most_one_new_status_query_per_check() -> None:
    state, (client,) = make_state()
    state.submit()
    client.lookups[0].set_result(None)
    checks = 20
    for _ in range(checks):
        state.on_scheduler_check()
        while (future := client.pending_query()) is not None:
            future.set_result(None)
    assert len(client.queries) <= checks + 1
    assert state.get_result() is None
    assert not client.violations


def test_unequal_hits_take_the_minimum_and_free_the_excess_tail_once() -> None:
    state, (high, low) = make_state(2)
    state.submit()
    for client in (high, low):
        client.lookups[0].set_result(None)
    high.queries[0].set_result(4)
    assert state.get_result() is None
    low.queries[0].set_result(2)

    assert state.get_result() == 2 * TOKENS_PER_CHUNK
    for _ in range(3):
        assert state.on_scheduler_check() == []
        state.on_query_prefetch_status_finished("s0", high.queries[0])
    assert [(key.start, key.end) for key in high.frees] == [(128, 256)]
    assert high.frees[0].request_id == "r" and high.frees[0].worker_id is None
    assert not low.frees


def test_reply_handled_at_check_is_not_handled_again_by_its_callback() -> None:
    state, (high, low) = make_state(2)
    silent = [SilentFuture(), SilentFuture()]
    for client, future in zip((high, low), silent, strict=True):
        client.next_queries.append(future)
    state.submit()
    for client in (high, low):
        client.lookups[0].set_result(None)
    high.queries[0].set_result(4)
    low.queries[0].set_result(2)

    assert state.on_scheduler_check() == []
    assert state.get_result() == 2 * TOKENS_PER_CHUNK
    for future in silent:
        future.fire()
    assert len(high.frees) == 1 and not low.frees


def test_already_completed_futures_resolve_inside_submit() -> None:
    state, clients = make_state(2)
    for client in clients:
        client.next_lookups.append(ready(None))
        client.next_queries.append(ready(3))
    run_bounded(state.submit)
    assert state.get_result() == 3 * TOKENS_PER_CHUNK


def test_completed_reply_whose_callback_never_ran_is_not_timed_out(
    clock: list[float],
) -> None:
    state, (client,) = make_state(timeout=5)
    client.next_queries.append(SilentFuture())
    state.submit()
    client.lookups[0].set_result(None)
    client.queries[0].set_result(2)
    clock[0] += 10

    assert state.on_scheduler_check() == []
    assert state.get_result() == 2 * TOKENS_PER_CHUNK


def test_failed_lookup_still_queries_status() -> None:
    state, (client,) = make_state()
    state.submit()
    client.lookups[0].set_exception(RuntimeError("send failed"))
    assert len(client.queries) == 1
    client.queries[0].set_result(0)
    assert state.get_result() == 0


def test_failed_status_is_requeried_with_its_own_deadline(clock: list[float]) -> None:
    state, (client,) = make_state(timeout=5)
    state.submit()
    client.lookups[0].set_result(None)
    for _ in range(3):
        clock[0] += 4
        assert state.on_scheduler_check() == []
        client.queries[-1].set_exception(RuntimeError("status failed"))
    assert len(client.queries) == 4
    assert state.get_result() is None

    clock[0] += 4.9
    assert state.on_scheduler_check() == []
    clock[0] += 0.2
    assert state.on_scheduler_check() == ["s0"]
    assert state.get_result() == 0


def test_waiting_for_a_check_does_not_count_against_the_deadline(
    clock: list[float],
) -> None:
    state, (client,) = make_state(timeout=5)
    state.submit()
    client.lookups[0].set_result(None)
    client.queries[0].set_result(None)

    clock[0] += 60
    assert state.on_scheduler_check() == []
    assert len(client.queries) == 2
    assert state.get_result() is None

    clock[0] += 4.9
    assert state.on_scheduler_check() == []
    clock[0] += 0.2
    assert state.on_scheduler_check() == ["s0"]


@pytest.mark.parametrize("acked", [False, True])
def test_timeout_is_reported_once_and_final(clock: list[float], acked: bool) -> None:
    state, (client,) = make_state(timeout=5)
    state.submit()
    if acked:
        client.lookups[0].set_result(None)
    clock[0] += 5

    assert state.on_scheduler_check() == ["s0"]
    assert state.get_result() == 0
    assert state.on_scheduler_check() == []

    sent = len(client.queries)
    client.lookups[0].set_result(None)
    if acked:
        client.queries[0].set_result(4)
    assert len(client.queries) == sent
    assert state.get_result() == 0
    assert state.drain() is None


def test_blocking_check_does_not_wait_before_the_ack() -> None:
    state, (client,) = make_state(blocking_lookup=True)
    state.submit()
    assert run_bounded(state.on_scheduler_check, timeout=1) == []
    assert state.get_result() is None


def test_blocking_check_waits_for_the_status_reply() -> None:
    state, (client,) = make_state(timeout=5, blocking_lookup=True)
    status = WaitRecorder()
    client.next_queries.append(status)
    state.submit()
    client.lookups[0].set_result(None)

    results: list[list[str]] = []
    thread = threading.Thread(target=lambda: results.append(state.on_scheduler_check()))
    thread.start()
    try:
        assert status.waiting.wait(1)
        assert thread.is_alive()
        status.set_result(2)
        thread.join(1)
        assert not thread.is_alive()
    finally:
        status.set_result(2)
        thread.join(6)
    assert results == [[]]
    timeout = status.waits[0]
    assert timeout is not None and 0 < timeout <= 5
    assert state.get_result() == 2 * TOKENS_PER_CHUNK


def test_blocking_check_does_not_wait_on_a_requery_it_triggered() -> None:
    state, (client,) = make_state(timeout=5, blocking_lookup=True)
    first = WaitRecorder()
    client.next_queries.append(first)
    state.submit()
    client.lookups[0].set_result(None)

    done = threading.Event()

    def check() -> None:
        state.on_scheduler_check()
        done.set()

    thread = threading.Thread(target=check)
    thread.start()
    try:
        assert first.waiting.wait(1)
        first.set_result(None)
        assert done.wait(1), "check waited on the query its reply re-sent"
    finally:
        first.set_result(None)
        thread.join(6)
    assert len(client.queries) == 2 and not client.queries[1].query()
    assert state.get_result() is None


def test_close_stops_sending_and_finalizes_an_unresolved_result() -> None:
    state, (client,) = make_state()
    state.submit()
    state.close()
    client.lookups[0].set_result(None)
    assert not client.queries
    assert state.get_result() == 0
    assert state.on_scheduler_check() == []


def test_close_keeps_a_resolved_result() -> None:
    state, (client,) = make_state()
    state.submit()
    client.lookups[0].set_result(None)
    client.queries[0].set_result(2)
    state.close()
    assert state.get_result() == 2 * TOKENS_PER_CHUNK


def test_drain_waits_for_rpcs_in_flight_without_sending_more() -> None:
    state, (acked, unacked) = make_state(2)
    state.submit()
    acked.lookups[0].set_result(None)

    done = threading.Event()

    def drain() -> None:
        state.drain()
        done.set()

    thread = threading.Thread(target=drain)
    thread.start()
    try:
        assert not done.wait(0.05)
        unacked.lookups[0].set_result(None)
        assert not done.wait(0.05)
        acked.queries[0].set_result(None)
        assert done.wait(1)
    finally:
        unacked.lookups[0].set_result(None)
        acked.queries[0].set_result(None)
        thread.join(6)
    assert len(acked.queries) == 1 and not unacked.queries


def test_drain_returns_the_server_that_missed_its_deadline() -> None:
    state, (client,) = make_state(timeout=0.05)
    state.submit()
    assert run_bounded(state.drain) == "s0"


def test_concurrent_replies_keep_one_query_in_flight_per_server() -> None:
    state, clients = make_state(4, timeout=30)
    rng = random.Random(0)
    final = {0: 5, 1: 3, 2: 7, 3: 3}
    stop = threading.Event()

    def reply_randomly() -> None:
        while not stop.is_set():
            index = rng.randrange(len(clients))
            client = clients[index]
            if not client.lookups[0].query():
                client.lookups[0].set_result(None)
            elif (future := client.pending_query()) is not None:
                future.set_result(final[index] if rng.random() < 0.05 else None)
            time.sleep(0)

    state.submit()
    replier = threading.Thread(target=reply_randomly)
    replier.start()
    try:
        deadline = time.monotonic() + 10
        while state.get_result() is None and time.monotonic() < deadline:
            assert state.on_scheduler_check() == []
            time.sleep(0.001)
    finally:
        stop.set()
        replier.join(5)
    assert state.get_result() == 3 * TOKENS_PER_CHUNK
    assert [len(client.frees) for client in clients] == [1, 0, 1, 0]
    assert not any(client.violations for client in clients)
