# SPDX-License-Identifier: Apache-2.0
"""LMCacheAsyncLookupClient returns a lookup's result as soon as every worker
has answered, instead of always sleeping lookup_backoff_time.

CPU only: the client is built without __init__, with fake push/pull sockets.
The real response thread (process_responses_from_workers) runs on the fake
pull socket, which yields one response per worker for every pushed request.
"""

# Standard
from typing import Optional
import queue
import threading
import time

# Third Party
import msgspec
import pytest

# First Party
from lmcache.inprocess.v1.lookup_client.async_lookup_message import (
    LookupRequestMsg,
    LookupResponseMsg,
)
from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)

BACKOFF = 2.0  # long, so a fixed sleep would be obvious


class _Workers:
    """Pull socket fed by the push sockets: each worker answers each request
    with its hit count (None: that worker never answers)."""

    def __init__(self, hits: list[Optional[int]]):
        self.hits = hits
        self.q: "queue.Queue[bytes]" = queue.Queue()

    def push_socket(self, rank: int):
        workers = self

        class _Push:
            def send(self, buf, copy=False):
                req = msgspec.msgpack.decode(buf, type=LookupRequestMsg)
                if workers.hits[rank] is not None:
                    workers.q.put(
                        msgspec.msgpack.encode(
                            LookupResponseMsg(
                                lookup_id=req.lookup_id,
                                num_hit_tokens=workers.hits[rank],
                            )
                        )
                    )

        return _Push()

    def recv(self, copy=False):
        while True:
            try:
                return self.q.get(timeout=0.05)
            except queue.Empty:
                continue


class _TokenDB:
    def process_tokens(self, token_ids, make_key=False):
        for start in range(0, len(token_ids), 256):
            yield start, min(start + 256, len(token_ids)), hash(start)


class _Config:
    lookup_timeout_ms = 60_000


def _client(hits: list[Optional[int]]) -> LMCacheAsyncLookupClient:
    c = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
    workers = _Workers(hits)
    c.config = _Config()
    c.world_size = len(hits)
    c.token_database = _TokenDB()
    c.push_sockets = [workers.push_socket(r) for r in range(len(hits))]
    c.pull_socket = workers
    c.lock = threading.Lock()
    c.lookup_done = threading.Condition(c.lock)
    c.reqs_status = {}
    c.res_for_each_worker = {}
    c.first_lookup_time = {}
    c.aborted_lookups = set()
    c.lookup_backoff_time = BACKOFF
    c.running = True
    c.thread = threading.Thread(target=c.process_responses_from_workers, daemon=True)
    c.thread.start()
    return c


@pytest.fixture
def make_client():
    clients = []

    def make(hits):
        c = _client(hits)
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.running = False


def _first_lookup(c: LMCacheAsyncLookupClient, req: str):
    # Same call order as the vLLM adapter: lookup_cache first, then lookup.
    assert c.lookup_cache(req) == -1
    t = time.perf_counter()
    res = c.lookup(list(range(1024)), lookup_id=req)
    return res, time.perf_counter() - t


def test_lookup_returns_result_without_waiting_backoff(make_client):
    c = make_client([768, 512, 1024, 768])
    res, dt = _first_lookup(c, "req-1")
    assert res == 512  # minimum over workers
    assert dt < BACKOFF / 4, f"lookup took {dt:.3f}s, expected an early return"
    # The scheduler's next check sees the same result, without waiting.
    t = time.perf_counter()
    assert c.lookup_cache("req-1") == 512
    assert time.perf_counter() - t < 0.1


def test_lookup_pending_returns_none_after_backoff(make_client):
    c = make_client([256, None])  # the second worker never answers
    res, dt = _first_lookup(c, "req-2")
    assert res is None
    assert dt >= BACKOFF * 0.9
    # Still pending: lookup_cache waits at most the backoff and reports None.
    t = time.perf_counter()
    assert c.lookup_cache("req-2") is None
    assert BACKOFF * 0.9 <= time.perf_counter() - t < BACKOFF * 2


def test_lookup_cache_wakes_up_when_result_arrives(make_client):
    c = make_client([256, 256])
    c.lookup_cache("req-3")  # marks the lookup as ongoing
    with c.lock:
        c.reqs_status["req-3"] = None
    c.first_lookup_time["req-3"] = time.time()

    def answer():
        time.sleep(0.2)
        with c.lock:
            c.reqs_status["req-3"] = 256
            c.lookup_done.notify_all()

    threading.Thread(target=answer, daemon=True).start()
    t = time.perf_counter()
    assert c.lookup_cache("req-3") == 256
    assert 0.15 <= time.perf_counter() - t < BACKOFF / 2
