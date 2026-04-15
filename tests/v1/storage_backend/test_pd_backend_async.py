# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PDBackendAsync (async PD sender/receiver).

No NIXL, CUDA, or real ZMQ peers required — all I/O is mocked with
asyncio.sleep stubs so tests run fast (< 1 s total) in CI.  Assertions
focus on timing and call ordering; data integrity is covered separately
by the NIXL integration tests.
"""

# Standard
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import itertools
import threading
import time

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.pd_backend import AllocRequest as SyncAllocRequest
from lmcache.v1.storage_backend.pd_backend import AllocResponse as SyncAllocResponse
from lmcache.v1.storage_backend.pd_backend import PDMsg as SyncPDMsg
from lmcache.v1.storage_backend.pd_backend_async import (
    AllocRequest as AsyncAllocRequest,
)
from lmcache.v1.storage_backend.pd_backend_async import (
    AllocResponse as AsyncAllocResponse,
)
from lmcache.v1.storage_backend.pd_backend_async import (
    PDBackendAsync,
)
from lmcache.v1.storage_backend.pd_backend_async import PDMsg as AsyncPDMsg
from lmcache.v1.storage_backend.pd_backend_async import (
    ProxyNotif,
)

TRANSFER_DELAY = 0.15
NONBLOCKING_THRESHOLD_RATIO = 0.25
CI_SERIAL_TIMEOUT_MARGIN = 3
_DEFAULT_SHAPE = [4, 2, 16, 8, 128]


def _make_key(i: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test",
        world_size=1,
        worker_id=0,
        chunk_hash=i,
        dtype=torch.bfloat16,
    )


def _make_mem_obj(idx: int = 0) -> MemoryObj:
    obj = MagicMock(spec=MemoryObj)
    obj.meta = SimpleNamespace(
        address=idx,
        fmt=MemoryFormat.KV_2LTD,
        shape=torch.Size(_DEFAULT_SHAPE),
        dtype=torch.bfloat16,
    )
    obj.get_ref_count.return_value = 1
    return obj


def _make_transfer_spec(
    receiver_host="127.0.0.1",
    init_port=9100,
    alloc_port=9101,
    req_id="req-0",
    is_last_prefill=True,
    num_transferred_tokens=0,
):
    return SimpleNamespace(
        receiver_host=receiver_host,
        receiver_init_port=[init_port],
        receiver_alloc_port=[alloc_port],
        req_id=req_id,
        is_last_prefill=is_last_prefill,
        num_transferred_tokens=num_transferred_tokens,
    )


def _make_alloc_req(
    keys,
    last_chunk_toks=16,
    req_id="",
    is_last_batch=False,
    shape=None,
):
    return AsyncAllocRequest(
        keys=[k.to_string() for k in keys],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=list(shape or _DEFAULT_SHAPE),
        dtype="bfloat16",
        last_chunk_toks=last_chunk_toks,
        req_id=req_id,
        is_last_batch=is_last_batch,
    )


def _auto_alloc():
    """Allocator that returns a fresh MemoryObj stub on every call."""
    c = itertools.count()

    def alloc(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        return _make_mem_obj(idx=next(c))

    return alloc


def _pd_backend_patches():
    return (
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.get_zmq_context",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.get_zmq_socket",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.CreateTransferChannel",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.get_correct_device",
            return_value="cpu",
        ),
    )


@contextmanager
def _patched_pd():
    p1, p2, p3, p4 = _pd_backend_patches()
    with p1, p2, p3, p4:
        yield


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def async_sender():
    p1, p2, p3, p4 = _pd_backend_patches()

    with p1, p2 as mock_zmq_sock, p3 as mock_create_tc, p4:
        alloc_socket = MagicMock()
        alloc_response = AsyncAllocResponse(remote_indexes=[0])
        alloc_socket.recv_multipart = AsyncMock(
            return_value=[b"", msgspec.msgpack.encode(alloc_response)]
        )
        alloc_socket.send_multipart = AsyncMock()
        mock_zmq_sock.return_value = alloc_socket

        tc = MagicMock()

        async def _slow_write(*a, **kw):
            await asyncio.sleep(TRANSFER_DELAY)
            return 1

        tc.async_batched_write = _slow_write
        mock_create_tc.return_value = tc

        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="sender",
            pd_proxy_host="127.0.0.1",
            pd_proxy_port=5555,
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
        )
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 16, 8, 128),
        )
        backend = PDBackendAsync(config, metadata)
        backend.proxy_side_channel = MagicMock()

        receiver_id = "127.0.0.1" + str(9100)
        backend.initialized_peers.add(receiver_id)
        backend._async_alloc_sockets[receiver_id] = alloc_socket

        yield backend
        backend.close()


@pytest.fixture
def async_receiver():
    p1, p2, p3, p4 = _pd_backend_patches()

    with p1, p2, p3, p4:
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="receiver",
            pd_peer_host="127.0.0.1",
            pd_peer_init_port=[9200],
            pd_peer_alloc_port=[9201],
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
        )
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 16, 8, 128),
        )
        backend = PDBackendAsync(config, metadata)
        yield backend
        for mem_obj in backend.data.values():
            try:
                mem_obj.ref_count_down()
            except Exception:
                pass
        backend.close()


# ── sender tests ──────────────────────────────────────────────────────────


def test_sender_nonblocking_fifo_transfers(async_sender):
    """batched_submit_put_task returns immediately; same-receiver requests
    are serialized in FIFO order."""
    N = 4
    done_events = [threading.Event() for _ in range(N)]
    completion_order = []
    lock = threading.Lock()

    def make_cb(i):
        def cb(key):
            with lock:
                completion_order.append(i)
            done_events[i].set()

        return cb

    t0 = time.monotonic()
    for i in range(N):
        async_sender.batched_submit_put_task(
            [_make_key(i)],
            [_make_mem_obj(i)],
            transfer_spec=_make_transfer_spec(req_id=f"req-{i}"),
            on_complete_callback=make_cb(i),
        )
    enqueue_elapsed = time.monotonic() - t0

    assert enqueue_elapsed < TRANSFER_DELAY * NONBLOCKING_THRESHOLD_RATIO

    timeout = TRANSFER_DELAY * N * CI_SERIAL_TIMEOUT_MARGIN
    for i, ev in enumerate(done_events):
        assert ev.wait(timeout=timeout), f"req-{i} did not complete"

    assert completion_order == list(range(N))


def test_sender_flow_control_backpressure(async_sender):
    """allocate() blocks when staging buffer is full, unblocks on release."""
    sentinel = _make_mem_obj(idx=77)
    async_sender.memory_allocator.allocate = MagicMock(return_value=sentinel)

    with async_sender._sender_staging_condition:
        async_sender._sender_inflight_chunks = async_sender._sender_max_inflight_chunks

    result = []
    blocked = threading.Event()
    unblocked = threading.Event()

    def worker():
        blocked.set()
        result.append(async_sender.allocate(torch.Size(_DEFAULT_SHAPE), torch.bfloat16))
        unblocked.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert blocked.wait(timeout=2.0)
    time.sleep(0.1)
    assert not unblocked.is_set(), "allocate() should be blocked"

    async_sender._release_sender_staging_chunks(1)
    assert unblocked.wait(timeout=2.0)
    assert result[0] is sentinel
    t.join(timeout=1.0)


def test_sender_chunk_ordering(async_sender):
    """Last prefill chunk waits for prior slow chunk before sending ProxyNotif."""
    SLOW, FAST = 0.30, 0.05
    REQ_ID = "req-chunked"

    call_count = 0
    call_lock = threading.Lock()

    async def controlled_write(*a, **kw):
        nonlocal call_count
        with call_lock:
            call_count += 1
            idx = call_count
        await asyncio.sleep(SLOW if idx == 1 else FAST)
        return 1

    async_sender.transfer_channel.async_batched_write = controlled_write

    notify_times = []
    sent_data = []

    def record_send(data):
        notify_times.append(time.monotonic())
        sent_data.append(data)

    async_sender.proxy_side_channel.send = record_send

    async_sender.batched_submit_put_task(
        [_make_key(0)],
        [_make_mem_obj(0)],
        transfer_spec=_make_transfer_spec(req_id=REQ_ID, is_last_prefill=False),
    )
    time.sleep(0.01)

    done = threading.Event()
    async_sender.batched_submit_put_task(
        [_make_key(1)],
        [_make_mem_obj(1)],
        transfer_spec=_make_transfer_spec(req_id=REQ_ID, is_last_prefill=True),
        on_complete_callback=lambda k: done.set(),
    )
    t_submit = time.monotonic()

    assert done.wait(timeout=SLOW * 3)
    assert len(notify_times) == 1

    notif = msgspec.msgpack.decode(sent_data[0], type=AsyncPDMsg)
    assert isinstance(notif, ProxyNotif) and notif.req_id == REQ_ID

    elapsed = notify_times[0] - t_submit
    assert elapsed >= SLOW * 0.8, (
        f"ProxyNotif too early ({elapsed:.3f}s) — fast chunk didn't wait for slow"
    )


def test_sender_per_receiver_concurrency(async_sender):
    """Different-receiver requests run concurrently; same-receiver serialized."""
    SLOW, FAST = 0.25, 0.05

    recv1_id = "127.0.0.1" + str(9100)
    recv2_id = "127.0.0.1" + str(9200)

    sock2 = MagicMock()
    sock2.recv_multipart = AsyncMock(
        return_value=[
            b"",
            msgspec.msgpack.encode(AsyncAllocResponse(remote_indexes=[0])),
        ]
    )
    sock2.send_multipart = AsyncMock()
    async_sender.initialized_peers.add(recv2_id)
    async_sender._async_alloc_sockets[recv2_id] = sock2

    delays = {recv1_id: SLOW, recv2_id: FAST}
    orig_transfer = async_sender._async_transfer_task

    async def patched_transfer(**kw):
        rid = kw.get("receiver_id", "")
        n = len(kw.get("memory_objs", []))
        await asyncio.sleep(delays.get(rid, FAST))
        cb = kw.get("on_complete_callback")
        for key in kw.get("keys", []):
            if cb:
                try:
                    cb(key)
                except Exception:
                    pass
        async_sender._release_sender_staging_chunks(n)

    async_sender._async_transfer_task = patched_transfer

    events = {"A": threading.Event(), "B": threading.Event(), "C": threading.Event()}
    times = {}
    lock = threading.Lock()

    def cb(name):
        def f(key):
            with lock:
                times[name] = time.monotonic()
            events[name].set()

        return f

    async_sender.batched_submit_put_task(
        [_make_key(10)],
        [_make_mem_obj(10)],
        transfer_spec=_make_transfer_spec(
            init_port=9100, alloc_port=9101, req_id="A", is_last_prefill=True
        ),
        on_complete_callback=cb("A"),
    )
    async_sender.batched_submit_put_task(
        [_make_key(20)],
        [_make_mem_obj(20)],
        transfer_spec=_make_transfer_spec(
            init_port=9200, alloc_port=9201, req_id="B", is_last_prefill=True
        ),
        on_complete_callback=cb("B"),
    )
    async_sender.batched_submit_put_task(
        [_make_key(30)],
        [_make_mem_obj(30)],
        transfer_spec=_make_transfer_spec(
            init_port=9100, alloc_port=9101, req_id="C", is_last_prefill=True
        ),
        on_complete_callback=cb("C"),
    )

    for name, ev in events.items():
        assert ev.wait(timeout=SLOW * 6), f"{name} timed out"

    assert times["B"] < times["A"], "B (fast recv2) should finish before A (slow recv1)"
    assert times["C"] >= times["A"], "C should finish after A (same recv1, FIFO)"

    async_sender._async_transfer_task = orig_transfer


# ── receiver tests ────────────────────────────────────────────────────────


def test_receiver_nonblocking_async_sleep(async_receiver):
    """Busy-wait retries yield via asyncio.sleep, not time.sleep."""
    RETRY_COUNT = 5
    TOKS_A, TOKS_B = 16, 8

    key_a, key_b = _make_key(100), _make_key(200)
    obj_a, obj_b = _make_mem_obj(idx=10), _make_mem_obj(idx=20)

    finish_order = []
    orig_put = async_receiver.put

    def tracked_put(key, mem_obj):
        if key == key_a:
            finish_order.append("a")
        elif key == key_b:
            finish_order.append("b")
        return orig_put(key, mem_obj)

    async_receiver.put = tracked_put

    calls = {}

    def patched_alloc(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        tok_dim = MemoryFormat.KV_2LTD.token_dim()
        toks = shapes[tok_dim] if isinstance(shapes, torch.Size) else shapes[tok_dim]
        calls[toks] = calls.get(toks, 0) + 1
        if toks == TOKS_A and calls[toks] <= RETRY_COUNT:
            return None
        return obj_a if toks == TOKS_A else obj_b

    async_receiver.allocate = patched_alloc

    req_a = _make_alloc_req(
        [key_a], last_chunk_toks=TOKS_A, shape=[4, 2, TOKS_A, 8, 128]
    )
    req_b = _make_alloc_req(
        [key_b], last_chunk_toks=TOKS_B, shape=[4, 2, TOKS_B, 8, 128]
    )

    async def _run():
        await asyncio.gather(
            async_receiver._async_allocate_and_put(req_a),
            async_receiver._async_allocate_and_put(req_b),
        )

    asyncio.run(_run())

    assert finish_order == ["b", "a"], (
        f"Got {finish_order}, busy-wait likely uses time.sleep instead of asyncio.sleep"
    )


def test_receiver_flow_control_inflight(async_receiver):
    """Allocation blocks when inflight is saturated, unblocks on notify."""
    mem_obj = _make_mem_obj(idx=60)
    async_receiver.allocate = (
        lambda shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw: mem_obj
    )

    alloc_req = _make_alloc_req([_make_key(600)])

    async def run():
        async with async_receiver._inflight_condition:
            async_receiver._inflight_chunks = async_receiver._max_inflight_chunks

        completed = asyncio.Event()
        holder = []

        async def do_alloc():
            holder.append(await async_receiver._async_allocate_and_put(alloc_req))
            completed.set()

        async def free_later():
            await asyncio.sleep(0.05)
            assert not completed.is_set(), "should still be blocked"
            async with async_receiver._inflight_condition:
                async_receiver._inflight_chunks -= 1
                async_receiver._inflight_condition.notify_all()

        await asyncio.gather(do_alloc(), free_later())
        assert holder[0].remote_indexes == [mem_obj.meta.address]

    asyncio.run(run())


def test_receiver_last_chunk_shape_override(async_receiver):
    """Last chunk's token dim is overridden to last_chunk_toks."""
    mem_obj = _make_mem_obj(idx=30)
    LAST_TOKS = 7

    shapes_seen = []

    def tracking_alloc(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        shapes_seen.append(shapes)
        return mem_obj

    async_receiver.allocate = tracking_alloc

    keys = [_make_key(300), _make_key(301), _make_key(302)]
    asyncio.run(
        async_receiver._async_allocate_and_put(
            _make_alloc_req(keys, last_chunk_toks=LAST_TOKS)
        )
    )

    assert len(shapes_seen) == 3
    tok_dim = MemoryFormat.KV_2LTD.token_dim()
    assert shapes_seen[-1][tok_dim] == LAST_TOKS


@pytest.mark.parametrize(
    "max_t, b1_n, b2_n, b1_off, b2_off",
    [(5, 5, 1, 0, 5), (4, 3, 2, 5000, 6000)],
    ids=["exact-then-overflow", "partial-then-overflow"],
)
def test_receiver_fail_fast_overflow(async_receiver, max_t, b1_n, b2_n, b1_off, b2_off):
    """Cumulative chunks > max_inflight → RuntimeError; prior batch kept."""
    async_receiver._max_inflight_chunks = max_t
    async_receiver.allocate = _auto_alloc()
    req_id = "req-failfast"

    b1_keys = [_make_key(b1_off + i) for i in range(b1_n)]

    async def run():
        r1 = await async_receiver._async_allocate_and_put(
            _make_alloc_req(b1_keys, req_id=req_id)
        )
        assert -1 not in r1.remote_indexes and len(r1.remote_indexes) == b1_n

        with pytest.raises(RuntimeError, match="max_inflight_chunks"):
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(b2_off + i) for i in range(b2_n)], req_id=req_id
                )
            )

    asyncio.run(run())

    for k in b1_keys:
        assert async_receiver.contains(k, pin=False)

    # unrelated req_id should work fine
    async_receiver._inflight_chunks = 0

    async def check_other():
        r = await async_receiver._async_allocate_and_put(
            _make_alloc_req([_make_key(20000)], req_id="req-other")
        )
        assert -1 not in r.remote_indexes

    asyncio.run(check_other())


def test_receiver_alloc_timeout(async_receiver):
    """allocate() returning None past deadline → RuntimeError; prior batch kept."""
    async_receiver._max_inflight_chunks = 10
    req_id = "req-timeout"
    async_receiver.allocate = _auto_alloc()

    b1_keys = [_make_key(1000 + i) for i in range(3)]
    r1 = asyncio.run(
        async_receiver._async_allocate_and_put(_make_alloc_req(b1_keys, req_id=req_id))
    )
    assert -1 not in r1.remote_indexes

    # second batch: first key ok, rest always None
    n = [0]

    def fail_after_first(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        n[0] += 1
        return _make_mem_obj(idx=999) if n[0] == 1 else None

    async_receiver.allocate = fail_after_first
    async_receiver._allocation_timeout = 0.05

    async def run():
        with pytest.raises(RuntimeError, match="timeout"):
            await async_receiver._async_allocate_and_put(
                _make_alloc_req([_make_key(2000 + i) for i in range(3)], req_id=req_id)
            )

    asyncio.run(run())

    for k in b1_keys:
        assert async_receiver.contains(k, pin=False)


def test_receiver_is_last_batch_cleanup(async_receiver):
    """is_last_batch=True removes req_id from _req_allocated_keys."""
    async_receiver._max_inflight_chunks = 10
    req_id = "req-lifecycle"
    async_receiver.allocate = _auto_alloc()

    asyncio.run(
        async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(3000 + i) for i in range(3)],
                req_id=req_id,
                is_last_batch=False,
            )
        )
    )
    assert req_id in async_receiver._req_allocated_keys
    assert len(async_receiver._req_allocated_keys[req_id]) == 3

    asyncio.run(
        async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(4000 + i) for i in range(2)],
                req_id=req_id,
                is_last_batch=True,
            )
        )
    )
    assert req_id not in async_receiver._req_allocated_keys


def test_receiver_admission_control(async_receiver):
    """Only one req_id allocates at a time; second waits for first to finish."""
    async_receiver._max_inflight_chunks = 20
    async_receiver.allocate = _auto_alloc()

    log = []

    async def run():
        log.append("A1-start")
        await async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(7000 + i) for i in range(2)],
                req_id="req-A",
                is_last_batch=False,
            )
        )
        log.append("A1-done")

        async def do_b():
            log.append("B-start")
            await async_receiver._async_allocate_and_put(
                _make_alloc_req([_make_key(8000)], req_id="req-B", is_last_batch=True)
            )
            log.append("B-done")

        async def do_a2():
            await asyncio.sleep(0.02)
            log.append("A2-start")
            await async_receiver._async_allocate_and_put(
                _make_alloc_req([_make_key(9000)], req_id="req-A", is_last_batch=True)
            )
            log.append("A2-done")

        await asyncio.gather(do_b(), do_a2())

    asyncio.run(run())
    assert log.index("A2-done") < log.index("B-done")


def test_receiver_error_response(async_receiver):
    """_handle_alloc_request sends error AllocResponse when allocation fails."""
    payload = msgspec.msgpack.encode(_make_alloc_req([_make_key(0)], req_id="req-err"))
    identity = b"fake-sender"

    frames_sent = []
    sock = MagicMock()
    sock.send_multipart = AsyncMock(side_effect=lambda f: frames_sent.append(f))

    orig = async_receiver._async_allocate_and_put

    async def failing(req):
        raise RuntimeError("boom")

    async_receiver._async_allocate_and_put = failing
    asyncio.run(async_receiver._handle_alloc_request(sock, identity, payload))
    async_receiver._async_allocate_and_put = orig

    assert len(frames_sent) == 1
    f = frames_sent[0]
    assert f[0] == identity and f[1] == b""
    resp = msgspec.msgpack.decode(f[2], type=AsyncPDMsg)
    assert isinstance(resp, AsyncAllocResponse) and resp.remote_indexes == [-1]


# ── close() ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role", ["sender", "receiver"])
def test_close_stops_thread(role, async_sender, async_receiver):
    """close() stops the background event-loop thread."""
    backend = async_sender if role == "sender" else async_receiver
    attr = "_sender_thread" if role == "sender" else "_recv_thread"
    assert getattr(backend, attr).is_alive()
    backend.close()
    assert not getattr(backend, attr).is_alive()
    backend.running = False


# ── pd_max_prefill_len init check ────────────────────────────────────────


def test_pd_max_prefill_len_check():
    """pd_max_prefill_len > buffer capacity → ValueError on init."""
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

    def recv_cfg(max_len):
        return LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="receiver",
            pd_peer_host="127.0.0.1",
            pd_peer_init_port=[9200],
            pd_peer_alloc_port=[9201],
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
            pd_max_prefill_len=max_len,
        )

    def send_cfg(max_len):
        return LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="sender",
            pd_proxy_host="127.0.0.1",
            pd_proxy_port=5555,
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
            pd_max_prefill_len=max_len,
        )

    meta = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 16, 8, 128),
    )

    with _patched_pd():
        with pytest.raises(ValueError, match="pd_max_prefill_len"):
            PDBackendAsync(recv_cfg(5000), meta)

    with _patched_pd():
        with pytest.raises(ValueError, match="pd_max_prefill_len"):
            PDBackendAsync(send_cfg(5000), meta)

    with _patched_pd():
        PDBackendAsync(recv_cfg(4096), meta).close()  # boundary ok

    with _patched_pd():
        PDBackendAsync(recv_cfg(0), meta).close()  # 0 skips check


# ── wire-format compatibility (sync ↔ async) ─────────────────────────────


def test_sync_request_decoded_as_async():
    req = SyncAllocRequest(
        keys=["k0", "k1"],
        fmt=0,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=7,
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=AsyncPDMsg)
    assert isinstance(decoded, AsyncAllocRequest)
    assert (
        decoded.keys == ["k0", "k1"]
        and decoded.req_id == ""
        and not decoded.is_last_batch
    )


def test_async_response_decoded_as_sync():
    resp = AsyncAllocResponse(remote_indexes=[100, 200])
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=SyncPDMsg)
    assert isinstance(decoded, SyncAllocResponse)
    assert decoded.already_sent_indexes == [] and decoded.remote_indexes == [100, 200]


def test_sync_response_decoded_as_async():
    resp = SyncAllocResponse(already_sent_indexes=[0], remote_indexes=[100])
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=AsyncPDMsg)
    assert isinstance(decoded, AsyncAllocResponse)
    assert decoded.already_sent_indexes == [0] and decoded.remote_indexes == [100]
