# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the nixl-backed transfer channel implementation.

Tests are written against the public contracts documented in
``transfer_channel/abstract.py`` and ``common/transfer_channel/api.py`` (the
``TransferChannelContext`` / ``TransferChannelServer`` / ``TransferChannelClient``
interfaces). They use only public methods and do not access private fields.

These tests require a working nixl runtime (with a UCX backend) and are skipped
when nixl is unavailable.

Most tests share a single module-scoped context-pair to avoid paying the (slow)
nixl agent initialization for every test. The reads are self-verifying -- each
reads fresh remote data into its target region and checks that region -- so a
reused buffer and connection do not affect their outcome. The one test that must
observe a pristine connection state (the 0 -> 1 connection-count transition)
uses its own function-scoped pair instead, since there is no public way to reset
an existing context's connected clients short of closing it.
"""

# Standard
from unittest.mock import MagicMock
import itertools
import threading
import time

# Third Party
import msgspec
import pytest
import torch
import zmq

nixl = pytest.importorskip("nixl")

# First Party
from lmcache import torch_dev, torch_device_type  # noqa: E402
from lmcache.v1.common.transfer_channel.api import TransferChannelAddress  # noqa: E402
from lmcache.v1.distributed.internal_api import L1MemoryDesc  # noqa: E402
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (  # noqa: E402
    NixlTransferChannelContext,
)

_ALIGN = 256
_BUF_SIZE = 4096

# Each context must bind to a distinct port so multiple pairs can coexist in one
# process without clashing.
_port_counter = itertools.count(17900)


def _next_url() -> str:
    return f"127.0.0.1:{next(_port_counter)}"


def _make_context_pair():
    """Create two contexts backed by two distinct CPU buffers.

    ``buf_b`` holds a known byte pattern so transfers can be content-verified;
    ``buf_a`` starts zeroed. Returns ``(ctx_a, buf_a, ctx_b, buf_b)``.
    """
    buf_a = torch.zeros(_BUF_SIZE, dtype=torch.uint8)
    buf_b = torch.arange(0, 256, dtype=torch.uint8).repeat(_BUF_SIZE // 256)
    desc_a = L1MemoryDesc(ptr=buf_a.data_ptr(), size=buf_a.numel(), align_bytes=_ALIGN)
    desc_b = L1MemoryDesc(ptr=buf_b.data_ptr(), size=buf_b.numel(), align_bytes=_ALIGN)

    url_a, url_b = _next_url(), _next_url()
    ctx_a = NixlTransferChannelContext(desc_a, listen_url=url_a, advertise_url=url_a)
    ctx_b = NixlTransferChannelContext(desc_b, listen_url=url_b, advertise_url=url_b)
    return ctx_a, buf_a, ctx_b, buf_b


@pytest.fixture(scope="module")
def shared_contexts():
    """A single context-pair reused across connection-state-insensitive tests."""
    ctx_a, buf_a, ctx_b, buf_b = _make_context_pair()
    try:
        yield ctx_a, buf_a, ctx_b, buf_b
    finally:
        ctx_a.close()
        ctx_b.close()


@pytest.fixture
def fresh_contexts():
    """A brand-new context-pair for tests that need pristine connection state."""
    ctx_a, buf_a, ctx_b, buf_b = _make_context_pair()
    try:
        yield ctx_a, buf_a, ctx_b, buf_b
    finally:
        ctx_a.close()
        ctx_b.close()


def _wait_finished(client, task_id, timeout_s=5.0):
    """Poll query_status until the task reports finished (or timeout)."""
    deadline = time.monotonic() + timeout_s
    result = client.query_status(task_id)
    while not result.is_finished() and time.monotonic() < deadline:
        time.sleep(0.01)
        result = client.query_status(task_id)
    return result


# =========================================================
# Address translation (get_transfer_channel_address)
# =========================================================
def test_get_address_returns_matching_offsets_and_sizes(shared_contexts):
    ctx_a, _, _, _ = shared_contexts
    addrs = ctx_a.get_transfer_channel_address([(0, 512), (1024, 256)])
    assert len(addrs) == 2
    assert (addrs[0].offset, addrs[0].size) == (0, 512)
    assert (addrs[1].offset, addrs[1].size) == (1024, 256)


def test_get_address_out_of_region_raises_value_error(shared_contexts):
    ctx_a, _, _, _ = shared_contexts
    with pytest.raises(ValueError):
        ctx_a.get_transfer_channel_address([(_BUF_SIZE, 256)])
    with pytest.raises(ValueError):
        ctx_a.get_transfer_channel_address([(_BUF_SIZE - 128, 256)])


# =========================================================
# End-to-end read (submit_read / query_status)
# =========================================================
def test_read_copies_remote_data_into_local_buffer(shared_contexts):
    ctx_a, buf_a, ctx_b, buf_b = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    local = ctx_a.get_transfer_channel_address([(0, 512)])
    remote = ctx_b.get_transfer_channel_address([(0, 512)])
    task_id = client.submit_read(local, remote)

    result = _wait_finished(client, task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True] * len(remote)
    assert torch.equal(buf_a[:512], buf_b[:512])


def test_read_into_offset_region(shared_contexts):
    ctx_a, buf_a, ctx_b, buf_b = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    # Read remote [0, 256) into local [1024, 1280).
    local = ctx_a.get_transfer_channel_address([(1024, 256)])
    remote = ctx_b.get_transfer_channel_address([(0, 256)])
    task_id = client.submit_read(local, remote)

    result = _wait_finished(client, task_id)
    assert result.is_finished() is True
    assert torch.equal(buf_a[1024:1280], buf_b[0:256])


def test_read_many_objects_lands_each_at_its_offset(shared_contexts):
    ctx_a, buf_a, ctx_b, buf_b = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    # Scatter remote [0, 1024) in four 256-byte objects, reversed, into local
    # [2048, 3072).
    remote = ctx_b.get_transfer_channel_address([(i * 256, 256) for i in range(4)])
    local = ctx_a.get_transfer_channel_address(
        [(2048 + (3 - i) * 256, 256) for i in range(4)]
    )
    task_id = client.submit_read(local, remote)

    result = _wait_finished(client, task_id)
    assert result.succeeded_mask == [True] * 4
    for i in range(4):
        dst = 2048 + (3 - i) * 256
        assert torch.equal(buf_a[dst : dst + 256], buf_b[i * 256 : (i + 1) * 256])


def test_peers_with_different_alignments_can_read():
    """The reader and the peer may register L1 with different alignments."""
    buf_a = torch.zeros(_BUF_SIZE, dtype=torch.uint8)
    buf_b = torch.arange(0, 256, dtype=torch.uint8).repeat(_BUF_SIZE // 256)
    url_a, url_b = _next_url(), _next_url()
    ctx_a = NixlTransferChannelContext(
        L1MemoryDesc(ptr=buf_a.data_ptr(), size=_BUF_SIZE, align_bytes=_ALIGN),
        listen_url=url_a,
        advertise_url=url_a,
    )
    ctx_b = NixlTransferChannelContext(
        L1MemoryDesc(ptr=buf_b.data_ptr(), size=_BUF_SIZE, align_bytes=2 * _ALIGN),
        listen_url=url_b,
        advertise_url=url_b,
    )
    try:
        client = ctx_a.get_transfer_channel_client(url_b)
        local = ctx_a.get_transfer_channel_address([(0, _ALIGN)])
        remote = ctx_b.get_transfer_channel_address([(_ALIGN, _ALIGN)])
        result = _wait_finished(client, client.submit_read(local, remote))
        assert result.succeeded_mask == [True]
        assert torch.equal(buf_a[:_ALIGN], buf_b[_ALIGN : 2 * _ALIGN])
    finally:
        ctx_a.close()
        ctx_b.close()


@pytest.mark.parametrize("offset", [_BUF_SIZE, _BUF_SIZE - _ALIGN // 2])
def test_read_outside_the_peer_region_is_rejected(shared_contexts, offset):
    """A remote address not inside the peer's registered L1 fails at submit."""
    ctx_a, buf_a, ctx_b, _ = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    before = buf_a.clone()
    local = ctx_a.get_transfer_channel_address([(0, _ALIGN)])
    remote = [TransferChannelAddress(offset=offset, size=_ALIGN)]
    with pytest.raises(Exception, match="NIXL_ERR"):
        client.submit_read(local, remote)
    assert torch.equal(buf_a, before)


def test_read_moves_exactly_the_object_size(shared_contexts):
    """An object whose size is not a multiple of the alignment leaves the
    local bytes after it untouched."""
    ctx_a, buf_a, ctx_b, buf_b = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    buf_a[3072:3584].fill_(0xEE)

    local = ctx_a.get_transfer_channel_address([(3072, 100)])
    remote = ctx_b.get_transfer_channel_address([(0, 100)])
    result = _wait_finished(client, client.submit_read(local, remote))

    assert result.succeeded_mask == [True]
    assert torch.equal(buf_a[3072:3172], buf_b[0:100])
    assert bool((buf_a[3172:3584] == 0xEE).all())


def test_submit_read_mismatched_lengths_raises_value_error(shared_contexts):
    ctx_a, _, ctx_b, _ = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    local = ctx_a.get_transfer_channel_address([(0, 256)])
    remote = ctx_b.get_transfer_channel_address([(0, 256), (256, 256)])
    with pytest.raises(ValueError):
        client.submit_read(local, remote)


def test_query_unknown_task_id_raises_key_error(shared_contexts):
    ctx_a, _, ctx_b, _ = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    with pytest.raises(KeyError):
        client.query_status(99999)


# =========================================================
# End-to-end write (submit_write / query_status)
# =========================================================
def test_write_copies_local_data_into_remote_buffer(fresh_contexts):
    ctx_a, buf_a, ctx_b, buf_b = fresh_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    buf_a[:512] = 0xAB

    # Write local [0, 512) into remote [0, 512).
    local = ctx_a.get_transfer_channel_address([(0, 512)])
    remote = ctx_b.get_transfer_channel_address([(0, 512)])
    task_id = client.submit_write(local, remote)

    result = _wait_finished(client, task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True] * len(remote)
    assert torch.all(buf_b[:512] == 0xAB)
    # Bytes outside the written region keep the peer's pattern.
    assert torch.equal(buf_b[512:768], torch.arange(0, 256, dtype=torch.uint8))


def test_write_into_offset_region(fresh_contexts):
    ctx_a, buf_a, ctx_b, buf_b = fresh_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    buf_a[0:256] = 0x5C

    # Write local [0, 256) into remote [1024, 1280).
    local = ctx_a.get_transfer_channel_address([(0, 256)])
    remote = ctx_b.get_transfer_channel_address([(1024, 256)])
    task_id = client.submit_write(local, remote)

    result = _wait_finished(client, task_id)
    assert result.is_finished() is True
    assert torch.all(buf_b[1024:1280] == 0x5C)
    assert torch.equal(buf_b[1280:1536], torch.arange(0, 256, dtype=torch.uint8))


def test_write_then_read_back_round_trips(fresh_contexts):
    """Bytes written into the peer can be read back through the same client."""
    ctx_a, buf_a, ctx_b, _ = fresh_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    buf_a[0:256] = torch.arange(255, -1, -1, dtype=torch.uint8)

    local_src = ctx_a.get_transfer_channel_address([(0, 256)])
    remote = ctx_b.get_transfer_channel_address([(2048, 256)])
    write_id = client.submit_write(local_src, remote)
    assert _wait_finished(client, write_id).succeeded_mask == [True]

    local_dst = ctx_a.get_transfer_channel_address([(2048, 256)])
    read_id = client.submit_read(local_dst, remote)
    assert _wait_finished(client, read_id).succeeded_mask == [True]
    # Read and write task ids come from one counter.
    assert read_id != write_id
    assert torch.equal(buf_a[2048:2304], buf_a[0:256])


def test_write_multiple_objects_reports_one_flag_each(fresh_contexts):
    ctx_a, buf_a, ctx_b, buf_b = fresh_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    buf_a[0:256] = 1
    buf_a[256:512] = 2

    local = ctx_a.get_transfer_channel_address([(0, 256), (256, 256)])
    remote = ctx_b.get_transfer_channel_address([(512, 256), (0, 256)])
    task_id = client.submit_write(local, remote)

    result = _wait_finished(client, task_id)
    assert result.succeeded_mask == [True, True]
    assert torch.all(buf_b[512:768] == 1)
    assert torch.all(buf_b[0:256] == 2)


def test_submit_write_mismatched_lengths_raises_value_error(shared_contexts):
    ctx_a, _, ctx_b, _ = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    local = ctx_a.get_transfer_channel_address([(0, 256)])
    remote = ctx_b.get_transfer_channel_address([(0, 256), (256, 256)])
    with pytest.raises(ValueError):
        client.submit_write(local, remote)


def test_finished_write_cannot_be_queried_again(fresh_contexts):
    ctx_a, _, ctx_b, _ = fresh_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    local = ctx_a.get_transfer_channel_address([(0, 256)])
    remote = ctx_b.get_transfer_channel_address([(0, 256)])
    task_id = client.submit_write(local, remote)
    assert _wait_finished(client, task_id).is_finished() is True
    with pytest.raises(KeyError):
        client.query_status(task_id)


# =========================================================
# Client / server management
# =========================================================
def test_get_client_is_idempotent(shared_contexts):
    ctx_a, _, ctx_b, _ = shared_contexts
    first = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    second = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    assert first is second


def test_register_client_keeps_first_and_closes_duplicate(fresh_contexts):
    """A racing second registration keeps the first client and closes the dup."""
    ctx_a, _, _, _ = fresh_contexts
    key = "10.255.255.1:65000"
    first = MagicMock()
    second = MagicMock()

    assert ctx_a.register_client(key, first) is first
    # The peer's inbound connection registers a duplicate for the same key.
    assert ctx_a.register_client(key, second) is first
    second.close.assert_called_once()
    first.close.assert_not_called()
    assert ctx_a.get_num_connected_clients() == 1


def test_remove_client_closes_and_drops_it(fresh_contexts):
    """Removing a peer closes its client and drops it from the context."""
    ctx_a, _, _, _ = fresh_contexts
    key = "10.255.255.1:65000"
    client = MagicMock()
    ctx_a.register_client(key, client)
    assert ctx_a.get_num_connected_clients() == 1

    ctx_a.remove_transfer_channel_client(key)

    client.close.assert_called_once()
    assert ctx_a.get_num_connected_clients() == 0


def test_remove_unknown_client_is_noop(fresh_contexts):
    """Removing a peer with no registered client does nothing and does not raise."""
    ctx_a, _, _, _ = fresh_contexts
    ctx_a.remove_transfer_channel_client("10.255.255.1:65000")
    assert ctx_a.get_num_connected_clients() == 0


def test_remove_client_after_connect(fresh_contexts):
    """A live client obtained via get_transfer_channel_client can be removed."""
    ctx_a, _, ctx_b, _ = fresh_contexts
    ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    assert ctx_a.get_num_connected_clients() == 1

    ctx_a.remove_transfer_channel_client(ctx_b.advertise_url)
    assert ctx_a.get_num_connected_clients() == 0


def test_connecting_registers_clients_on_both_sides(fresh_contexts):
    ctx_a, _, ctx_b, _ = fresh_contexts
    assert ctx_a.get_num_connected_clients() == 0
    assert ctx_b.get_num_connected_clients() == 0

    ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    # A actively dialed B; B passively learned a reverse client for A.
    assert ctx_a.get_num_connected_clients() == 1
    assert ctx_b.get_num_connected_clients() == 1


def test_submits_succeed_while_new_peers_handshake(fresh_contexts):
    """Submits on several threads keep succeeding while peers dial the reader.

    Each submit looks up the remote agent while the reader's handshake server
    adds the joining peers' agents.
    """
    ctx_a, _, ctx_b, _ = fresh_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    half = _BUF_SIZE // 2
    local = ctx_a.get_transfer_channel_address(
        [(i, _ALIGN) for i in range(0, half, _ALIGN)]
    )
    remote = ctx_b.get_transfer_channel_address(
        [(half + i, _ALIGN) for i in range(0, half, _ALIGN)]
    )
    stop = threading.Event()
    errors: list[str] = []
    completed = [0, 0, 0]

    def submit_loop(slot: int, write: bool) -> None:
        submit = client.submit_write if write else client.submit_read
        while not stop.is_set():
            try:
                result = _wait_finished(client, submit(local, remote))
                if result.succeeded_mask != [True] * len(local):
                    errors.append(f"mask {result.succeeded_mask}")
                completed[slot] += 1
            except Exception as err:  # noqa: BLE001 - collected for the assert
                errors.append(repr(err))

    threads = [
        threading.Thread(target=submit_loop, args=(slot, write))
        for slot, write in enumerate([False, False, True])
    ]
    for thread in threads:
        thread.start()
    peers = []
    try:
        for _ in range(8):
            buf = torch.zeros(_BUF_SIZE, dtype=torch.uint8)
            url = _next_url()
            peer = NixlTransferChannelContext(
                L1MemoryDesc(ptr=buf.data_ptr(), size=_BUF_SIZE, align_bytes=_ALIGN),
                listen_url=url,
                advertise_url=url,
            )
            peers.append((peer, buf))
            peer.get_transfer_channel_client(ctx_a.advertise_url)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
        for peer, _ in peers:
            peer.close()

    assert errors == []
    assert all(count > 0 for count in completed)
    assert ctx_a.get_num_connected_clients() == 1 + len(peers)


def test_server_is_available_from_context(shared_contexts):
    ctx_a, _, _, _ = shared_contexts
    assert ctx_a.get_transfer_channel_server() is not None


# =========================================================
# Extra memory regions
# =========================================================
_REGION_KINDS = [
    pytest.param(None, "cpu", id="host"),
    pytest.param(
        0,
        "cuda",
        id="device",
        marks=pytest.mark.skipif(
            torch_device_type != "cuda", reason="needs a CUDA device"
        ),
    ),
]


@pytest.mark.parametrize("device_id, device", _REGION_KINDS)
def test_read_into_registered_region(fresh_contexts, device_id, device):
    ctx_a, buf_a, ctx_b, buf_b = fresh_contexts
    target = torch.zeros(_BUF_SIZE, dtype=torch.uint8, device=device)
    region = ctx_a.register_memory_region(target.data_ptr(), target.numel(), device_id)
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    local = ctx_a.get_transfer_channel_address([(512, 1024)], region=region)
    remote = ctx_b.get_transfer_channel_address([(256, 1024)])
    task_id = client.submit_read(local, remote, local_region=region)

    assert _wait_finished(client, task_id).succeeded_mask == [True]
    result = target.cpu()
    assert torch.equal(result[512:1536], buf_b[256:1280])
    assert not result[:512].any() and not result[1536:].any()
    assert not buf_a.any()

    ctx_a.deregister_memory_region(region)
    with pytest.raises(KeyError):
        ctx_a.get_transfer_channel_address([(0, 256)], region=region)


@pytest.mark.skipif(
    torch_device_type != "cuda" or torch_dev.device_count() < 2,
    reason="needs two CUDA devices",
)
def test_concurrent_reads_into_one_region_per_device(fresh_contexts):
    ctx_a, _, ctx_b, buf_b = fresh_contexts
    targets = [
        torch.zeros(_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{device_id}")
        for device_id in range(torch_dev.device_count())
    ]
    regions = [
        ctx_a.register_memory_region(t.data_ptr(), t.numel(), device_id)
        for device_id, t in enumerate(targets)
    ]
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    # Each device reads a different slice, so a read landing in the wrong
    # region shows up as wrong bytes.
    offsets = [_ALIGN * i for i in range(len(regions))]
    task_ids = [
        client.submit_read(
            ctx_a.get_transfer_channel_address([(offset, 512)], region=region),
            ctx_b.get_transfer_channel_address([(offset, 512)]),
            local_region=region,
        )
        for offset, region in zip(offsets, regions, strict=True)
    ]

    for task_id in task_ids:
        assert _wait_finished(client, task_id).succeeded_mask == [True]
    for offset, target in zip(offsets, targets, strict=True):
        result = target.cpu()
        assert torch.equal(result[offset : offset + 512], buf_b[offset : offset + 512])
        assert not result[:offset].any() and not result[offset + 512 :].any()


def test_region_size_must_be_multiple_of_alignment(shared_contexts):
    ctx_a, buf_a, _, _ = shared_contexts
    with pytest.raises(ValueError):
        ctx_a.register_memory_region(buf_a.data_ptr(), _ALIGN + 1)


# =========================================================
# Handshake protocol
# =========================================================


def _raw_request(url: str, msg: bytes) -> bytes:
    """Send one raw handshake request and return the raw reply."""
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"tcp://{url}")
    try:
        sock.send(msg)
        return sock.recv()
    finally:
        sock.close(linger=0)


@pytest.mark.parametrize(
    "request_bytes",
    [
        # An InitReq from a peer that predates the protocol version.
        msgspec.msgpack.encode(
            {"type": "InitReq", "agent_name": "a", "agent_meta": b""}
        ),
        b"not a handshake message",
    ],
    ids=["unversioned-peer", "garbage"],
)
def test_rejected_request_is_answered_and_later_handshakes_work(
    fresh_contexts, request_bytes
):
    """A request the server cannot handle gets an error reply, and the server
    keeps answering later handshakes."""
    ctx_a, _, ctx_b, _ = fresh_contexts

    reply = msgspec.msgpack.decode(_raw_request(ctx_b.advertise_url, request_bytes))
    assert reply["type"] == "ErrorResp"

    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    local = ctx_a.get_transfer_channel_address([(0, _ALIGN)])
    remote = ctx_b.get_transfer_channel_address([(0, _ALIGN)])
    assert _wait_finished(client, client.submit_read(local, remote)).succeeded_mask == [
        True
    ]


def test_dialing_a_peer_of_another_protocol_version_fails_fast(fresh_contexts):
    """A server answering with an unversioned InitResp is rejected at once."""
    ctx_a, _, _, _ = fresh_contexts
    url = _next_url()
    server = zmq.Context.instance().socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    server.bind(f"tcp://{url}")
    try:
        reply = msgspec.msgpack.encode(
            {"type": "InitResp", "agent_name": "old", "agent_meta": b""}
        )

        def answer_once() -> None:
            server.recv()
            server.send(reply)

        responder = threading.Thread(target=answer_once, daemon=True)
        responder.start()
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="protocol mismatch"):
            ctx_a.get_transfer_channel_client(url)
        assert time.monotonic() - start < 5.0
        responder.join(timeout=5.0)
    finally:
        server.close(linger=0)
