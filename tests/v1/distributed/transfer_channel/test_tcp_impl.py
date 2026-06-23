# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the TCP-backed transfer channel implementation.

Tests are written against the public contracts documented in
``transfer_channel/abstract.py`` and ``transfer_channel/api.py`` (the
``TransferChannelContext`` / ``TransferChannelServer`` / ``TransferChannelClient``
interfaces). They use only public methods and do not access private fields.

Unlike the nixl implementation, TCP does not require special hardware or
third-party libraries, so these tests run unconditionally.

Most tests share a single module-scoped context-pair to avoid paying the
connection setup cost for every test. The reads are self-verifying -- each
reads fresh remote data into its target region and checks that region -- so a
reused buffer and connection do not affect their outcome. Tests that must
observe a pristine connection state use their own function-scoped pair.
"""

# Standard
import itertools
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
)
from lmcache.v1.distributed.transfer_channel.impl.tcp_impl import (
    TcpTransferChannelContext,
)

_ALIGN = 256
_BUF_SIZE = 4096

# Each context must bind to a distinct port so multiple pairs can coexist in one
# process without clashing.
_port_counter = itertools.count(18900)


def _next_url() -> str:
    return f"127.0.0.1:{next(_port_counter)}"


def _make_context_pair():
    """Create two contexts backed by two distinct CPU buffers.

    ``buf_b`` holds a known byte pattern so transfers can be content-verified;
    ``buf_a`` starts zeroed. Returns ``(ctx_a, buf_a, url_a, ctx_b, buf_b, url_b)``.
    """
    buf_a = torch.zeros(_BUF_SIZE, dtype=torch.uint8)
    buf_b = torch.arange(0, 256, dtype=torch.uint8).repeat(_BUF_SIZE // 256)
    desc_a = L1MemoryDesc(ptr=buf_a.data_ptr(), size=buf_a.numel(), align_bytes=_ALIGN)
    desc_b = L1MemoryDesc(ptr=buf_b.data_ptr(), size=buf_b.numel(), align_bytes=_ALIGN)

    url_a, url_b = _next_url(), _next_url()
    ctx_a = TcpTransferChannelContext(desc_a, listen_url=url_a, advertise_url=url_a)
    ctx_b = TcpTransferChannelContext(desc_b, listen_url=url_b, advertise_url=url_b)
    return ctx_a, buf_a, url_a, ctx_b, buf_b, url_b


@pytest.fixture(scope="module")
def shared_contexts():
    """A single context-pair reused across connection-state-insensitive tests."""
    ctx_a, buf_a, url_a, ctx_b, buf_b, url_b = _make_context_pair()
    try:
        yield ctx_a, buf_a, url_a, ctx_b, buf_b, url_b
    finally:
        ctx_a.close()
        ctx_b.close()


@pytest.fixture
def fresh_contexts():
    """A brand-new context-pair for tests that need pristine connection state."""
    ctx_a, buf_a, url_a, ctx_b, buf_b, url_b = _make_context_pair()
    try:
        yield ctx_a, buf_a, url_a, ctx_b, buf_b, url_b
    finally:
        ctx_a.close()
        ctx_b.close()


def _wait_finished(client, task_id, timeout_s=5.0):
    """Poll query_read_status until the task reports finished (or timeout)."""
    deadline = time.monotonic() + timeout_s
    result = client.query_read_status(task_id)
    while not result.is_finished() and time.monotonic() < deadline:
        time.sleep(0.01)
        result = client.query_read_status(task_id)
    return result


# =========================================================
# Address translation (get_transfer_channel_address)
# =========================================================
class TestAddressTranslation:
    """Tests for get_transfer_channel_address."""

    def test_returns_matching_offsets_and_sizes(self, shared_contexts):
        ctx_a, _, _, _, _, _ = shared_contexts
        addrs = ctx_a.get_transfer_channel_address([(0, 512), (1024, 256)])
        assert len(addrs) == 2
        assert (addrs[0].offset, addrs[0].size) == (0, 512)
        assert (addrs[1].offset, addrs[1].size) == (1024, 256)

    def test_out_of_region_raises_value_error(self, shared_contexts):
        ctx_a, _, _, _, _, _ = shared_contexts
        with pytest.raises(ValueError):
            ctx_a.get_transfer_channel_address([(_BUF_SIZE, 256)])
        with pytest.raises(ValueError):
            ctx_a.get_transfer_channel_address([(_BUF_SIZE - 128, 256)])

    def test_negative_offset_raises_value_error(self, shared_contexts):
        ctx_a, _, _, _, _, _ = shared_contexts
        with pytest.raises(ValueError):
            ctx_a.get_transfer_channel_address([(-1, 256)])

    def test_zero_size_at_valid_offset(self, shared_contexts):
        """Zero-size objects should still be translatable (edge case)."""
        ctx_a, _, _, _, _, _ = shared_contexts
        # This depends on implementation; TCP passes through directly
        # A zero-size read is technically valid (no data to transfer)
        addrs = ctx_a.get_transfer_channel_address([(0, 0)])
        assert len(addrs) == 1
        assert addrs[0].offset == 0
        assert addrs[0].size == 0

    def test_multiple_addresses_at_boundary(self, shared_contexts):
        """Addresses exactly at the end of the region should be valid."""
        ctx_a, _, _, _, _, _ = shared_contexts
        addrs = ctx_a.get_transfer_channel_address([(_BUF_SIZE - 256, 256)])
        assert len(addrs) == 1
        assert addrs[0].offset == _BUF_SIZE - 256
        assert addrs[0].size == 256


# =========================================================
# End-to-end read (submit_read / query_read_status)
# =========================================================
class TestEndToEndRead:
    """Tests for the full read path: submit_read + query_read_status."""

    def test_read_copies_remote_data_into_local_buffer(self, shared_contexts):
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)

        local = ctx_a.get_transfer_channel_address([(0, 512)])
        remote = ctx_b.get_transfer_channel_address([(0, 512)])
        task_id = client.submit_read(local, remote)

        result = _wait_finished(client, task_id)
        assert result.is_finished() is True
        assert result.succeeded_mask == [True] * len(remote)
        assert torch.equal(buf_a[:512], buf_b[:512])

    def test_read_into_offset_region(self, shared_contexts):
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)

        # Read remote [0, 256) into local [1024, 1280).
        local = ctx_a.get_transfer_channel_address([(1024, 256)])
        remote = ctx_b.get_transfer_channel_address([(0, 256)])
        task_id = client.submit_read(local, remote)

        result = _wait_finished(client, task_id)
        assert result.is_finished() is True
        assert torch.equal(buf_a[1024:1280], buf_b[0:256])

    def test_read_multiple_entries_in_one_batch(self, shared_contexts):
        """A single submit_read with multiple entries should transfer all."""
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)

        # Read two disjoint regions in one batch
        local = ctx_a.get_transfer_channel_address([(2048, 256), (2304, 256)])
        remote = ctx_b.get_transfer_channel_address([(256, 256), (512, 256)])
        task_id = client.submit_read(local, remote)

        result = _wait_finished(client, task_id)
        assert result.is_finished() is True
        assert result.succeeded_mask == [True, True]
        assert torch.equal(buf_a[2048:2304], buf_b[256:512])
        assert torch.equal(buf_a[2304:2560], buf_b[512:768])

    def test_read_full_buffer(self, shared_contexts):
        """Reading the entire buffer should work."""
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)

        local = ctx_a.get_transfer_channel_address([(0, _BUF_SIZE)])
        remote = ctx_b.get_transfer_channel_address([(0, _BUF_SIZE)])
        task_id = client.submit_read(local, remote)

        result = _wait_finished(client, task_id)
        assert result.is_finished() is True
        assert result.succeeded_mask == [True]
        assert torch.equal(buf_a[:_BUF_SIZE], buf_b[:_BUF_SIZE])

    def test_multiple_sequential_reads(self, shared_contexts):
        """Multiple sequential reads on the same client should all succeed."""
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)

        for i in range(5):
            offset = i * 256
            local = ctx_a.get_transfer_channel_address([(offset, 256)])
            remote = ctx_b.get_transfer_channel_address([(offset, 256)])
            task_id = client.submit_read(local, remote)

            result = _wait_finished(client, task_id)
            assert result.is_finished() is True
            assert result.succeeded_mask == [True]
            assert torch.equal(
                buf_a[offset : offset + 256], buf_b[offset : offset + 256]
            )

    def test_submit_read_mismatched_lengths_raises_value_error(self, shared_contexts):
        ctx_a, _, _, ctx_b, _, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)
        local = ctx_a.get_transfer_channel_address([(0, 256)])
        remote = ctx_b.get_transfer_channel_address([(0, 256), (256, 256)])
        with pytest.raises(ValueError):
            client.submit_read(local, remote)

    def test_query_unknown_task_id_raises_key_error(self, shared_contexts):
        ctx_a, _, _, _, _, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)
        with pytest.raises(KeyError):
            client.query_read_status(99999)


# =========================================================
# Client / server management
# =========================================================
class TestClientServerManagement:
    """Tests for client creation, removal, and server access."""

    def test_get_client_is_idempotent(self, shared_contexts):
        ctx_a, _, _, _, _, url_b = shared_contexts
        first = ctx_a.get_transfer_channel_client(url_b)
        second = ctx_a.get_transfer_channel_client(url_b)
        assert first is second

    def test_remove_unknown_client_is_noop(self, fresh_contexts):
        """Removing a peer with no registered client does nothing and does not raise."""
        ctx_a, _, _, _, _, _ = fresh_contexts
        ctx_a.remove_transfer_channel_client("10.255.255.1:65000")
        assert ctx_a.get_num_connected_clients() == 0

    def test_remove_client_after_connect(self, fresh_contexts):
        """A live client obtained via get_transfer_channel_client can be removed."""
        ctx_a, _, _, _, _, url_b = fresh_contexts
        ctx_a.get_transfer_channel_client(url_b)
        assert ctx_a.get_num_connected_clients() == 1

        ctx_a.remove_transfer_channel_client(url_b)
        assert ctx_a.get_num_connected_clients() == 0

    def test_reconnect_after_remove(self, fresh_contexts):
        """After removing a client, a new one can be created for the same peer."""
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = fresh_contexts
        ctx_a.get_transfer_channel_client(url_b)
        ctx_a.remove_transfer_channel_client(url_b)
        assert ctx_a.get_num_connected_clients() == 0

        # Reconnect and verify data transfer still works
        client = ctx_a.get_transfer_channel_client(url_b)
        assert ctx_a.get_num_connected_clients() == 1

        local = ctx_a.get_transfer_channel_address([(0, 256)])
        remote = ctx_b.get_transfer_channel_address([(0, 256)])
        task_id = client.submit_read(local, remote)

        result = _wait_finished(client, task_id)
        assert result.is_finished() is True
        assert result.succeeded_mask == [True]
        assert torch.equal(buf_a[:256], buf_b[:256])

    def test_connecting_does_not_register_reverse_client(self, fresh_contexts):
        """TCP connections are unidirectional: A connecting to B does not give B
        a client to A (unlike nixl's passive client creation)."""
        ctx_a, _, _, ctx_b, _, url_b = fresh_contexts
        assert ctx_a.get_num_connected_clients() == 0
        assert ctx_b.get_num_connected_clients() == 0

        ctx_a.get_transfer_channel_client(url_b)

        # A has a client to B, but B does NOT have a reverse client to A
        assert ctx_a.get_num_connected_clients() == 1
        assert ctx_b.get_num_connected_clients() == 0

    def test_server_is_available_from_context(self, shared_contexts):
        ctx_a, _, _, _, _, _ = shared_contexts
        assert ctx_a.get_transfer_channel_server() is not None

    def test_close_cleans_up_all_clients(self, fresh_contexts):
        """Closing the context should clean up all clients."""
        ctx_a, _, _, _, _, url_b = fresh_contexts
        ctx_a.get_transfer_channel_client(url_b)
        assert ctx_a.get_num_connected_clients() == 1

        ctx_a.close()
        assert ctx_a.get_num_connected_clients() == 0


# =========================================================
# Bidirectional reads (both sides read from each other)
# =========================================================
class TestBidirectionalReads:
    """Tests that both peers can read from each other (with separate connections)."""

    def test_both_peers_can_read_from_each_other(self, fresh_contexts):
        """Both A and B can read from each other using separate client connections."""
        ctx_a, buf_a, url_a, ctx_b, buf_b, url_b = fresh_contexts

        # Write a known pattern into buf_a for B to read
        buf_a[:256] = torch.arange(0, 256, dtype=torch.uint8)

        # A reads from B
        client_a_to_b = ctx_a.get_transfer_channel_client(url_b)
        local_a = ctx_a.get_transfer_channel_address([(256, 256)])
        remote_b = ctx_b.get_transfer_channel_address([(0, 256)])
        task_id_1 = client_a_to_b.submit_read(local_a, remote_b)

        result_1 = _wait_finished(client_a_to_b, task_id_1)
        assert result_1.is_finished() is True
        assert result_1.succeeded_mask == [True]
        assert torch.equal(buf_a[256:512], buf_b[0:256])

        # B reads from A
        client_b_to_a = ctx_b.get_transfer_channel_client(url_a)
        local_b = ctx_b.get_transfer_channel_address([(256, 256)])
        remote_a = ctx_a.get_transfer_channel_address([(0, 256)])
        task_id_2 = client_b_to_a.submit_read(local_b, remote_a)

        result_2 = _wait_finished(client_b_to_a, task_id_2)
        assert result_2.is_finished() is True
        assert result_2.succeeded_mask == [True]
        assert torch.equal(buf_b[256:512], buf_a[0:256])


# =========================================================
# Concurrent reads
# =========================================================
class TestConcurrentReads:
    """Tests for concurrent read operations."""

    def test_multiple_concurrent_reads(self, shared_contexts):
        """Multiple reads submitted without waiting should all complete."""
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = shared_contexts
        client = ctx_a.get_transfer_channel_client(url_b)

        task_ids = []
        for i in range(4):
            offset = i * 256
            local = ctx_a.get_transfer_channel_address([(offset, 256)])
            remote = ctx_b.get_transfer_channel_address([(offset, 256)])
            task_id = client.submit_read(local, remote)
            task_ids.append((task_id, offset))

        # Wait for all to complete
        for task_id, offset in task_ids:
            result = _wait_finished(client, task_id)
            assert result.is_finished() is True
            assert result.succeeded_mask == [True]
            assert torch.equal(
                buf_a[offset : offset + 256], buf_b[offset : offset + 256]
            )


# =========================================================
# Error handling
# =========================================================
class TestErrorHandling:
    """Tests for error conditions and edge cases."""

    def test_connect_to_nonexistent_server_raises(self):
        """Connecting to a port with no server should raise."""
        buf = torch.zeros(_BUF_SIZE, dtype=torch.uint8)
        desc = L1MemoryDesc(ptr=buf.data_ptr(), size=buf.numel(), align_bytes=_ALIGN)
        url = _next_url()
        ctx = TcpTransferChannelContext(desc, listen_url=url, advertise_url=url)
        try:
            # Use a port that is very unlikely to have a server
            bad_url = "127.0.0.1:1"
            with pytest.raises((ConnectionRefusedError, OSError)):
                ctx.get_transfer_channel_client(bad_url)
        finally:
            ctx.close()

    def test_read_after_server_close_marks_task_failed(self, fresh_contexts):
        """If the server closes, eventually new reads should fail gracefully."""
        ctx_a, buf_a, _, ctx_b, buf_b, url_b = fresh_contexts
        client = ctx_a.get_transfer_channel_client(url_b)

        # Verify the connection works first
        local = ctx_a.get_transfer_channel_address([(0, 256)])
        remote = ctx_b.get_transfer_channel_address([(0, 256)])
        task_id = client.submit_read(local, remote)
        result = _wait_finished(client, task_id)
        assert result.is_finished() is True
        assert result.succeeded_mask == [True]

        # Close the server side
        ctx_b.close()

        # Wait until the client's recv loop detects the broken connection.
        # We poll by submitting reads until one fails, with a deadline.
        deadline = time.monotonic() + 5.0
        saw_failure = False
        while time.monotonic() < deadline:
            time.sleep(0.2)
            local_n = ctx_a.get_transfer_channel_address([(256, 256)])
            remote_n = [TransferChannelAddress(offset=256, size=256)]
            tid = client.submit_read(local_n, remote_n)
            res = _wait_finished(client, tid, timeout_s=2.0)
            if res.is_finished() and res.succeeded_mask == [False]:
                saw_failure = True
                break

        assert saw_failure, "Expected at least one read to fail after server close"
