# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the P2P L2 adapter (mocked MQ client + transfer channel)."""

# Standard
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.client.transport.errors import MPRequestError
from lmcache.v1.common.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelTransferResult,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    P2PL2Adapter,
    P2PL2AdapterConfig,
    PeerL2AdapterBase,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.transport.futures import MessagingFuture

_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])
_STORE_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([10])], dtypes=[torch.uint8])
_PEER_BUFFERS = [
    TransferChannelAddress(offset=100, size=10),
    TransferChannelAddress(offset=200, size=10),
]
_NOT_RESERVED = TransferChannelAddress(offset=-1, size=0)


def _key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=f"hash{i}".encode(), model_name="m", kv_rank=1)


class _FakeFuture:
    """Minimal MessagingFuture stand-in returning a value or raising."""

    def __init__(self, value=None, exc=None):
        self._value = value
        self._exc = exc

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._value

    def query(self) -> bool:
        return self._exc is None

    def add_done_callback(self, fn):
        fn(self)


class _PendingFuture:
    """MessagingFuture stand-in that the test completes explicitly."""

    def __init__(self) -> None:
        self._value = None
        self._exc: BaseException | None = None
        self._callbacks: list = []

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._value

    def add_done_callback(self, fn) -> None:
        self._callbacks.append(fn)

    def resolve(self, value=None, exc: BaseException | None = None) -> None:
        self._value, self._exc = value, exc
        for fn in self._callbacks:
            fn(self)


def _fd_ready(fd: int, timeout_s: float = 0.0) -> bool:
    """Return whether ``fd`` is readable within ``timeout_s``."""
    return bool(select.select([fd], [], [], timeout_s)[0])


def _answered(outcome) -> MessagingFuture:
    """A request future the peer already answered, with a value or a failure."""
    future: MessagingFuture = MessagingFuture()
    if isinstance(outcome, BaseException):
        future.set_exception(outcome)
    else:
        future.set_result(outcome)
    return future


def _store_objects(count: int) -> list[MagicMock]:
    """Local objects of ``_STORE_LAYOUT`` at distinct offsets."""
    objects = []
    for i in range(count):
        obj = MagicMock(shm_offset=1000 * (i + 1), shm_byte_length=10)
        obj.get_shapes.return_value = _STORE_LAYOUT.shapes
        obj.get_dtypes.return_value = _STORE_LAYOUT.dtypes
        objects.append(obj)
    return objects


def _store_fd_signaled(adapter: P2PL2Adapter) -> bool:
    """Whether the store event fd would wake the store loop right now."""
    readable, _, _ = select.select([adapter.get_store_event_fd()], [], [], 0)
    return bool(readable)


def _poll_store_result(adapter: P2PL2Adapter, task_id: int):
    """Poll the store path until ``task_id`` completes and return its result."""
    for _ in range(5):
        completed = adapter.pop_completed_store_tasks()
        if task_id in completed:
            return completed[task_id]
    raise AssertionError("the store task did not complete")


@contextmanager
def _adapter(
    lookup_timeout_s: float = 10.0,
    load_timeout_s: float = 10.0,
    write_enabled: bool = False,
    store_timeout_s: float = 10.0,
):
    req_client = MagicMock(spec=RequestClient)
    tc_ctx = MagicMock()
    tc_client = MagicMock()
    tc_ctx.get_transfer_channel_client.return_value = tc_client
    tc_ctx.get_transfer_channel_address.side_effect = lambda pairs: [
        TransferChannelAddress(offset=o, size=s) for o, s in pairs
    ]
    notifier = MagicMock()

    with (
        patch.object(
            p2p_mod.RequestClientFactory,
            "create",
            return_value=req_client,
        ),
        patch.object(p2p_mod, "get_transfer_channel_context", return_value=tc_ctx),
        patch.object(p2p_mod, "PeriodicEventNotifier") as mock_pen,
    ):
        mock_pen.get.return_value = notifier
        config = P2PL2AdapterConfig(
            peer_mq_server_url="tcp://peer:5555",
            peer_transfer_channel_server_url="peer:7600",
            lookup_timeout_s=lookup_timeout_s,
            load_timeout_s=load_timeout_s,
            write_enabled=write_enabled,
            store_timeout_s=store_timeout_s,
        )
        adapter = P2PL2Adapter(config)
        try:
            yield adapter, req_client, tc_ctx, tc_client, notifier
        finally:
            adapter.close()


# ---------------------------------------------------------------------------
# Config + factory
# ---------------------------------------------------------------------------


def test_config_from_dict_roundtrip():
    config = P2PL2AdapterConfig.from_dict(
        {
            "type": "p2p",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
            "lookup_timeout_s": 4,
            "load_timeout_s": 6,
            "write_enabled": True,
            "store_timeout_s": 3,
        }
    )
    assert config.peer_mq_server_url == "tcp://peer:5555"
    assert config.peer_transfer_channel_server_url == "peer:7600"
    assert config.lookup_timeout_s == 4.0
    assert config.load_timeout_s == 6.0
    assert config.write_enabled is True
    assert config.store_timeout_s == 3.0


def test_config_writes_are_off_by_default():
    config = P2PL2AdapterConfig.from_dict(
        {
            "type": "p2p",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
        }
    )
    assert config.write_enabled is False


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "p2p", "peer_transfer_channel_server_url": "peer:7600"},
        {"type": "p2p", "peer_mq_server_url": "tcp://peer:5555"},
        {
            "type": "p2p",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
            "lookup_timeout_s": -1,
        },
        {
            "type": "p2p",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
            "write_enabled": "yes",
        },
        {
            "type": "p2p",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
            "store_timeout_s": 0,
        },
    ],
)
def test_config_from_dict_rejects_invalid(bad):
    with pytest.raises(ValueError):
        P2PL2AdapterConfig.from_dict(bad)


def test_factory_creates_adapter():
    # First Party
    from lmcache.v1.distributed.l2_adapters import create_l2_adapter

    with (
        patch.object(
            p2p_mod.RequestClientFactory,
            "create",
            return_value=MagicMock(spec=RequestClient),
        ),
        patch.object(p2p_mod, "get_transfer_channel_context", return_value=MagicMock()),
        patch.object(p2p_mod, "PeriodicEventNotifier") as mock_pen,
    ):
        mock_pen.get.return_value = MagicMock()
        config = P2PL2AdapterConfig("tcp://peer:5555", "peer:7600")
        adapter = create_l2_adapter(config)
        assert isinstance(adapter, P2PL2Adapter)
        adapter.close()


def test_failed_region_registration_releases_rpc_resources() -> None:
    """A failed transfer-channel handshake unwinds the RPC client, event fds
    and notifier registrations acquired before it."""
    client = MagicMock(spec=RequestClient)
    events = [MagicMock() for _ in range(3)]
    for fd, event in enumerate(events):
        event.fileno.return_value = fd
    notifier = MagicMock()
    context = MagicMock()
    context.get_transfer_channel_client.side_effect = RuntimeError(
        "registration failed"
    )
    with (
        patch.object(p2p_mod.RequestClientFactory, "create", return_value=client),
        patch.object(p2p_mod, "create_event_notifier", side_effect=events),
        patch.object(p2p_mod, "PeriodicEventNotifier") as periodic,
        patch.object(p2p_mod, "get_transfer_channel_context", return_value=context),
    ):
        periodic.get.return_value = notifier
        with pytest.raises(RuntimeError, match="registration failed"):
            P2PL2Adapter(P2PL2AdapterConfig("tcp://peer:5555", "peer:7600"))
    client.close.assert_called_once()
    for event in events:
        event.close.assert_called_once()
    assert {call.args[0] for call in notifier.unregister_fd.call_args_list} == {1, 2}


def test_missing_periodic_notifier_unwinds_before_the_region_hook() -> None:
    """If the notifier singleton is unavailable, construction fails before the
    region hook runs and releases the RPC client and fds."""
    client = MagicMock(spec=RequestClient)
    events = [MagicMock() for _ in range(3)]
    context = MagicMock()
    with (
        patch.object(p2p_mod.RequestClientFactory, "create", return_value=client),
        patch.object(p2p_mod, "create_event_notifier", side_effect=events),
        patch.object(p2p_mod, "PeriodicEventNotifier") as periodic,
        patch.object(p2p_mod, "get_transfer_channel_context", return_value=context),
    ):
        periodic.get.return_value = None
        with pytest.raises(RuntimeError, match="PeriodicEventNotifier"):
            P2PL2Adapter(P2PL2AdapterConfig("tcp://peer:5555", "peer:7600"))
    client.close.assert_called_once()
    for event in events:
        event.close.assert_called_once()
    context.get_transfer_channel_client.assert_not_called()


# ---------------------------------------------------------------------------
# Base-class contract through a minimal subclass
# ---------------------------------------------------------------------------


class _RecordingPeerAdapter(PeerL2AdapterBase):
    """The smallest lender-shaped subclass: records the hook calls."""

    def __init__(self, config: P2PL2AdapterConfig, log: list[str]) -> None:
        self.log = log
        super().__init__(config)

    def _register_peer_region(self) -> None:
        # The RPC client and the base's own state exist before the hook runs.
        self.log.append(f"register:{self._req_client is not None}")
        self.log.append(f"open:{not self._closed}")

    def _unregister_peer_region(self) -> None:
        self.log.append("unregister")

    def submit_load_task(self, keys, objects):
        raise NotImplementedError

    def query_load_result(self, task_id):
        raise NotImplementedError

    def get_memory_tags(self) -> MemoryTags:
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE)

    def _process_lookup_result(self, keys, addresses) -> Bitmap:
        self.log.append(f"lookup:{len(keys)}")
        bitmap = Bitmap(len(keys))
        for i, address in enumerate(addresses):
            if address.is_valid():
                bitmap.set(i)
        return bitmap


def test_base_contract_hook_order_and_one_lookup_result_per_lookup() -> None:
    """The base constructs RPC resources, runs the region hook once, hands each
    completed lookup to _process_lookup_result exactly once, reports the
    status keys the L2 interface requires, and unregisters the region before
    closing the RPC client."""
    log: list[str] = []
    req_client = MagicMock(spec=RequestClient)
    req_client.close.side_effect = lambda: log.append("rpc-closed")
    req_client.p2p_lookup_and_lock.return_value = _FakeFuture(9)
    with (
        patch.object(p2p_mod.RequestClientFactory, "create", return_value=req_client),
        patch.object(p2p_mod, "PeriodicEventNotifier") as periodic,
    ):
        periodic.get.return_value = MagicMock()
        adapter = _RecordingPeerAdapter(
            P2PL2AdapterConfig("tcp://peer:5555", "peer:7600", lookup_timeout_s=10.0),
            log,
        )
        assert log == ["register:True", "open:True"]
        assert adapter.report_status()["is_healthy"] is True

        keys = [_key(0), _key(1)]
        # The query chained from the lookup acknowledgement finds no result yet.
        req_client.p2p_query_lookup_results.return_value = _FakeFuture(None)
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        assert adapter.query_lookup_and_lock_result(task_id) is None
        req_client.p2p_query_lookup_results.return_value = _FakeFuture(
            [TransferChannelAddress(0, 8), TransferChannelAddress(-1, 0)]
        )
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.test(0) and not bitmap.test(1)
        # The task is forgotten: a second query of the same id is not a result.
        assert adapter.query_lookup_and_lock_result(task_id) is None
        assert log == ["register:True", "open:True", "lookup:2"]

        adapter.close()
        adapter.close()
        assert log[2:] == ["lookup:2", "unregister", "rpc-closed"]


# ---------------------------------------------------------------------------
# Event fds + notifier registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("write_enabled", [False, True])
def test_event_fds_are_distinct_and_registered(write_enabled: bool):
    with _adapter(write_enabled=write_enabled) as (
        adapter,
        _mq,
        _tc_ctx,
        _tc,
        notifier,
    ):
        store_fd = adapter.get_store_event_fd()
        fds = {
            store_fd,
            adapter.get_lookup_and_lock_event_fd(),
            adapter.get_load_event_fd(),
        }
        assert len(fds) == 3
        registered = {c.args[0] for c in notifier.register_fd.call_args_list}
        assert adapter.get_lookup_and_lock_event_fd() in registered
        assert adapter.get_load_event_fd() in registered
        # The store fd is pulsed only when stores reach the peer and are
        # therefore completed by polling.
        assert (store_fd in registered) is write_enabled

        adapter.close()
        unregistered = {c.args[0] for c in notifier.unregister_fd.call_args_list}
        assert (store_fd in unregistered) is write_enabled


def test_memory_tags_are_remote_dram():
    with _adapter() as (adapter, _mq, _tc_ctx, _tc, _notifier):
        assert adapter.get_memory_tags() == MemoryTags(
            MemorySegment.DRAM, MemoryResidency.REMOTE
        )


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_lookup_forwards_group_layout_descs_and_returns_addresses():
    keys = [_key(0), _key(1), _key(2)]
    addresses = [
        TransferChannelAddress(offset=100, size=10),
        TransferChannelAddress(offset=200, size=20),
        TransferChannelAddress(offset=-1, size=0),  # not found on peer
    ]
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(value=42)
        req_client.p2p_query_lookup_results.return_value = _FakeFuture(value=addresses)

        group_layout_descs = {0: _LAYOUT}
        task_id = adapter.submit_lookup_and_lock_task(keys, group_layout_descs)

        # The real group_layout_descs dict is forwarded into the lookup
        # payload.
        req_client.p2p_lookup_and_lock.assert_called_once_with(keys, group_layout_descs)
        assert req_client.p2p_lookup_and_lock.call_args.args[1] is group_layout_descs

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True
        assert bitmap.test(1) is True
        assert bitmap.test(2) is False

        # Valid addresses are stashed by key for the load phase.
        assert adapter._remote_addresses[keys[0]] == addresses[0]
        assert adapter._remote_addresses[keys[1]] == addresses[1]
        assert keys[2] not in adapter._remote_addresses


def test_lookup_query_not_ready_then_ready():
    keys = [_key(0)]
    addresses = [TransferChannelAddress(offset=100, size=10)]
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        responses = iter(
            [
                _FakeFuture(value=None),  # chained from the lookup ack
                _FakeFuture(value=None),  # first pulse
                _FakeFuture(value=addresses),  # second pulse
            ]
        )
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(value=7)
        req_client.p2p_query_lookup_results.side_effect = lambda _task_id: next(
            responses
        )
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})

        # First pulse: peer not ready yet.
        assert adapter.query_lookup_and_lock_result(task_id) is None
        # Second pulse: ready.
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.test(0) is True


def test_lookup_submit_timeout_yields_miss():
    keys = [_key(0), _key(1)]
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(exc=TimeoutError())
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0
        # No P2P_QUERY_LOOKUP_RESULTS was issued for a failed lookup.
        req_client.p2p_query_lookup_results.assert_not_called()


def test_lookup_deadline_expired_yields_miss():
    keys = [_key(0)]
    with _adapter(lookup_timeout_s=0.01) as (
        adapter,
        req_client,
        _tc_ctx,
        _tc,
        _notifier,
    ):
        # Lookup id resolves, but the query is never ready before the deadline.
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(value=7)
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        time.sleep(0.02)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


def test_lookup_result_signals_the_lookup_fd_without_polling():
    """The lookup ack chains the result query, and the result signals the
    lookup event fd, so no periodic poll is needed to complete a lookup."""
    keys = [_key(0)]
    addresses = [TransferChannelAddress(offset=100, size=10)]
    ack, result = _PendingFuture(), _PendingFuture()
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        req_client.p2p_lookup_and_lock.return_value = ack
        req_client.p2p_query_lookup_results.return_value = result
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        req_client.p2p_query_lookup_results.assert_not_called()

        ack.resolve(value=7)
        req_client.p2p_query_lookup_results.assert_called_once_with(7)
        assert not _fd_ready(lookup_fd)

        result.resolve(value=addresses)
        assert _fd_ready(lookup_fd, timeout_s=1.0)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.test(0) is True


def test_lookup_submit_does_not_wait_for_the_peer():
    """Submitting a lookup returns at once even if the peer never answers."""
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        req_client.p2p_lookup_and_lock.return_value = _PendingFuture()
        start = time.monotonic()
        task_id = adapter.submit_lookup_and_lock_task([_key(0)], {0: _LAYOUT})
        assert time.monotonic() - start < 0.5
        assert adapter.query_lookup_and_lock_result(task_id) is None


def test_unanswered_lookup_yields_miss_after_deadline():
    """A lookup whose ack never arrives becomes a miss at the lookup deadline."""
    keys = [_key(0), _key(1)]
    with _adapter(lookup_timeout_s=0.01) as (
        adapter,
        req_client,
        _tc_ctx,
        _tc,
        _notifier,
    ):
        req_client.p2p_lookup_and_lock.return_value = _PendingFuture()
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        time.sleep(0.02)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


def test_failed_lookup_ack_yields_miss_and_signals():
    """A lookup ack that fails is a miss, signalled without a result query."""
    keys = [_key(0)]
    ack = _PendingFuture()
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        req_client.p2p_lookup_and_lock.return_value = ack
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        ack.resolve(exc=RuntimeError("peer rejected the RPC"))
        assert _fd_ready(adapter.get_lookup_and_lock_event_fd(), timeout_s=1.0)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.popcount() == 0
        req_client.p2p_query_lookup_results.assert_not_called()


def test_lookup_completions_after_close_are_ignored():
    """RPC completions that land after close neither raise nor signal."""
    ack = _PendingFuture()
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        req_client.p2p_lookup_and_lock.return_value = ack
        adapter.submit_lookup_and_lock_task([_key(0)], {0: _LAYOUT})
        adapter.close()
        ack.resolve(value=7)
        req_client.p2p_query_lookup_results.assert_not_called()


def test_unanswered_query_is_resent_after_the_rpc_timeout():
    """A result query whose reply never arrives is sent again once the RPC
    timeout has passed, instead of waiting for the lookup deadline."""
    keys = [_key(0)]
    addresses = [TransferChannelAddress(offset=100, size=10)]
    lost, answered = _PendingFuture(), _FakeFuture(value=addresses)
    with (
        patch.object(p2p_mod, "_LOOKUP_RPC_TIMEOUT_S", 0.05),
        _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier),
    ):
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(value=7)
        req_client.p2p_query_lookup_results.side_effect = [lost, answered]
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        assert adapter.query_lookup_and_lock_result(task_id) is None
        assert req_client.p2p_query_lookup_results.call_count == 1

        time.sleep(0.1)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert req_client.p2p_query_lookup_results.call_count == 2
        assert bitmap is not None and bitmap.test(0) is True


def test_unacknowledged_lookup_is_a_miss_at_the_rpc_timeout():
    """A peer that never acknowledges costs the RPC timeout, not the much
    longer lookup deadline."""
    with (
        patch.object(p2p_mod, "_LOOKUP_RPC_TIMEOUT_S", 0.05),
        _adapter(lookup_timeout_s=30.0) as (adapter, req_client, *_),
    ):
        req_client.p2p_lookup_and_lock.return_value = _PendingFuture()
        task_id = adapter.submit_lookup_and_lock_task([_key(0)], {0: _LAYOUT})
        assert adapter.query_lookup_and_lock_result(task_id) is None
        time.sleep(0.1)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.popcount() == 0
        req_client.p2p_query_lookup_results.assert_not_called()


def test_result_that_landed_before_the_deadline_poll_is_used():
    """A result already stored when the deadline passes is a hit, not a miss:
    the peer holds read locks for it that only the load/unlock path frees."""
    keys = [_key(0)]
    addresses = [TransferChannelAddress(offset=100, size=10)]
    with _adapter(lookup_timeout_s=0.05) as (adapter, req_client, *_):
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(value=7)
        req_client.p2p_query_lookup_results.return_value = _FakeFuture(value=addresses)
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        time.sleep(0.1)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.test(0) is True


def test_result_after_the_deadline_miss_is_dropped():
    """A query reply that lands after the lookup already expired as a miss is
    ignored: no error, and the finished task stays finished."""
    query = _PendingFuture()
    with _adapter(lookup_timeout_s=0.05) as (adapter, req_client, *_):
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(value=7)
        req_client.p2p_query_lookup_results.return_value = query
        task_id = adapter.submit_lookup_and_lock_task([_key(0)], {0: _LAYOUT})
        time.sleep(0.1)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.popcount() == 0
        query.resolve(value=[TransferChannelAddress(offset=100, size=10)])
        assert adapter.query_lookup_and_lock_result(task_id) is None


def test_second_failed_query_yields_miss():
    """One failed result query is retried; a second failure is a miss."""
    keys = [_key(0)]
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        req_client.p2p_lookup_and_lock.return_value = _FakeFuture(value=7)
        req_client.p2p_query_lookup_results.return_value = _FakeFuture(
            exc=RuntimeError("peer rejected the RPC")
        )
        task_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        # The chained query failed once; the next poll retries and fails again.
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        if bitmap is None:
            bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.popcount() == 0
        assert req_client.p2p_query_lookup_results.call_count == 2


def test_query_unknown_task_returns_none():
    with _adapter() as (adapter, _mq, _tc_ctx, _tc, _notifier):
        assert adapter.query_lookup_and_lock_result(999) is None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def test_load_reads_into_local_objects():
    keys = [_key(0), _key(1)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=100, size=10)
        adapter._remote_addresses[keys[1]] = TransferChannelAddress(offset=200, size=20)
        tc_client.submit_read.return_value = 55
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=True, succeeded_mask=[True, False]
        )

        objects = [
            MagicMock(shm_offset=1000, shm_byte_length=10),
            MagicMock(shm_offset=2000, shm_byte_length=20),
        ]
        task_id = adapter.submit_load_task(keys, objects)

        local, remote = tc_client.submit_read.call_args.args
        assert local == [
            TransferChannelAddress(offset=1000, size=10),
            TransferChannelAddress(offset=2000, size=20),
        ]
        assert remote == [
            adapter._remote_addresses[keys[0]],
            adapter._remote_addresses[keys[1]],
        ]

        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True
        assert bitmap.test(1) is False


def test_load_not_finished_returns_none():
    keys = [_key(0)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=100, size=10)
        tc_client.submit_read.return_value = 1
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=False
        )
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        assert adapter.query_load_result(task_id) is None


def test_load_deadline_expired_yields_failure():
    keys = [_key(0)]
    with _adapter(load_timeout_s=0.01) as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=100, size=10)
        tc_client.submit_read.return_value = 1
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        time.sleep(0.02)
        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


# ---------------------------------------------------------------------------
# Unlock + store no-ops + lifecycle
# ---------------------------------------------------------------------------


def test_unlock_sends_rpc_and_clears_addresses():
    keys = [_key(0), _key(1)]
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=1, size=1)
        adapter.submit_unlock(keys)
        req_client.p2p_unlock_objects.assert_called_once_with(keys)
        assert keys[0] not in adapter._remote_addresses


def test_unlock_empty_is_noop():
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        adapter.submit_unlock([])
        req_client.p2p_unlock_objects.assert_not_called()


def test_store_completes_immediately_without_leaking():
    with _adapter() as (adapter, req_client, _tc_ctx, _tc, _notifier):
        task_id = adapter.submit_store_task([_key(0)], [MagicMock()])
        completed = adapter.pop_completed_store_tasks()
        assert set(completed) == {task_id}
        assert completed[task_id].is_successful() is True
        assert completed[task_id].bytes_transferred() == 0
        # Draining clears the completed set.
        assert adapter.pop_completed_store_tasks() == {}
        # With writes disabled the peer is never asked for anything.
        req_client.p2p_reserve_write.assert_not_called()


# Store with writes enabled
# ---------------------------------------------------------------------------


def test_store_writes_to_peer_and_succeeds_only_after_commit():
    keys = [_key(0), _key(1)]
    with _adapter(write_enabled=True) as (
        adapter,
        req_client,
        _tc_ctx,
        tc_client,
        _notifier,
    ):
        reservation: MessagingFuture = MessagingFuture()
        commit: MessagingFuture = MessagingFuture()
        req_client.p2p_reserve_write.return_value = reservation
        req_client.p2p_finish_write.return_value = commit
        tc_client.submit_write.return_value = 7
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=False
        )

        task_id = adapter.submit_store_task(keys, _store_objects(2))
        req_client.p2p_reserve_write.assert_called_once_with(keys, _STORE_LAYOUT)
        assert adapter.report_status()["in_flight_stores"] == 1

        # The peer has not answered: polling returns at once, with nothing.
        assert adapter.pop_completed_store_tasks() == {}
        tc_client.submit_write.assert_not_called()

        # Reserved: the answer itself starts the transfer into the buffers the
        # peer returned, without waiting for the store loop to poll.
        reservation.set_result(_PEER_BUFFERS)
        local, remote = tc_client.submit_write.call_args.args
        assert local == [
            TransferChannelAddress(offset=1000, size=10),
            TransferChannelAddress(offset=2000, size=10),
        ]
        assert remote == _PEER_BUFFERS

        # Transfer still running: nothing is committed yet.
        assert adapter.pop_completed_store_tasks() == {}
        req_client.p2p_finish_write.assert_not_called()

        # Transfer done: the peer is asked to commit, the task stays open.
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=True, succeeded_mask=[True, True]
        )
        assert adapter.pop_completed_store_tasks() == {}
        tc_client.query_status.assert_called_with(7)
        req_client.p2p_finish_write.assert_called_once_with(keys, [True, True])
        assert _store_fd_signaled(adapter) is False

        # Committed: the answer wakes the store loop, and the task succeeds
        # with the bytes written.
        commit.set_result([True, True])
        assert _store_fd_signaled(adapter) is True
        completed = adapter.pop_completed_store_tasks()
        assert set(completed) == {task_id}
        assert completed[task_id].is_successful() is True
        assert completed[task_id].bytes_transferred() == 20
        assert adapter.pop_completed_store_tasks() == {}
        assert adapter.report_status()["in_flight_stores"] == 0


@pytest.mark.parametrize(
    ("reservation", "write_mask", "commit", "finish_call"),
    [
        pytest.param(
            [_NOT_RESERVED, _NOT_RESERVED], None, None, None, id="nothing-reserved"
        ),
        pytest.param(
            [_PEER_BUFFERS[0], _NOT_RESERVED],
            [True],
            None,
            ([0], [True]),
            id="partial-reservation",
        ),
        pytest.param(
            _PEER_BUFFERS,
            [True, False],
            None,
            ([0, 1], [True, False]),
            id="write-failed-for-a-key",
        ),
        pytest.param(
            _PEER_BUFFERS,
            [True, True],
            [True, False],
            ([0, 1], [True, True]),
            id="peer-did-not-commit-a-key",
        ),
        pytest.param(
            _PEER_BUFFERS,
            [True, True],
            MPRequestError("INTERNAL", "boom"),
            ([0, 1], [True, True]),
            id="commit-request-rejected",
        ),
        pytest.param(
            TimeoutError("no answer"), None, None, None, id="reserve-request-lost"
        ),
        pytest.param(
            MPRequestError("UNIMPLEMENTED", "old peer"),
            None,
            None,
            None,
            id="reserve-request-rejected",
        ),
    ],
)
def test_store_fails_unless_every_key_reaches_the_peer(
    reservation, write_mask, commit, finish_call
):
    keys = [_key(0), _key(1)]
    with _adapter(write_enabled=True) as (
        adapter,
        req_client,
        _tc_ctx,
        tc_client,
        _notifier,
    ):
        req_client.p2p_reserve_write.return_value = _answered(reservation)
        req_client.p2p_finish_write.return_value = _answered(commit)
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=True, succeeded_mask=write_mask or []
        )

        task_id = adapter.submit_store_task(keys, _store_objects(2))
        # A store that fails on the reservation's answer wakes the store loop.
        assert _store_fd_signaled(adapter) is (write_mask is None)
        result = _poll_store_result(adapter, task_id)

        assert result.is_successful() is False
        assert result.bytes_transferred() == 0
        if write_mask is None:
            tc_client.submit_write.assert_not_called()
        else:
            # Only the buffers the peer reserved are written.
            _local, remote = tc_client.submit_write.call_args.args
            assert remote == [a for a in reservation if a.is_valid()]
        if finish_call is None:
            req_client.p2p_finish_write.assert_not_called()
        else:
            key_indices, mask = finish_call
            req_client.p2p_finish_write.assert_called_once_with(
                [keys[i] for i in key_indices], mask
            )


@pytest.mark.parametrize("stuck_step", ["reservation", "transfer", "commit"])
def test_store_past_its_deadline_fails(stuck_step: str):
    keys = [_key(0), _key(1)]
    with _adapter(write_enabled=True, store_timeout_s=0.01) as (
        adapter,
        req_client,
        _tc_ctx,
        tc_client,
        _notifier,
    ):
        req_client.p2p_reserve_write.return_value = (
            MessagingFuture()
            if stuck_step == "reservation"
            else _answered(_PEER_BUFFERS)
        )
        # The commit request is never answered.
        req_client.p2p_finish_write.return_value = MessagingFuture()
        tc_client.query_status.return_value = (
            TransferChannelTransferResult(finished=False)
            if stuck_step == "transfer"
            else TransferChannelTransferResult(
                finished=True, succeeded_mask=[True, True]
            )
        )

        task_id = adapter.submit_store_task(keys, _store_objects(2))
        time.sleep(0.02)
        result = _poll_store_result(adapter, task_id)

        assert result.is_successful() is False
        if stuck_step == "reservation":
            req_client.p2p_finish_write.assert_not_called()
        elif stuck_step == "transfer":
            # The peer is told to drop the buffers of the abandoned transfer.
            req_client.p2p_finish_write.assert_called_once_with(keys, [False, False])
        else:
            req_client.p2p_finish_write.assert_called_once_with(keys, [True, True])


def test_reservation_answered_after_the_deadline_starts_no_transfer():
    with _adapter(write_enabled=True, store_timeout_s=0.01) as (
        adapter,
        req_client,
        _tc_ctx,
        tc_client,
        _notifier,
    ):
        reservation: MessagingFuture = MessagingFuture()
        req_client.p2p_reserve_write.return_value = reservation
        task_id = adapter.submit_store_task([_key(0), _key(1)], _store_objects(2))
        time.sleep(0.02)
        assert _poll_store_result(adapter, task_id).is_successful() is False

        # The store already failed: a late answer must not write anything.
        reservation.set_result(_PEER_BUFFERS)
        tc_client.submit_write.assert_not_called()
        assert adapter.pop_completed_store_tasks() == {}


def test_store_after_close_fails_without_asking_the_peer():
    with _adapter(write_enabled=True) as (
        adapter,
        req_client,
        _tc_ctx,
        _tc,
        _notifier,
    ):
        adapter.close()
        task_id = adapter.submit_store_task([_key(0)], _store_objects(1))
        completed = adapter.pop_completed_store_tasks()
        assert completed[task_id].is_successful() is False
        req_client.p2p_reserve_write.assert_not_called()


def test_load_missing_address_is_failure_not_raise():
    keys = [_key(0)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        # No lookup ran, so no remote address is stashed for this key.
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        tc_client.submit_read.assert_not_called()
        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


def test_load_invalid_address_is_failure_not_raise():
    keys = [_key(0)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        # An invalid stashed address must be rejected, not read.
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=-1, size=0)
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        tc_client.submit_read.assert_not_called()
        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


def test_report_status():
    with _adapter() as (adapter, _mq, _tc_ctx, _tc, _notifier):
        status = adapter.report_status()
        assert status["is_healthy"] is True
        assert status["peer_mq_server_url"] == "tcp://peer:5555"
        assert status["peer_transfer_channel_server_url"] == "peer:7600"
        assert status["in_flight_lookups"] == 0
        assert status["in_flight_loads"] == 0
        assert status["write_enabled"] is False
        assert status["in_flight_stores"] == 0


def test_close_unregisters_fds_and_closes_client():
    with _adapter() as (adapter, req_client, tc_ctx, _tc, notifier):
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()
        adapter.close()
        unregistered = {c.args[0] for c in notifier.unregister_fd.call_args_list}
        assert lookup_fd in unregistered
        assert load_fd in unregistered
        req_client.close.assert_called()
        # The transfer-channel client is dropped from the context on close.
        tc_ctx.remove_transfer_channel_client.assert_called_once_with("peer:7600")


def test_close_is_idempotent():
    with _adapter() as (adapter, req_client, tc_ctx, _tc, _notifier):
        adapter.close()
        adapter.close()
        # The second close is a no-op: no duplicate teardown of shared resources.
        req_client.close.assert_called_once()
        tc_ctx.remove_transfer_channel_client.assert_called_once_with("peer:7600")


def test_region_teardown_failure_keeps_rpc_open_for_retry() -> None:
    """A failed region teardown leaves the adapter open so close can retry."""
    with _adapter() as (adapter, client, context, _tc, notifier):
        context.remove_transfer_channel_client.side_effect = [
            RuntimeError("busy"),
            None,
        ]
        with pytest.raises(RuntimeError, match="busy"):
            adapter.close()
        client.close.assert_not_called()
        notifier.unregister_fd.assert_not_called()
        adapter.close()
        client.close.assert_called_once()
