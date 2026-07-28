# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.distributed.transfer_channel.impl import verbs_impl


class _FakeClient:
    def __init__(
        self,
        statuses=None,
        submit_error: Exception | None = None,
        close_errors: int = 0,
    ):
        self.statuses = list(statuses or [(True, True, 1)])
        self.submit_error = submit_error
        self.close_errors = close_errors
        self.submissions: list[tuple[list[int], list[int], list[int]]] = []
        self.closed = False
        self.healthy = True

    def submit_read(self, local_offsets, remote_offsets, sizes):
        if self.submit_error is not None:
            raise self.submit_error
        self.submissions.append((local_offsets, remote_offsets, sizes))
        return len(self.submissions)

    def query_read_status(self, task_id):
        assert task_id > 0
        return self.statuses.pop(0)

    def close(self):
        if self.close_errors:
            self.close_errors -= 1
            raise RuntimeError("injected close failure")
        self.closed = True


class _FakeContext:
    created: list["_FakeContext"] = []
    close_errors = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.connections = []
        self.closed = False
        self.created.append(self)

    def connect(self, peer_url):
        self.connections.append(peer_url)
        return _FakeClient()

    def close(self):
        if self.__class__.close_errors:
            self.__class__.close_errors -= 1
            raise RuntimeError("injected context close failure")
        self.closed = True


def _address(offset: int, size: int) -> TransferChannelAddress:
    return TransferChannelAddress(offset=offset, size=size)


def test_context_registers_host_l1_on_each_rail(monkeypatch):
    _FakeContext.created = []
    monkeypatch.setattr(
        verbs_impl,
        "_load_native",
        lambda: SimpleNamespace(RdmaContext=_FakeContext),
    )
    memory = L1MemoryDesc(ptr=0x100000, size=1 << 20, align_bytes=4096)

    context = verbs_impl.VerbsTransferChannelContext(
        memory,
        listen_url="0.0.0.0:7600",
        advertise_url="10.0.0.1:7600",
        device_name="mlx5_0,mlx5_1",
        gid_indices="3,5",
    )

    assert [item.kwargs["device_name"] for item in _FakeContext.created] == [
        "mlx5_0",
        "mlx5_1",
    ]
    assert [item.kwargs["listen_url"] for item in _FakeContext.created] == [
        "0.0.0.0:7600",
        "0.0.0.0:7601",
    ]
    assert [item.kwargs["advertise_url"] for item in _FakeContext.created] == [
        "10.0.0.1:7600",
        "10.0.0.1:7601",
    ]
    assert [item.kwargs["gid_index"] for item in _FakeContext.created] == [3, 5]
    assert all(
        item.kwargs["base_address"] == memory.ptr for item in _FakeContext.created
    )
    assert all(item.kwargs["length"] == memory.size for item in _FakeContext.created)

    client = context.get_transfer_channel_client("10.0.0.2:7600")
    assert client is context.get_transfer_channel_client("10.0.0.2:7600")
    assert [item.connections for item in _FakeContext.created] == [
        ["10.0.0.2:7600"],
        ["10.0.0.2:7601"],
    ]
    assert context.get_transfer_channel_address([(4096, 8192)]) == [
        _address(4096, 8192)
    ]
    context.close()
    assert all(item.closed for item in _FakeContext.created)


def test_dual_rail_read_stripes_and_merges_completion():
    rail0 = _FakeClient(statuses=[(False, False, 2), (True, True, 2)])
    rail1 = _FakeClient(statuses=[(True, True, 2)])
    client = verbs_impl.VerbsTransferChannelClient([rail0, rail1], reconnect=lambda: [])

    task_id = client.submit_read(
        [_address(0, 8), _address(32, 6)],
        [_address(64, 8), _address(96, 6)],
    )

    assert rail0.submissions == [([0, 32], [64, 96], [4, 3])]
    assert rail1.submissions == [([4, 35], [68, 99], [4, 3])]
    assert not client.query_read_status(task_id).finished
    assert client.query_read_status(task_id).succeeded_mask == [True, True]


def test_failed_rail_only_fails_objects_using_that_rail():
    rail0 = _FakeClient(statuses=[(True, True, 2)])
    rail1 = _FakeClient(statuses=[(True, False, 1)])
    client = verbs_impl.VerbsTransferChannelClient([rail0, rail1], reconnect=lambda: [])

    task_id = client.submit_read(
        [_address(0, 1), _address(16, 4)],
        [_address(32, 1), _address(48, 4)],
    )

    assert client.query_read_status(task_id).succeeded_mask == [True, False]


def test_failed_completion_reconnects_before_next_read():
    old = _FakeClient(statuses=[(True, False, 1)])
    replacement = _FakeClient(statuses=[(True, True, 1)])
    reconnects = []

    def reconnect():
        reconnects.append(True)
        return [replacement]

    client = verbs_impl.VerbsTransferChannelClient([old], reconnect=reconnect)
    first = client.submit_read([_address(0, 8)], [_address(16, 8)])
    assert client.query_read_status(first).succeeded_mask == [False]

    second = client.submit_read([_address(0, 8)], [_address(16, 8)])
    assert reconnects == [True]
    assert old.closed
    assert client.query_read_status(second).succeeded_mask == [True]


def test_unhealthy_rail_waits_for_active_read_before_reconnect():
    old = _FakeClient(statuses=[(True, True, 1)])
    replacement = _FakeClient()
    reconnects = []

    def reconnect():
        reconnects.append(True)
        return [replacement]

    client = verbs_impl.VerbsTransferChannelClient([old], reconnect=reconnect)
    first = client.submit_read([_address(0, 8)], [_address(16, 8)])
    old.healthy = False

    with pytest.raises(RuntimeError, match="reads in flight"):
        client.submit_read([_address(32, 8)], [_address(48, 8)])

    assert reconnects == []
    assert not old.closed
    assert client.query_read_status(first).succeeded_mask == [True]

    second = client.submit_read([_address(32, 8)], [_address(48, 8)])
    assert reconnects == [True]
    assert old.closed
    assert client.query_read_status(second).succeeded_mask == [True]


def test_queue_depth_rejection_does_not_fail_active_read():
    rail0 = _FakeClient(statuses=[(True, True, 1)])
    rail1 = _FakeClient(statuses=[(False, False, 1), (True, True, 1)])
    client = verbs_impl.VerbsTransferChannelClient(
        [rail0, rail1],
        reconnect=lambda: [],
        queue_depth=1,
    )
    first = client.submit_read([_address(0, 2)], [_address(16, 2)])
    assert not client.query_read_status(first).finished

    with pytest.raises(RuntimeError, match="send queue is full"):
        client.submit_read([_address(32, 2)], [_address(48, 2)])

    assert len(rail0.submissions) == 1
    assert len(rail1.submissions) == 1
    assert not rail0.closed
    assert not rail1.closed
    assert client.query_read_status(first).succeeded_mask == [True]

    client.submit_read([_address(32, 2)], [_address(48, 2)])
    assert len(rail0.submissions) == 2
    assert len(rail1.submissions) == 2


def test_submit_failure_reconnects_before_failed_task_is_queried():
    old = _FakeClient()
    replacement = _FakeClient()
    reconnects = []

    def reconnect():
        reconnects.append(True)
        return [replacement]

    client = verbs_impl.VerbsTransferChannelClient([old], reconnect=reconnect)
    first = client.submit_read([_address(0, 8)], [_address(16, 8)])
    old.submit_error = RuntimeError("injected submit failure")

    with pytest.raises(RuntimeError, match="injected submit failure"):
        client.submit_read([_address(0, 8)], [_address(16, 8)])

    second = client.submit_read([_address(0, 8)], [_address(16, 8)])

    assert reconnects == [True]
    assert client.query_read_status(first).succeeded_mask == [False]
    assert client.query_read_status(second).succeeded_mask == [True]


def test_partial_submit_close_failure_stays_nonterminal_until_quiesced():
    rail0 = _FakeClient(close_errors=3)
    rail1 = _FakeClient()
    replacement0 = _FakeClient()
    replacement1 = _FakeClient()
    reconnects = []

    def reconnect():
        reconnects.append(True)
        return [replacement0, replacement1]

    client = verbs_impl.VerbsTransferChannelClient(
        [rail0, rail1],
        reconnect=reconnect,
        queue_depth=4,
    )
    first = client.submit_read(
        [_address(0, 2)],
        [_address(16, 2)],
    )
    rail1.submit_error = RuntimeError("injected submit failure")

    poisoned = client.submit_read(
        [_address(32, 2)],
        [_address(48, 2)],
    )

    assert poisoned != first
    assert not client.query_read_status(first).finished

    with pytest.raises(RuntimeError, match="injected close failure"):
        client.submit_read(
            [_address(64, 2)],
            [_address(80, 2)],
        )
    assert len(rail0.submissions) == 2
    assert len(rail1.submissions) == 1

    assert client.query_read_status(poisoned).succeeded_mask == [False]
    assert client.query_read_status(first).succeeded_mask == [False]

    recovered = client.submit_read(
        [_address(64, 2)],
        [_address(80, 2)],
    )
    assert reconnects == [True]
    assert client.query_read_status(recovered).succeeded_mask == [True]


def test_client_close_can_be_retried_after_native_failure():
    native = _FakeClient(close_errors=1)
    client = verbs_impl.VerbsTransferChannelClient([native], reconnect=lambda: [])
    client.submit_read([_address(0, 8)], [_address(16, 8)])

    with pytest.raises(RuntimeError, match="injected close"):
        client.close()
    assert not client._closed

    client.close()
    assert client._closed
    assert native.closed
    with pytest.raises(RuntimeError, match="client is closed"):
        client.submit_read([_address(32, 8)], [_address(48, 8)])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"device_name": "mlx5_0,mlx5_0"}, "unique names"),
        ({"device_name": "mlx5_0,mlx5_1", "gid_indices": "3"}, "count"),
        ({"device_name": "mlx5_0", "queue_depth": 0}, "queue depth"),
        ({"device_name": "mlx5_0", "port_num": 256}, "port"),
    ],
)
def test_context_rejects_invalid_rdma_options(monkeypatch, kwargs, match):
    monkeypatch.setattr(
        verbs_impl,
        "_load_native",
        lambda: SimpleNamespace(RdmaContext=_FakeContext),
    )
    with pytest.raises(ValueError, match=match):
        verbs_impl.VerbsTransferChannelContext(
            L1MemoryDesc(ptr=1, size=4096, align_bytes=4096),
            listen_url="0.0.0.0:7600",
            advertise_url="10.0.0.1:7600",
            **kwargs,
        )


def test_context_rejects_out_of_bounds_l1_address(monkeypatch):
    _FakeContext.created = []
    monkeypatch.setattr(
        verbs_impl,
        "_load_native",
        lambda: SimpleNamespace(RdmaContext=_FakeContext),
    )
    context = verbs_impl.VerbsTransferChannelContext(
        L1MemoryDesc(ptr=1, size=4096, align_bytes=4096),
        listen_url="0.0.0.0:7600",
        advertise_url="10.0.0.1:7600",
        device_name="mlx5_0",
    )
    with pytest.raises(ValueError, match="outside"):
        context.get_transfer_channel_address([(4090, 16)])
    context.close()


def test_context_close_can_be_retried_after_native_failure(monkeypatch):
    _FakeContext.created = []
    _FakeContext.close_errors = 1
    monkeypatch.setattr(
        verbs_impl,
        "_load_native",
        lambda: SimpleNamespace(RdmaContext=_FakeContext),
    )
    context = verbs_impl.VerbsTransferChannelContext(
        L1MemoryDesc(ptr=1, size=4096, align_bytes=4096),
        listen_url="0.0.0.0:7600",
        advertise_url="10.0.0.1:7600",
        device_name="mlx5_0",
    )

    with pytest.raises(RuntimeError, match="injected context close"):
        context.close()
    assert context._closing
    with pytest.raises(RuntimeError, match="context is closed"):
        context.get_transfer_channel_client("10.0.0.2:7600")
    context.close()
    assert context._closed
    assert not context._closing


def test_context_close_cancels_connect_without_waiting_for_state_lock(monkeypatch):
    connect_started = threading.Event()
    release_connect = threading.Event()
    native_close_called = threading.Event()

    class BlockingContext:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def connect(self, peer_url):
            del peer_url
            connect_started.set()
            assert release_connect.wait(timeout=5)
            raise RuntimeError("connect cancelled")

        def close(self):
            native_close_called.set()
            release_connect.set()

    monkeypatch.setattr(
        verbs_impl,
        "_load_native",
        lambda: SimpleNamespace(RdmaContext=BlockingContext),
    )
    context = verbs_impl.VerbsTransferChannelContext(
        L1MemoryDesc(ptr=1, size=4096, align_bytes=4096),
        listen_url="0.0.0.0:7600",
        advertise_url="10.0.0.1:7600",
        device_name="mlx5_0",
    )
    connect_errors: list[Exception] = []

    def connect():
        try:
            context.get_transfer_channel_client("10.0.0.2:7600")
        except Exception as error:  # noqa: BLE001 - captured from worker thread
            connect_errors.append(error)

    connect_thread = threading.Thread(target=connect)
    close_thread = threading.Thread(target=context.close)
    connect_thread.start()
    assert connect_started.wait(timeout=1)
    close_thread.start()
    try:
        assert native_close_called.wait(timeout=1)
    finally:
        release_connect.set()
    connect_thread.join(timeout=1)
    close_thread.join(timeout=1)

    assert not connect_thread.is_alive()
    assert not close_thread.is_alive()
    assert connect_errors
    assert context._closed


def test_context_closes_native_context_before_clients(monkeypatch):
    close_order: list[str] = []

    class OrderedClient(_FakeClient):
        def close(self):
            close_order.append("client")
            super().close()

    class OrderedContext(_FakeContext):
        def connect(self, peer_url):
            self.connections.append(peer_url)
            return OrderedClient()

        def close(self):
            close_order.append("context")
            super().close()

    OrderedContext.created = []
    OrderedContext.close_errors = 0
    monkeypatch.setattr(
        verbs_impl,
        "_load_native",
        lambda: SimpleNamespace(RdmaContext=OrderedContext),
    )
    context = verbs_impl.VerbsTransferChannelContext(
        L1MemoryDesc(ptr=1, size=4096, align_bytes=4096),
        listen_url="0.0.0.0:7600",
        advertise_url="10.0.0.1:7600",
        device_name="mlx5_0",
    )
    context.get_transfer_channel_client("10.0.0.2:7600")

    context.close()

    assert close_order == ["context", "client"]


def test_remove_client_retains_native_client_after_close_failure():
    client = _FakeClient(close_errors=1)
    context = object.__new__(verbs_impl.VerbsTransferChannelContext)
    context._lock = verbs_impl.threading.Lock()
    context._clients = {"peer:7600": client}

    context.remove_transfer_channel_client("peer:7600")
    assert context._clients["peer:7600"] is client
    context.remove_transfer_channel_client("peer:7600")
    assert "peer:7600" not in context._clients


def test_remove_client_does_not_hold_context_lock_while_closing():
    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingClient(_FakeClient):
        def close(self):
            close_started.set()
            assert release_close.wait(timeout=5)
            super().close()

    client = BlockingClient()
    context = object.__new__(verbs_impl.VerbsTransferChannelContext)
    context._lock = threading.Lock()
    context._clients = {"peer:7600": client}
    remove_thread = threading.Thread(
        target=context.remove_transfer_channel_client,
        args=("peer:7600",),
    )
    remove_thread.start()
    assert close_started.wait(timeout=1)

    acquired = context._lock.acquire(timeout=1)
    if acquired:
        context._lock.release()
    release_close.set()
    remove_thread.join(timeout=1)

    assert acquired
    assert not remove_thread.is_alive()
    assert "peer:7600" not in context._clients
