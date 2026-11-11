# SPDX-License-Identifier: Apache-2.0
"""HIXL Python contracts with native/ACL doubles, not data-transfer validation."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import itertools
import socket
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel import (
    delete_transfer_channel_context,
    get_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.distributed.transfer_channel.impl.hixl_impl import (
    HixlChannelMetadata,
    HixlTransferChannelClient,
    HixlTransferChannelContext,
)


class _HixlError(RuntimeError):
    pass


class _Operation(IntEnum):
    READ = 0
    WRITE = 1


class _Status(IntEnum):
    WAITING = 0
    COMPLETED = 1
    FAILED = 2


@dataclass
class _Descriptor:
    local_addr: int
    remote_addr: int
    len: int


_PEER = HixlChannelMetadata(
    advertise_url="127.0.0.1:19001",
    engine_id="127.0.0.1:19101",
    buffer_base_ptr=0x20000,
    buffer_size=4096,
    instance_id="peer-instance",
)
_LOCAL = [TransferChannelAddress(64, 128), TransferChannelAddress(512, 64)]
_REMOTE = [TransferChannelAddress(256, 128), TransferChannelAddress(768, 64)]


@pytest.fixture
def native_sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace only Ascend dependencies; keep Channel and ZMQ code real."""
    engine = Mock(
        spec=[
            "initialize",
            "register_mem",
            "connect",
            "transfer_async",
            "get_transfer_status",
            "disconnect",
            "deregister_mem",
            "finalize",
        ]
    )
    engine.register_mem.return_value = 7
    requests = itertools.count(100)
    engine.transfer_async.side_effect = lambda *args: next(requests)
    engine.get_transfer_status.return_value = _Status.COMPLETED
    runtime = SimpleNamespace(
        prepare_acl_context=Mock(return_value=1),
        use_acl_context=Mock(side_effect=lambda context: nullcontext()),
    )
    for name in ("lmcache_ascend", "lmcache_ascend.v1"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules,
        "lmcache_ascend.hixl_npu_comms",
        SimpleNamespace(
            Hixl=Mock(return_value=engine),
            MEM_HOST=1,
            HixlError=_HixlError,
            TransferOp=_Operation,
            TransferStatus=_Status,
            TransferOpDesc=_Descriptor,
        ),
    )
    monkeypatch.setitem(sys.modules, "lmcache_ascend.v1.hixl_runtime", runtime)
    monkeypatch.setenv("LMCACHE_HIXL_ENGINE_ID", "127.0.0.1:19100")
    for suffix in ("DEVICE_ID", "CONNECT_TIMEOUT_MS", "COMM_LISTEN_PORT"):
        monkeypatch.delenv("LMCACHE_HIXL_" + suffix, raising=False)
    return SimpleNamespace(engine=engine, runtime=runtime)


@pytest.fixture
def channel_url() -> str:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


@pytest.fixture
def context(
    native_sdk: SimpleNamespace, channel_url: str
) -> Iterator[HixlTransferChannelContext]:
    ctx = initialize_transfer_channel_context(
        "hixl",
        L1MemoryDesc(ptr=0x10000, size=4096, align_bytes=256),
        channel_url,
        channel_url,
    )
    try:
        assert isinstance(ctx, HixlTransferChannelContext)
        yield ctx
    finally:
        delete_transfer_channel_context()


@pytest.fixture
def client(context: HixlTransferChannelContext) -> HixlTransferChannelClient:
    return context.register_client(_PEER.advertise_url, context.create_client(_PEER))


def test_factory_registers_and_releases_l1(
    context: HixlTransferChannelContext,
    native_sdk: SimpleNamespace,
) -> None:
    """Selecting hixl must reach its factory and clean up its registration."""
    assert get_transfer_channel_context() is context
    native_sdk.engine.register_mem.assert_called_once_with(0x10000, 4096, 1)
    assert context.metadata.buffer_base_ptr == 0x10000
    assert context.metadata.buffer_size == 4096
    delete_transfer_channel_context()
    native_sdk.engine.deregister_mem.assert_called_once_with(7)
    native_sdk.engine.finalize.assert_called_once()


@pytest.mark.parametrize("operation", ["read", "write"])
def test_submission_translates_offsets_without_swapping_addresses(
    client: HixlTransferChannelClient,
    native_sdk: SimpleNamespace,
    operation: str,
) -> None:
    task = getattr(client, f"submit_{operation}")(_LOCAL, _REMOTE)
    native_sdk.engine.transfer_async.assert_called_once_with(
        "127.0.0.1:19101",
        _Operation[operation.upper()],
        [_Descriptor(0x10040, 0x20100, 128), _Descriptor(0x10200, 0x20300, 64)],
    )
    assert client.query_status(task).succeeded_mask == [True, True]


@pytest.mark.parametrize(
    "terminal,mask",
    [(_Status.COMPLETED, [True, True]), (_Status.FAILED, [False, False])],
)
def test_waiting_preserves_task_and_terminal_status_consumes_it(
    client: HixlTransferChannelClient,
    native_sdk: SimpleNamespace,
    terminal: _Status,
    mask: list[bool],
) -> None:
    native_sdk.engine.get_transfer_status.side_effect = [_Status.WAITING, terminal]
    task = client.submit_read(_LOCAL, _REMOTE)
    waiting = client.query_status(task)
    assert not waiting.is_finished()
    assert waiting.succeeded_mask == []
    result = client.query_status(task)
    assert result.is_finished()
    assert result.succeeded_mask == mask
    with pytest.raises(KeyError):
        client.query_status(task)


@pytest.mark.parametrize(
    "native_failure", [True, False], ids=["native-query", "acl-preparation"]
)
def test_query_failure_consumption_and_other_task_isolation(
    client: HixlTransferChannelClient,
    native_sdk: SimpleNamespace,
    native_failure: bool,
) -> None:
    first = client.submit_read(_LOCAL, _REMOTE)
    other = client.submit_write(_LOCAL, _REMOTE)
    assert first != other
    error_type: type[RuntimeError]
    if native_failure:
        native_sdk.engine.get_transfer_status.side_effect = _HixlError("query failed")
        error_type = _HixlError
    else:
        native_sdk.runtime.use_acl_context.side_effect = RuntimeError("ACL unavailable")
        error_type = RuntimeError
    with pytest.raises(error_type):
        client.query_status(first)
    native_sdk.engine.get_transfer_status.side_effect = None
    native_sdk.runtime.use_acl_context.side_effect = lambda context: nullcontext()
    if native_failure:
        with pytest.raises(KeyError):
            client.query_status(first)
    else:
        assert client.query_status(first).is_finished()
    assert client.query_status(other).succeeded_mask == [True, True]


@pytest.mark.parametrize(
    "local,remote",
    [
        ([TransferChannelAddress(4090, 128)], _REMOTE[:1]),
        (_LOCAL, _REMOTE[:1]),
        (_LOCAL[:1], [TransferChannelAddress(256, 64)]),
    ],
    ids=["outside-pool", "count-mismatch", "size-mismatch"],
)
def test_invalid_batch_is_rejected_before_native_submission(
    client: HixlTransferChannelClient,
    native_sdk: SimpleNamespace,
    local: list[TransferChannelAddress],
    remote: list[TransferChannelAddress],
) -> None:
    with pytest.raises(ValueError):
        client.submit_read(local, remote)
    native_sdk.engine.transfer_async.assert_not_called()


def test_duplicate_and_removed_clients_leave_connection_owned_by_context(
    context: HixlTransferChannelContext,
    client: HixlTransferChannelClient,
    native_sdk: SimpleNamespace,
) -> None:
    duplicate = context.create_client(_PEER)
    assert context.register_client(_PEER.advertise_url, duplicate) is client
    assert context.get_transfer_channel_client(_PEER.advertise_url) is client
    with pytest.raises(RuntimeError):
        duplicate.submit_read(_LOCAL, _REMOTE)
    native_sdk.engine.connect.assert_called_once()
    context.remove_transfer_channel_client(_PEER.advertise_url)
    assert context.get_num_connected_clients() == 0
    with pytest.raises(RuntimeError):
        client.submit_read(_LOCAL, _REMOTE)
    native_sdk.engine.disconnect.assert_not_called()


def test_close_waits_for_query_and_rejects_late_registration(
    context: HixlTransferChannelContext,
    client: HixlTransferChannelClient,
    native_sdk: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize must not race a native call held by a registered Client."""
    candidate = context.create_client(_PEER)
    task = client.submit_read(_LOCAL, _REMOTE)
    querying, closing, release = Event(), Event(), Event()
    original_close = client.close

    def blocked_query(req: int) -> _Status:
        querying.set()
        assert release.wait(10), "test did not release native query"
        return _Status.COMPLETED

    def observed_close() -> None:
        closing.set()
        original_close()

    native_sdk.engine.get_transfer_status.side_effect = blocked_query
    monkeypatch.setattr(client, "close", observed_close)
    with ThreadPoolExecutor(max_workers=2) as executor:
        query = executor.submit(client.query_status, task)
        try:
            assert querying.wait(5)
            shutdown = executor.submit(context.close)
            assert closing.wait(5)
            with pytest.raises(RuntimeError):
                context.register_client(_PEER.advertise_url, candidate)
            with pytest.raises(RuntimeError):
                candidate.submit_read(_LOCAL, _REMOTE)
            # Give shutdown time to progress while the native query stays blocked.
            with pytest.raises(TimeoutError):
                shutdown.result(timeout=0.2)
            native_sdk.engine.finalize.assert_not_called()
        finally:
            release.set()
        assert query.result(timeout=5).is_finished()
        shutdown.result(timeout=5)
    native_sdk.engine.finalize.assert_called_once()
    assert context.get_num_connected_clients() == 0


def test_registration_failure_finalizes_engine_without_publishing_context(
    native_sdk: SimpleNamespace,
    channel_url: str,
) -> None:
    native_sdk.engine.register_mem.side_effect = _HixlError("registration failed")
    try:
        with pytest.raises(_HixlError):
            initialize_transfer_channel_context(
                "hixl",
                L1MemoryDesc(ptr=0x10000, size=4096, align_bytes=256),
                channel_url,
                channel_url,
            )
        native_sdk.engine.finalize.assert_called_once()
        native_sdk.engine.deregister_mem.assert_not_called()
        with pytest.raises(RuntimeError):
            get_transfer_channel_context()
    finally:
        delete_transfer_channel_context()
