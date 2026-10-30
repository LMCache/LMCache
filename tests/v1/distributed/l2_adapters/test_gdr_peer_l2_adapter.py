# SPDX-License-Identifier: Apache-2.0
"""GDR peer adapter: borrow a peer's DRAM objects as remote handles read at
retrieve time (mocked RPC client and transfer channel)."""

# Standard
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest
import torch

# First Party
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
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    ShadowL1ManagerConfig,
    StorageManagerConfig,
    with_shadow_l1,
)
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.gdr_peer_l2_adapter import (
    GdrPeerL2Adapter,
    GdrPeerL2AdapterConfig,
)
from lmcache.v1.distributed.memory_manager import ShadowL1MemoryManager
from lmcache.v1.distributed.p2p import P2PClient
from lmcache.v1.memory_management import BorrowedMemoryObj

PAGE = 4096
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([PAGE])], dtypes=[torch.uint8])
TTL = 300


def _key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=f"hash{i}".encode(), model_name="m", kv_rank=0)


class _FakeFuture:
    def __init__(self, value=None):
        self._value = value

    def result(self, timeout=None):
        return self._value


@contextmanager
def _adapter(load_timeout_s: float = 10.0, read_ttl_s: float = TTL):
    req_client = MagicMock(spec=P2PClient)
    req_client.p2p_lookup_and_lock.return_value = _FakeFuture(7)
    req_client.p2p_unlock_objects.return_value = _FakeFuture(None)
    tc_ctx = MagicMock()
    tc_client = MagicMock()
    tc_ctx.get_transfer_channel_client.return_value = tc_client
    with (
        patch.object(p2p_mod, "P2PClient", return_value=req_client),
        patch(
            "lmcache.v1.distributed.l2_adapters.gdr_peer_l2_adapter."
            "get_transfer_channel_context",
            return_value=tc_ctx,
        ),
        patch.object(p2p_mod, "PeriodicEventNotifier") as mock_pen,
    ):
        mock_pen.get.return_value = MagicMock()
        config = GdrPeerL2AdapterConfig(
            peer_mq_server_url="grpc://peer:5555",
            peer_transfer_channel_server_url="peer:7600",
            lookup_timeout_s=10.0,
            load_timeout_s=load_timeout_s,
        )
        config.read_ttl_s = read_ttl_s
        adapter = GdrPeerL2Adapter(config)
        try:
            yield adapter, req_client, tc_ctx, tc_client
        finally:
            adapter.close()


def _lookup(adapter, req_client, keys, addresses):
    """Run one lookup to completion and return its task id."""
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: LAYOUT})
    req_client.p2p_query_lookup_results.return_value = _FakeFuture(addresses)
    bitmap = adapter.query_lookup_and_lock_result(task_id)
    assert bitmap is not None
    return task_id, bitmap


def _wait_unlocks(req_client, count: int) -> None:
    deadline = time.monotonic() + 5
    while req_client.p2p_unlock_objects.call_count < count:
        assert time.monotonic() < deadline, "unlock RPC was not sent"
        time.sleep(0.005)


def _load(adapter, keys):
    """Run the borrower's load over fresh shadow placeholders; the bound ones."""
    _, objs = ShadowL1MemoryManager().allocate(LAYOUT, len(keys))
    bitmap = adapter.query_load_result(adapter.submit_load_task(keys, objs))
    assert bitmap is not None
    return {
        key: obj
        for i, (key, obj) in enumerate(zip(keys, objs, strict=True))
        if bitmap.test(i)
    }


def test_hits_become_remote_handles_that_read_into_a_device_region() -> None:
    """A lookup hit is borrowed as a remote handle; reading it submits one
    transfer-channel read from the peer's address into the given region."""
    with _adapter() as (adapter, req_client, _tc_ctx, tc_client):
        keys = [_key(0), _key(1)]
        addresses = [
            TransferChannelAddress(offset=PAGE, size=PAGE),
            TransferChannelAddress(offset=-1, size=0),
        ]
        _, bitmap = _lookup(adapter, req_client, keys, addresses)
        assert bitmap.test(0) and not bitmap.test(1)

        objects = _load(adapter, keys)
        assert list(objects) == [keys[0]]
        obj = objects[keys[0]]
        assert isinstance(obj, BorrowedMemoryObj) and obj.is_remote
        assert obj.is_valid()
        assert obj.get_size() == PAGE
        assert obj.get_shapes() == LAYOUT.shapes
        assert obj.raw_tensor is None
        assert adapter.get_memory_tags() == MemoryTags(
            MemorySegment.DRAM, MemoryResidency.REMOTE
        )
        assert adapter.get_local_borrow_count() == 1

        tc_client.submit_read.return_value = 42
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=False
        )
        read = obj.read_into(3, 2 * PAGE)
        tc_client.submit_read.assert_called_once_with(
            [TransferChannelAddress(offset=2 * PAGE, size=PAGE)],
            [addresses[0]],
            local_region=3,
        )
        assert not read.is_finished()
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=True, succeeded_mask=[True]
        )
        assert read.is_finished() and read.succeeded()

        # The controller returns the keys that became no object: the miss had
        # no reservation, so nothing is sent.
        adapter.submit_unlock([keys[1]])
        req_client.p2p_unlock_objects.assert_not_called()
        obj.release()
        _wait_unlocks(req_client, 1)
        req_client.p2p_unlock_objects.assert_called_once_with([keys[0]])
        assert not obj.is_valid()
        assert adapter.get_local_borrow_count() == 0


def test_a_timed_out_read_fails_only_once_its_cancel_is_confirmed() -> None:
    """A read past its deadline is cancelled, and stays unfinished until the
    transport confirms it stopped: it could land in a reused staging slot."""
    with _adapter(load_timeout_s=0.01) as (adapter, req_client, _tc_ctx, tc_client):
        key = _key(0)
        _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE)],
        )
        obj = _load(adapter, [key])[key]
        tc_client.submit_read.return_value = 1
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=True, succeeded_mask=[False]
        )
        read = obj.read_into(1, 0)
        assert read.is_finished() and not read.succeeded()

        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=False
        )
        tc_client.cancel.return_value = False
        read = obj.read_into(1, 0)
        assert not read.is_finished()
        tc_client.cancel.assert_not_called()
        time.sleep(0.02)
        assert not read.is_finished()
        tc_client.cancel.assert_called_once_with(1)
        # A read in flight keeps the peer's transfer-channel client.
        with pytest.raises(RuntimeError, match="reads in flight"):
            adapter.close()
        tc_client.cancel.return_value = True
        time.sleep(0.02)
        assert read.is_finished() and not read.succeeded()
        obj.release()


def test_a_transport_error_fails_the_read_once_it_is_stopped() -> None:
    """The read is cancelled and no longer counts as in flight, so the peer
    can still be removed."""
    with _adapter() as (adapter, req_client, _tc_ctx, tc_client):
        key = _key(0)
        _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE)],
        )
        obj = _load(adapter, [key])[key]
        tc_client.submit_read.return_value = 1
        tc_client.query_status.side_effect = RuntimeError("backend error")
        tc_client.cancel.return_value = False
        read = obj.read_into(1, 0)
        assert not read.is_finished()
        tc_client.cancel.return_value = True
        assert read.is_finished() and not read.succeeded()
        obj.release()
        _wait_unlocks(req_client, 1)
        adapter.close()


def test_expired_lease_refuses_to_read_and_skips_the_unlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hit whose owner TTL may have lapsed is never read, and no unlock is
    sent for it: the owner's own TTL cleanup covers it."""
    with _adapter(read_ttl_s=30) as (adapter, req_client, _tc_ctx, tc_client):
        key = _key(0)
        _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE)],
        )
        obj = _load(adapter, [key])[key]
        now = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: now + 30.5)
        assert not obj.is_valid()
        read = obj.read_into(1, 0)
        assert read.is_finished() and not read.succeeded()
        tc_client.submit_read.assert_not_called()
        obj.release()
        # The unlock worker sees the same clock: the reservation is past its
        # local expiry, so it leaves the owner's TTL to reclaim it.
        time.sleep(0.05)
        req_client.p2p_unlock_objects.assert_not_called()


def test_an_adopted_lease_outlives_the_placeholder_it_came_with() -> None:
    """The shadow L1 moves a fresh lease onto a resident handle and releases
    the placeholder under the older lease: that returns the older reservation
    only, and the resident handle stays valid and readable under the fresh
    one."""
    with _adapter() as (adapter, req_client, _tc_ctx, tc_client):
        key = _key(0)
        hit = [TransferChannelAddress(offset=0, size=PAGE)]
        _lookup(adapter, req_client, [key], hit)
        resident = _load(adapter, [key])[key]
        _lookup(adapter, req_client, [key], hit)
        placeholder = _load(adapter, [key])[key]
        assert adapter.get_local_borrow_count() == 2
        older = resident.lease
        resident.replace_lease(placeholder.lease)
        placeholder.replace_lease(older)
        placeholder.release()
        _wait_unlocks(req_client, 1)
        assert resident.is_valid()
        tc_client.query_status.return_value = TransferChannelTransferResult(
            finished=True, succeeded_mask=[True]
        )
        assert resident.read_into(3, 0).is_finished()
        tc_client.submit_read.assert_called_once()
        assert adapter.get_local_borrow_count() == 1
        resident.release()
        resident.release()
        _wait_unlocks(req_client, 2)
        assert adapter.get_local_borrow_count() == 0


def test_close_refuses_while_handles_are_live_then_drains() -> None:
    with _adapter() as (adapter, req_client, tc_ctx, _tc_client):
        key = _key(0)
        _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE)],
        )
        obj = _load(adapter, [key])[key]
        with pytest.raises(RuntimeError, match="live borrowed objects"):
            adapter.close()
        tc_ctx.remove_transfer_channel_client.assert_not_called()
        obj.release()
        _wait_unlocks(req_client, 1)
        adapter.close()
        tc_ctx.remove_transfer_channel_client.assert_called_once_with("peer:7600")
        assert adapter.report_status()["local_borrows"] == 0


def test_gdr_mode_adds_a_dram_shadow_l1_once() -> None:
    """The server derives the shadow L1 a GDR borrower needs from the load
    mode: appended after the owned L1s, with their lock TTLs, and left alone
    when the operator configured one."""
    dram = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False),
        read_ttl_seconds=7,
        write_ttl_seconds=9,
    )
    config = StorageManagerConfig(
        l1_manager_config=dram,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[]),
    )
    shadowed = with_shadow_l1(config, MemorySegment.DRAM)
    assert [c.tag for c in shadowed.l1_manager_configs] == ["_default", "shadow.dram"]
    shadow = shadowed.l1_manager_configs[1]
    assert isinstance(shadow, ShadowL1ManagerConfig)
    assert shadow.segment is MemorySegment.DRAM
    assert (shadow.read_ttl_seconds, shadow.write_ttl_seconds) == (7, 9)
    assert with_shadow_l1(shadowed, MemorySegment.DRAM) is shadowed
    assert [c.tag for c in config.l1_manager_configs] == ["_default"]
