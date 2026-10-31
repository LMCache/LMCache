# SPDX-License-Identifier: Apache-2.0
"""GDR peer adapter: lend a peer's DRAM objects as remote handles read at
retrieve time (mocked RPC client and transfer channel)."""

# Standard
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import gc
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.config import get_type_name_for_config
from lmcache.v1.distributed.l2_adapters.gdr_peer_l2_adapter import (
    GdrPeerL2Adapter,
    GdrPeerL2AdapterConfig,
    TransferChannelRegionRegistrar,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)
from lmcache.v1.memory_management import RemoteBorrowedMemoryObj
from lmcache.v1.multiprocess.transport.base import RequestClient

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
def _adapter(load_timeout_s: float = 10.0):
    req_client = MagicMock(spec=RequestClient)
    req_client.p2p_lookup_and_lock.return_value = _FakeFuture(7)
    req_client.p2p_unlock_objects.return_value = _FakeFuture(None)
    tc_ctx = MagicMock()
    tc_client = MagicMock()
    tc_ctx.get_transfer_channel_client.return_value = tc_client
    with (
        patch.object(p2p_mod.RequestClientFactory, "create", return_value=req_client),
        patch(
            "lmcache.v1.distributed.l2_adapters.gdr_peer_l2_adapter."
            "get_transfer_channel_context",
            return_value=tc_ctx,
        ),
        patch.object(p2p_mod, "PeriodicEventNotifier") as mock_pen,
    ):
        mock_pen.get.return_value = MagicMock()
        adapter = GdrPeerL2Adapter(
            GdrPeerL2AdapterConfig(
                peer_mq_server_url="tcp://peer:5555",
                peer_transfer_channel_server_url="peer:7600",
                lookup_timeout_s=10.0,
                load_timeout_s=load_timeout_s,
            )
        )
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


def test_config_is_the_p2p_config_bound_to_the_dram_shadow() -> None:
    config = GdrPeerL2AdapterConfig.from_dict(
        {
            "type": "p2p_gdr",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
        }
    )
    assert isinstance(config, GdrPeerL2AdapterConfig)
    assert get_type_name_for_config(config) == "p2p_gdr"
    assert config.get_wanted_l1_memory_tags() == (
        MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE),
    )
    assert "GPU" in GdrPeerL2AdapterConfig.help()


def test_hits_become_remote_handles_that_read_into_a_device_region() -> None:
    """A lookup hit is lent as a remote handle; reading it submits one
    transfer-channel read from the peer's address into the given region."""
    with _adapter() as (adapter, req_client, _tc_ctx, tc_client):
        keys = [_key(0), _key(1)]
        addresses = [
            TransferChannelAddress(offset=PAGE, size=PAGE, read_ttl_seconds=TTL),
            TransferChannelAddress(offset=-1, size=0),
        ]
        task_id, bitmap = _lookup(adapter, req_client, keys, addresses)
        assert bitmap.test(0) and not bitmap.test(1)

        objects = adapter.take_borrowed_objects(task_id, keys, {0: LAYOUT})
        assert list(objects) == [keys[0]]
        obj = objects[keys[0]]
        assert isinstance(obj, RemoteBorrowedMemoryObj)
        assert obj.is_valid()
        assert obj.get_size() == PAGE
        assert obj.get_shapes() == LAYOUT.shapes
        assert obj.tensor is None
        assert adapter.get_memory_tags() == MemoryTags(
            MemorySegment.DRAM, MemoryResidency.REMOTE
        )
        assert adapter.get_active_borrow_count() == 1

        tc_client.submit_read.return_value = 42
        tc_client.query_read_status.return_value = TransferChannelReadResult(
            finished=False
        )
        read = obj.read_into(3, 2 * PAGE)
        tc_client.submit_read.assert_called_once_with(
            [TransferChannelAddress(offset=2 * PAGE, size=PAGE)],
            [addresses[0]],
            local_region=3,
        )
        assert not read.is_finished()
        tc_client.query_read_status.return_value = TransferChannelReadResult(
            finished=True, succeeded_mask=[True]
        )
        assert read.is_finished() and read.succeeded()

        adapter.release_lookup(task_id, keys)
        req_client.p2p_unlock_objects.assert_not_called()
        obj.release()
        _wait_unlocks(req_client, 1)
        req_client.p2p_unlock_objects.assert_called_once_with([keys[0]])
        assert not obj.is_valid()
        assert adapter.get_active_borrow_count() == 0


def test_failed_or_timed_out_read_reports_failure() -> None:
    with _adapter(load_timeout_s=0.01) as (adapter, req_client, _tc_ctx, tc_client):
        key = _key(0)
        task_id, _ = _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE, read_ttl_seconds=TTL)],
        )
        obj = adapter.take_borrowed_objects(task_id, [key], {0: LAYOUT})[key]
        tc_client.submit_read.return_value = 1
        tc_client.query_read_status.return_value = TransferChannelReadResult(
            finished=True, succeeded_mask=[False]
        )
        read = obj.read_into(1, 0)
        assert read.is_finished() and not read.succeeded()

        tc_client.query_read_status.return_value = TransferChannelReadResult(
            finished=False
        )
        read = obj.read_into(1, 0)
        time.sleep(0.02)
        assert read.is_finished() and not read.succeeded()
        obj.release()


def test_expired_lease_refuses_to_read_and_skips_the_unlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hit whose owner TTL may have lapsed is never read, and no unlock is
    sent for it: the owner's own TTL cleanup covers it."""
    with _adapter() as (adapter, req_client, _tc_ctx, tc_client):
        key = _key(0)
        task_id, _ = _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE, read_ttl_seconds=10)],
        )
        obj = adapter.take_borrowed_objects(task_id, [key], {0: LAYOUT})[key]
        now = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: now + 9.5)
        assert not obj.is_valid()
        read = obj.read_into(1, 0)
        assert read.is_finished() and not read.succeeded()
        tc_client.submit_read.assert_not_called()
        obj.release()
        # The unlock worker sees the same clock: the reservation is past its
        # local expiry, so it leaves the owner's TTL to reclaim it.
        time.sleep(0.05)
        req_client.p2p_unlock_objects.assert_not_called()


def test_hits_without_an_owner_ttl_are_misses() -> None:
    """A peer that reports no read TTL cannot lend: the lease would be
    unbounded. Its reservation is returned at once."""
    with _adapter() as (adapter, req_client, _tc_ctx, _tc_client):
        key = _key(0)
        _, bitmap = _lookup(
            adapter, req_client, [key], [TransferChannelAddress(offset=0, size=PAGE)]
        )
        assert not bitmap.test(0)
        _wait_unlocks(req_client, 1)


def test_dropped_handle_is_released_by_its_finalizer() -> None:
    with _adapter() as (adapter, req_client, _tc_ctx, _tc_client):
        key = _key(0)
        task_id, _ = _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE, read_ttl_seconds=TTL)],
        )
        objects = adapter.take_borrowed_objects(task_id, [key], {0: LAYOUT})
        assert adapter.get_active_borrow_count() == 1
        del objects
        gc.collect()
        _wait_unlocks(req_client, 1)
        assert adapter.get_active_borrow_count() == 0


def test_close_refuses_while_handles_are_live_then_drains() -> None:
    with _adapter() as (adapter, req_client, tc_ctx, _tc_client):
        key = _key(0)
        task_id, _ = _lookup(
            adapter,
            req_client,
            [key],
            [TransferChannelAddress(offset=0, size=PAGE, read_ttl_seconds=TTL)],
        )
        obj = adapter.take_borrowed_objects(task_id, [key], {0: LAYOUT})[key]
        with pytest.raises(RuntimeError, match="live borrowed objects"):
            adapter.close()
        tc_ctx.remove_transfer_channel_client.assert_not_called()
        obj.release()
        _wait_unlocks(req_client, 1)
        adapter.close()
        tc_ctx.remove_transfer_channel_client.assert_called_once_with("peer:7600")
        assert adapter.report_status()["active_borrows"] == 0


def test_region_registrar_forwards_to_the_transfer_context() -> None:
    context = MagicMock()
    context.register_device_region.return_value = 5
    registrar = TransferChannelRegionRegistrar(context, align_bytes=PAGE)
    assert registrar.align_bytes == PAGE
    assert registrar.register(0x1000, 2 * PAGE, 1) == 5
    context.register_device_region.assert_called_once_with(0x1000, 2 * PAGE, 1)
    registrar.unregister(5)
    context.unregister_device_region.assert_called_once_with(5)
