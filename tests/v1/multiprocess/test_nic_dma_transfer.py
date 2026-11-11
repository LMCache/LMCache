# SPDX-License-Identifier: Apache-2.0
"""Retrieve of objects a peer owns through the cache context's landing banks.

Everything native and device-side is faked, so these run without CUDA. They
check the ordering contract of the NIC DMA engine: the stream is drained
before the first read, the kernel is queued only after every read of the
batch finished, a bank's last kernel is waited for before reads land in it
again, and banks alternate.
"""

# Standard
from itertools import count
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    PendingRead,
)
from lmcache.v1.multiprocess.object_group_transfer import nic_dma
from lmcache.v1.platform.base.cache_context import LandingSlot
import lmcache.lmcache_native as lmcache_native

CHUNK = 16
OBJ_BYTES = 256
SLOTS = 2
REGION = 9
H2D = lmcache_native.TransferDirection.H2D


class _Read(PendingRead):
    """Finishes after ``polls`` polls, with the given outcome."""

    def __init__(self, polls: int, ok: bool, log: list[str], name: str) -> None:
        self._left = polls
        self._ok = ok
        self._log = log
        self._name = name

    def is_finished(self) -> bool:
        self._left -= 1
        if self._left <= 0:
            self._log.append(f"done:{self._name}")
            return True
        return False

    def succeeded(self) -> bool:
        return self._ok


class _Landing:
    """Two banks of ``SLOTS`` object-group slots over one CPU tensor."""

    def __init__(self) -> None:
        self.buffer = torch.zeros(2 * SLOTS * OBJ_BYTES, dtype=torch.uint8)

    def slot(self, bank: int, index: int, object_group: int) -> LandingSlot:
        assert object_group == 0
        offset = (bank * SLOTS + index) * OBJ_BYTES
        return LandingSlot(REGION, offset, self.buffer[offset : offset + OBJ_BYTES])


class _Event:
    """A completion event that is complete when polled; logs the poll."""

    def __init__(self, log: list[str], name: int) -> None:
        self._log = log
        self._name = name

    def is_complete(self) -> bool:
        self._log.append(f"wait:{self._name}")
        return True


def _peer_object(log: list[str], name: str, polls: int = 2, ok: bool = True):
    meta = MemoryObjMetadata(
        shape=torch.Size([OBJ_BYTES]),
        dtype=torch.uint8,
        address=0x1000,
        phy_size=OBJ_BYTES,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([OBJ_BYTES])],
        dtypes=[torch.uint8],
    )

    def read_into(region_id: int, offset: int) -> PendingRead:
        log.append(f"read:{name}:{region_id}:{offset}")
        return _Read(polls, ok, log, name)

    obj = BorrowedMemoryObj(meta)
    obj.bind(
        (lambda: True, lambda: None),
        read_into=read_into,
        address=meta.address,
        source="peer",
    )
    return obj


@pytest.fixture
def harness(monkeypatch):
    log: list[str] = []
    landing = _Landing()
    cc = MagicMock()
    cc.landing_banks = 2
    cc.max_batch_size = SLOTS
    cc.get_landing_slot.side_effect = landing.slot
    cc.lmcache_tokens_per_chunk = CHUNK
    cc.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[SimpleNamespace(kernel_group_indices=[0, 1])],
        get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda og: True),
        get_subchunk_sw_size_tokens=lambda kg: CHUNK,
    )
    cc.get_kernel_group_offset_in_object.side_effect = lambda og, kg: kg * 64
    cc.calculate_num_blocks.side_effect = lambda tokens, kg: tokens // 4
    cc.device = torch.device("cpu")
    cc.stream = None
    ops = MagicMock()
    ops.multi_layer_block_kv_transfer.side_effect = lambda *a, **k: log.append("kernel")
    monkeypatch.setattr(nic_dma, "device_ops", ops)
    events = count()

    def record(device, stream) -> _Event:
        name = next(events)
        log.append(f"record:{name}")
        return _Event(log, name)

    monkeypatch.setattr(nic_dma, "record_completion_event", record)
    return cc, landing, ops, log


def _run(cc, objs, skip_first_n_tokens: int = 0, chunk_range=None, direction=H2D):
    block_ids = [torch.arange(len(objs) * CHUNK // 4) for _ in range(2)]
    nic_dma.transfer(
        cc,
        block_ids,
        objs,
        0,
        SLOTS,
        skip_first_n_tokens,
        direction,
        transfer_key="test",
        block_ids_host=[],
        chunk_range=chunk_range or (0, len(objs)),
    )


def test_reads_land_in_alternating_banks_before_each_kernel(harness) -> None:
    cc, landing, ops, log = harness
    _run(cc, [_peer_object(log, name) for name in "abcde"])
    assert log == [
        # The stream is drained before anything lands in the banks.
        "record:0",
        "wait:0",
        f"read:a:{REGION}:{0 * OBJ_BYTES}",
        f"read:b:{REGION}:{1 * OBJ_BYTES}",
        "done:a",
        "done:b",
        "kernel",
        "kernel",
        "record:1",
        # Bank 1 was never used: no wait before its reads.
        f"read:c:{REGION}:{2 * OBJ_BYTES}",
        f"read:d:{REGION}:{3 * OBJ_BYTES}",
        "done:c",
        "done:d",
        "kernel",
        "kernel",
        "record:2",
        # Bank 0 again: its kernel (event 1) must have finished first.
        "wait:1",
        f"read:e:{REGION}:{0 * OBJ_BYTES}",
        "done:e",
        "kernel",
        "kernel",
        "record:3",
    ]
    # The kernel reads each slot at the kernel group's offset within the object.
    first_call = ops.multi_layer_block_kv_transfer.call_args_list[0].args
    second_call = ops.multi_layer_block_kv_transfer.call_args_list[1].args
    base = landing.buffer.data_ptr()
    assert first_call[1] == [base, base + OBJ_BYTES]
    assert second_call[1] == [base + 64, base + OBJ_BYTES + 64]


def test_only_the_chunk_range_is_read_at_its_own_block_ids(harness) -> None:
    cc, _landing, ops, log = harness
    objs = [None, None, None, _peer_object(log, "d"), _peer_object(log, "e"), None]
    _run(cc, objs, chunk_range=(3, 5))
    assert [e for e in log if e.startswith("read")] == [
        f"read:d:{REGION}:0",
        f"read:e:{REGION}:{OBJ_BYTES}",
    ]
    # Chunks 3 and 4 own blocks 12..19 of each kernel group (4 per chunk).
    block_ids = ops.multi_layer_block_kv_transfer.call_args_list[0].args[2]
    assert block_ids.tolist() == list(range(12, 20))


def test_failed_read_raises_before_the_kernel(harness) -> None:
    """A failure aborts the transfer, but only once every read of the batch
    settled: an abandoned read would land in the bank when it is reused."""
    cc, _landing, ops, log = harness
    with pytest.raises(RuntimeError, match="failed"):
        _run(
            cc,
            [_peer_object(log, "slow", polls=4), _peer_object(log, "bad", ok=False)],
        )
    ops.multi_layer_block_kv_transfer.assert_not_called()
    assert "done:slow" in log


def test_object_not_owned_by_a_peer_is_rejected_after_started_reads_settle(
    harness,
) -> None:
    cc, _landing, ops, log = harness
    with pytest.raises(ValueError, match="peer owns"):
        _run(cc, [_peer_object(log, "a", polls=3), None])
    assert "done:a" in log
    ops.multi_layer_block_kv_transfer.assert_not_called()


def test_store_direction_is_rejected(harness) -> None:
    cc, _landing, _ops, log = harness
    with pytest.raises(ValueError, match="retrieved"):
        _run(
            cc, [_peer_object(log, "a")], direction=lmcache_native.TransferDirection.D2H
        )
    assert log == []


def test_context_without_a_landing_buffer_is_an_error(harness) -> None:
    cc, _landing, _ops, log = harness
    cc.landing_banks = 0
    with pytest.raises(RuntimeError, match="landing buffer"):
        _run(cc, [_peer_object(log, "a")])
    cc.get_landing_slot.assert_not_called()


def test_fully_skipped_batches_touch_no_slot(harness) -> None:
    cc, _landing, _ops, log = harness
    objs = [_peer_object(log, name) for name in "abc"]
    _run(cc, objs, skip_first_n_tokens=2 * CHUNK)
    # The skipped batch never took a bank, so the remaining chunk lands in bank 0.
    assert [e for e in log if e.startswith("read")] == [f"read:c:{REGION}:0"]
    cc.get_landing_slot.assert_called_once_with(0, 0, 0)
