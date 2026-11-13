# SPDX-License-Identifier: Apache-2.0
"""Retrieve of objects a peer owns through the cache context's staging buffer.

Everything native and device-side is faked, so these run without CUDA. They
check the ordering contract of the NIC DMA engine: before the reads of a
batch the stream is synchronized, which covers the previous batch's kernel,
and the kernel is queued only after every read of the batch finished.
"""

# Standard
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
from lmcache.v1.multiprocess.object_group_transfer import nic_dma, staged
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


class _Staging:
    """``SLOTS`` object-group slots over one CPU tensor; a kernel group is
    64 bytes into its object."""

    def __init__(self) -> None:
        self.buffer = torch.zeros(SLOTS * OBJ_BYTES, dtype=torch.uint8)

    def object_group_buffer(self, index: int, object_group: int) -> torch.Tensor:
        assert object_group == 0
        return self.buffer[index * OBJ_BYTES : (index + 1) * OBJ_BYTES]

    def kernel_group_buffer(self, index: int, kernel_group: int) -> torch.Tensor:
        return self.buffer[index * OBJ_BYTES + kernel_group * 64 :]


def _peer_object(
    log: list[str],
    name: str,
    polls: int = 2,
    ok: bool = True,
    size: int = OBJ_BYTES,
    bound: bool = True,
):
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.uint8,
        address=0x1000,
        phy_size=size,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([OBJ_BYTES])],
        dtypes=[torch.uint8],
    )

    def read_into(region_id: int, offset: int) -> PendingRead:
        log.append(f"read:{name}:{region_id}:{offset}")
        return _Read(polls, ok, log, name)

    obj = BorrowedMemoryObj(meta)
    if not bound:
        return obj
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
    staging = _Staging()
    cc = MagicMock()
    cc.get_temp_object_group_buffer.side_effect = staging.object_group_buffer
    cc.get_temp_kernel_group_buffer.side_effect = staging.kernel_group_buffer
    nic_dma.set_staging_region(
        cc,
        nic_dma.StagingRegion(REGION, staging.buffer.data_ptr(), staging.buffer.nbytes),
    )
    cc.lmcache_tokens_per_chunk = CHUNK
    cc.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[SimpleNamespace(kernel_group_indices=[0, 1])],
        get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda og: True),
        get_subchunk_sw_size_tokens=lambda kg: CHUNK,
    )
    cc.calculate_num_blocks.side_effect = lambda tokens, kg: tokens // 4
    cc.device = torch.device("cpu")
    cc.stream = None
    ops = MagicMock()
    ops.multi_layer_block_kv_transfer.side_effect = lambda *a, **k: log.append("kernel")
    monkeypatch.setattr(staged, "device_ops", ops)
    monkeypatch.setattr(
        nic_dma, "synchronize_stream", lambda device, stream: log.append("sync")
    )
    return cc, staging, ops, log


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


def test_reads_land_in_idle_staging_slots_before_each_kernel(harness) -> None:
    cc, staging, ops, log = harness
    _run(cc, [_peer_object(log, name) for name in "abc"])
    assert log == [
        # Whatever an earlier transfer queued has left the staging buffer.
        "sync",
        f"read:a:{REGION}:{0 * OBJ_BYTES}",
        f"read:b:{REGION}:{1 * OBJ_BYTES}",
        "done:a",
        "done:b",
        "kernel",
        "kernel",
        # The same slots again: the kernel reading them must have finished.
        "sync",
        f"read:c:{REGION}:{0 * OBJ_BYTES}",
        "done:c",
        "kernel",
        "kernel",
    ]
    # The kernel reads each kernel group where it sits in the staged object.
    first_call = ops.multi_layer_block_kv_transfer.call_args_list[0].args
    second_call = ops.multi_layer_block_kv_transfer.call_args_list[1].args
    base = staging.buffer.data_ptr()
    assert first_call[1] == [base, base + OBJ_BYTES]
    assert second_call[1] == [base + 64, base + OBJ_BYTES + 64]


def test_only_the_chunk_range_is_read_at_its_own_block_ids(harness) -> None:
    cc, _staging, ops, log = harness
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
    settled: an abandoned read would land in the staging buffer when it is
    reused."""
    cc, _staging, ops, log = harness
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
    cc, _staging, ops, log = harness
    with pytest.raises(ValueError, match="peer owns"):
        _run(cc, [_peer_object(log, "a", polls=3), None])
    assert "done:a" in log
    ops.multi_layer_block_kv_transfer.assert_not_called()


def test_store_direction_is_rejected(harness) -> None:
    cc, _staging, _ops, log = harness
    with pytest.raises(ValueError, match="retrieved"):
        _run(
            cc, [_peer_object(log, "a")], direction=lmcache_native.TransferDirection.D2H
        )
    assert log == []


def test_unregistered_staging_buffer_is_an_error(harness) -> None:
    cc, _staging, _ops, log = harness
    nic_dma.clear_staging_region(cc)
    with pytest.raises(RuntimeError, match="staging buffer registered"):
        _run(cc, [_peer_object(log, "a")])
    assert log == []


def test_fully_skipped_batches_start_no_read(harness) -> None:
    cc, _staging, _ops, log = harness
    objs = [_peer_object(log, name) for name in "abc"]
    _run(cc, objs, skip_first_n_tokens=2 * CHUNK)
    assert [e for e in log if e.startswith("read")] == [f"read:c:{REGION}:0"]


def test_chunks_before_the_first_written_token_are_not_read(harness) -> None:
    """A batch that starts inside the skipped prefix reads only the chunks
    the kernel writes; the kernel still gets the batch's skip."""
    cc, _staging, ops, log = harness
    _run(cc, [_peer_object(log, name) for name in "ab"], skip_first_n_tokens=CHUNK + 4)
    assert [e for e in log if e.startswith("read")] == [f"read:b:{REGION}:{OBJ_BYTES}"]
    # 20 skipped tokens are 5 blocks of the batch.
    assert ops.multi_layer_block_kv_transfer.call_args_list[0].args[-1] == 5


def test_sliding_window_group_reads_only_the_window(harness) -> None:
    cc, _staging, ops, log = harness
    cc.kv_layer_groups_manager.get_attn_desc = lambda: SimpleNamespace(
        is_full_attention=lambda og: False, num_chunks_in_sw=[2]
    )
    objs = [None, None, None, _peer_object(log, "d"), _peer_object(log, "e")]
    _run(cc, objs)
    assert [e for e in log if e.startswith("read")] == [
        f"read:d:{REGION}:0",
        f"read:e:{REGION}:{OBJ_BYTES}",
    ]
    block_ids = ops.multi_layer_block_kv_transfer.call_args_list[0].args[2]
    assert block_ids.tolist() == list(range(12, 20))


def test_error_starting_a_read_is_raised_after_started_reads_settle(harness) -> None:
    """The error of the read that could not start is the one reported, even
    when an earlier read of the batch also fails."""
    cc, _staging, ops, log = harness
    objs = [
        _peer_object(log, "a", polls=3, ok=False),
        _peer_object(log, "b", bound=False),
    ]
    with pytest.raises(RuntimeError, match="not bound"):
        _run(cc, objs)
    assert "done:a" in log
    ops.multi_layer_block_kv_transfer.assert_not_called()


def test_object_that_does_not_fit_its_slot_is_rejected(harness) -> None:
    cc, _staging, _ops, log = harness
    with pytest.raises(ValueError, match="Size mismatch"):
        _run(cc, [_peer_object(log, "a", size=OBJ_BYTES // 2)])
    assert log == ["sync"]


def test_slot_outside_the_registered_region_is_rejected(harness) -> None:
    cc, staging, _ops, log = harness
    nic_dma.set_staging_region(
        cc, nic_dma.StagingRegion(REGION, staging.buffer.data_ptr(), OBJ_BYTES)
    )
    with pytest.raises(ValueError, match="outside the registered region"):
        _run(cc, [_peer_object(log, name) for name in "ab"])
    # The first chunk's slot is inside the region and its read is awaited.
    assert "done:a" in log


def test_slow_reads_are_logged_once_and_still_awaited(harness, monkeypatch) -> None:
    cc, _staging, ops, log = harness
    monkeypatch.setattr(nic_dma, "_SLOW_READ_WARNING_S", 0.0)
    warnings = MagicMock()
    monkeypatch.setattr(nic_dma.logger, "warning", warnings)
    _run(cc, [_peer_object(log, "a", polls=5)])
    warnings.assert_called_once()
    assert "done:a" in log
    ops.multi_layer_block_kv_transfer.assert_called()
