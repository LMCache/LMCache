# SPDX-License-Identifier: Apache-2.0
"""Retrieve of borrowed remote objects through the GDR landing ring.

Everything native and device-side is faked, so these run without CUDA. They
check the ordering contract of ``run_borrowed_transfer``: a bank is acquired
before reads land in it, the kernel is queued only after every read of the
batch finished, local objects in the same batch are staged the ordinary way,
and banks alternate.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gdr_context import LandingSlot
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    PendingRead,
    RemoteBorrowedMemoryObj,
)
import lmcache.lmcache_native as lmcache_native
import lmcache.v1.multiprocess.object_group_transfer as mod

CHUNK = 16
OBJ_BYTES = 256
SLOTS = 2


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


class _Ring:
    """Two banks of ``SLOTS`` slots over one CPU tensor; records the protocol."""

    def __init__(self, log: list[str]) -> None:
        self.buffer = torch.zeros(2 * SLOTS * OBJ_BYTES, dtype=torch.uint8)
        self.log = log
        self.slots_per_bank = SLOTS

    def slot(self, bank: int, index: int) -> LandingSlot:
        offset = (bank * SLOTS + index) * OBJ_BYTES
        return LandingSlot(9, offset, self.buffer[offset : offset + OBJ_BYTES])

    def acquire(self, bank: int) -> None:
        self.log.append(f"acquire:{bank}")

    def release(self, bank: int, device, stream) -> None:
        self.log.append(f"release:{bank}")


def _remote(log: list[str], name: str, polls: int = 2, ok: bool = True):
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

    return RemoteBorrowedMemoryObj(
        meta, lease_is_valid=lambda: True, on_release=lambda: None, read_into=read_into
    )


def _local(size: int = OBJ_BYTES):
    obj = MagicMock()
    obj.get_size.return_value = size
    return obj


@pytest.fixture
def harness(monkeypatch):
    log: list[str] = []
    ring = _Ring(log)
    cc = MagicMock()
    cc.gdr_landing = ring
    cc.lmcache_tokens_per_chunk = CHUNK
    cc.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[SimpleNamespace(kernel_group_indices=[0, 1])],
        get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda og: True),
        get_subchunk_sw_size_tokens=lambda kg: CHUNK,
    )
    cc.get_kernel_group_offset_in_object.side_effect = lambda og, kg: kg * 64
    cc.calculate_num_blocks.side_effect = lambda tokens, kg: tokens // 4
    cc.device = torch.device("cpu")
    ops = MagicMock()
    ops.multi_layer_block_kv_transfer.side_effect = lambda *a, **k: log.append("kernel")
    monkeypatch.setattr(mod, "device_ops", ops)
    monkeypatch.setattr(
        mod,
        "lmcache_memcpy_async_h2d",
        lambda obj, buf: log.append(f"h2d:{buf.nbytes}"),
    )
    monkeypatch.setattr(mod, "_READ_POLL_INTERVAL_S", 0)
    return cc, ring, ops, log


def _run(cc, objs, skip_first_n_tokens: int = 0):
    block_ids = [torch.arange(len(objs) * CHUNK // 4) for _ in range(2)]
    mod.run_borrowed_transfer(
        cc,
        block_ids,
        objs,
        0,
        skip_first_n_tokens,
        lmcache_native.TransferDirection.H2D,
    )


def test_reads_land_in_alternating_banks_before_each_kernel(harness) -> None:
    cc, ring, ops, log = harness
    objs = [_remote(log, "a"), _remote(log, "b"), _remote(log, "c")]
    _run(cc, objs)
    assert log == [
        "acquire:0",
        f"read:a:9:{0 * OBJ_BYTES}",
        f"read:b:9:{1 * OBJ_BYTES}",
        "done:a",
        "done:b",
        "kernel",
        "kernel",
        "release:0",
        "acquire:1",
        f"read:c:9:{2 * OBJ_BYTES}",
        "done:c",
        "kernel",
        "kernel",
        "release:1",
    ]
    # The kernel reads each slot at the kernel group's offset within the object.
    first_call = ops.multi_layer_block_kv_transfer.call_args_list[0].args
    second_call = ops.multi_layer_block_kv_transfer.call_args_list[1].args
    base = ring.buffer.data_ptr()
    assert first_call[1] == [base, base + OBJ_BYTES]
    assert second_call[1] == [base + 64, base + OBJ_BYTES + 64]


def test_local_objects_in_a_borrowed_batch_are_staged_normally(harness) -> None:
    cc, _ring, _ops, log = harness
    _run(cc, [_local(), _remote(log, "r")])
    assert log[:3] == ["acquire:0", f"h2d:{OBJ_BYTES}", f"read:r:9:{OBJ_BYTES}"]
    assert log[3:] == ["done:r", "kernel", "kernel", "release:0"]


def test_failed_read_raises_before_the_kernel(harness) -> None:
    cc, _ring, ops, log = harness
    with pytest.raises(RuntimeError, match="failed"):
        _run(cc, [_remote(log, "ok"), _remote(log, "bad", ok=False)])
    ops.multi_layer_block_kv_transfer.assert_not_called()


def test_missing_object_and_wrong_direction_are_rejected(harness) -> None:
    cc, _ring, _ops, log = harness
    with pytest.raises(ValueError, match="None"):
        _run(cc, [_remote(log, "a"), None])
    with pytest.raises(ValueError, match="retrieved"):
        mod.run_borrowed_transfer(
            cc, [], [_remote(log, "a")], 0, 0, lmcache_native.TransferDirection.D2H
        )


def test_no_landing_ring_is_an_error(harness) -> None:
    cc, _ring, _ops, log = harness
    cc.gdr_landing = None
    with pytest.raises(RuntimeError, match="landing ring"):
        _run(cc, [_remote(log, "a")])


def test_fully_skipped_batches_touch_no_slot(harness) -> None:
    cc, _ring, _ops, log = harness
    objs = [_remote(log, "a"), _remote(log, "b"), _remote(log, "c")]
    _run(cc, objs, skip_first_n_tokens=2 * CHUNK)
    # The skipped batch never took a bank, so the remaining chunk lands in bank 0.
    assert [e for e in log if e.startswith("read")] == ["read:c:9:0"]
    assert log.count("acquire:0") == 1 and log.count("acquire:1") == 0


def test_dispatch_routes_remote_objects_to_the_borrowed_path(
    harness, monkeypatch
) -> None:
    cc, _ring, _ops, log = harness
    called = []
    monkeypatch.setattr(
        mod, "run_borrowed_transfer", lambda *a, **k: called.append(a[2])
    )
    monkeypatch.setattr(mod, "_HAS_BATCH_MEMCPY_ASYNC", True)
    objs = [_remote(log, "a")]
    mod.transfer_kv_per_object_group(
        cc,
        [],
        objs,
        0,
        4,
        0,
        lmcache_native.TransferDirection.H2D,
        transfer_key="k",
        block_ids_host=[[1]],
    )
    assert called == [objs]
    assert not mod.direct_transfer_supported(cc, 0, objs, [[1]])
