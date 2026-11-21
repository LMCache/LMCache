# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine under the staged loop: ordering of stream sync, reads and
kernels, and its error paths. Native and device code is faked, so these run
without CUDA."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.transfer_channel import (
    disable_staging_buffers,
    enable_staging_buffers,
    register_staging_buffer,
    unregister_staging_buffer,
)
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    PendingRead,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    nic_dma,
    staged,
    transfer_unfull_chunk,
    unfull_chunk,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import TransferEngine
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
):
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
    staging = _Staging()
    cc = MagicMock()
    cc.get_temp_object_group_buffer.side_effect = staging.object_group_buffer
    cc.get_temp_kernel_group_buffer.side_effect = staging.kernel_group_buffer
    context = MagicMock()
    context.register_memory_region.return_value = REGION
    enable_staging_buffers(context, 1)
    staging.region = register_staging_buffer(staging.buffer, None)
    disable_staging_buffers()
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
    monkeypatch.setattr(unfull_chunk, "device_ops", ops)
    monkeypatch.setattr(
        nic_dma, "synchronize_stream", lambda device, stream: log.append("sync")
    )
    yield cc, staging, ops, log
    unregister_staging_buffer(staging.region)


def _run(cc, objs):
    block_ids = [torch.arange(len(objs) * CHUNK // 4) for _ in range(2)]
    staged.run_staged_loop(
        cc,
        block_ids,
        objs,
        0,
        SLOTS,
        0,
        H2D,
        engine=TransferEngine.NIC_DMA,
        chunk_range=(0, len(objs)),
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


def test_failed_read_raises_before_the_kernel(harness) -> None:
    """A failure aborts the transfer only after every read of the batch
    settled."""
    cc, _staging, ops, log = harness
    with pytest.raises(RuntimeError, match="failed"):
        _run(
            cc,
            [_peer_object(log, "slow", polls=4), _peer_object(log, "bad", ok=False)],
        )
    ops.multi_layer_block_kv_transfer.assert_not_called()
    assert "done:slow" in log


def test_unregistered_staging_buffer_is_an_error(harness) -> None:
    cc, staging, _ops, log = harness
    unregister_staging_buffer(staging.region)
    with pytest.raises(RuntimeError, match="staging buffer registered"):
        _run(cc, [_peer_object(log, "a")])
    assert log == []


def _run_unfull(cc, memory_obj, direction=H2D):
    transfer_unfull_chunk(
        cc,
        [torch.arange(4) for _ in range(2)],
        [0, 0],
        [0, 0],
        memory_obj,
        0,
        cc.get_temp_object_group_buffer(0, 0),
        direction,
    )


def test_unfull_chunk_a_peer_owns_is_read_before_its_kernels(harness) -> None:
    cc, _staging, _ops, log = harness
    _run_unfull(cc, _peer_object(log, "tail"))
    assert log == ["sync", f"read:tail:{REGION}:0", "done:tail", "kernel", "kernel"]
