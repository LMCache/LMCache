# SPDX-License-Identifier: Apache-2.0
"""The pipelined retrieve: what is queued on the load stream and on the
scatter stream, how the caller's stream is tied to both, and when the
dispatcher uses it. Streams, the engine and the kernel are faked, so these
run without CUDA."""

# Standard
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.multiprocess.object_group_transfer import (
    dispatch,
    staged,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer.dispatch import (
    open_retrieve_streams,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import TransferEngine
from lmcache.v1.platform.base.staging_lane import BankSignals, StagingLane
import lmcache.lmcache_native as lmcache_native

CHUNK = 16
BLOCK = 4
SLOTS = 2
H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H
CALLER = "caller"
SCATTER = "scatter"


class _Slot:
    """A staging slot that reports ``(bank, slot)`` as its address."""

    def __init__(self, bank: str, slot: int) -> None:
        self.address = (bank, slot)

    def data_ptr(self) -> tuple[str, int]:
        return self.address


class _Bank:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_temp_kernel_group_buffer(self, batch_idx: int, kernel_group: int):
        return _Slot(self._name, batch_idx)

    def get_temp_object_group_buffer(self, batch_idx: int, object_group: int):
        return _Slot(self._name, batch_idx)


class _LoggedSignals(BankSignals):
    def __init__(self, log: list[tuple]) -> None:
        self._log = log

    def wait_free(self, bank: int, stream: object) -> None:
        self._log.append(("wait_free", bank, stream))

    def wait_free_on_host(self, bank: int) -> None:
        self._log.append(("wait_free_on_host", bank))

    def signal_staged(self, bank: int, stream: object) -> None:
        self._log.append(("signal_staged", bank, stream))

    def wait_staged(self, bank: int, stream: object) -> None:
        self._log.append(("wait_staged", bank, stream))

    def signal_free(self, bank: int, stream: object) -> None:
        self._log.append(("signal_free", bank, stream))


@pytest.fixture
def harness(monkeypatch):
    """A cache context with one kernel group, a lane with two banks, and a
    log of the signals plus ``("stage", stream, chunk names, slots)`` and
    ``("kernel", stream, slots, block ids)`` entries. ``stream`` is the
    stream that was current when the fake ran."""
    log: list[tuple] = []
    current = [CALLER]

    @contextmanager
    def use_stream(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous

    cc = MagicMock()
    cc.lmcache_tokens_per_chunk = CHUNK
    cc.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[SimpleNamespace(kernel_group_indices=[0])],
        get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda og: True),
        get_subchunk_sw_size_tokens=lambda kg: CHUNK,
    )
    cc.calculate_num_blocks.side_effect = lambda tokens, kg: tokens // BLOCK

    def fake_stage(engine, memory_objs, buffers, direction):
        log.append(
            (
                "stage",
                current[0],
                [obj.name for obj in memory_objs],
                [buffer.address for buffer in buffers],
            )
        )

    def fake_kernel(pointers, buffers, block_ids, *rest):
        log.append(("kernel", current[0], buffers, block_ids.tolist()))

    ops = MagicMock()
    ops.multi_layer_block_kv_transfer.side_effect = fake_kernel
    monkeypatch.setattr(staged, "device_ops", ops)
    monkeypatch.setattr(staged, "stage", fake_stage)
    monkeypatch.setattr(staged, "host_loader", lambda engine, objs: None)
    monkeypatch.setattr(staged, "torch_dev", SimpleNamespace(stream=use_stream))
    lane = StagingLane("load", [_Bank("a"), _Bank("b")], _LoggedSignals(log))
    return SimpleNamespace(cc=cc, lane=lane, log=log, ops=ops)


def _objects(count: int) -> list[MemoryObj | None]:
    objs: list[MemoryObj | None] = []
    for index in range(count):
        obj = MagicMock(spec=MemoryObj)
        obj.name = f"c{index}"
        objs.append(obj)
    return objs


def _run(harness, objs, skip_first_n_tokens=0) -> None:
    staged.run_staged_loop(
        harness.cc,
        [torch.arange(len(objs) * (CHUNK // BLOCK))],
        objs,
        0,
        SLOTS,
        skip_first_n_tokens,
        H2D,
        engine=TransferEngine.NVME_DMA,
        lane=harness.lane,
        scatter_stream=SCATTER,
    )


def test_batches_alternate_banks_and_split_over_the_two_streams(harness):
    _run(harness, _objects(3))
    assert harness.log == [
        ("wait_free", 0, "load"),
        ("stage", "load", ["c0", "c1"], [("a", 0), ("a", 1)]),
        ("signal_staged", 0, "load"),
        ("wait_staged", 0, SCATTER),
        ("kernel", SCATTER, [("a", 0), ("a", 1)], list(range(0, 8))),
        ("signal_free", 0, SCATTER),
        ("wait_free", 1, "load"),
        ("stage", "load", ["c2"], [("b", 0)]),
        ("signal_staged", 1, "load"),
        ("wait_staged", 1, SCATTER),
        ("kernel", SCATTER, [("b", 0)], list(range(8, 12))),
        ("signal_free", 1, SCATTER),
    ]


def test_bank_rotation_carries_over_to_the_next_call(harness):
    """A request that ends on bank 0 leaves bank 1 to the next request."""
    _run(harness, _objects(1))
    harness.log.clear()
    _run(harness, _objects(1))
    assert [entry[:2] for entry in harness.log if entry[0] == "wait_free"] == [
        ("wait_free", 1)
    ]


def test_chunk_before_the_token_skip_is_not_loaded_but_keeps_its_slot(harness):
    _run(harness, _objects(2), skip_first_n_tokens=CHUNK)
    assert ("stage", "load", ["c1"], [("a", 1)]) in harness.log


def test_bank_is_freed_when_the_kernel_launch_fails(harness):
    harness.ops.multi_layer_block_kv_transfer.side_effect = RuntimeError("launch")
    with pytest.raises(RuntimeError):
        _run(harness, _objects(1))
    assert harness.log[-2:] == [
        ("wait_staged", 0, SCATTER),
        ("signal_free", 0, SCATTER),
    ]


def test_failed_load_signals_nothing_and_launches_no_kernel(harness, monkeypatch):
    def failing_stage(engine, memory_objs, buffers, direction):
        raise ValueError("size mismatch")

    monkeypatch.setattr(staged, "stage", failing_stage)
    with pytest.raises(ValueError):
        _run(harness, _objects(1))
    assert harness.log == [("wait_free", 0, "load")]


@pytest.fixture
def host_harness(harness, monkeypatch):
    """The harness with an engine that loads on host threads; a load is
    logged as ``("load_from_host", chunk names, slots)``."""
    harness.load_error = None

    def load_from_host(memory_objs, buffers, device):
        if harness.load_error is not None:
            raise harness.load_error
        harness.log.append(
            (
                "load_from_host",
                [obj.name for obj in memory_objs],
                [buffer.address for buffer in buffers],
            )
        )

    loader = SimpleNamespace(load_from_host=load_from_host)
    monkeypatch.setattr(staged, "host_loader", lambda engine, objs: loader)
    return harness


def test_host_loaded_batch_is_in_its_bank_before_anything_is_queued_for_it(
    host_harness,
):
    """No stream may wait for a host thread, so the host waits for the bank,
    loads it, and only then queues the batch on the scatter stream. Nothing
    goes on the lane's stream."""
    _run(host_harness, _objects(3))
    assert host_harness.log == [
        ("wait_free_on_host", 0),
        ("load_from_host", ["c0", "c1"], [("a", 0), ("a", 1)]),
        ("signal_staged", 0, SCATTER),
        ("wait_staged", 0, SCATTER),
        ("kernel", SCATTER, [("a", 0), ("a", 1)], list(range(0, 8))),
        ("signal_free", 0, SCATTER),
        ("wait_free_on_host", 1),
        ("load_from_host", ["c2"], [("b", 0)]),
        ("signal_staged", 1, SCATTER),
        ("wait_staged", 1, SCATTER),
        ("kernel", SCATTER, [("b", 0)], list(range(8, 12))),
        ("signal_free", 1, SCATTER),
    ]


def test_failed_host_load_is_raised_and_queues_nothing_for_its_batch(host_harness):
    host_harness.load_error = OSError("short slab read")
    with pytest.raises(OSError, match="short slab read"):
        _run(host_harness, _objects(1))
    assert host_harness.log == [("wait_free_on_host", 0)]


@pytest.fixture
def stream_events(monkeypatch):
    """Fake the platform stream helpers for the dispatcher; returns
    the log of ``(recorded on, waited on)`` stream pairs."""
    log: list[tuple] = []

    def record_completion_event(device, stream):
        return SimpleNamespace(wait_on=lambda waiter: log.append((stream, waiter)))

    monkeypatch.setattr(
        dispatch,
        "platform_stream",
        SimpleNamespace(
            current_stream=lambda device: CALLER,
            record_completion_event=record_completion_event,
        ),
    )
    return log


def _lane_context() -> MagicMock:
    """A cache context whose lanes run on a stream named after the lane."""
    cc = MagicMock()
    cc.get_scatter_stream.return_value = SCATTER
    cc.get_staging_lane.side_effect = lambda name, file_io: StagingLane(
        name, [], MagicMock()
    )
    return cc


def test_scatter_stream_is_gated_once_and_every_used_stream_is_joined(
    stream_events,
):
    cc = _lane_context()
    with open_retrieve_streams(cc, H2D) as streams:
        assert stream_events == []
        streams.lane(TransferEngine.GPU_DMA)
        streams.lane(TransferEngine.NVME_DMA)
        streams.lane(TransferEngine.GPU_DMA)
        assert stream_events == [(CALLER, SCATTER)]
    assert stream_events[1:] == [
        ("gpu_dma", CALLER),
        ("nvme_dma", CALLER),
        (SCATTER, CALLER),
    ]


def test_streams_are_joined_when_the_transfer_raises(stream_events):
    with pytest.raises(ValueError):
        with open_retrieve_streams(_lane_context(), H2D) as streams:
            streams.lane(TransferEngine.NVME_DMA)
            raise ValueError("missing object")
    assert stream_events[1:] == [("nvme_dma", CALLER), (SCATTER, CALLER)]


def test_call_that_stages_nothing_touches_no_stream(stream_events):
    with open_retrieve_streams(_lane_context(), H2D):
        pass
    assert stream_events == []


def test_only_the_file_engine_asks_for_file_io_banks(stream_events):
    cc = _lane_context()
    with open_retrieve_streams(cc, H2D) as streams:
        streams.lane(TransferEngine.GPU_DMA)
        streams.lane(TransferEngine.NVME_DMA)
    assert [call.kwargs["file_io"] for call in cc.get_staging_lane.call_args_list] == [
        False,
        True,
    ]


def _dispatch(monkeypatch, stream_events, direction, direct_copy=False):
    """Run the dispatcher on a context with a scatter stream, over a
    host-memory chunk followed by a slab chunk; return each run's path."""
    calls: list[tuple] = []
    monkeypatch.setattr(dispatch, "batch_memcpy_available", lambda: direct_copy)
    monkeypatch.setattr(dispatch, "direct_transfer_supported", lambda *a: True)
    monkeypatch.setattr(dispatch, "native_plan_available", lambda: True)
    monkeypatch.setattr(
        dispatch, "run_direct_transfer", lambda *a, **k: calls.append(("direct",))
    )
    monkeypatch.setattr(
        dispatch,
        "run_object_group_transfer_plan",
        lambda *a, **k: calls.append(("plan",)),
    )
    monkeypatch.setattr(
        dispatch,
        "run_staged_loop",
        lambda *a, **k: calls.append(
            (
                "staged",
                k["engine"].value,
                k["lane"] and k["lane"].stream,
                k["scatter_stream"],
            )
        ),
    )
    transfer_kv_per_object_group(
        _lane_context(),
        block_ids_gpu=[MagicMock()],
        memory_objs=[MagicMock(spec=MemoryObj), MagicMock(spec=GDSMemoryObject)],
        object_group_id=0,
        batch_size=4,
        skip_first_n_tokens=0,
        direction=direction,
        transfer_key="test",
        block_ids_host=[[1, 2]],
    )
    return calls


def test_retrieve_stages_every_run_on_its_engines_lane(monkeypatch, stream_events):
    assert _dispatch(monkeypatch, stream_events, H2D) == [
        ("staged", "gpu_dma", "gpu_dma", SCATTER),
        ("staged", "nvme_dma", "nvme_dma", SCATTER),
    ]
    assert stream_events[-1] == (SCATTER, CALLER)


def test_retrieve_keeps_the_direct_copy_for_host_memory(monkeypatch, stream_events):
    assert _dispatch(monkeypatch, stream_events, H2D, direct_copy=True) == [
        ("direct",),
        ("staged", "nvme_dma", "nvme_dma", SCATTER),
    ]


def test_store_stays_on_the_callers_stream(monkeypatch, stream_events):
    assert _dispatch(monkeypatch, stream_events, D2H) == [
        ("plan",),
        ("staged", "nvme_dma", None, None),
    ]
    assert stream_events == []
