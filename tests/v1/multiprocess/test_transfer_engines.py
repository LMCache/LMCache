# SPDX-License-Identifier: Apache-2.0
"""Per-chunk engine selection: object type to engine, the split into runs,
and the copy path of each run. Copy paths are stubbed, so no CUDA build is
needed."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    dispatch,
    engines,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    run_step,
    select_engine,
    split_into_runs,
    stage,
)
from lmcache.v1.multiprocess.object_group_transfer.transfer_step import (
    TransferStep,
    collect_off_stream_steps,
    hand_off_step,
)
import lmcache.lmcache_native as lmcache_native

GPU = TransferEngine.GPU_DMA
NVME = TransferEngine.NVME_DMA
NIC = TransferEngine.NIC_DMA


def _host() -> MemoryObj:
    return MagicMock(spec=MemoryObj)


def _storage() -> MemoryObj:
    return MagicMock(spec=GDSMemoryObject)


def _peer() -> MemoryObj:
    return MagicMock(spec=BorrowedMemoryObj)


def _objects(pattern: str) -> list[MemoryObj | None]:
    """``h`` host object, ``s`` storage-backed object, ``p`` object a peer
    owns, ``-`` missing."""
    make = {"h": _host, "s": _storage, "p": _peer, "-": lambda: None}
    return [make[c]() for c in pattern]


def test_engine_follows_the_object_type():
    assert select_engine(_host()) is GPU
    assert select_engine(_storage()) is NVME
    assert select_engine(_peer()) is NIC


def test_staging_an_object_a_peer_owns_is_not_implemented():
    with pytest.raises(NotImplementedError):
        stage(
            NIC,
            [_peer()],
            [MagicMock()],
            lmcache_native.TransferDirection.H2D,
        )


def _step(
    memory_objs: list, block_ids: list[torch.Tensor], calls: list
) -> TransferStep:
    """A step that records what it is asked, with one region per object."""

    def targets(staging: object) -> list:
        calls.append(("targets", staging))
        return [f"region-{index}" for index in range(len(memory_objs))]

    def launch(staging: object, ids: list[torch.Tensor]) -> None:
        calls.append(("launch", staging, ids))

    return TransferStep(memory_objs, block_ids, targets, launch)


@pytest.mark.parametrize("engine, make", [(GPU, _host), (NVME, _storage)])
def test_an_on_stream_engine_stages_a_step_then_launches_it(monkeypatch, engine, make):
    """The step lands in the context's own staging and its kernels are queued
    right after, with the step's block ids; nothing is handed off."""
    calls: list = []
    monkeypatch.setattr(
        engines,
        "stage",
        lambda *args: calls.append(("stage", *args)),
    )
    context, objs, ids = MagicMock(), [make()], [torch.arange(2)]
    collected: list[TransferStep] = []
    with collect_off_stream_steps(collected):
        run_step(engine, context, _step(objs, ids, calls))
    assert calls == [
        ("targets", context),
        ("stage", engine, objs, ["region-0"], lmcache_native.TransferDirection.H2D),
        ("launch", context, ids),
    ]
    assert collected == []


def test_an_off_stream_engine_hands_a_step_to_the_collector_untouched():
    """Its bytes land by a path the stream does not order, so the step is
    neither staged nor launched here. Without a collector the caller cannot
    run it, as before the step interface."""
    calls: list = []
    objs = [_peer()]
    step = _step(objs, [torch.arange(2)], calls)
    with pytest.raises(NotImplementedError):
        run_step(NIC, MagicMock(), step)
    collected: list[TransferStep] = []
    with collect_off_stream_steps(collected):
        run_step(NIC, MagicMock(), step)
    assert [taken.memory_objs for taken in collected] == [objs]
    assert calls == []


def _handed_off(block_ids: list[torch.Tensor]) -> TransferStep:
    return TransferStep(
        [_peer()], block_ids, lambda staging: [], lambda staging, ids: None
    )


def test_steps_of_one_staging_share_one_copy_of_the_block_ids():
    """The batches of a transfer pass the same staged block ids: they are
    copied once, and restaging the buffer afterwards does not reach the copy."""
    buffer = torch.arange(8)
    staged_ids = [buffer[:4], buffer[4:]]
    collected: list[TransferStep] = []
    with collect_off_stream_steps(collected):
        hand_off_step(_handed_off(staged_ids))
        hand_off_step(_handed_off(staged_ids))
        buffer.fill_(-1)
    first, second = collected
    assert first.block_ids is second.block_ids
    assert first.block_ids is not staged_ids
    assert [ids.tolist() for ids in first.block_ids] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_restaged_buffer_gets_its_own_copy_of_the_block_ids():
    """A later transfer restages the buffer and passes new views of it: its
    steps get the new ids and the earlier steps keep theirs. A new collect
    block starts without a copy, even for the same staged list."""
    buffer = torch.arange(4)
    first_staging = [buffer[:]]
    collected: list[TransferStep] = []
    with collect_off_stream_steps(collected):
        hand_off_step(_handed_off(first_staging))
        buffer.add_(10)
        hand_off_step(_handed_off([buffer[:]]))
    with collect_off_stream_steps(collected):
        buffer.add_(10)
        hand_off_step(_handed_off(first_staging))
    assert [step.block_ids[0].tolist() for step in collected] == [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("", []),
        ("hhhh", [(GPU, 0, 4)]),
        ("ssss", [(NVME, 0, 4)]),
        ("hhsshh", [(GPU, 0, 2), (NVME, 2, 4), (GPU, 4, 6)]),
        ("shhs", [(NVME, 0, 1), (GPU, 1, 3), (NVME, 3, 4)]),
        ("hppsp", [(GPU, 0, 1), (NIC, 1, 3), (NVME, 3, 4), (NIC, 4, 5)]),
        # A missing chunk joins the run of the next object ...
        ("--ss", [(NVME, 0, 4)]),
        ("hh-ss", [(GPU, 0, 2), (NVME, 2, 5)]),
        # ... or the previous one when nothing follows it.
        ("ss--", [(NVME, 0, 4)]),
        ("---", [(GPU, 0, 3)]),
    ],
)
def test_chunks_split_into_consecutive_same_engine_runs(pattern, expected):
    assert split_into_runs(_objects(pattern)) == expected


def _transfer(monkeypatch, pattern: str) -> list[tuple[str, tuple[int, int]]]:
    """Run the dispatcher with the copy paths stubbed and no direct copy;
    return each run's path and chunk range, in call order."""
    calls: list[tuple[str, tuple[int, int]]] = []
    monkeypatch.setattr(dispatch, "batch_memcpy_available", lambda: False)
    monkeypatch.setattr(dispatch, "native_plan_available", lambda: True)
    monkeypatch.setattr(
        dispatch,
        "run_object_group_transfer_plan",
        lambda *a, **k: calls.append(("plan", k["chunk_range"])),
    )
    monkeypatch.setattr(
        dispatch,
        "run_staged_loop",
        lambda *a, **k: calls.append((k["engine"].value, k["chunk_range"])),
    )
    transfer_kv_per_object_group(
        MagicMock(),
        block_ids_gpu=[MagicMock()],
        memory_objs=_objects(pattern),
        object_group_id=0,
        batch_size=4,
        skip_first_n_tokens=0,
        direction=lmcache_native.TransferDirection.H2D,
        transfer_key="test",
        block_ids_host=[[1, 2]],
    )
    return calls


def test_single_engine_group_is_one_transfer_over_every_chunk(monkeypatch):
    assert _transfer(monkeypatch, "hhhhh") == [("plan", (0, 5))]


def test_empty_group_runs_no_copy_path(monkeypatch):
    assert _transfer(monkeypatch, "") == []


def test_each_run_goes_to_its_engine_in_chunk_order(monkeypatch):
    """Only host memory may use the native plan; the other engines stage one
    batch at a time."""
    assert _transfer(monkeypatch, "hhpppsh") == [
        ("plan", (0, 2)),
        ("nic_dma", (2, 5)),
        ("nvme_dma", (5, 6)),
        ("plan", (6, 7)),
    ]


def test_host_memory_stages_per_batch_without_the_native_plan(monkeypatch):
    calls: list[tuple[str, tuple[int, int]]] = []
    monkeypatch.setattr(dispatch, "batch_memcpy_available", lambda: False)
    monkeypatch.setattr(dispatch, "native_plan_available", lambda: False)
    monkeypatch.setattr(
        dispatch,
        "run_staged_loop",
        lambda *a, **k: calls.append((k["engine"].value, k["chunk_range"])),
    )
    transfer_kv_per_object_group(
        MagicMock(),
        block_ids_gpu=[MagicMock()],
        memory_objs=_objects("hh"),
        object_group_id=0,
        batch_size=4,
        skip_first_n_tokens=0,
        direction=lmcache_native.TransferDirection.H2D,
        transfer_key="test",
    )
    assert calls == [("gpu_dma", (0, 2))]
