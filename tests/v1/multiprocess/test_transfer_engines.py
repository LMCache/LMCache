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
    retrieve_transfer,
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


class _OffStreamEngine:
    """Records what it is given; a test finishes a submission by hand."""

    def __init__(self) -> None:
        self.submitted: list[tuple] = []

    def submit(self, cache_context, steps, on_done) -> None:
        self.submitted.append((cache_context, list(steps), on_done))


@pytest.fixture
def nic_engine(monkeypatch) -> _OffStreamEngine:
    """An off-stream engine for objects a peer owns."""
    engine = _OffStreamEngine()
    monkeypatch.setitem(engines.OFF_STREAM_ENGINES, NIC, engine)
    return engine


@pytest.mark.parametrize("engine, make", [(GPU, _host), (NVME, _storage)])
def test_an_on_stream_step_is_staged_then_launched_and_queued_at_once(
    monkeypatch, engine, make
):
    """Staged into the context's buffer, launched with the step's block ids,
    and reported queued right away."""
    calls: list = []
    monkeypatch.setattr(
        engines,
        "stage",
        lambda *args: calls.append(("stage", *args)),
    )
    context, objs, ids = MagicMock(), [make()], [torch.arange(2)]
    done: list[bool] = []
    with retrieve_transfer() as device_work:
        run_step(engine, context, _step(objs, ids, calls))
        device_work.when_queued(context, True, done.append)
    assert calls == [
        ("targets", context),
        ("stage", engine, objs, ["region-0"], lmcache_native.TransferDirection.H2D),
        ("launch", context, ids),
    ]
    assert done == [True]


@pytest.mark.parametrize("result", [True, False])
def test_an_off_stream_step_is_queued_when_its_engine_says_so(nic_engine, result):
    """The transfer neither stages nor launches it; the engine gets it after
    the transfers ran and reports when it is queued."""
    calls: list = []
    context, objs = MagicMock(), [_peer()]
    done: list[bool] = []
    with retrieve_transfer() as device_work:
        run_step(NIC, context, _step(objs, [torch.arange(2)], calls))
        assert nic_engine.submitted == []
        device_work.when_queued(context, True, done.append)
    ((submitted_context, steps, on_done),) = nic_engine.submitted
    assert submitted_context is context
    assert [step.memory_objs for step in steps] == [objs]
    assert calls == [] and done == []
    on_done(result)
    assert done == [result]


def test_an_off_stream_step_without_an_engine_fails_the_retrieve():
    """Also outside ``retrieve_transfer``, where such a step cannot run at all."""
    step = _step([_peer()], [torch.arange(2)], [])
    with pytest.raises(NotImplementedError):
        run_step(NIC, MagicMock(), step)
    done: list[bool] = []
    with retrieve_transfer() as device_work:
        run_step(NIC, MagicMock(), step)
        device_work.when_queued(MagicMock(), True, done.append)
    assert done == [False]


def test_a_failed_retrieve_gives_its_off_stream_steps_to_no_engine(nic_engine):
    done: list[bool] = []
    with retrieve_transfer() as device_work:
        run_step(NIC, MagicMock(), _step([_peer()], [torch.arange(2)], []))
        device_work.when_queued(MagicMock(), False, done.append)
    assert done == [False]
    assert nic_engine.submitted == []


def _handed_off(block_ids: list[torch.Tensor]) -> TransferStep:
    return TransferStep(
        [_peer()], block_ids, lambda staging: [], lambda staging, ids: None
    )


def _submitted_steps(nic_engine: _OffStreamEngine) -> list[TransferStep]:
    return [step for _, steps, _ in nic_engine.submitted for step in steps]


def test_steps_of_one_staging_share_one_copy_of_the_block_ids(nic_engine):
    """Batches of one staging pass the same block ids: copied once, and
    restaging the buffer does not reach the copy."""
    buffer = torch.arange(8)
    staged_ids = [buffer[:4], buffer[4:]]
    with retrieve_transfer() as device_work:
        run_step(NIC, MagicMock(), _handed_off(staged_ids))
        run_step(NIC, MagicMock(), _handed_off(staged_ids))
        buffer.fill_(-1)
        device_work.when_queued(MagicMock(), True, lambda result: None)
    first, second = _submitted_steps(nic_engine)
    assert first.block_ids is second.block_ids
    assert first.block_ids is not staged_ids
    assert [ids.tolist() for ids in first.block_ids] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_restaged_buffer_gets_its_own_copy_of_the_block_ids(nic_engine):
    """A restaged buffer and a new retrieve each get a fresh copy; earlier
    steps keep theirs."""
    buffer = torch.arange(4)
    first_staging = [buffer[:]]
    with retrieve_transfer() as device_work:
        run_step(NIC, MagicMock(), _handed_off(first_staging))
        buffer.add_(10)
        run_step(NIC, MagicMock(), _handed_off([buffer[:]]))
        device_work.when_queued(MagicMock(), True, lambda result: None)
    with retrieve_transfer() as device_work:
        buffer.add_(10)
        run_step(NIC, MagicMock(), _handed_off(first_staging))
        device_work.when_queued(MagicMock(), True, lambda result: None)
    assert [step.block_ids[0].tolist() for step in _submitted_steps(nic_engine)] == [
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
