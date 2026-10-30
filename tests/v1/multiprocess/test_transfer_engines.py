# SPDX-License-Identifier: Apache-2.0
"""Per-chunk engine selection: object type to engine, the split into runs,
and the copy path of each run. Copy paths are stubbed, so no CUDA build is
needed."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    dispatch,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    select_engine,
    split_into_runs,
    stage,
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


def test_storing_into_an_object_a_peer_owns_is_rejected():
    with pytest.raises(ValueError, match="retrieved"):
        stage(
            NIC,
            [_peer()],
            [MagicMock()],
            lmcache_native.TransferDirection.D2H,
        )


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
