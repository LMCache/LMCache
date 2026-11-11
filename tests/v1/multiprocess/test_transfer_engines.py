# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-chunk transfer engine selection.

Covers which engine a chunk's L1 object calls for, how an object group's
chunks are split into same-engine runs, and that
``transfer_kv_per_object_group`` hands each run to its engine. The engines
are stubbed, so these run without a CUDA build.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.multiprocess.object_group_transfer import (
    gpu_dma,
    nvme_dma,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    select_engine,
    split_into_runs,
)
import lmcache.lmcache_native as lmcache_native

GPU = TransferEngine.GPU_DMA
NVME = TransferEngine.NVME_DMA


def _host() -> MemoryObj:
    return MagicMock(spec=MemoryObj)


def _storage() -> MemoryObj:
    return MagicMock(spec=GDSMemoryObject)


def _objects(pattern: str) -> list[MemoryObj | None]:
    """``h`` host object, ``s`` storage-backed object, ``-`` missing."""
    make = {"h": _host, "s": _storage, "-": lambda: None}
    return [make[c]() for c in pattern]


def test_engine_follows_the_object_type():
    assert select_engine(_host()) is GPU
    assert select_engine(_storage()) is NVME


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("", []),
        ("hhhh", [(GPU, 0, 4)]),
        ("ssss", [(NVME, 0, 4)]),
        ("hhsshh", [(GPU, 0, 2), (NVME, 2, 4), (GPU, 4, 6)]),
        ("shs", [(NVME, 0, 1), (GPU, 1, 2), (NVME, 2, 3)]),
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


def _transfer(monkeypatch, memory_objs):
    """Run the dispatcher with both engines stubbed; return the stubs."""
    gpu, nvme = MagicMock(), MagicMock()
    monkeypatch.setattr(gpu_dma, "transfer", gpu)
    monkeypatch.setattr(nvme_dma, "transfer", nvme)
    transfer_kv_per_object_group(
        MagicMock(),
        block_ids_gpu=[MagicMock()],
        memory_objs=memory_objs,
        object_group_id=0,
        batch_size=4,
        skip_first_n_tokens=0,
        direction=lmcache_native.TransferDirection.H2D,
        transfer_key="test",
        block_ids_host=[[1, 2]],
    )
    return gpu, nvme


def _ranges(stub: MagicMock) -> list[tuple[int, int]]:
    return [call.kwargs["chunk_range"] for call in stub.call_args_list]


def test_single_engine_group_is_one_transfer_over_every_chunk(monkeypatch):
    objs = _objects("hhhhh")
    gpu, nvme = _transfer(monkeypatch, objs)
    assert _ranges(gpu) == [(0, 5)]
    nvme.assert_not_called()
    # The engine gets the whole list, so chunk positions stay absolute.
    assert gpu.call_args.args[2] is objs


def test_mixed_group_sends_each_run_to_its_engine(monkeypatch):
    gpu, nvme = _transfer(monkeypatch, _objects("hhsshhs"))
    assert _ranges(gpu) == [(0, 2), (4, 6)]
    assert _ranges(nvme) == [(2, 4), (6, 7)]


def test_empty_group_runs_no_engine(monkeypatch):
    gpu, nvme = _transfer(monkeypatch, [])
    gpu.assert_not_called()
    nvme.assert_not_called()
