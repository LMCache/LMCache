# SPDX-License-Identifier: Apache-2.0
"""All ranks and object groups must be ready before a shared prefix is served."""

# Standard
from collections.abc import Callable

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_orchestrator.api import RegionUsage

pytestmark = pytest.mark.no_shared_allocator

Factory = Callable[[str], StorageManager]
Usage = Callable[[], RegionUsage]
LAYOUTS = [
    MemoryLayoutDesc([torch.Size([2, 16])], [torch.float16]),
    MemoryLayoutDesc([torch.Size([4, 8])], [torch.float32]),
]


def _rows(tp: int, state_window: int = -1) -> list[GroupedObjectKeys]:
    """Use the serving key encoding for two groups and two chunks per rank."""
    return [
        GroupedObjectKeys(
            [
                ObjectKey(
                    ObjectKey.IntHash2Bytes(chunk + 1),
                    "shared-tp",
                    ObjectKey.ComputeKVRank(tp, rank, tp, rank),
                    group,
                )
                for chunk in range(2)
            ],
            group,
            layout,
            sliding_window_size=state_window if group == 1 else -1,
        )
        for group, layout in enumerate(LAYOUTS)
        for rank in range(tp)
    ]


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("state_window", [-1, 1])
def test_shared_prefix_waits_for_every_rank_and_group(
    make_store: Factory, usage: Usage, tp: int, state_window: int
) -> None:
    """A late shard truncates all rows; overlapping lookups keep separate leases."""
    writer, reader = make_store("writer"), make_store("reader")
    rows = _rows(tp, state_window)
    delayed = {}
    for index, row in enumerate(rows):
        reserved = writer.reserve_write(row.keys, row.layout_desc)
        assert len(reserved) == 2
        for obj in reserved.values():
            assert obj.tensor is not None
            obj.tensor.fill_(index + 1)
        if index == len(rows) - 1:
            delayed[row.keys[-1]] = reserved.pop(row.keys[-1])
        writer.finish_write_by_owner(writer.prepare_write_completion(reserved))

    short = reader.query_prefetch_status(
        reader.submit_prefetch_task(PrefetchTaskSpec(rows), skip_l2=True)
    )
    assert short is not None
    assert [bitmap.get_indices_list() for bitmap in short.hit_cells] == [[0]] * len(
        rows
    )
    writer.finish_write_by_owner(writer.prepare_write_completion(delayed))
    full = reader.query_prefetch_status(
        reader.submit_prefetch_task(PrefetchTaskSpec(rows), skip_l2=True)
    )
    assert full is not None
    assert [bitmap.get_indices_list() for bitmap in full.hit_cells] == [
        [1] if row.sliding_window_size == 1 else [0, 1] for row in rows
    ]

    first_keys = [row.keys[0] for row in rows]
    reader.finish_read_prefetched(first_keys, l1_owners=short.l1_owners)
    # Finishing the short lookup cannot release the overlapping full lookup.
    for index, row in enumerate(rows):
        retained = full.hit_cells[index].gather(row.keys)
        with reader.read_prefetched_results(retained, full.l1_owners) as objects:
            assert objects is not None and len(objects) == len(retained)
            assert all(torch.all(obj.tensor == index + 1) for obj in objects)
        reader.finish_read_prefetched(retained, l1_owners=full.l1_owners)
    assert usage().read_leases == 0


def test_peer_layout_mismatch_releases_every_reserved_read(
    make_store: Factory, usage: Usage
) -> None:
    """A differently configured decoder must not copy an incompatible extent."""
    writer, reader = make_store("writer"), make_store("reader")
    row = _rows(1)[0]
    reserved = writer.reserve_write(row.keys, row.layout_desc)
    writer.finish_write_by_owner(writer.prepare_write_completion(reserved))
    incompatible = GroupedObjectKeys(row.keys, 0, LAYOUTS[1])
    with pytest.raises(ValueError, match="Shared L1 layout differs"):
        reader.submit_prefetch_task(PrefetchTaskSpec([incompatible]), skip_l2=True)
    assert usage().read_leases == 0
