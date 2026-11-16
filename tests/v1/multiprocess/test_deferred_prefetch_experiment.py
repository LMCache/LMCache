# SPDX-License-Identifier: Apache-2.0
"""Exercise lookup-only prefetch against a full shared DRAM L1 and real FS."""

# Standard
import time

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
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.key_codec import encode_fs_filename
from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.multiprocess.streaming import ReservedArenaAllocator
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator


def test_full_l1_returns_mixed_lookup_without_loading_or_eviction(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=8192, use_lazy=False, align_bytes=4096, shm_name=""
        )
    )
    owner = DramL1Manager(config)
    reserved = ReservedArenaAllocator(owner, 4096)
    disk_config = FSL2AdapterConfig(str(tmp_path))
    disk = FSL2Adapter(disk_config)
    ctrl = PrefetchController(
        [owner],
        [L1ManagerDescriptor(index=0, config=config)],
        [disk],
        [L2AdapterDescriptor(index=0, config=disk_config)],
        DefaultPrefetchPolicy(),
    )
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    hot, cold = [
        ObjectKey(chunk_hash=bytes([i]) * 32, model_name="m", kv_rank=0) for i in (1, 2)
    ]
    submitted = []
    lookup = disk.submit_lookup_and_lock_task

    def capture(keys, layouts):
        submitted.append(list(keys))
        return lookup(keys, layouts)

    monkeypatch.setattr(disk, "submit_lookup_and_lock_task", capture)
    try:
        written = owner.reserve_write([hot], [False], layout, "hot")
        assert written[hot][0] == L1Error.SUCCESS
        written[hot][1].raw_tensor.fill_(11)
        owner.finish_write([hot], tag="hot")
        (tmp_path / encode_fs_filename(hot)).write_bytes(bytes([11]) * 4096)
        (tmp_path / encode_fs_filename(cold)).write_bytes(bytes([22]) * 4096)
        assert owner.get_memory_usage() == (4096, 4096)
        ctrl.start()
        spec = PrefetchTaskSpec(
            [GroupedObjectKeys([hot, cold], 0, layout)], defer_l2_load=True
        )
        req = ctrl.submit_prefetch_request(spec)
        deadline = time.monotonic() + 5
        result = None
        while result is None and time.monotonic() < deadline:
            result = ctrl.query_prefetch_result(req)
            if result is None:
                time.sleep(0.01)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == [0, 1]
        assert result.l1_hit_cells[0].get_indices_list() == [0]
        assert result.l2_hit_cells[0].get_indices_list() == [1]
        assert submitted == [[cold]]
        assert result.l1_owners == {hot: owner.l1_manager_id}
        assert owner.get_memory_usage() == (4096, 4096)
        assert owner.unsafe_read([hot])[hot][0] == L1Error.SUCCESS
        assert owner.reserve_read([cold])[cold][0] == L1Error.KEY_NOT_EXIST
        owner.finish_read([hot])
    finally:
        ctrl.stop()
        disk.close()
        reserved.close()
        owner.close()


def test_lookup_trims_after_gap_and_releases_late_hot_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=12288, use_lazy=False, align_bytes=4096, shm_name=""
        )
    )
    owner = DramL1Manager(config)
    reserved = ReservedArenaAllocator(owner, 4096)
    disk_config = FSL2AdapterConfig(str(tmp_path))
    disk = FSL2Adapter(disk_config)
    ctrl = PrefetchController(
        [owner],
        [L1ManagerDescriptor(index=0, config=config)],
        [disk],
        [L2AdapterDescriptor(index=0, config=disk_config)],
        DefaultPrefetchPolicy(),
    )
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    hot, cold, missing, later = [
        ObjectKey(chunk_hash=bytes([i]) * 32, model_name="m", kv_rank=0)
        for i in (1, 2, 3, 4)
    ]
    try:
        written = owner.reserve_write([hot, later], [False, False], layout, "hot")
        assert all(written[k][0] == L1Error.SUCCESS for k in (hot, later))
        owner.finish_write([hot, later], tag="hot")
        (tmp_path / encode_fs_filename(cold)).write_bytes(bytes([22]) * 4096)
        ctrl.start()
        req = ctrl.submit_prefetch_request(
            PrefetchTaskSpec(
                [GroupedObjectKeys([hot, cold, missing, later], 0, layout)],
                defer_l2_load=True,
            )
        )
        deadline = time.monotonic() + 5
        result = None
        while result is None and time.monotonic() < deadline:
            result = ctrl.query_prefetch_result(req)
            if result is None:
                time.sleep(0.01)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == [0, 1]
        assert result.l1_owners == {hot: owner.l1_manager_id}
        assert owner.unsafe_read([hot])[hot][0] == L1Error.SUCCESS
        assert owner.unsafe_read([later])[later][0] == L1Error.KEY_NOT_READABLE
        owner.finish_read([hot])
        assert owner.get_memory_usage() == (8192, 8192)
    finally:
        ctrl.stop()
        disk.close()
        reserved.close()
        owner.close()


def test_deferred_lookup_requires_prefix_and_read_locks():
    # First Party
    from lmcache.v1.distributed.api import PrefetchLockMode

    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    key = ObjectKey(chunk_hash=b"x", model_name="m", kv_rank=0)
    row = GroupedObjectKeys([key], 0, layout)
    with pytest.raises(ValueError, match="requires prefix"):
        PrefetchTaskSpec([row], fetching_policy="full", defer_l2_load=True)
    with pytest.raises(ValueError, match="read locks"):
        PrefetchTaskSpec([row], lock_mode=PrefetchLockMode.NO_LOCK, defer_l2_load=True)


def test_concurrent_deferred_lookups_progress_when_l1_is_full(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=8192, use_lazy=False, align_bytes=4096, shm_name=""
        )
    )
    owner = DramL1Manager(config)
    reserved = ReservedArenaAllocator(owner, 4096)
    disk_config = FSL2AdapterConfig(str(tmp_path))
    disk = FSL2Adapter(disk_config)
    ctrl = PrefetchController(
        [owner],
        [L1ManagerDescriptor(index=0, config=config)],
        [disk],
        [L2AdapterDescriptor(index=0, config=disk_config)],
        DefaultPrefetchPolicy(),
    )
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    hot, cold = [
        ObjectKey(chunk_hash=bytes([i]) * 32, model_name="m", kv_rank=0) for i in (1, 2)
    ]
    try:
        assert (
            owner.reserve_write([hot], [False], layout, "hot")[hot][0]
            == L1Error.SUCCESS
        )
        owner.finish_write([hot], tag="hot")
        (tmp_path / encode_fs_filename(cold)).write_bytes(bytes([22]) * 4096)
        ctrl.start()
        spec = PrefetchTaskSpec(
            [GroupedObjectKeys([hot, cold], 0, layout)], defer_l2_load=True
        )
        pending = {ctrl.submit_prefetch_request(spec) for _ in range(8)}
        results = []
        deadline = time.monotonic() + 5
        while pending and time.monotonic() < deadline:
            for req in list(pending):
                result = ctrl.query_prefetch_result(req)
                if result is not None:
                    results.append(result)
                    pending.remove(req)
            time.sleep(0.01)
        assert not pending
        assert len(results) == 8
        assert all(x.hit_cells[0].get_indices_list() == [0, 1] for x in results)
        assert owner.get_memory_usage() == (4096, 4096)
        for _ in results:
            owner.finish_read([hot])
    finally:
        ctrl.stop()
        disk.close()
        reserved.close()
        owner.close()


@pytest.mark.parametrize("hot_positions", [[(0, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)]])
def test_multi_rank_lookup_stops_at_missing_shard_and_releases_locks(
    tmp_path, monkeypatch, hot_positions
):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=16384, use_lazy=False, align_bytes=4096, shm_name=""
        )
    )
    owner = DramL1Manager(config)
    reserved = ReservedArenaAllocator(owner, 4096)
    disk_config = FSL2AdapterConfig(str(tmp_path))
    disk = FSL2Adapter(disk_config)
    ctrl = PrefetchController(
        [owner],
        [L1ManagerDescriptor(index=0, config=config)],
        [disk],
        [L2AdapterDescriptor(index=0, config=disk_config)],
        DefaultPrefetchPolicy(),
    )
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    rows = [
        [
            ObjectKey(chunk_hash=bytes([i]) * 32, model_name="m", kv_rank=rank)
            for i in (1, 2, 3)
        ]
        for rank in (0, 1)
    ]
    hot = [rows[rank][index] for rank, index in hot_positions]
    cold = [
        rows[rank][index]
        for rank in (0, 1)
        for index in (0, 1)
        if (rank, index) not in hot_positions
    ]
    submitted = []
    lookup = disk.submit_lookup_and_lock_task

    def capture(keys, layouts):
        submitted.append(list(keys))
        return lookup(keys, layouts)

    monkeypatch.setattr(disk, "submit_lookup_and_lock_task", capture)
    try:
        written = owner.reserve_write(hot, [False] * len(hot), layout, "hot")
        assert all(written[k][0] == L1Error.SUCCESS for k in hot)
        owner.finish_write(hot, tag="hot")
        for k in cold + [rows[0][2]]:
            (tmp_path / encode_fs_filename(k)).write_bytes(bytes([22]) * 4096)
        ctrl.start()
        groups = [GroupedObjectKeys(row, 0, layout, num_kv_readers=2) for row in rows]
        req = ctrl.submit_prefetch_request(PrefetchTaskSpec(groups, defer_l2_load=True))
        deadline = time.monotonic() + 5
        result = None
        while result is None and time.monotonic() < deadline:
            result = ctrl.query_prefetch_result(req)
            if result is None:
                time.sleep(0.01)
        assert result is not None
        assert [row.get_indices_list() for row in result.hit_cells] == [[0, 1], [0, 1]]
        assert result.l1_owners == {k: owner.l1_manager_id for k in hot}
        assert result.l2_hit_count == len(cold)
        assert submitted == [
            [
                rows[rank][index]
                for rank in (0, 1)
                for index in (0, 1, 2)
                if (rank, index) not in hot_positions
            ]
        ]
        owner.finish_read(hot, read_locks=2)
        assert owner.get_memory_usage() == (len(hot) * 4096, 12288)
    finally:
        ctrl.stop()
        disk.close()
        reserved.close()
        owner.close()
