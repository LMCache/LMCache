# SPDX-License-Identifier: Apache-2.0
"""Exercise lookup-only prefetch against a full shared DRAM L1 and real FS."""

# Standard
from contextlib import ExitStack
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchResult,
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
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator


LAYOUT = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])


@pytest.fixture
def prefetch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    with ExitStack() as cleanup:

        def create(hot: list[ObjectKey], disk_keys: list[ObjectKey], hot_capacity: int):
            config = L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=(hot_capacity + 1) * 4096,
                    use_lazy=False,
                    align_bytes=4096,
                    shm_name="",
                )
            )
            owner = DramL1Manager(config)
            cleanup.callback(owner.close)
            reserved = owner.reserve_streaming_arena(4096)
            cleanup.callback(owner.release_streaming_arena, reserved)
            disk_config = FSL2AdapterConfig(str(tmp_path))
            disk = FSL2Adapter(disk_config)
            cleanup.callback(disk.close)
            lookup = Mock(wraps=disk.submit_lookup_and_lock_task)
            monkeypatch.setattr(disk, "submit_lookup_and_lock_task", lookup)
            ctrl = PrefetchController(
                [owner],
                [L1ManagerDescriptor(index=0, config=config)],
                [disk],
                [L2AdapterDescriptor(index=0, config=disk_config)],
                DefaultPrefetchPolicy(),
            )
            reservation = owner.reserve_write(hot, [False] * len(hot), LAYOUT, "hot")
            assert all(reservation[k][0] == L1Error.SUCCESS for k in hot)
            owner.finish_write(hot, tag="hot")
            for key in disk_keys:
                (tmp_path / encode_fs_filename(key)).write_bytes(bytes(4096))
            ctrl.start()
            cleanup.callback(ctrl.stop)
            return ctrl, owner, lookup

        yield create


def _key(index: int, rank: int = 0) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([index]) * 32, model_name="m", kv_rank=rank)


def _query(ctrl: PrefetchController, request: int) -> PrefetchResult:
    assert ctrl.wait_prefetch_result(request, timeout=5)
    result = ctrl.query_prefetch_result(request)
    assert result is not None
    return result


@pytest.mark.parametrize("num_requests", [1, 8])
def test_full_l1_maps_only_missing_keys_to_fs_without_admission(prefetch, num_requests):
    hot, cold = _key(1), _key(2)
    ctrl, owner, lookup = prefetch([hot], [hot, cold], 1)
    assert owner.get_memory_usage() == (4096, 4096)
    spec = PrefetchTaskSpec(
        [GroupedObjectKeys([hot, cold], 0, LAYOUT)], defer_l2_load=True
    )
    requests = [ctrl.submit_prefetch_request(spec) for _ in range(num_requests)]
    for req in requests:
        result = _query(ctrl, req)
        assert result.hit_cells[0].get_indices_list() == [0, 1]
        assert result.l1_hit_cells[0].get_indices_list() == [0]
        assert result.l2_hit_cells[0].get_indices_list() == [1]
        assert result.l1_owners == {hot: owner.l1_manager_id}
    assert [call.args[0] for call in lookup.call_args_list] == [[cold]] * num_requests
    assert owner.get_memory_usage() == (4096, 4096)
    assert owner.unsafe_read([hot])[hot][0] == L1Error.SUCCESS
    assert owner.reserve_read([cold])[cold][0] == L1Error.KEY_NOT_EXIST
    owner.finish_read([hot], read_locks=num_requests)


def test_lookup_trims_after_gap_and_releases_late_hot_lock(prefetch):
    hot, cold, missing, later = [_key(i) for i in range(1, 5)]
    ctrl, owner, _ = prefetch([hot, later], [cold], 2)
    req = ctrl.submit_prefetch_request(
        PrefetchTaskSpec(
            [GroupedObjectKeys([hot, cold, missing, later], 0, LAYOUT)],
            defer_l2_load=True,
        )
    )
    result = _query(ctrl, req)
    assert result.hit_cells[0].get_indices_list() == [0, 1]
    assert result.l1_owners == {hot: owner.l1_manager_id}
    assert owner.unsafe_read([hot])[hot][0] == L1Error.SUCCESS
    assert owner.unsafe_read([later])[later][0] == L1Error.KEY_NOT_READABLE
    owner.finish_read([hot])


def test_deferred_lookup_requires_prefix_and_read_locks():
    row = GroupedObjectKeys([_key(1)], 0, LAYOUT)
    with pytest.raises(ValueError, match="requires prefix"):
        PrefetchTaskSpec([row], fetching_policy="full", defer_l2_load=True)
    with pytest.raises(ValueError, match="read locks"):
        PrefetchTaskSpec([row], lock_mode=PrefetchLockMode.NO_LOCK, defer_l2_load=True)


@pytest.mark.parametrize("hot_positions", [[(0, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)]])
def test_multi_rank_lookup_stops_at_missing_shard_and_releases_locks(
    prefetch, hot_positions
):
    rows = [[_key(i, rank) for i in (1, 2, 3)] for rank in (0, 1)]
    hot = [rows[rank][index] for rank, index in hot_positions]
    cold = [key for row in rows for key in row[:2] if key not in hot]
    ctrl, owner, lookup = prefetch(hot, cold + [rows[0][2]], 3)
    groups = [GroupedObjectKeys(row, 0, LAYOUT, num_kv_readers=2) for row in rows]
    req = ctrl.submit_prefetch_request(PrefetchTaskSpec(groups, defer_l2_load=True))
    result = _query(ctrl, req)
    assert [row.get_indices_list() for row in result.hit_cells] == [[0, 1], [0, 1]]
    assert result.l1_owners == {k: owner.l1_manager_id for k in hot}
    assert result.l2_hit_count == len(cold)
    assert lookup.call_args.args[0] == [
        key for row in rows for key in row if key not in hot
    ]
    owner.finish_read(hot, read_locks=2)
    assert owner.get_memory_usage() == (len(hot) * 4096, 12288)
