# SPDX-License-Identifier: Apache-2.0
"""Segmented prefetch through the real controller and L1 lock state machine."""

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
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.l2_adapters.fault_inject_l2_adapter import (
    FaultInjectL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)


@pytest.mark.parametrize("window", [-1, 1])
@pytest.mark.parametrize("source", ["l1", "l2", "failed_l2"])
def test_segmented_prefix_retains_complete_tail_and_releases_other_locks(
    window: int, source: str
) -> None:
    """Keep whole tail chunks without expanding the prefix's window locks."""
    use_l2 = source != "l1"
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            init_size_in_bytes=1 << 20,
            use_lazy=False,
            mlock=False,
        )
    )
    l1 = DramL1Manager(cfg)
    l2_cfg = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    l2 = FaultInjectL2Adapter(
        MockL2Adapter(l2_cfg),
        rate=0.0,
        seed=0,
        gap_indices=(0,) if source == "failed_l2" else (),
    )
    ctrl = PrefetchController(
        [l1],
        [L1ManagerDescriptor(0, cfg)],
        [l2] if use_l2 else [],
        [L2AdapterDescriptor(0, l2_cfg)] if use_l2 else [],
        RetainPrefetchPolicy(),
    )
    ctrl.start()
    layout = MemoryLayoutDesc([torch.Size([4, 2])], [torch.bfloat16])
    groups = [
        GroupedObjectKeys(
            keys=[
                ObjectKey(
                    chunk_hash=ObjectKey.IntHash2Bytes(col),
                    model_name="model",
                    kv_rank=0,
                    object_group_id=row,
                )
                for col in range(6)
            ],
            object_group_id=row,
            layout_desc=layout,
            sliding_window_size=w,
            num_kv_readers=2,
        )
        for row, w in enumerate([-1, window])
    ]
    present = [[0, 1, 2, 4, 5], [0, 2, 4]]
    expected = [[0, 2, 4], [0, 2, 4]] if window == -1 else [[0, 1, 2, 4], [2, 4]]
    if source == "failed_l2":
        expected = [[2, 4], [2, 4]] if window == -1 else [[4], [4]]
    try:
        for row, cols in enumerate(present):
            keys = [groups[row].keys[col] for col in cols]
            written = l1.reserve_write(keys, [False] * len(keys), layout)
            assert all(written[key][0] == L1Error.SUCCESS for key in keys)
            l1.finish_write(keys)
            if use_l2 and row == 1:
                objects = l1.reserve_read(keys)
                task = l2.submit_store_task(keys, [objects[key][1] for key in keys])
                deadline = time.monotonic() + 5
                while task not in l2.pop_completed_store_tasks():
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                l1.finish_read(keys)
                assert all(
                    error == L1Error.SUCCESS for error in l1.delete(keys).values()
                )
        request = ctrl.submit_prefetch_request(
            PrefetchTaskSpec(groups, fetching_policy="segmented_prefix")
        )
        assert ctrl.wait_prefetch_result(request, 5)
        result = ctrl.query_prefetch_result(request)
        assert result is not None
        assert [row.get_indices_list() for row in result.hit_cells] == expected
        assert result.l2_hit_count == (len(expected[1]) if use_l2 else 0)
        for row, cols in enumerate(present):
            for col in cols:
                key = groups[row].keys[col]
                if col in expected[row]:
                    assert l1.delete([key])[key] == L1Error.KEY_IS_LOCKED
                    l1.finish_read([key])
                    assert l1.delete([key])[key] == L1Error.KEY_IS_LOCKED
                    l1.finish_read([key])
                    assert l1.delete([key])[key] == L1Error.SUCCESS
                elif not use_l2 or row == 0:
                    assert l1.delete([key])[key] == L1Error.SUCCESS
                else:
                    assert l1.delete([key])[key] == L1Error.KEY_NOT_EXIST
    finally:
        ctrl.stop()
        l2.close()
        l1.close()
