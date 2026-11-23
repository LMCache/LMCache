# SPDX-License-Identifier: Apache-2.0
"""Abandoned prefetch results release their exact reservations without I/O waits."""

# Standard
from collections import Counter
from threading import Event
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
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
from lmcache.v1.platform import create_event_notifier


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("lock_mode", [PrefetchLockMode.LOCK, PrefetchLockMode.NO_LOCK])
def test_discard_releases_only_retained_locks(
    pending: bool, lock_mode: PrefetchLockMode
) -> None:
    """Both publication orders preserve row reader counts and actual L1 owners."""
    layout = MemoryLayoutDesc(shapes=[torch.Size([1])], dtypes=[torch.float32])
    groups = [
        GroupedObjectKeys(
            keys=[ObjectKey(ObjectKey.IntHash2Bytes(i), "m", 0, gid) for i in range(3)],
            object_group_id=gid,
            layout_desc=layout,
            num_kv_readers=gid + 2,
        )
        for gid in range(2)
    ]
    managers: list[L1ManagerInterface] = []
    released, balances = [], []
    for owner, group in zip((3, 7), groups, strict=True):
        manager = Mock(l1_manager_id=owner)
        manager.get_memory_tags.return_value = MemoryTags(
            MemorySegment.DRAM, MemoryResidency.LOCAL
        )
        remaining: Counter = Counter()
        done = Event()
        resident = {group.keys[0], group.keys[2]}

        def reserve(
            keys: list[ObjectKey],
            read_locks: int,
            *,
            resident: set[ObjectKey] = resident,
            remaining: Counter[ObjectKey] = remaining,
        ) -> dict[ObjectKey, tuple[L1Error, None]]:
            result = {}
            for key in keys:
                if key in resident:
                    remaining[key] += read_locks
                    result[key] = (L1Error.SUCCESS, None)
                else:
                    result[key] = (L1Error.KEY_NOT_EXIST, None)
            return result

        def finish(
            keys: list[ObjectKey],
            read_locks: int,
            *,
            remaining: Counter[ObjectKey] = remaining,
            done: Event = done,
        ) -> None:
            for key in keys:
                remaining[key] -= read_locks
                assert remaining[key] == 0
            if not any(remaining.values()):
                done.set()

        manager.reserve_read.side_effect = reserve
        manager.finish_read.side_effect = finish
        managers.append(manager)
        released.append(done)
        balances.append(remaining)

    lookup_fd, load_fd = create_event_notifier(), create_event_notifier()
    adapter = Mock()
    adapter.get_lookup_and_lock_event_fd.return_value = lookup_fd.fileno()
    adapter.get_load_event_fd.return_value = load_fd.fileno()
    controller = PrefetchController(
        l1_managers=managers,
        l1_manager_descriptors=[
            L1ManagerDescriptor(
                index=index, config=L1ManagerConfig(L1MemoryManagerConfig(4096, False))
            )
            for index in range(2)
        ],
        l2_adapters=[adapter],
        adapter_descriptors=[L2AdapterDescriptor(0, MockL2AdapterConfig(0.01, 10.0))],
        policy=DefaultPrefetchPolicy(),
    )
    # Keep submission asynchronous, but exclude the fake L2 from execution.
    spec = PrefetchTaskSpec(
        groups,
        fetching_policy="prefix",
        lock_mode=lock_mode,
        allowed_l2_memory_tags=frozenset(
            {MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)}
        ),
    )
    adapter.get_memory_tags.return_value = MemoryTags(
        MemorySegment.STORAGE, MemoryResidency.LOCAL
    )
    request_id = controller.submit_prefetch_request(spec)
    if pending:
        controller.discard_prefetch_result(request_id)
    controller.start()
    try:
        if not pending:
            assert controller.wait_prefetch_result(request_id, timeout=5)
            controller.discard_prefetch_result(request_id)
        assert all(event.wait(5) for event in released)
    finally:
        controller.stop()
        lookup_fd.close()
        load_fd.close()
    assert all(not any(balance.values()) for balance in balances)
    assert controller.query_prefetch_result(request_id) is None
    assert controller.report_status()["completed_results_count"] == 0
    adapter.submit_lookup_and_lock_task.assert_not_called()
