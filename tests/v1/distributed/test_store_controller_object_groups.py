# SPDX-License-Identifier: Apache-2.0
"""Batch L2 stores only across objects with the same group layout."""

# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.memory_management import MemoryObj


@pytest.mark.no_shared_allocator
def test_store_batches_keep_object_group_layouts_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One listener drain must not mix differently sized object groups."""
    l1 = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=4 << 20,
                use_lazy=True,
                init_size_in_bytes=1 << 20,
            )
        )
    )
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10)
    adapter = MockL2Adapter(config)
    controller = StoreController(
        l1_manager=l1,
        l2_adapters=[adapter],
        adapter_descriptors=[L2AdapterDescriptor(index=0, config=config)],
        policy=DefaultStorePolicy(),
    )
    batches: list[list[tuple[int, int]]] = []
    submitted = threading.Event()
    original_submit = adapter.submit_store_task

    def capture_batch(keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        batches.append(
            [
                (key.object_group_id, obj.get_size())
                for key, obj in zip(keys, objects, strict=False)
            ]
        )
        if sum(len(batch) for batch in batches) == 2:
            submitted.set()
        return original_submit(keys, objects)

    monkeypatch.setattr(adapter, "submit_store_task", capture_batch)
    keys = [
        ObjectKey(
            chunk_hash=b"same-chunk",
            model_name="hybrid",
            kv_rank=0,
            object_group_id=group,
        )
        for group in range(2)
    ]
    try:
        for key, width in zip(keys, [8, 16], strict=True):
            result = l1.reserve_write(
                keys=[key],
                is_temporary=[False],
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([width])], dtypes=[torch.float16]
                ),
            )
            assert result[key][0] == L1Error.SUCCESS
        # Both notifications are queued before starting the controller.
        assert all(error == L1Error.SUCCESS for error in l1.finish_write(keys).values())
        controller.start()
        assert submitted.wait(5), "controller did not submit both groups"
        assert sorted(batches) == [[(0, 16)], [(1, 32)]]
    finally:
        controller.stop()
        adapter.close()
        l1.close()
