# SPDX-License-Identifier: Apache-2.0
"""Multi-L1 configuration and routing regressions."""

# Standard
import json
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import DRAML1Config, parse_args
from lmcache.v1.distributed.l1_manager import L1Manager, L1OperationResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
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
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import GDSMemoryObject
from tests.v1.distributed.utils import single_row_spec


class SelectL1:
    """A mutable policy exercises routing independently of config order."""

    index = 1

    def select_write_targets(
        self, keys: list[ObjectKey], managers: list[L1ManagerDescriptor]
    ) -> dict[int, list[ObjectKey]]:
        return {self.index: keys}


@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("with_l2", [False, True])
def test_parallel_l1_lookup_and_failed_lookup_cleanup(
    fail_first: bool, with_l2: bool
) -> None:
    # A sequential implementation cannot pass this barrier. No timing-based
    # speed assertion is needed, and failure still leaves the other L1 locked.
    barrier = threading.Barrier(2, timeout=5)

    class BarrierL1(L1Manager):
        def reserve_read(
            self, keys: list[ObjectKey], read_locks: int = 1
        ) -> dict[ObjectKey, L1OperationResult]:
            barrier.wait()
            if fail_first and self is managers[0]:
                raise RuntimeError("lookup unavailable")
            return super().reserve_read(keys, read_locks)

    config = parse_args(
        [
            "--eviction-policy",
            "noop",
            "--l1-manager",
            '{"type":"DRAM","tag":"left","size_gb":0.002,"use_lazy":false}',
            "--l1-manager",
            '{"type":"DRAM","tag":"right","size_gb":0.002,"use_lazy":false}',
        ]
    )
    managers: list[L1Manager] = [BarrierL1(c) for c in config.l1_manager_configs]
    adapter_config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=1)
    adapter_config.affinity_tag = "right"
    adapters: list[L2AdapterInterface] = (
        [MockL2Adapter(adapter_config)] if with_l2 else []
    )
    controller = PrefetchController(
        l1_managers=managers,
        # The default policy picks index 0 (right), not the first list entry.
        l1_manager_descriptors=[
            L1ManagerDescriptor(1 - i, c)
            for i, c in enumerate(config.l1_manager_configs)
        ],
        l2_adapters=adapters,
        adapter_descriptors=[L2AdapterDescriptor(0, adapter_config)] if with_l2 else [],
        policy=DefaultPrefetchPolicy(),
    )
    key = ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(3), model_name="multi", kv_rank=0
    )
    layout = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
    controller.start()
    try:
        for manager in managers:
            assert manager.reserve_write([key], [False], layout)[key][1] is not None
            manager.finish_write([key])
        request = controller.submit_prefetch_request(single_row_spec([key], layout))
        assert controller.wait_prefetch_result(request, timeout=10)
        result = controller.query_prefetch_result(request)
        assert result is not None
        assert result.hit_cells[0].popcount() == (0 if fail_first else 1)
        assert managers[0].report_status()["read_locked_count"] == 0
        assert managers[1].report_status()["read_locked_count"] == (
            0 if fail_first else 1
        )
        if not fail_first:
            managers[1].finish_read([key])
    finally:
        controller.stop()
        for adapter in adapters:
            adapter.close()
        for manager in managers:
            manager.close()


def test_dram_and_gds_are_peer_managers() -> None:
    config = parse_args(
        [
            "--eviction-policy",
            "noop",
            "--l1-manager",
            '{"type":"DRAM","tag":"_default","size_gb":0.002,"use_lazy":false}',
            "--l1-manager",
            '{"type":"GDS","tag":"nvme","size_gb":0.002,"path":"/unused"}',
        ]
    )
    manager = StorageManager(config, write_policy=SelectL1())
    key = ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(2), model_name="multi", kv_rank=0
    )
    layout = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
    try:
        assert isinstance(manager.reserve_write([key], layout)[key], GDSMemoryObject)
        manager.finish_write([key])
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        assert manager.wait_prefetch_status(handle, timeout=5)
        result = manager.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].popcount() == 1
        with manager.read_prefetched_results([key]) as objects:
            assert objects is not None and isinstance(objects[0], GDSMemoryObject)
        manager.finish_read_prefetched([key])
        assert manager.report_status()["l1_managers"]["nvme"]["read_locked_count"] == 0
        with pytest.raises(ValueError, match="single"):
            _ = manager.l1_memory_desc
    finally:
        manager.close()


def test_multi_l1_store_prefetch_and_overlapping_reads() -> None:
    args = ["--eviction-policy", "noop"]
    for tag in ("_default", "other"):
        args += [
            "--l1-manager",
            json.dumps(
                {
                    "type": "DRAM",
                    "tag": tag,
                    "size_gb": 0.002,
                    "use_lazy": False,
                }
            ),
            "--l2-adapter",
            json.dumps(
                {
                    "type": "mock",
                    "affinity_tag": tag,
                    "max_size_gb": 0.01,
                    "mock_bandwidth_gb": 1,
                }
            ),
        ]
    policy = SelectL1()
    manager = StorageManager(parse_args(args), write_policy=policy)
    key = ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(1), model_name="multi", kv_rank=0
    )
    layout = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
    try:
        reserved = manager.reserve_write([key], layout)
        assert key in reserved
        tensor = reserved[key].tensor
        assert tensor is not None
        tensor.fill_(7)
        policy.index = 0  # finish_write must still commit the reservation in L1 1.
        manager.finish_write([key])
        assert (
            manager.report_status()["l1_managers"]["other"]["total_object_count"] == 1
        )
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        assert manager.wait_prefetch_status(handle, timeout=5)
        result = manager.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].popcount() == 1

        # A concurrent write creates another resident copy in L1 0. A second
        # lookup must not steal the first lookup's read lock or return freed data.
        reserved = manager.reserve_write([key], layout)
        tensor = reserved[key].tensor
        assert tensor is not None
        tensor.fill_(7)
        manager.finish_write([key])
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        assert manager.wait_prefetch_status(handle, timeout=5)
        result = manager.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].popcount() == 1
        for _ in range(2):
            with manager.read_prefetched_results([key]) as objects:
                assert objects is not None and objects[0].tensor is not None
                assert torch.all(objects[0].tensor == 7)
            manager.finish_read_prefetched([key])

        # Store completion is asynchronous. Both affinity adapters must receive
        # the data, after which clear+prefetch exercises L2 -> its designated L1.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            statuses = manager.report_status()
            if all(
                c["read_locked_count"] == 0 for c in statuses["l1_managers"].values()
            ) and all(
                isinstance(a, MockL2Adapter) and a.debug_has_key(key)
                for _, a in manager.l2_adapters()
            ):
                break
            time.sleep(0.01)
        assert manager.memcheck()
        assert manager.get_l1_usage()[0] == 2 * 4096
        manager.clear()
        assert manager.get_l1_usage()[0] == 0
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        assert manager.wait_prefetch_status(handle, timeout=5)
        result = manager.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].popcount() == 1
        with manager.read_prefetched_results([key]) as objects:
            assert objects is not None and objects[0].tensor is not None
            assert torch.all(objects[0].tensor == 7)
        manager.finish_read_prefetched([key])
        manager.delete_l2_adapter(0)
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        assert manager.wait_prefetch_status(handle, timeout=5)
        result = manager.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].popcount() == 1
        status = manager.report_status()["l1_managers"]
        assert status["other"]["read_locked_count"] == 1
        assert status["_default"]["read_locked_count"] == 0
        with manager.read_prefetched_results([key]) as objects:
            assert objects is not None and objects[0].tensor is not None
            assert torch.all(objects[0].tensor == 7)
        manager.finish_read_prefetched([key])

        adapter_config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=1)
        adapter_config.affinity_tag = "other"
        added = manager.add_l2_adapter(adapter_config)
        policy.index = 1
        tensor = manager.reserve_write([key], layout)[key].tensor
        assert tensor is not None
        tensor.fill_(7)
        manager.finish_write([key])
        deadline = time.monotonic() + 5
        added_adapter = next(a for d, a in manager.l2_adapters() if d.index == added)
        assert isinstance(added_adapter, MockL2Adapter)
        while time.monotonic() < deadline and not added_adapter.debug_has_key(key):
            time.sleep(0.01)
        assert added_adapter.debug_has_key(key)
        manager.delete_l2_adapter(added)
    finally:
        manager.close()


def test_multi_l1_cli() -> None:
    config = parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--l1-read-ttl-seconds",
            "45",
            "--l1-manager",
            json.dumps(
                {
                    "type": "DRAM",
                    "tag": "other",
                    "size_gb": 2,
                    "read_ttl_seconds": 60,
                    "eviction": {"eviction_policy": "noop"},
                }
            ),
            "--l2-adapter",
            '{"type":"mock","max_size_gb":1,"mock_bandwidth_gb":1,"affinity_tag":"other"}',
        ]
    )
    first, second = config.l1_manager_configs
    assert isinstance(first, DRAML1Config)
    assert first.tag == "_default" and first.read_ttl_seconds == 45
    assert second.tag == "other" and second.read_ttl_seconds == 60
    assert second.memory_config.size_in_bytes == 2 << 30
    assert first.eviction is not None and second.eviction is not None
    assert first.eviction.eviction_policy == "LRU"
    assert second.eviction.eviction_policy == "noop"
    assert config.l2_adapter_config.adapters[0].affinity_tag == "other"
    assert config.l1_manager_config is None


@pytest.mark.parametrize(
    "override",
    [
        {"type": "unknown"},
        {"tag": ""},
        {"size_gb": -1},
        {"size_gb": float("nan")},
        {"size_gb": True},
        {"align_bytes": 3},
        {"use_lazy": "false"},
        {"use_lazy": True, "shm_name": "pool"},
        {"eviction": {"eviction_ratio": 2}},
        {"eviction": []},
        {"typo": 1},
    ],
)
def test_invalid_l1_json(override: dict) -> None:
    spec = {"type": "DRAM", "tag": "_default", "size_gb": 1, **override}
    with pytest.raises(ValueError):
        parse_args(["--l1-manager", json.dumps(spec), "--eviction-policy", "LRU"])


def test_duplicate_tags_and_missing_affinity() -> None:
    spec = '{"type":"DRAM","tag":"_default","size_gb":1}'
    with pytest.raises(ValueError, match="unique"):
        parse_args(
            ["--l1-manager", spec, "--l1-manager", spec, "--eviction-policy", "LRU"]
        )
    with pytest.raises(ValueError, match="affinity_tag"):
        parse_args(
            [
                "--l1-manager",
                spec,
                "--eviction-policy",
                "LRU",
                "--l2-adapter",
                '{"type":"mock","max_size_gb":1,"mock_bandwidth_gb":1,"affinity_tag":"missing"}',
            ]
        )


def test_json_only_and_removed_gds_flags() -> None:
    spec = {
        "type": "DRAM",
        "tag": "custom",
        "size_gb": 1,
        "eviction": {"eviction_policy": "LRU"},
    }
    config = parse_args(["--l1-manager", json.dumps(spec)])
    assert config.l1_manager_config is not None
    assert config.l1_manager_config.tag == "custom"
    with pytest.raises(SystemExit):
        parse_args(["--gds-l1-path", "/tmp/slab"])
    with pytest.raises(ValueError, match="eviction"):
        parse_args(["--l1-manager", '{"type":"DRAM","tag":"x","size_gb":1}'])
