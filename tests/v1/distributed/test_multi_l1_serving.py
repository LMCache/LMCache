# SPDX-License-Identifier: Apache-2.0
"""Serving ownership, backend configuration and fixed L2 affinity contracts."""

# Standard
from pathlib import Path
import json
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    DRAML1ManagerConfig,
    GDSL1ManagerConfig,
    parse_args,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import single_row_spec
import lmcache.v1.distributed.storage_manager as storage_module

pytestmark = pytest.mark.no_shared_allocator
LAYOUT = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])


def _args(specs: list[dict[str, object]]) -> list[str]:
    """Build CLI arguments with eviction disabled for controlled placement."""
    return [
        "--eviction-policy",
        "noop",
        *[item for spec in specs for item in ("--l1-manager", json.dumps(spec))],
    ]


def test_backend_config_subclasses_and_affinity(tmp_path: Path) -> None:
    specs = [
        {"type": "DRAM", "tag": "_default", "size_gb": 1},
        {"type": "DEVDAX", "tag": "dax", "size_gb": 1, "path": str(tmp_path / "dax")},
        {"type": "GDS", "tag": "gds", "size_gb": 1, "path": str(tmp_path / "gds")},
    ]
    config = parse_args(
        _args(specs)
        + ["--l2-adapter", '{"type":"fs","base_path":"/unused","affinity_tag":"dax"}']
    )
    assert [type(c) for c in config.l1_manager_configs] == [
        DRAML1ManagerConfig,
        DevDaxL1ManagerConfig,
        GDSL1ManagerConfig,
    ]
    assert config.l2_adapter_config.adapters[0].affinity_tag == "dax"
    with pytest.raises(ValueError, match="affinity_tag"):
        parse_args(
            _args(specs)
            + [
                "--l2-adapter",
                '{"type":"fs","base_path":"/unused","affinity_tag":"missing"}',
            ]
        )
    with pytest.raises(ValueError, match="host-backed"):
        parse_args(
            _args(specs)
            + [
                "--l2-adapter",
                '{"type":"fs","base_path":"/unused","affinity_tag":"gds"}',
            ]
        )


def test_dram_devdax_serving_round_trip(tmp_path: Path) -> None:
    backing = tmp_path / "dax"
    backing.write_bytes(b"\0" * 8192)
    config = parse_args(
        _args(
            [
                {
                    "type": "DRAM",
                    "tag": "_default",
                    "size_gb": 8192 / (1 << 30),
                    "use_lazy": False,
                },
                {
                    "type": "DEVDAX",
                    "tag": "dax",
                    "size_gb": 8192 / (1 << 30),
                    "path": str(backing),
                },
            ]
        )
    )
    storage = StorageManager(config)
    keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), "multi-serving", 0) for i in range(3)]
    try:
        owners = []
        for index, key in enumerate(keys):
            objects = storage.reserve_write([key], LAYOUT)
            tensor = objects[key].tensor
            assert tensor is not None
            tensor.fill_(index + 1)
            del tensor
            owners.append(objects[key].get_l1_manager())
            storage.finish_write_by_owner(storage.prepare_write_completion(objects))
        assert owners[0] == owners[1] and owners[1] != owners[2]
        handle = storage.submit_prefetch_task(
            single_row_spec(keys, LAYOUT, num_kv_readers=2)
        )
        assert storage.wait_prefetch_status(handle, 5)
        result = storage.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].popcount() == 3
        for _ in range(2):
            with storage.read_prefetched_results(
                keys, result.l1_owners
            ) as read_objects:
                assert read_objects is not None
                for index, obj in enumerate(read_objects):
                    assert obj.tensor is not None and bool(
                        torch.all(obj.tensor == index + 1)
                    )
                completion = storage.prepare_read_completion(keys, result.l1_owners)
            storage.finish_read_by_owner(completion)
        assert storage.get_l1_usage() == (12288, 16384)
        assert storage.memcheck()
        statuses = storage.report_status()["l1_managers"]
        assert set(statuses) == {"_default", "dax"}
        assert all(s["read_locked_count"] == 0 for s in statuses.values())
        storage.clear()
        assert storage.get_l1_usage()[0] == 0
    finally:
        storage.close()


def test_fixed_affinity_loads_into_nondefault_l1(tmp_path: Path) -> None:
    config = parse_args(
        _args(
            [
                {
                    "type": "DRAM",
                    "tag": "_default",
                    "size_gb": 8192 / (1 << 30),
                    "use_lazy": False,
                },
                {
                    "type": "DRAM",
                    "tag": "archive",
                    "size_gb": 8192 / (1 << 30),
                    "use_lazy": False,
                },
            ]
        )
        + [
            "--l2-adapter",
            json.dumps(
                {"type": "fs", "base_path": str(tmp_path), "affinity_tag": "archive"}
            ),
        ]
    )
    storage = StorageManager(config)
    keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), "affinity", 0) for i in range(3)]
    try:
        for key in keys:
            objects = storage.reserve_write([key], LAYOUT)
            tensor = objects[key].tensor
            assert tensor is not None
            tensor.fill_(9)
            storage.finish_write_by_owner(storage.prepare_write_completion(objects))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = storage.report_status()
            if (
                status["store_controllers"]["archive"]["pending_keys_count"] == 0
                and status["store_controllers"]["archive"]["in_flight_task_count"] == 0
                and status["l1_managers"]["archive"]["read_locked_count"] == 0
                and list(tmp_path.glob("*.data"))
            ):
                break
            time.sleep(0.01)
        assert list(tmp_path.glob("*.data"))
        storage.clear()
        handle = storage.submit_prefetch_task(single_row_spec([keys[2]], LAYOUT))
        assert storage.wait_prefetch_status(handle, 5)
        result = storage.query_prefetch_status(handle)
        assert result is not None and result.l2_hit_count == 1
        status = storage.report_status()["l1_managers"]
        assert status["_default"]["memory_used_bytes"] == 0
        assert status["archive"]["read_locked_count"] == 1
        with storage.read_prefetched_results(
            [keys[2]], result.l1_owners
        ) as read_objects:
            assert read_objects is not None and read_objects[0].tensor is not None
            assert bool(torch.all(read_objects[0].tensor == 9))
        storage.finish_read_prefetched([keys[2]], l1_owners=result.l1_owners)
        assert storage.get_l1_usage()[0] == 0
        storage.delete_l2_adapter(0)
        assert storage.report_status()["num_l2_adapters"] == 0
        adapter_id = storage.add_l2_adapter(config.l2_adapter_config.adapters[0])
        controllers = storage.report_status()["store_controllers"]
        assert controllers["_default"]["num_active_adapters"] == 0
        assert controllers["archive"]["num_active_adapters"] == 1
        storage.delete_l2_adapter(adapter_id)
    finally:
        storage.close()


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"type": []}, "type DRAM"),
        ({"tag": " "}, "non-empty"),
        ({"size_gb": True}, "positive number"),
        ({"size_gb": float("inf")}, "positive number"),
        ({"align_bytes": 3}, "power of two"),
        ({"read_ttl_seconds": 0}, "positive integer"),
        ({"use_lazy": "false"}, "boolean"),
        ({"extra_field": 1}, "Unknown L1 fields"),
        ({"eviction": {"trigger_watermark": 2}}, "between zero and one"),
    ],
)
def test_invalid_l1_json(changes: dict[str, object], error: str) -> None:
    spec = {"type": "DRAM", "tag": "_default", "size_gb": 1, **changes}
    with pytest.raises(ValueError, match=error):
        parse_args(_args([spec]))


def test_duplicate_tags_gds_limit_and_shared_memory_names() -> None:
    dram = {"type": "DRAM", "tag": "_default", "size_gb": 1}
    with pytest.raises(ValueError, match="unique"):
        parse_args(_args([dram, dram]))
    gds = {"type": "GDS", "tag": "gds", "size_gb": 1, "path": "/unused"}
    with pytest.raises(ValueError, match="one GDS"):
        parse_args(_args([gds, {**gds, "tag": "second", "path": "/another"}]))
    shm = {**dram, "use_lazy": False, "shm_name": "duplicate"}
    with pytest.raises(ValueError, match="distinct shm_name"):
        parse_args(_args([shm, {**shm, "tag": "second"}]))
    with pytest.raises(ValueError, match="at least 4096"):
        parse_args(_args([{**gds, "align_bytes": 1024}]))


def test_adapter_startup_failure_closes_all_created_l1s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backing = tmp_path / "dax"
    backing.write_bytes(b"\0" * 8192)
    config = parse_args(
        _args(
            [
                {
                    "type": "DRAM",
                    "tag": "_default",
                    "size_gb": 8192 / (1 << 30),
                    "use_lazy": False,
                },
                {
                    "type": "DEVDAX",
                    "tag": "dax",
                    "size_gb": 8192 / (1 << 30),
                    "path": str(backing),
                },
            ]
        )
        + [
            "--l2-adapter",
            json.dumps(
                {"type": "fs", "base_path": str(tmp_path), "affinity_tag": "dax"}
            ),
        ]
    )
    closed = []
    close = L1Manager.close

    def record_close(manager: L1Manager) -> None:
        closed.append(manager.config.tag)
        close(manager)

    def fail_adapter(*args: object, **kwargs: object) -> None:
        raise ValueError("adapter failed to open")

    monkeypatch.setattr(L1Manager, "close", record_close)
    monkeypatch.setattr(storage_module, "create_l2_adapter", fail_adapter)
    with pytest.raises(ValueError, match="adapter failed to open"):
        StorageManager(config)
    assert closed == ["dax", "_default"]


def test_l1_manager_inherits_global_eviction_settings_by_field() -> None:
    """--l1-manager inherits the global eviction flags field by field.

    The defaults were once built positionally, so adding a field to
    EvictionConfig silently shifted every later flag into the wrong field.
    """
    config = parse_args(
        [
            "--eviction-policy",
            "LRU",
            "--eviction-trigger-watermark",
            "0.9",
            "--eviction-target-watermark",
            "0.6",
            "--eviction-ratio",
            "0.3",
            "--l1-manager",
            json.dumps({"type": "DRAM", "tag": "_default", "size_gb": 1}),
        ]
    )
    eviction = config.eviction_config
    assert eviction.trigger_watermark == 0.9
    assert eviction.target_watermark == 0.6
    assert eviction.eviction_ratio == 0.3
    assert eviction.extra_logging_enabled is False
