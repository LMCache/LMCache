# SPDX-License-Identifier: Apache-2.0
"""Restart identity and directory-scan tests for ``fs_native``."""

# Standard
from pathlib import Path
import os
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.fs_key_codec import object_key_to_filename
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
    get_or_create_disk_uuid,
    scan_existing_cache_entries,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


def make_key(chunk_id: int) -> ObjectKey:
    """Create a reversible filesystem test key."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="org/test-model",
        kv_rank=0x01020304,
        object_group_id=2,
        cache_salt="tenant-a",
    )


def write_entry(path: Path, key: ObjectKey, size: int, mtime_ns: int) -> Path:
    """Write one cache file and set a deterministic modification time."""
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / object_key_to_filename(key)
    file_path.write_bytes(bytes(size))
    os.utime(file_path, ns=(mtime_ns, mtime_ns))
    return file_path


def make_memory_obj(size: int = 16) -> TensorMemoryObj:
    """Create a small CPU object accepted by the native connector."""
    raw_data = torch.arange(size, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=raw_data.shape,
        dtype=raw_data.dtype,
        address=0,
        phy_size=size,
        ref_count=1,
        fmt=MemoryFormat.BINARY,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_store(
    adapter: L2AdapterInterface,
    task_id: int,
    timeout: float = 5.0,
) -> None:
    """Wait for one native adapter store and assert that it succeeded."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        select.select(
            [adapter.get_store_event_fd()], [], [], min(max(remaining, 0), 0.1)
        )
        result = adapter.pop_completed_store_tasks().get(task_id)
        if result is not None:
            assert result.is_successful()
            return
    raise TimeoutError(f"timed out waiting for store task {task_id}")


def test_disk_uuid_is_persistent(tmp_path: Path) -> None:
    first = get_or_create_disk_uuid(str(tmp_path))
    second = get_or_create_disk_uuid(str(tmp_path))

    assert first == second
    assert (tmp_path / ".lmcache_disk_uuid").read_text().strip() == first


def test_invalid_disk_uuid_fails_closed(tmp_path: Path) -> None:
    (tmp_path / ".lmcache_disk_uuid").write_text("not-a-uuid\n")

    with pytest.raises(RuntimeError, match="refusing to remap"):
        get_or_create_disk_uuid(str(tmp_path))


def test_scan_recovers_only_native_connector_flat_entries(tmp_path: Path) -> None:
    first = make_key(1)
    second = make_key(2)
    write_entry(tmp_path, first, size=11, mtime_ns=100)
    write_entry(tmp_path / "ab" / "cd", second, size=22, mtime_ns=200)

    entries, skipped = scan_existing_cache_entries(str(tmp_path))
    by_key = {key: (size, mtime_ns) for key, size, mtime_ns in entries}

    assert by_key == {first: (11, 100)}
    assert skipped == 0


def test_scan_prefers_access_time_for_recovered_lru_order(tmp_path: Path) -> None:
    key = make_key(1)
    file_path = write_entry(tmp_path, key, size=11, mtime_ns=100)
    os.utime(file_path, ns=(300, 100))

    entries, skipped = scan_existing_cache_entries(str(tmp_path))

    assert entries == [(key, 11, 300)]
    assert skipped == 0


def test_scan_ignores_subdirectories_and_counts_bad_flat_candidates(
    tmp_path: Path,
) -> None:
    key = make_key(1)
    write_entry(tmp_path, key, size=11, mtime_ns=100)
    write_entry(tmp_path / ".tmp", make_key(2), size=33, mtime_ns=300)
    (tmp_path / "malformed.data").write_bytes(b"bad")
    (tmp_path / "symlink.data").symlink_to(object_key_to_filename(key))
    noncanonical = object_key_to_filename(key).replace("@2@", "@02@")
    (tmp_path / noncanonical).write_bytes(b"duplicate")

    entries, skipped = scan_existing_cache_entries(str(tmp_path))

    assert entries == [(key, 11, 100)]
    assert skipped == 3


def test_real_fs_restart_recovery_exposes_entry_to_lru_gc(tmp_path: Path) -> None:
    """A persisted salted entry must recover into accounting and LRU GC."""
    pytest.importorskip("lmcache.lmcache_fs")
    key = make_key(7)
    config = FSNativeL2AdapterConfig(
        base_path=str(tmp_path),
        max_capacity_gb=0.001,
    )
    adapter = create_l2_adapter_from_registry(config)
    task_id = adapter.submit_store_task([key], [make_memory_obj()])
    wait_for_store(adapter, task_id)
    placement_id = config.placement_id
    adapter.close()

    stored_path = tmp_path / object_key_to_filename(key)
    assert stored_path.exists()

    restarted_config = FSNativeL2AdapterConfig(
        base_path=str(tmp_path),
        max_capacity_gb=0.001,
    )
    restarted = create_l2_adapter_from_registry(restarted_config)
    try:
        state = L2AdapterEvictionState(
            adapter_id=0,
            adapter=restarted,
            eviction_config=EvictionConfig(
                eviction_policy="LRU",
                trigger_watermark=0.8,
                eviction_ratio=1.0,
            ),
        )
        usage = restarted.get_usage()
        status = restarted.report_status()

        assert restarted_config.placement_id == placement_id
        assert usage.total_bytes_used == 16
        assert status["recovered_keys"] == 1
        assert status["recovered_bytes"] == 16

        actions = state.eviction_policy.get_eviction_actions(1.0)
        assert len(actions) == 1
        assert actions[0].keys == [key]
        restarted.delete(actions[0].keys)

        assert not stored_path.exists()
        assert restarted.get_usage().total_bytes_used == 0
    finally:
        restarted.close()
