# SPDX-License-Identifier: Apache-2.0
"""Restart recovery for the fs_native L2 adapter.

Chunk files persist across restarts and stay readable (lookup checks the
file), so a new process must also register them: otherwise they never count
toward ``max_capacity_gb`` and are never evicted, and disk usage grows by up
to a full cap on every restart.
"""

# Standard
from pathlib import Path
import argparse
import json
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import _object_key_to_filename
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
    _scan_persisted_objects,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
    PersistedObject,
    _object_key_to_string,
)
from tests.v1.distributed.test_native_connector_l2_adapter import (
    MockNativeConnector,
    create_memory_obj,
    wait_for_event_fd,
)

MODEL = "org/model"


def make_key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=MODEL,
        kv_rank=0x08000800,
        cache_salt=cache_salt,
    )


def write_chunk_file(base: Path, key: ObjectKey, size: int, mtime: float) -> Path:
    """Write a chunk file the way the native FS connector names it."""
    path = base / _object_key_to_filename(key)
    path.write_bytes(b"\0" * size)
    os.utime(path, (mtime, mtime))
    return path


# =============================================================================
# Directory scan (pure Python)
# =============================================================================


class TestScanPersistedObjects:
    def test_returns_chunk_files_with_size_and_mtime(self, tmp_path):
        """Every decodable ``.data`` file is reported with its size and mtime."""
        plain, salted = make_key(1), make_key(2, cache_salt="tenant-a")
        write_chunk_file(tmp_path, plain, 4096, 1000.0)
        write_chunk_file(tmp_path, salted, 8192, 2000.0)

        found = {obj.key: obj for obj in _scan_persisted_objects(str(tmp_path))}

        assert found == {
            plain: PersistedObject(key=plain, size=4096, mtime=1000.0),
            salted: PersistedObject(key=salted, size=8192, mtime=2000.0),
        }

    def test_skips_tmp_undecodable_and_directories(self, tmp_path):
        """In-flight ``.tmp`` writes, foreign files, and sub-directories are ignored."""
        key = make_key(1)
        write_chunk_file(tmp_path, key, 4096, 1000.0)
        (
            tmp_path / (_object_key_to_filename(make_key(2))[: -len(".data")] + ".tmp")
        ).write_bytes(b"x")
        (tmp_path / "notes.data").write_bytes(b"x")
        (tmp_path / "subdir.data").mkdir()

        assert [obj.key for obj in _scan_persisted_objects(str(tmp_path))] == [key]


# =============================================================================
# NativeConnectorL2Adapter.recover_persisted_objects
# =============================================================================


def objects(*specs: tuple[int, int, float]) -> list[PersistedObject]:
    return [PersistedObject(key=make_key(i), size=s, mtime=m) for i, s, m in specs]


@pytest.fixture
def mock_client():
    return MockNativeConnector()


def make_adapter(mock_client, found: list[PersistedObject] | None):
    scanner = None if found is None else (lambda: list(found))
    return NativeConnectorL2Adapter(
        mock_client, max_capacity_gb=1, persisted_object_scanner=scanner
    )


class TestNativeAdapterRecovery:
    def test_recovered_objects_count_toward_usage(self, mock_client):
        """Recovered bytes show up in get_usage like stored ones."""
        adapter = make_adapter(mock_client, objects((1, 100, 1.0), (2, 250, 2.0)))
        try:
            assert adapter.recover_persisted_objects() == 2
            assert adapter.get_usage().total_bytes_used == 350
        finally:
            adapter.close()

    def test_eviction_order_is_oldest_file_first(self, mock_client):
        """LRU evicts recovered objects in mtime order, oldest first."""
        found = objects((1, 10, 30.0), (2, 10, 10.0), (3, 10, 20.0))
        adapter = make_adapter(mock_client, found)
        policy = LRUEvictionPolicy()
        adapter.register_listener(L2EvictionPolicy(policy))
        try:
            adapter.recover_persisted_objects()
            actions = policy.get_eviction_actions(1.0)
            evicted = [key for action in actions for key in action.keys]
            assert evicted == [make_key(2), make_key(3), make_key(1)]
        finally:
            adapter.close()

    def test_delete_of_recovered_object_releases_its_bytes(self, mock_client):
        """Evicting a recovered object decrements usage (its size was recorded)."""
        found = objects((1, 100, 1.0), (2, 250, 2.0))
        # The backend already holds both objects, as after a restart.
        mock_client.submit_batch_set(
            [_object_key_to_string(obj.key) for obj in found],
            [memoryview(b"\0" * obj.size) for obj in found],
        )
        mock_client.drain_completions()
        adapter = make_adapter(mock_client, found)
        try:
            adapter.recover_persisted_objects()
            adapter.delete([make_key(2)])
            assert adapter.get_usage().total_bytes_used == 100
        finally:
            adapter.close()

    def test_already_tracked_key_is_not_counted_twice(self, mock_client):
        """A key this process stored before recovery keeps a single count."""
        stored = create_memory_obj()  # 4096 bytes
        found = objects((1, 4096, 1.0), (2, 250, 2.0))
        adapter = make_adapter(mock_client, found)
        try:
            adapter.submit_store_task([make_key(1)], [stored])
            assert wait_for_event_fd(adapter.get_store_event_fd())
            adapter.pop_completed_store_tasks()

            assert adapter.recover_persisted_objects() == 1
            # The store's usage update lands just after its completion signal.
            deadline = time.time() + 5
            while adapter.get_usage().total_bytes_used < 4096 + 250:
                assert time.time() < deadline
                time.sleep(0.01)
            time.sleep(0.1)
            assert adapter.get_usage().total_bytes_used == 4096 + 250
        finally:
            adapter.close()

    def test_no_scanner_recovers_nothing(self, mock_client):
        """Without a scanner (recovery disabled) usage starts at zero."""
        adapter = make_adapter(mock_client, None)
        try:
            assert adapter.recover_persisted_objects() == 0
            assert adapter.get_usage().total_bytes_used == 0
        finally:
            adapter.close()


# =============================================================================
# Config
# =============================================================================


class TestRecoverOnStartConfig:
    def test_defaults_to_true(self):
        """Recovery is on unless disabled."""
        cfg = FSNativeL2AdapterConfig.from_dict(
            {"type": "fs_native", "base_path": "/d"}
        )
        assert cfg.recover_on_start is True

    def test_can_be_disabled(self):
        cfg = FSNativeL2AdapterConfig.from_dict(
            {"type": "fs_native", "base_path": "/d", "recover_on_start": False}
        )
        assert cfg.recover_on_start is False

    def test_rejects_non_bool(self):
        with pytest.raises(ValueError, match="recover_on_start"):
            FSNativeL2AdapterConfig.from_dict(
                {"type": "fs_native", "base_path": "/d", "recover_on_start": "yes"}
            )


# =============================================================================
# End to end through StorageManager (needs the native FS extension)
# =============================================================================


def _storage_manager(base_path: Path, capacity_gb: float, recover: bool):
    # First Party
    from lmcache.v1.distributed.config import (  # noqa: PLC0415
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import (  # noqa: PLC0415
        add_l2_adapters_args,
        parse_args_to_l2_adapters_config,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager  # noqa: PLC0415
    from tests.v1.distributed.utils import should_use_lazy_alloc  # noqa: PLC0415

    # The server's own --l2-adapter parse path, which attaches the
    # eviction block that from_dict() alone leaves unset.
    spec = {
        "type": "fs_native",
        "base_path": str(base_path),
        "max_capacity_gb": capacity_gb,
        "recover_on_start": recover,
        "eviction": {
            "eviction_policy": "LRU",
            "trigger_watermark": 0.9,
            "eviction_ratio": 0.5,
        },
    }
    parser = argparse.ArgumentParser()
    add_l2_adapters_args(parser)
    l2_config = parse_args_to_l2_adapters_config(
        parser.parse_args(["--l2-adapter", json.dumps(spec)])
    )
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024 * 1024,
                    use_lazy=should_use_lazy_alloc(),
                    init_size_in_bytes=32 * 1024 * 1024,
                    align_bytes=0x1000,
                ),
                write_ttl_seconds=600,
                read_ttl_seconds=300,
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=l2_config,
        )
    )


def _usage(storage_manager) -> int:
    return sum(
        u.total_bytes_used
        for usages in storage_manager.get_l2_usages_by_type().values()
        for u in usages
    )


@pytest.fixture
def leftover_files(tmp_path):
    """Four 1 MiB chunk files left by a previous process, oldest = chunk 0."""
    pytest.importorskip("lmcache.lmcache_fs")
    return [
        write_chunk_file(tmp_path, make_key(i), 1024 * 1024, 1000.0 + i)
        for i in range(4)
    ]


class TestStorageManagerRestartRecovery:
    def test_leftover_files_are_accounted(self, tmp_path, leftover_files):
        """A new process counts the previous process's files toward its cap."""
        sm = _storage_manager(tmp_path, capacity_gb=1, recover=True)
        try:
            assert _usage(sm) == 4 * 1024 * 1024
        finally:
            sm.close()

    def test_leftover_files_over_cap_are_evicted_oldest_first(
        self, tmp_path, leftover_files
    ):
        """Over the cap, eviction deletes leftover files, oldest first."""
        # 4 MiB on disk against a 4.2 MiB cap: 95% > the 0.9 watermark.
        sm = _storage_manager(tmp_path, capacity_gb=4.2 / 1024, recover=True)
        try:
            deadline = time.time() + 10
            while leftover_files[0].exists() and time.time() < deadline:
                time.sleep(0.2)
            assert not leftover_files[0].exists()
            assert leftover_files[3].exists()
            assert _usage(sm) == sum(
                p.stat().st_size for p in leftover_files if p.exists()
            )
        finally:
            sm.close()

    def test_recovery_disabled_leaves_files_unaccounted(self, tmp_path, leftover_files):
        """With recover_on_start false, usage starts at zero and files stay."""
        sm = _storage_manager(tmp_path, capacity_gb=4.2 / 1024, recover=False)
        try:
            time.sleep(2)  # two eviction-loop passes
            assert _usage(sm) == 0
            assert all(p.exists() for p in leftover_files)
        finally:
            sm.close()
