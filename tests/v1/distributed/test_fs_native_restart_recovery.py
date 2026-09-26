# SPDX-License-Identifier: Apache-2.0
"""Restart recovery for the fs_native L2 adapter.

Chunk files persist across restarts and stay readable (lookup checks the
file), so a new process must also register them: otherwise they never count
toward ``max_capacity_gb`` and are never evicted, and disk usage grows by up
to a full cap on every restart.
"""

# Standard
from pathlib import Path
from unittest.mock import patch
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
from lmcache.v1.distributed.l2_adapters import fs_native_l2_adapter
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)
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
MIB = 1024 * 1024


def make_key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=MODEL,
        kv_rank=0x08000800,
        cache_salt=cache_salt,
    )


def parse_spec(spec: dict) -> L2AdapterConfigBase:
    """Parse an adapter spec through the server's own --l2-adapter path, which
    also attaches ``shared`` and the ``eviction`` block."""
    parser = argparse.ArgumentParser()
    add_l2_adapters_args(parser)
    return parse_args_to_l2_adapters_config(
        parser.parse_args(["--l2-adapter", json.dumps(spec)])
    ).adapters[0]


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

    def test_skips_directories_and_symlinks_with_chunk_names(self, tmp_path):
        """Only regular files count, even when a name decodes to a valid key."""
        key = make_key(1)
        write_chunk_file(tmp_path, key, 4096, 1000.0)
        (tmp_path / _object_key_to_filename(make_key(2))).mkdir()
        target = tmp_path / "elsewhere"
        target.write_bytes(b"\0" * 4096)
        (tmp_path / _object_key_to_filename(make_key(3))).symlink_to(target)

        assert [obj.key for obj in _scan_persisted_objects(str(tmp_path))] == [key]

    def test_skips_names_that_are_not_canonical(self, tmp_path):
        """A name that parses but is not the key's canonical filename is skipped:
        eviction would delete the canonical path and never free this file."""
        key = make_key(1)
        canonical = _object_key_to_filename(key)
        (tmp_path / canonical.replace("@0x08000800@", "@0x8000800@")).write_bytes(b"x")

        assert _scan_persisted_objects(str(tmp_path)) == []

    def test_unlistable_base_path_recovers_nothing(self, tmp_path):
        """A listing error skips recovery (logged) instead of failing startup."""
        write_chunk_file(tmp_path, make_key(1), 4096, 1000.0)
        with (
            patch.object(
                fs_native_l2_adapter.os,
                "scandir",
                side_effect=PermissionError("denied"),
            ),
            patch.object(fs_native_l2_adapter.logger, "error") as error,
        ):
            assert _scan_persisted_objects(str(tmp_path)) == []
        error.assert_called_once()


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


def _adapter_spec(base_path: Path, capacity_gb: float, recover: bool) -> dict:
    return {
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


def _storage_manager(specs: list[dict]):
    # First Party
    from lmcache.v1.distributed.config import (  # noqa: PLC0415
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import (  # noqa: PLC0415
        L2AdaptersConfig,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager  # noqa: PLC0415
    from tests.v1.distributed.utils import should_use_lazy_alloc  # noqa: PLC0415

    l2_config = L2AdaptersConfig(adapters=[parse_spec(spec) for spec in specs])
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


def _wait_until(condition, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            return False
        time.sleep(0.2)
    return True


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
    return [write_chunk_file(tmp_path, make_key(i), MIB, 1000.0 + i) for i in range(4)]


class TestStorageManagerRestartRecovery:
    def test_leftover_files_are_accounted(self, tmp_path, leftover_files):
        """A new process counts the previous process's files toward its cap."""
        sm = _storage_manager([_adapter_spec(tmp_path, 1, recover=True)])
        try:
            assert _usage(sm) == 4 * 1024 * 1024
        finally:
            sm.close()

    def test_runtime_added_adapter_accounts_leftover_files(
        self, tmp_path, leftover_files
    ):
        """add_l2_adapter recovers too, before the adapter serves traffic."""
        sm = _storage_manager([])
        try:
            sm.add_l2_adapter(parse_spec(_adapter_spec(tmp_path, 1, recover=True)))
            assert _usage(sm) == 4 * 1024 * 1024
        finally:
            sm.close()

    def test_leftover_files_over_cap_are_evicted_oldest_first(
        self, tmp_path, leftover_files
    ):
        """Over the cap, eviction deletes leftover files, oldest first."""
        # 4 MiB on disk against a 4.2 MiB cap: 95% > the 0.9 watermark.
        sm = _storage_manager([_adapter_spec(tmp_path, 4.2 / 1024, recover=True)])
        try:
            assert _wait_until(lambda: not leftover_files[0].exists())
            assert leftover_files[3].exists()
            # Unlinks precede the batch's accounting update; wait for both to settle.
            # exists() only (no stat): files keep vanishing while this polls.
            assert _wait_until(
                lambda: _usage(sm) == MIB * sum(p.exists() for p in leftover_files)
            ), "L2 accounting did not settle to the files left on disk"
        finally:
            sm.close()

    @pytest.mark.parametrize(
        "wrapped",
        [False, True],
        ids=["direct", "fault_inject_wrapped"],
    )
    @pytest.mark.parametrize(
        "settings",
        [
            {"shared": True, "eviction": {"eviction_policy": "LRU"}},
            {"eviction": {"eviction_policy": "IsolatedLRU"}},
        ],
        ids=["shared", "isolated_lru"],
    )
    def test_shared_or_isolated_lru_skips_recovery(
        self, tmp_path, leftover_files, settings, wrapped
    ):
        """With shared: true or IsolatedLRU on the adapter's effective (outer)
        spec, leftover files are neither accounted nor evicted, also when a
        wrapper holds those settings instead of the inner fs_native spec."""
        fs = {
            "type": "fs_native",
            "base_path": str(tmp_path),
            "max_capacity_gb": 4.2 / 1024,
        }
        spec = (
            {"type": "fault_inject", "inner": fs, **settings}
            if wrapped
            else {**fs, **settings}
        )
        sm = _storage_manager([spec])
        try:
            time.sleep(2)  # two eviction-loop passes
            assert _usage(sm) == 0
            assert all(p.exists() for p in leftover_files)
        finally:
            sm.close()

    def test_recovery_disabled_leaves_files_unaccounted(self, tmp_path, leftover_files):
        """With recover_on_start false, usage starts at zero and files stay."""
        sm = _storage_manager([_adapter_spec(tmp_path, 4.2 / 1024, recover=False)])
        try:
            time.sleep(2)  # two eviction-loop passes
            assert _usage(sm) == 0
            assert all(p.exists() for p in leftover_files)
        finally:
            sm.close()
