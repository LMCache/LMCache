# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fs_native L2 adapter config (no native extension required).

``max_capacity_gb`` only declares capacity for usage accounting; eviction runs
only when the adapter spec also carries an ``eviction`` block. These tests pin
the warning that makes that non-enforcement visible, so a declared cap can no
longer be silently ignored.
"""

# Standard
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters import fs_native_l2_adapter
from lmcache.v1.distributed.l2_adapters.config import PersistConfig
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)

BASE = {"type": "fs_native", "base_path": "/data/lmcache"}


def _from_dict(**extra):
    """Parse a config spec, returning (config, warning_calls)."""
    with patch.object(fs_native_l2_adapter.logger, "warning") as warn:
        cfg = FSNativeL2AdapterConfig.from_dict({**BASE, **extra})
    return cfg, warn.call_args_list


class TestFSNativeCapacityEvictionWarning:
    def test_capacity_without_eviction_warns(self):
        cfg, warnings = _from_dict(max_capacity_gb=4000)
        assert cfg.max_capacity_gb == 4000
        assert len(warnings) == 1
        msg = warnings[0][0][0] % tuple(warnings[0][0][1:])
        assert "eviction" in msg
        assert "4000" in msg
        assert "/data/lmcache" in msg

    def test_capacity_with_eviction_does_not_warn(self):
        cfg, warnings = _from_dict(
            max_capacity_gb=4000,
            eviction={"eviction_policy": "LRU"},
        )
        assert cfg.max_capacity_gb == 4000
        assert warnings == []

    def test_no_capacity_does_not_warn(self):
        cfg, warnings = _from_dict()
        assert cfg.max_capacity_gb == 0
        assert warnings == []

    def test_zero_capacity_does_not_warn(self):
        cfg, warnings = _from_dict(max_capacity_gb=0)
        assert cfg.max_capacity_gb == 0
        assert warnings == []


class TestFSNativeCapacityHelpText:
    def test_help_does_not_claim_to_bound_disk_usage(self):
        help_text = FSNativeL2AdapterConfig.help()
        assert "max_capacity_gb" in help_text
        # The old wording ("max L2 capacity ... for usage tracking / eviction")
        # read as an enforced cap; it must name the eviction requirement.
        assert "eviction" in help_text
        assert "max L2 capacity" not in help_text


class TestFSNativePersistEnabled:
    def test_delete_cache_files_removes_only_cache_files(self, tmp_path):
        (tmp_path / "a.data").write_bytes(b"x")
        (tmp_path / "b.tmp").write_bytes(b"x")
        (tmp_path / "notes.txt").write_bytes(b"x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.data").write_bytes(b"x")
        tmp_dir = tmp_path / "staging"
        tmp_dir.mkdir()
        (tmp_dir / "d.tmp").write_bytes(b"x")
        (tmp_dir / "e.data").write_bytes(b"x")

        removed = fs_native_l2_adapter.delete_cache_files(str(tmp_path), str(tmp_dir))

        # In-flight writes in the tmp dir keep the .data name, so both
        # suffixes go in both directories.
        assert removed == 4
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "notes.txt",
            "staging",
            "sub",
        ]
        assert (tmp_path / "sub" / "c.data").exists()
        assert list(tmp_dir.iterdir()) == []

    def test_delete_cache_files_tolerates_missing_directory(self, tmp_path):
        assert fs_native_l2_adapter.delete_cache_files(str(tmp_path / "missing")) == 0

    def test_persist_disabled_on_shared_adapter_is_rejected(self):
        cfg, _ = _from_dict()
        cfg.shared = True
        cfg.persist_config = PersistConfig(persist_enabled=False)
        with pytest.raises(ValueError, match="shared"):
            fs_native_l2_adapter.validate_persist_config(cfg)

    def test_persist_disabled_on_private_adapter_is_allowed(self):
        cfg, _ = _from_dict()
        cfg.persist_config = PersistConfig(persist_enabled=False)
        fs_native_l2_adapter.validate_persist_config(cfg)

    def test_help_mentions_persist_enabled(self):
        assert "persist_enabled" in FSNativeL2AdapterConfig.help()
