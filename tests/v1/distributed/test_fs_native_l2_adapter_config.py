# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fs_native L2 adapter config (no native extension required).

``max_capacity_gb`` only declares capacity for usage accounting; eviction runs
only when the adapter spec also carries an ``eviction`` block. These tests pin
the warning that makes that non-enforcement visible, so a declared cap can no
longer be silently ignored.
"""

# Standard
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters import create_l2_adapter, fs_native_l2_adapter
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from tests.v1.distributed.test_native_value_capacity import ControlledConnector

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


class TestFSNativeReadPoolDefault:
    def test_read_pool_is_on_by_default(self):
        cfg, _ = _from_dict()
        assert cfg.num_read_workers > 0
        assert FSNativeL2AdapterConfig(base_path="/data").num_read_workers > 0

    def test_zero_selects_the_legacy_path(self):
        cfg, _ = _from_dict(num_read_workers=0)
        assert cfg.num_read_workers == 0

    def test_budget_without_read_pool_warns_that_it_is_ignored(self):
        _, warnings = _from_dict(num_read_workers=0, read_max_bytes_in_flight=1 << 30)
        assert len(warnings) == 1
        assert "ignored" in warnings[0].args[0]

    def test_budget_with_read_pool_does_not_warn(self):
        _, warnings = _from_dict(read_max_bytes_in_flight=1 << 30)
        assert warnings == []


class TestFSNativeReadPoolValidation:
    @pytest.mark.parametrize("field", ["num_read_workers", "read_max_bytes_in_flight"])
    @pytest.mark.parametrize("bad", [-1, True, 1.5, "256", None])
    def test_from_dict_rejects(self, field, bad):
        with pytest.raises(ValueError, match=field):
            FSNativeL2AdapterConfig.from_dict({**BASE, field: bad})

    @pytest.mark.parametrize("field", ["num_read_workers", "read_max_bytes_in_flight"])
    def test_constructor_rejects(self, field):
        with pytest.raises(ValueError, match=field):
            FSNativeL2AdapterConfig(base_path="/data", **{field: -1})


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_require_empty_is_strict_bool(bad: object) -> None:
    with pytest.raises(ValueError, match="require_empty"):
        FSNativeL2AdapterConfig.from_dict({**BASE, "require_empty": bad})
    with pytest.raises(ValueError, match="require_empty"):
        FSNativeL2AdapterConfig(base_path="/data", require_empty=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "extra,qualified",
    [
        ({}, True),
        ({"require_empty": False}, False),
        ({"shared": True}, False),
        ({"max_capacity_gb": 0}, False),
        ({"max_capacity_gb": 1e-20}, False),
        ({"use_odirect": True}, False),
    ],
)
def test_factory_only_qualifies_cold_local_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict, qualified: bool
) -> None:
    class Client(ControlledConnector):
        def __init__(self, *args: object) -> None:
            super().__init__()

        def read_budget_bytes(self) -> int:
            return 0

    module = ModuleType("lmcache.lmcache_fs")
    monkeypatch.setattr(module, "LMCacheFSClient", Client, raising=False)
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_fs", module)
    config = FSNativeL2AdapterConfig.from_dict(
        {
            **BASE,
            "base_path": str(tmp_path),
            "require_empty": True,
            "max_capacity_gb": 1,
            **extra,
        }
    )
    config.shared = extra.get("shared", False)
    adapter = create_l2_adapter(config)
    try:
        assert adapter.supports_value_capacity is qualified
    finally:
        adapter.close()


def test_occupied_cold_directory_rejected_before_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    occupied = tmp_path / "existing"
    occupied.write_bytes(b"preserve")
    config = FSNativeL2AdapterConfig(base_path=str(tmp_path), require_empty=True)
    module = ModuleType("lmcache.lmcache_fs")
    with patch.object(module, "LMCacheFSClient", create=True) as client:
        monkeypatch.setitem(sys.modules, "lmcache.lmcache_fs", module)
        with pytest.raises(ValueError, match="empty"):
            create_l2_adapter(config)
        client.assert_not_called()
    assert list(tmp_path.iterdir()) == [occupied]
    assert occupied.read_bytes() == b"preserve"


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), True, -1, 1e308])
def test_invalid_capacity_rejected_before_factory(bad: float) -> None:
    with pytest.raises(ValueError, match="max_capacity_gb"):
        FSNativeL2AdapterConfig.from_dict({**BASE, "max_capacity_gb": bad})
    with pytest.raises(ValueError, match="max_capacity_gb"):
        FSNativeL2AdapterConfig(base_path="/data", max_capacity_gb=bad)
