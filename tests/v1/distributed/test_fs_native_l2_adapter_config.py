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
