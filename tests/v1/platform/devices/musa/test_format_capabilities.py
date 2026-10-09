# SPDX-License-Identifier: Apache-2.0
"""Tests for MUSA transfer capability decisions derived from KV format traits."""

# Standard
from collections.abc import Callable

# Third Party
import pytest

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.platform.devices.musa.format_capabilities import (
    is_supported_musa_mp_block_transfer_format,
    is_supported_musa_native_block_transfer_format,
)

F = lmcache_native.EngineKVFormat


@pytest.mark.parametrize(
    "engine_kv_format",
    [
        F.NL_X_NB_BS_HS,
        F.NL_X_TWO_NB_BS_NH_HS,
        F.TWO_X_NL_X_NB_BS_NH_HS,
    ],
)
def test_current_musa_mp_layouts_remain_supported(
    engine_kv_format: lmcache_native.EngineKVFormat,
) -> None:
    """Keep the existing three MUSA MP layouts accepted."""
    assert is_supported_musa_mp_block_transfer_format(engine_kv_format) is True


@pytest.mark.parametrize(
    "engine_kv_format",
    [
        F.NL_X_NB_TWO_BS_NH_HS,
        F.NL_X_TWO_NB_NH_BS_HS,
        F.NL_X_NBBS_ONE_HS,
        F.NL_X_NB_BSV_BSS,
        F.NL_X_TWO_X_NB_BS_NH_HS,
    ],
)
def test_musa_mp_rejects_other_physical_layouts(
    engine_kv_format: lmcache_native.EngineKVFormat,
) -> None:
    """Reject layouts whose physical contract is outside the current path."""
    assert is_supported_musa_mp_block_transfer_format(engine_kv_format) is False


@pytest.mark.parametrize(
    ("engine_kv_format", "expected"),
    [
        (F.NL_X_NB_BS_HS, True),
        (F.NL_X_TWO_NB_BS_NH_HS, True),
        (F.TWO_X_NL_X_NB_BS_NH_HS, False),
        (F.NL_X_NB_BSV_BSS, False),
    ],
)
def test_native_musa_capability_preserves_current_abi(
    engine_kv_format: lmcache_native.EngineKVFormat,
    expected: bool,
) -> None:
    """Keep native support limited to ordinary MLA and NHD two-major layouts."""
    assert is_supported_musa_native_block_transfer_format(engine_kv_format) is expected


def test_capability_helpers_are_callable_without_format_name_tables() -> None:
    """Expose capability predicates as callable layout-based decisions."""
    predicates: tuple[Callable[[lmcache_native.EngineKVFormat], bool], ...] = (
        is_supported_musa_mp_block_transfer_format,
        is_supported_musa_native_block_transfer_format,
    )
    assert all(callable(predicate) for predicate in predicates)
