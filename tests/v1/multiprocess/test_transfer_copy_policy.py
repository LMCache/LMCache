# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the transfer copy policy (kernel vs direct copy path).

Covers the config surface (``TransferCopyPolicy``, CLI flags) and the
per-object-group path selection in ``lmcache_driven_transfer``. Everything
native is mocked, so these run without a CUDA build.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import argparse

# Third Party
import pytest

# First Party
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import GDSMemoryObject
from lmcache.v1.multiprocess.config import (
    TransferCopyPolicy,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    TransferCopyPath,
    select_transfer_copy_path,
)

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #


def _parse(argv: list[str]):
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    return parse_args_to_mp_server_config(parser.parse_args(argv))


def test_policy_defaults_to_kernel_path():
    config = _parse([])
    assert config.transfer_copy_policy == TransferCopyPolicy()
    assert config.transfer_copy_policy.mode == "kernel"
    assert config.transfer_copy_policy.direct_min_block_bytes == 128 * 1024


def test_policy_flags_are_parsed():
    config = _parse(
        ["--transfer-copy-mode", "auto", "--direct-copy-min-block-bytes", "65536"]
    )
    assert config.transfer_copy_policy == TransferCopyPolicy(
        mode="auto", direct_min_block_bytes=65536
    )


def test_policy_rejects_unknown_mode_and_negative_threshold():
    with pytest.raises(ValueError, match="transfer copy mode"):
        TransferCopyPolicy(mode="fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min block bytes"):
        TransferCopyPolicy(direct_min_block_bytes=-1)
    with pytest.raises(SystemExit):
        _parse(["--transfer-copy-mode", "fast"])


# ------------------------------------------------------------------ #
#  Path selection                                                      #
# ------------------------------------------------------------------ #

_FMT_ELIGIBLE = 3  # NL_X_NB_BS_HS
_FMT_HND = 6  # NL_X_TWO_NB_NH_BS_HS


def _cache_context(formats: list[int], block_bytes: list[int]):
    """A cache context with one object group spanning all kernel groups."""
    cc = MagicMock()
    cc.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[ObjectGroupInfo(kernel_group_indices=list(range(len(formats))))]
    )
    cc.get_engine_kv_format.side_effect = lambda kg: formats[kg]
    # bs * nh * hs * element_size == block_bytes with nh = hs = 1, elem = 1.
    cc.get_shape_desc.side_effect = lambda kg: SimpleNamespace(
        bs=block_bytes[kg], nh=1, hs=1, element_size=1
    )
    return cc


@pytest.fixture
def native(monkeypatch):
    """Pretend the native direct-copy support is present and usable."""
    ops = MagicMock()
    ops.batch_memcpy_supported.return_value = True
    ops.direct_copy_format_supported.side_effect = lambda f: f == _FMT_ELIGIBLE
    monkeypatch.setattr(mod, "device_ops", ops)
    monkeypatch.setattr(mod, "_HAS_NATIVE_DIRECT_COPY", True)
    monkeypatch.setattr(mod, "_direct_copy_fallback_logged", set())
    return ops


def _select(cc, mode, block_ids_host=((1, 2),), memory_objs=()):
    objs = list(memory_objs) if memory_objs else [MagicMock()]
    return select_transfer_copy_path(
        cc, 0, objs, list(block_ids_host), TransferCopyPolicy(mode=mode)
    )


def test_kernel_mode_never_selects_direct(native):
    cc = _cache_context([_FMT_ELIGIBLE], [1 << 20])
    assert _select(cc, "kernel") is TransferCopyPath.KERNEL
    native.batch_memcpy_supported.assert_not_called()


def test_direct_mode_selects_direct_for_eligible_layout(native):
    cc = _cache_context([_FMT_ELIGIBLE], [4096])
    assert _select(cc, "direct") is TransferCopyPath.DIRECT


def test_direct_mode_falls_back_for_ineligible_layout(native, caplog):
    cc = _cache_context([_FMT_ELIGIBLE, _FMT_HND], [1 << 20, 1 << 20])
    with caplog.at_level("WARNING"):
        assert _select(cc, "direct") is TransferCopyPath.KERNEL
        # Repeated decisions do not repeat the warning.
        assert _select(cc, "direct") is TransferCopyPath.KERNEL
    assert sum("not eligible" in r.message for r in caplog.records) == 1


def test_direct_mode_falls_back_without_native_support(monkeypatch, caplog):
    ops = MagicMock()
    ops.batch_memcpy_supported.return_value = False
    monkeypatch.setattr(mod, "device_ops", ops)
    monkeypatch.setattr(mod, "_HAS_NATIVE_DIRECT_COPY", True)
    monkeypatch.setattr(mod, "_direct_copy_fallback_logged", set())
    cc = _cache_context([_FMT_ELIGIBLE], [1 << 20])
    with caplog.at_level("WARNING"):
        assert _select(cc, "direct") is TransferCopyPath.KERNEL
    assert any("cudaMemcpyBatchAsync" in r.message for r in caplog.records)


def test_direct_mode_falls_back_without_host_block_ids(native):
    cc = _cache_context([_FMT_ELIGIBLE], [1 << 20])
    assert _select(cc, "direct", block_ids_host=()) is TransferCopyPath.KERNEL


def test_direct_mode_falls_back_for_gds_objects(native):
    cc = _cache_context([_FMT_ELIGIBLE], [1 << 20])
    gds = MagicMock(spec=GDSMemoryObject)
    assert _select(cc, "direct", memory_objs=(gds,)) is TransferCopyPath.KERNEL


def test_auto_mode_applies_block_size_threshold(native):
    big = _cache_context([_FMT_ELIGIBLE], [128 * 1024])
    small = _cache_context([_FMT_ELIGIBLE], [128 * 1024 - 1])
    assert _select(big, "auto") is TransferCopyPath.DIRECT
    assert _select(small, "auto") is TransferCopyPath.KERNEL
    # One small group in the object group keeps the whole group on the kernel.
    mixed = _cache_context([_FMT_ELIGIBLE, _FMT_ELIGIBLE], [1 << 20, 4096])
    assert _select(mixed, "auto") is TransferCopyPath.KERNEL


def test_transfer_dispatches_to_direct_plan(monkeypatch, native):
    """``transfer_kv_per_object_group`` routes to the direct planner and never
    touches the staged path when the direct path is selected."""
    cc = _cache_context([_FMT_ELIGIBLE], [1 << 20])
    direct_calls = []
    monkeypatch.setattr(
        mod, "_run_direct_copy_plan", lambda *a, **k: direct_calls.append(a)
    )
    staged = MagicMock()
    monkeypatch.setattr(mod, "_run_object_group_transfer_plan", staged)
    mod.transfer_kv_per_object_group(
        cc,
        block_ids_gpu=[MagicMock()],
        memory_objs=[MagicMock()],
        object_group_id=0,
        batch_size=1,
        skip_first_n_tokens=0,
        direction=mod.lmcache_native.TransferDirection.D2H,
        transfer_key="test",
        block_ids_host=[[1, 2]],
        copy_policy=TransferCopyPolicy(mode="direct"),
    )
    assert len(direct_calls) == 1
    staged.assert_not_called()
