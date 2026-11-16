# SPDX-License-Identifier: Apache-2.0
"""Unit tests for direct copy-path eligibility and dispatch.

Covers ``direct_transfer_supported`` and the branch it guards in
``transfer_kv_per_object_group``. Everything native is mocked, so these run
without a CUDA build.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import GDSMemoryObject
from lmcache.v1.multiprocess.object_group_transfer import direct_transfer_supported
import lmcache.v1.multiprocess.object_group_transfer as mod

_FMT_ELIGIBLE = 3  # NL_X_NB_BS_HS
_FMT_HND = 6  # NL_X_TWO_NB_NH_BS_HS


def _cache_context(
    formats: list[int],
    num_heads: list[int] | None = None,
    block_size: int = 16,
) -> MagicMock:
    """A cache context with one object group spanning all kernel groups. The
    default geometry moves 16 x 1 x 1024 x 2 B = 32 KB per layer and block."""
    heads = num_heads or [1] * len(formats)
    cc = MagicMock()
    cc.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[ObjectGroupInfo(kernel_group_indices=list(range(len(formats))))]
    )
    cc.get_engine_kv_format.side_effect = lambda kg: formats[kg]
    cc.get_shape_desc.side_effect = lambda kg: SimpleNamespace(
        nh=heads[kg], bs=block_size, hs=1024, element_size=2
    )
    return cc


@pytest.fixture
def native(monkeypatch):
    """Pretend the native direct-copy support is present and usable."""
    ops = MagicMock()
    # Stands in for the native rule: HND qualifies only with a single head.
    ops.direct_copy_format_supported.side_effect = lambda f, sd: (
        f == _FMT_ELIGIBLE or (f == _FMT_HND and sd.nh == 1)
    )
    monkeypatch.setattr(mod, "device_ops", ops)
    monkeypatch.setattr(mod, "_HAS_BATCH_MEMCPY_ASYNC", True)
    monkeypatch.setattr(mod, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", True)
    monkeypatch.setattr(mod, "_direct_copy_logged_rejections", set())
    return ops


# ------------------------------------------------------------------ #
#  Eligibility                                                         #
# ------------------------------------------------------------------ #


def _supported(cc, block_ids_host=((1, 2),), memory_objs=()):
    objs = list(memory_objs) if memory_objs else [MagicMock()]
    return direct_transfer_supported(cc, 0, objs, list(block_ids_host))


def test_eligible_layout_is_supported(native):
    assert _supported(_cache_context([_FMT_ELIGIBLE])) is True


class _RecordingHandler(logging.Handler):
    """Collects formatted messages emitted on the module logger."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def recorded_logs() -> Iterator[list[str]]:
    """Messages the module logger emits during the test."""
    # lmcache's init_logger sets propagate=False and levels the handlers at
    # LMCACHE_LOG_LEVEL, so caplog sees nothing when that is above WARNING. Attach
    # our own handler and force the level for the duration of the test.
    handler = _RecordingHandler()
    original_level = mod.logger.level
    mod.logger.setLevel(logging.INFO)
    mod.logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        mod.logger.removeHandler(handler)
        mod.logger.setLevel(original_level)


def test_every_kernel_group_must_be_eligible(
    native: MagicMock, recorded_logs: list[str]
) -> None:
    """One ineligible group disqualifies the whole object group, logged once."""
    cc = _cache_context([_FMT_ELIGIBLE, _FMT_HND], num_heads=[1, 8])
    assert _supported(cc) is False
    # Repeated decisions do not repeat the message.
    assert _supported(cc) is False
    assert sum("not eligible" in m for m in recorded_logs) == 1


def test_eligibility_uses_each_kernel_group_shape(
    native: MagicMock, recorded_logs: list[str]
) -> None:
    """The decision sees each kernel group's own geometry, and a rejection
    names the head count that disqualified it."""
    assert _supported(_cache_context([_FMT_HND], num_heads=[1])) is True
    assert _supported(_cache_context([_FMT_HND, _FMT_HND], num_heads=[1, 8])) is False
    rejections = [m for m in recorded_logs if "not eligible" in m]
    assert len(rejections) == 1
    assert "with 8 heads" in rejections[0] and "kernel group 1" in rejections[0]


def test_small_blocks_stay_on_kernel_path(native: MagicMock) -> None:
    """An eligible layout whose block is below the size floor is not direct;
    the floor itself still qualifies."""
    assert _supported(_cache_context([_FMT_ELIGIBLE], block_size=16)) is True
    assert _supported(_cache_context([_FMT_ELIGIBLE], block_size=8)) is False


def test_missing_host_block_ids_is_unsupported(native):
    assert _supported(_cache_context([_FMT_ELIGIBLE]), block_ids_host=()) is False


def test_gds_objects_are_unsupported(native):
    gds = MagicMock(spec=GDSMemoryObject)
    cc = _cache_context([_FMT_ELIGIBLE])
    assert _supported(cc, memory_objs=[MagicMock(), gds]) is False


# ------------------------------------------------------------------ #
#  Dispatch                                                            #
# ------------------------------------------------------------------ #


def _dispatch(monkeypatch, block_ids_host):
    """Run ``transfer_kv_per_object_group`` with both planners stubbed out."""
    cc = _cache_context([_FMT_ELIGIBLE])
    direct = MagicMock()
    staged = MagicMock()
    monkeypatch.setattr(mod, "run_direct_transfer", direct)
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
        block_ids_host=block_ids_host,
    )
    return direct, staged


def test_dispatch_prefers_direct_when_supported(monkeypatch, native):
    direct, staged = _dispatch(monkeypatch, [[1, 2]])
    direct.assert_called_once()
    staged.assert_not_called()


def test_dispatch_falls_back_to_kernel_without_host_block_ids(monkeypatch, native):
    direct, staged = _dispatch(monkeypatch, [])
    direct.assert_not_called()
    staged.assert_called_once()


def test_dispatch_skips_direct_without_native_support(monkeypatch, native):
    """``_HAS_BATCH_MEMCPY_ASYNC`` short-circuits before any eligibility work."""
    monkeypatch.setattr(mod, "_HAS_BATCH_MEMCPY_ASYNC", False)
    direct, staged = _dispatch(monkeypatch, [[1, 2]])
    direct.assert_not_called()
    staged.assert_called_once()
    native.direct_copy_format_supported.assert_not_called()
