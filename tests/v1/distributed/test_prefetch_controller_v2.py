# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the per-request key state of the v2 prefetch controller.
"""

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.storage_controllers.prefetch_controller_v2 import (
    PrefetchKeyState,
)


def _row(size: int, *indices: int) -> Bitmap:
    """Build a bitmap of ``size`` bits with ``indices`` set."""
    bitmap = Bitmap(size)
    for index in indices:
        bitmap.set(index)
    return bitmap


class TestPrefetchKeyState:
    def test_defaults_are_empty_maps(self) -> None:
        """A fresh key state starts with no index in any of its five maps."""
        state = PrefetchKeyState()

        for name in (
            "l1_locked_keys",
            "l2_locked_keys",
            "l1_planned_keys",
            "l2_planned_keys",
            "l1_reserved_keys",
        ):
            with pytest.raises(KeyError):
                getattr(state, name)[0]

    def test_maps_are_distinct_objects(self) -> None:
        """The five maps of one key state are independent instances."""
        state = PrefetchKeyState()
        maps = [
            state.l1_locked_keys,
            state.l2_locked_keys,
            state.l1_planned_keys,
            state.l2_planned_keys,
            state.l1_reserved_keys,
        ]

        assert len({id(m) for m in maps}) == len(maps)

    def test_instances_do_not_share_maps(self) -> None:
        """Two key states never share a map through the dataclass defaults."""
        first = PrefetchKeyState()
        second = PrefetchKeyState()
        first.l1_locked_keys[0] = [_row(4, 0)]

        with pytest.raises(KeyError):
            second.l1_locked_keys[0]
