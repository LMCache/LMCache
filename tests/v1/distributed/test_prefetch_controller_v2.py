# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the per-request key state of the v2 prefetch controller.

The state map keeps, per L1 manager or L2 adapter index, one bitmap per key
group of a prefetch request. Tests cover layout validation, the set-algebra
operators the phase transitions rely on, and value independence between
maps (a derived map must never alias the bitmaps of its operands).
"""

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.storage_controllers.prefetch_controller_v2 import (
    PrefetchKeyState,
    _MapState,
)


def _bits(bitmap: Bitmap) -> list[int]:
    """Return the set bit indices of ``bitmap`` for value comparison."""
    return bitmap.get_indices_list()


def _row(size: int, *indices: int) -> Bitmap:
    """Build a bitmap of ``size`` bits with ``indices`` set."""
    bitmap = Bitmap(size)
    for index in indices:
        bitmap.set(index)
    return bitmap


def _rows(state: _MapState, adapter_idx: int) -> list[list[int]]:
    """Return the set bit indices of every row stored for ``adapter_idx``."""
    return [_bits(bitmap) for bitmap in state[adapter_idx]]


def _sizes(state: _MapState, adapter_idx: int) -> list[int]:
    """Return the size of every row stored for ``adapter_idx``."""
    return [bitmap.size() for bitmap in state[adapter_idx]]


class TestMapStateStorage:
    def test_set_and_get_roundtrip(self) -> None:
        """Rows stored under an index are returned with the same bits."""
        state = _MapState()
        state[0] = [_row(4, 0, 2), _row(4, 3)]

        assert _rows(state, 0) == [[0, 2], [3]]
        assert _sizes(state, 0) == [4, 4]

    def test_get_unknown_index_raises_key_error(self) -> None:
        """Reading an index that was never stored raises KeyError."""
        state = _MapState()
        state[0] = [_row(4)]

        with pytest.raises(KeyError):
            state[1]

    def test_overwrite_replaces_rows(self) -> None:
        """Storing the same index twice keeps only the latest rows."""
        state = _MapState()
        state[0] = [_row(4, 0)]
        state[0] = [_row(4, 3)]

        assert _rows(state, 0) == [[3]]

    def test_zero_width_rows_are_accepted(self) -> None:
        """A request with zero keys per group stores empty rows."""
        state = _MapState()
        state[0] = [Bitmap(0), Bitmap(0)]

        assert _sizes(state, 0) == [0, 0]
        assert _rows(state, 0) == [[], []]

    def test_contains_reports_stored_indices_only(self) -> None:
        """Membership is true for stored indices and false otherwise."""
        state = _MapState()
        state[3] = [_row(4)]

        assert 3 in state
        assert 0 not in state
        assert 4 not in state


class TestMapStateMerge:
    def test_merge_of_empty_map_is_empty_list(self) -> None:
        """A map with no index has nothing to merge."""
        assert _MapState().merge() == []

    def test_merge_of_single_index_copies_rows(self) -> None:
        """One index merges to rows with the same bits and sizes."""
        state = _MapState()
        state[0] = [_row(4, 0, 2), _row(4, 3)]

        merged = state.merge()

        assert [_bits(b) for b in merged] == [[0, 2], [3]]
        assert [b.size() for b in merged] == [4, 4]

    def test_merge_unions_rows_across_indices(self) -> None:
        """Each merged row is the OR of that row over every index."""
        state = _MapState()
        state[0] = [_row(4, 0), _row(4, 1)]
        state[1] = [_row(4, 2), _row(4, 1)]
        state[2] = [_row(4), _row(4, 3)]

        merged = state.merge()

        assert [_bits(b) for b in merged] == [[0, 2], [1, 3]]

    def test_merge_does_not_alias_stored_rows(self) -> None:
        """Mutating a merged row leaves the stored rows unchanged."""
        state = _MapState()
        state[0] = [_row(4, 0)]

        merged = state.merge()
        merged[0].set(3)

        assert _rows(state, 0) == [[0]]

    def test_merge_of_zero_width_rows(self) -> None:
        """Zero-width rows merge to zero-width rows, one per group."""
        state = _MapState()
        state[0] = [Bitmap(0), Bitmap(0)]
        state[1] = [Bitmap(0), Bitmap(0)]

        merged = state.merge()

        assert [b.size() for b in merged] == [0, 0]


class TestMapStateLayoutValidation:
    def test_rows_with_different_sizes_rejected(self) -> None:
        """Rows of one index must all have the same number of columns."""
        state = _MapState()

        with pytest.raises(ValueError):
            state[0] = [_row(4), _row(5)]

    def test_row_count_mismatch_across_indices_rejected(self) -> None:
        """Every index must carry the same number of rows."""
        state = _MapState()
        state[0] = [_row(4), _row(4)]

        with pytest.raises(ValueError):
            state[1] = [_row(4)]

    def test_column_count_mismatch_across_indices_rejected(self) -> None:
        """Every index must carry rows of the same width."""
        state = _MapState()
        state[0] = [_row(4), _row(4)]

        with pytest.raises(ValueError):
            state[1] = [_row(8), _row(8)]

    def test_empty_row_list_rejected(self) -> None:
        """A request always has at least one key group, so no rows is invalid."""
        state = _MapState()

        with pytest.raises(ValueError):
            state[0] = []

    def test_rejected_store_leaves_state_unchanged(self) -> None:
        """A layout violation neither stores the rows nor changes the layout."""
        state = _MapState()
        state[0] = [_row(4, 1)]

        with pytest.raises(ValueError):
            state[1] = [_row(8)]

        assert _rows(state, 0) == [[1]]
        with pytest.raises(KeyError):
            state[1]
        # The layout is still the one established by the first store.
        state[2] = [_row(4, 2)]
        assert _rows(state, 2) == [[2]]


class TestMapStateUnion:
    def test_add_unions_shared_index(self) -> None:
        """Bits of a shared index are OR-ed row by row."""
        left = _MapState()
        left[0] = [_row(4, 0), _row(4, 1)]
        right = _MapState()
        right[0] = [_row(4, 2), _row(4, 1, 3)]

        result = left + right

        assert _rows(result, 0) == [[0, 2], [1, 3]]

    def test_add_keeps_indices_present_on_one_side(self) -> None:
        """An index present on only one side is copied into the result."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[1] = [_row(4, 3)]

        result = left + right

        assert _rows(result, 0) == [[0]]
        assert _rows(result, 1) == [[3]]

    def test_add_is_commutative(self) -> None:
        """Both operand orders produce the same bits per index."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        left[1] = [_row(4, 1)]
        right = _MapState()
        right[1] = [_row(4, 2)]
        right[2] = [_row(4, 3)]

        forward = left + right
        backward = right + left

        for index in (0, 1, 2):
            assert _rows(forward, index) == _rows(backward, index)

    def test_add_with_empty_state_copies_bits(self) -> None:
        """Adding an empty map yields the same bits as the non-empty operand."""
        left = _MapState()
        left[0] = [_row(4, 0, 3)]

        result = left + _MapState()

        assert _rows(result, 0) == [[0, 3]]

    def test_add_does_not_mutate_operands(self) -> None:
        """A binary union leaves both operands untouched."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[0] = [_row(4, 2)]

        left + right

        assert _rows(left, 0) == [[0]]
        assert _rows(right, 0) == [[2]]

    def test_add_result_does_not_alias_operand_rows(self) -> None:
        """Mutating a result row must not change the operand it came from."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[1] = [_row(4, 1)]

        result = left + right
        result[0][0].set(3)
        result[1][0].set(3)

        assert _rows(left, 0) == [[0]]
        assert _rows(right, 1) == [[1]]

    def test_add_rejects_layout_mismatch(self) -> None:
        """Union across maps with different layouts raises instead of
        silently truncating the wider rows."""
        left = _MapState()
        left[0] = [_row(4, 3)]
        right = _MapState()
        right[0] = [_row(8, 7)]

        with pytest.raises(ValueError):
            left + right

    def test_iadd_unions_in_place_and_returns_self(self) -> None:
        """In-place union updates the receiver and evaluates to it."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[0] = [_row(4, 2)]
        right[1] = [_row(4, 1)]

        original = left
        left += right

        assert left is original
        assert _rows(left, 0) == [[0, 2]]
        assert _rows(left, 1) == [[1]]

    def test_iadd_does_not_alias_other_rows(self) -> None:
        """Rows adopted from the other map are independent copies."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[1] = [_row(4, 1)]

        left += right
        left[1][0].set(3)

        assert _rows(right, 1) == [[1]]

    def test_iadd_rejects_layout_mismatch(self) -> None:
        """In-place union enforces the same layout rule as a store."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[1] = [_row(8, 0)]

        with pytest.raises(ValueError):
            left += right

    def test_iadd_into_empty_state_adopts_layout(self) -> None:
        """A map filled only through in-place union still validates later stores."""
        left = _MapState()
        right = _MapState()
        right[0] = [_row(4, 0)]

        left += right

        assert _rows(left, 0) == [[0]]
        with pytest.raises(ValueError):
            left[1] = [_row(8)]


class TestMapStateDifference:
    def test_sub_clears_bits_of_shared_index(self) -> None:
        """Bits set in the right map are cleared row by row."""
        left = _MapState()
        left[0] = [_row(4, 0, 1, 2), _row(4, 3)]
        right = _MapState()
        right[0] = [_row(4, 1), _row(4, 3)]

        result = left - right

        assert _rows(result, 0) == [[0, 2], []]

    def test_sub_ignores_indices_only_in_right(self) -> None:
        """An index the left map does not hold never appears in the result."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[1] = [_row(4, 0)]

        result = left - right

        assert _rows(result, 0) == [[0]]
        with pytest.raises(KeyError):
            result[1]

    def test_sub_keeps_indices_only_in_left(self) -> None:
        """An index missing from the right map is copied unchanged."""
        left = _MapState()
        left[0] = [_row(4, 0, 2)]
        right = _MapState()

        result = left - right

        assert _rows(result, 0) == [[0, 2]]

    def test_sub_does_not_mutate_operands(self) -> None:
        """A binary difference leaves both operands untouched."""
        left = _MapState()
        left[0] = [_row(4, 0, 1)]
        right = _MapState()
        right[0] = [_row(4, 1)]

        left - right

        assert _rows(left, 0) == [[0, 1]]
        assert _rows(right, 0) == [[1]]

    def test_sub_result_does_not_alias_operand_rows(self) -> None:
        """Rows copied unchanged into the result are independent copies."""
        left = _MapState()
        left[0] = [_row(4, 0)]

        result = left - _MapState()
        result[0][0].set(3)

        assert _rows(left, 0) == [[0]]

    def test_sub_rejects_layout_mismatch(self) -> None:
        """Difference across maps with different layouts raises."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[0] = [_row(8, 0)]

        with pytest.raises(ValueError):
            left - right

    def test_isub_clears_in_place_and_returns_self(self) -> None:
        """In-place difference updates the receiver and evaluates to it."""
        left = _MapState()
        left[0] = [_row(4, 0, 1)]
        left[1] = [_row(4, 2)]
        right = _MapState()
        right[0] = [_row(4, 1)]
        right[2] = [_row(4, 2)]

        original = left
        left -= right

        assert left is original
        assert _rows(left, 0) == [[0]]
        assert _rows(left, 1) == [[2]]
        with pytest.raises(KeyError):
            left[2]

    def test_isub_rejects_layout_mismatch(self) -> None:
        """In-place difference enforces the same layout rule as a store."""
        left = _MapState()
        left[0] = [_row(4, 0)]
        right = _MapState()
        right[0] = [_row(8, 0)]

        with pytest.raises(ValueError):
            left -= right


class TestMapStatePhaseAlgebra:
    def test_plan_then_release_leaves_planned_bits(self) -> None:
        """The planning step of the design table: locked - planned is what
        gets released and locked - released equals planned."""
        locked = _MapState()
        locked[0] = [_row(6, 0, 1, 2, 3), _row(6, 0, 1, 2)]
        locked[1] = [_row(6, 4), _row(6, 4, 5)]
        planned = _MapState()
        planned[0] = [_row(6, 0, 1), _row(6, 0, 1)]
        planned[1] = [_row(6), _row(6)]

        to_release = locked - planned
        locked -= to_release

        assert _rows(to_release, 0) == [[2, 3], [2]]
        assert _rows(to_release, 1) == [[4], [4, 5]]
        assert _rows(locked, 0) == [[0, 1], [0, 1]]
        assert _rows(locked, 1) == [[], []]

    def test_finalize_unions_loaded_into_locked(self) -> None:
        """The finalize step: locked keys plus successfully loaded keys."""
        locked = _MapState()
        locked[0] = [_row(6, 0, 1), _row(6, 0)]
        loaded = _MapState()
        loaded[0] = [_row(6, 2, 3), _row(6, 1, 2)]

        locked += loaded

        assert _rows(locked, 0) == [[0, 1, 2, 3], [0, 1, 2]]


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
