# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the storage-controller utilities.

``Bitmap2D`` is a grid with one bitmap per key group of a prefetch request.
``MapState`` keeps one such grid per L1 manager or L2 adapter index. Tests
cover layout validation, the set-algebra operators the phase transitions rely
on, the cross-index merge, and value independence between maps (a derived map
must never alias the bitmaps of its operands). The descriptor tests cover the
type names derived from the configuration objects.
"""

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import L1BackendType
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    Bitmap2D,
    L1ManagerDescriptor,
    MapState,
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


def _rows(state: MapState, adapter_idx: int) -> list[list[int]]:
    """Return the set bit indices of every row stored for ``adapter_idx``."""
    return [_bits(bitmap) for bitmap in state[adapter_idx]]


def _sizes(state: MapState, adapter_idx: int) -> list[int]:
    """Return the size of every row stored for ``adapter_idx``."""
    return [bitmap.size() for bitmap in state[adapter_idx]]


class TestMapStateStorage:
    def test_set_and_get_roundtrip(self) -> None:
        """Rows stored under an index are returned with the same bits."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0, 2), _row(4, 3)])

        assert _rows(state, 0) == [[0, 2], [3]]
        assert _sizes(state, 0) == [4, 4]

    def test_get_unknown_index_raises_key_error(self) -> None:
        """Reading an index that was never stored raises KeyError."""
        state = MapState()
        state[0] = Bitmap2D([_row(4)])

        with pytest.raises(KeyError):
            state[1]

    def test_overwrite_replaces_rows(self) -> None:
        """Storing the same index twice keeps only the latest rows."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0)])
        state[0] = Bitmap2D([_row(4, 3)])

        assert _rows(state, 0) == [[3]]

    def test_zero_width_rows_are_accepted(self) -> None:
        """A request with zero keys per group stores empty rows."""
        state = MapState()
        state[0] = Bitmap2D([Bitmap(0), Bitmap(0)])

        assert _sizes(state, 0) == [0, 0]
        assert _rows(state, 0) == [[], []]

    def test_contains_reports_stored_indices_only(self) -> None:
        """Membership is true for stored indices and false otherwise."""
        state = MapState()
        state[3] = Bitmap2D([_row(4)])

        assert 3 in state
        assert 0 not in state
        assert 4 not in state


class TestMapStateDelete:
    def test_delete_removes_only_that_index(self) -> None:
        """Deleting one index leaves the others untouched."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0)])
        state[1] = Bitmap2D([_row(4, 1)])

        del state[0]

        assert 0 not in state
        assert _rows(state, 1) == [[1]]
        assert [index for index, _grid in state.items()] == [1]

    def test_delete_unknown_index_raises(self) -> None:
        """Deleting an index that was never stored raises KeyError."""
        state = MapState()
        state[0] = Bitmap2D([_row(4)])

        with pytest.raises(KeyError):
            del state[1]

    def test_delete_last_index_keeps_layout(self) -> None:
        """After every index is removed, a store of another shape is still
        rejected and a store of the same shape is accepted."""
        state = MapState()
        state[0] = Bitmap2D([_row(4)])

        del state[0]

        assert state.merge().popcount() == 0
        with pytest.raises(ValueError):
            state[1] = Bitmap2D([_row(8)])
        state[1] = Bitmap2D([_row(4, 2)])
        assert _rows(state, 1) == [[2]]

    def test_delete_then_readd(self) -> None:
        """An index can be stored again after deletion."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0)])

        del state[0]
        state[0] = Bitmap2D([_row(4, 3)])

        assert _rows(state, 0) == [[3]]


class TestMapStateItems:
    def test_empty_map_has_no_items(self) -> None:
        """A fresh map yields nothing."""
        assert list(MapState().items()) == []

    def test_items_pairs_index_with_rows(self) -> None:
        """Each pair holds an index and that index's rows."""
        state = MapState()
        state[2] = Bitmap2D([_row(4, 0), _row(4, 3)])
        state[0] = Bitmap2D([_row(4, 1), _row(4)])

        items = {index: [_bits(b) for b in rows] for index, rows in state.items()}

        assert items == {2: [[0], [3]], 0: [[1], []]}

    def test_items_follow_insertion_order(self) -> None:
        """Indices come back in the order they were stored."""
        state = MapState()
        for index in (5, 1, 3):
            state[index] = Bitmap2D([_row(2)])

        assert [index for index, _rows_ in state.items()] == [5, 1, 3]

    def test_items_reflect_overwrite(self) -> None:
        """Re-storing an index shows the new rows without a duplicate pair."""
        state = MapState()
        state[0] = Bitmap2D([_row(3, 0)])
        state[0] = Bitmap2D([_row(3, 2)])

        assert [(i, _bits(r[0])) for i, r in state.items()] == [(0, [2])]

    def test_items_agree_with_contains_and_getitem(self) -> None:
        """Every yielded index is a member and yields the same rows."""
        state = MapState()
        state[1] = Bitmap2D([_row(3, 1)])
        state[4] = Bitmap2D([_row(3, 0, 2)])

        for index, rows in state.items():
            assert index in state
            assert [_bits(b) for b in rows] == _rows(state, index)


class TestMapStateCopy:
    def test_copy_of_empty_map_is_empty(self) -> None:
        """Copying an empty map yields a map with no index."""
        assert MapState().copy().merge().size() == (0, 0)

    def test_copy_has_same_indices_and_bits(self) -> None:
        """Every index and every bit is carried over."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0, 2), _row(4, 3)])
        state[2] = Bitmap2D([_row(4, 1), _row(4)])

        duplicate = state.copy()

        assert 0 in duplicate and 2 in duplicate and 1 not in duplicate
        assert _rows(duplicate, 0) == [[0, 2], [3]]
        assert _rows(duplicate, 2) == [[1], []]

    def test_copy_is_independent(self) -> None:
        """Mutating rows of either map leaves the other unchanged."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0)])

        duplicate = state.copy()
        duplicate[0][0].set(3)
        state[0][0].set(1)

        assert _rows(state, 0) == [[0, 1]]
        assert _rows(duplicate, 0) == [[0, 3]]

    def test_copy_keeps_layout(self) -> None:
        """The copy enforces the same shape as the original."""
        state = MapState()
        state[0] = Bitmap2D([_row(4)])

        duplicate = state.copy()

        with pytest.raises(ValueError):
            duplicate[1] = Bitmap2D([_row(8)])


class TestMapStateMerge:
    def test_merge_of_empty_map_is_empty_list(self) -> None:
        """A map with no index merges to an empty grid."""
        assert MapState().merge().size() == (0, 0)

    def test_merge_of_single_index_copies_rows(self) -> None:
        """One index merges to rows with the same bits and sizes."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0, 2), _row(4, 3)])

        merged = state.merge()

        assert [_bits(b) for b in merged] == [[0, 2], [3]]
        assert [b.size() for b in merged] == [4, 4]

    def test_merge_unions_rows_across_indices(self) -> None:
        """Each merged row is the OR of that row over every index."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0), _row(4, 1)])
        state[1] = Bitmap2D([_row(4, 2), _row(4, 1)])
        state[2] = Bitmap2D([_row(4), _row(4, 3)])

        merged = state.merge()

        assert [_bits(b) for b in merged] == [[0, 2], [1, 3]]

    def test_merge_does_not_alias_stored_rows(self) -> None:
        """Mutating a merged row leaves the stored rows unchanged."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 0)])

        merged = state.merge()
        merged[0].set(3)

        assert _rows(state, 0) == [[0]]

    def test_merge_of_zero_width_rows(self) -> None:
        """Zero-width rows merge to zero-width rows, one per group."""
        state = MapState()
        state[0] = Bitmap2D([Bitmap(0), Bitmap(0)])
        state[1] = Bitmap2D([Bitmap(0), Bitmap(0)])

        merged = state.merge()

        assert [b.size() for b in merged] == [0, 0]


class TestMapStateLayoutValidation:
    def test_rows_with_different_sizes_rejected(self) -> None:
        """Rows of one index must all have the same number of columns."""
        state = MapState()

        with pytest.raises(ValueError):
            state[0] = Bitmap2D([_row(4), _row(5)])

    def test_row_count_mismatch_across_indices_rejected(self) -> None:
        """Every index must carry the same number of rows."""
        state = MapState()
        state[0] = Bitmap2D([_row(4), _row(4)])

        with pytest.raises(ValueError):
            state[1] = Bitmap2D([_row(4)])

    def test_column_count_mismatch_across_indices_rejected(self) -> None:
        """Every index must carry rows of the same width."""
        state = MapState()
        state[0] = Bitmap2D([_row(4), _row(4)])

        with pytest.raises(ValueError):
            state[1] = Bitmap2D([_row(8), _row(8)])

    def test_empty_row_list_rejected(self) -> None:
        """A request always has at least one key group, so no rows is invalid."""
        state = MapState()

        with pytest.raises(ValueError):
            state[0] = Bitmap2D([])

    def test_rejected_store_leaves_state_unchanged(self) -> None:
        """A layout violation neither stores the rows nor changes the layout."""
        state = MapState()
        state[0] = Bitmap2D([_row(4, 1)])

        with pytest.raises(ValueError):
            state[1] = Bitmap2D([_row(8)])

        assert _rows(state, 0) == [[1]]
        with pytest.raises(KeyError):
            state[1]
        # The layout is still the one established by the first store.
        state[2] = Bitmap2D([_row(4, 2)])
        assert _rows(state, 2) == [[2]]


class TestMapStateUnion:
    def test_add_unions_shared_index(self) -> None:
        """Bits of a shared index are OR-ed row by row."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0), _row(4, 1)])
        right = MapState()
        right[0] = Bitmap2D([_row(4, 2), _row(4, 1, 3)])

        result = left + right

        assert _rows(result, 0) == [[0, 2], [1, 3]]

    def test_add_keeps_indices_present_on_one_side(self) -> None:
        """An index present on only one side is copied into the result."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[1] = Bitmap2D([_row(4, 3)])

        result = left + right

        assert _rows(result, 0) == [[0]]
        assert _rows(result, 1) == [[3]]

    def test_add_is_commutative(self) -> None:
        """Both operand orders produce the same bits per index."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        left[1] = Bitmap2D([_row(4, 1)])
        right = MapState()
        right[1] = Bitmap2D([_row(4, 2)])
        right[2] = Bitmap2D([_row(4, 3)])

        forward = left + right
        backward = right + left

        for index in (0, 1, 2):
            assert _rows(forward, index) == _rows(backward, index)

    def test_add_with_empty_state_copies_bits(self) -> None:
        """Adding an empty map yields the same bits as the non-empty operand."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0, 3)])

        result = left + MapState()

        assert _rows(result, 0) == [[0, 3]]

    def test_add_does_not_mutate_operands(self) -> None:
        """A binary union leaves both operands untouched."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[0] = Bitmap2D([_row(4, 2)])

        left + right

        assert _rows(left, 0) == [[0]]
        assert _rows(right, 0) == [[2]]

    def test_add_result_does_not_alias_operand_rows(self) -> None:
        """Mutating a result row must not change the operand it came from."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[1] = Bitmap2D([_row(4, 1)])

        result = left + right
        result[0][0].set(3)
        result[1][0].set(3)

        assert _rows(left, 0) == [[0]]
        assert _rows(right, 1) == [[1]]

    def test_add_rejects_layout_mismatch(self) -> None:
        """Union across maps with different layouts raises instead of
        silently truncating the wider rows."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 3)])
        right = MapState()
        right[0] = Bitmap2D([_row(8, 7)])

        with pytest.raises(ValueError):
            left + right

    def test_iadd_unions_in_place_and_returns_self(self) -> None:
        """In-place union updates the receiver and evaluates to it."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[0] = Bitmap2D([_row(4, 2)])
        right[1] = Bitmap2D([_row(4, 1)])

        original = left
        left += right

        assert left is original
        assert _rows(left, 0) == [[0, 2]]
        assert _rows(left, 1) == [[1]]

    def test_iadd_does_not_alias_other_rows(self) -> None:
        """Rows adopted from the other map are independent copies."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[1] = Bitmap2D([_row(4, 1)])

        left += right
        left[1][0].set(3)

        assert _rows(right, 1) == [[1]]

    def test_iadd_rejects_layout_mismatch(self) -> None:
        """In-place union enforces the same layout rule as a store."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[1] = Bitmap2D([_row(8, 0)])

        with pytest.raises(ValueError):
            left += right

    def test_iadd_into_empty_state_adopts_layout(self) -> None:
        """A map filled only through in-place union still validates later stores."""
        left = MapState()
        right = MapState()
        right[0] = Bitmap2D([_row(4, 0)])

        left += right

        assert _rows(left, 0) == [[0]]
        with pytest.raises(ValueError):
            left[1] = Bitmap2D([_row(8)])


class TestMapStateDifference:
    def test_sub_clears_bits_of_shared_index(self) -> None:
        """Bits set in the right map are cleared row by row."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0, 1, 2), _row(4, 3)])
        right = MapState()
        right[0] = Bitmap2D([_row(4, 1), _row(4, 3)])

        result = left - right

        assert _rows(result, 0) == [[0, 2], []]

    def test_sub_ignores_indices_only_in_right(self) -> None:
        """An index the left map does not hold never appears in the result."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[1] = Bitmap2D([_row(4, 0)])

        result = left - right

        assert _rows(result, 0) == [[0]]
        with pytest.raises(KeyError):
            result[1]

    def test_sub_keeps_indices_only_in_left(self) -> None:
        """An index missing from the right map is copied unchanged."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0, 2)])
        right = MapState()

        result = left - right

        assert _rows(result, 0) == [[0, 2]]

    def test_sub_does_not_mutate_operands(self) -> None:
        """A binary difference leaves both operands untouched."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0, 1)])
        right = MapState()
        right[0] = Bitmap2D([_row(4, 1)])

        left - right

        assert _rows(left, 0) == [[0, 1]]
        assert _rows(right, 0) == [[1]]

    def test_sub_result_does_not_alias_operand_rows(self) -> None:
        """Rows copied unchanged into the result are independent copies."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])

        result = left - MapState()
        result[0][0].set(3)

        assert _rows(left, 0) == [[0]]

    def test_sub_rejects_layout_mismatch(self) -> None:
        """Difference across maps with different layouts raises."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[0] = Bitmap2D([_row(8, 0)])

        with pytest.raises(ValueError):
            left - right

    def test_isub_clears_in_place_and_returns_self(self) -> None:
        """In-place difference updates the receiver and evaluates to it."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0, 1)])
        left[1] = Bitmap2D([_row(4, 2)])
        right = MapState()
        right[0] = Bitmap2D([_row(4, 1)])
        right[2] = Bitmap2D([_row(4, 2)])

        original = left
        left -= right

        assert left is original
        assert _rows(left, 0) == [[0]]
        assert _rows(left, 1) == [[2]]
        with pytest.raises(KeyError):
            left[2]

    def test_isub_rejects_layout_mismatch(self) -> None:
        """In-place difference enforces the same layout rule as a store."""
        left = MapState()
        left[0] = Bitmap2D([_row(4, 0)])
        right = MapState()
        right[0] = Bitmap2D([_row(8, 0)])

        with pytest.raises(ValueError):
            left -= right


class TestMapStatePhaseAlgebra:
    def test_plan_then_release_leaves_planned_bits(self) -> None:
        """The planning step of the design table: locked - planned is what
        gets released and locked - released equals planned."""
        locked = MapState()
        locked[0] = Bitmap2D([_row(6, 0, 1, 2, 3), _row(6, 0, 1, 2)])
        locked[1] = Bitmap2D([_row(6, 4), _row(6, 4, 5)])
        planned = MapState()
        planned[0] = Bitmap2D([_row(6, 0, 1), _row(6, 0, 1)])
        planned[1] = Bitmap2D([_row(6), _row(6)])

        to_release = locked - planned
        locked -= to_release

        assert _rows(to_release, 0) == [[2, 3], [2]]
        assert _rows(to_release, 1) == [[4], [4, 5]]
        assert _rows(locked, 0) == [[0, 1], [0, 1]]
        assert _rows(locked, 1) == [[], []]

    def test_finalize_unions_loaded_into_locked(self) -> None:
        """The finalize step: locked keys plus successfully loaded keys."""
        locked = MapState()
        locked[0] = Bitmap2D([_row(6, 0, 1), _row(6, 0)])
        loaded = MapState()
        loaded[0] = Bitmap2D([_row(6, 2, 3), _row(6, 1, 2)])

        locked += loaded

        assert _rows(locked, 0) == [[0, 1, 2, 3], [0, 1, 2]]


# =============================================================================
# Bitmap2D
# =============================================================================


def _grid(*rows: Bitmap) -> Bitmap2D:
    """Build a grid from rows."""
    return Bitmap2D(list(rows))


def _grid_bits(grid: Bitmap2D) -> list[list[int]]:
    """Return the set bit indices of every row of ``grid``."""
    return [_bits(row) for row in grid]


class TestBitmap2DConstruction:
    def test_size_and_len(self) -> None:
        """Shape is (rows, columns) and len is the row count."""
        grid = _grid(_row(4, 0), _row(4, 3))

        assert grid.size() == (2, 4)
        assert len(grid) == 2

    def test_empty_grid(self) -> None:
        """A grid with no rows has shape (0, 0) and no bits."""
        grid = Bitmap2D([])

        assert grid.size() == (0, 0)
        assert len(grid) == 0
        assert grid.popcount() == 0
        assert list(grid) == []

    def test_inconsistent_widths_rejected(self) -> None:
        """Rows of different widths cannot form a grid."""
        with pytest.raises(ValueError):
            _grid(_row(4), _row(5))

    def test_zeros_shape(self) -> None:
        """zeros() builds an all-clear grid of the requested shape."""
        grid = Bitmap2D.zeros(3, 5)

        assert grid.size() == (3, 5)
        assert grid.popcount() == 0

    @pytest.mark.parametrize(("rows", "cols"), [(-1, 2), (2, -1)])
    def test_zeros_rejects_negative_shape(self, rows: int, cols: int) -> None:
        """Negative dimensions are invalid."""
        with pytest.raises(ValueError):
            Bitmap2D.zeros(rows, cols)

    def test_constructor_does_not_alias_the_list(self) -> None:
        """Appending to the list passed in does not change the grid."""
        rows = [_row(3, 0)]
        grid = Bitmap2D(rows)
        rows.append(_row(3, 1))

        assert len(grid) == 1

    def test_constructor_keeps_row_objects(self) -> None:
        """Rows are stored as given, so mutating one is visible in the grid."""
        row = _row(3, 0)
        grid = Bitmap2D([row])
        row.set(2)

        assert _grid_bits(grid) == [[0, 2]]


class TestBitmap2DAccess:
    def test_getitem_and_iter_yield_rows_in_order(self) -> None:
        """Indexing and iteration expose the rows in construction order."""
        grid = _grid(_row(3, 0), _row(3, 1), _row(3, 2))

        assert _bits(grid[1]) == [1]
        assert _grid_bits(grid) == [[0], [1], [2]]

    def test_getitem_out_of_range_raises(self) -> None:
        """A row index past the end raises IndexError."""
        with pytest.raises(IndexError):
            _grid(_row(3))[1]

    def test_popcount_sums_all_rows(self) -> None:
        """popcount counts set bits across every row."""
        assert _grid(_row(4, 0, 1), _row(4, 3)).popcount() == 3

    def test_to_list_returns_new_list_of_stored_rows(self) -> None:
        """The list is fresh but holds the stored Bitmap objects."""
        grid = _grid(_row(3, 0))
        rows = grid.to_list()
        rows.append(_row(3))
        rows[0].set(2)

        assert len(grid) == 1
        assert _grid_bits(grid) == [[0, 2]]

    def test_copy_is_independent(self) -> None:
        """Mutating a copy leaves the original untouched and vice versa."""
        grid = _grid(_row(3, 0))
        duplicate = grid.copy()
        duplicate[0].set(1)
        grid[0].set(2)

        assert _grid_bits(grid) == [[0, 2]]
        assert _grid_bits(duplicate) == [[0, 1]]

    def test_zeros_like_and_ones_like(self) -> None:
        """Both keep the shape; one clears every bit, the other sets every bit."""
        grid = _grid(_row(3, 1), _row(3))

        zeros = grid.zeros_like()
        ones = grid.ones_like()

        assert zeros.size() == (2, 3)
        assert zeros.popcount() == 0
        assert ones.size() == (2, 3)
        assert _grid_bits(ones) == [[0, 1, 2], [0, 1, 2]]


class TestBitmap2DAlgebra:
    def test_add_is_row_wise_union(self) -> None:
        """Union sets a bit where either grid has it."""
        result = _grid(_row(4, 0), _row(4, 1)) + _grid(_row(4, 2), _row(4, 1, 3))

        assert _grid_bits(result) == [[0, 2], [1, 3]]

    def test_sub_is_row_wise_difference(self) -> None:
        """Difference clears the bits set in the right operand."""
        result = _grid(_row(4, 0, 1, 2), _row(4, 3)) - _grid(_row(4, 1), _row(4, 3))

        assert _grid_bits(result) == [[0, 2], []]

    def test_and_is_row_wise_intersection(self) -> None:
        """Intersection keeps a bit only where both grids have it."""
        result = _grid(_row(4, 0, 1, 2), _row(4, 3)) & _grid(_row(4, 1, 2, 3), _row(4))

        assert _grid_bits(result) == [[1, 2], []]

    def test_binary_ops_do_not_mutate_operands(self) -> None:
        """Operands keep their bits after every binary operator."""
        left = _grid(_row(4, 0, 1))
        right = _grid(_row(4, 1, 2))

        left + right
        left - right
        left & right

        assert _grid_bits(left) == [[0, 1]]
        assert _grid_bits(right) == [[1, 2]]

    def test_binary_op_results_do_not_alias_operands(self) -> None:
        """Mutating a result row leaves both operands untouched."""
        left = _grid(_row(4, 0))
        right = _grid(_row(4, 1))

        (left + right)[0].set(3)
        (left - right)[0].set(3)
        (left & right)[0].set(3)

        assert _grid_bits(left) == [[0]]
        assert _grid_bits(right) == [[1]]

    def test_inplace_ops_update_and_return_self(self) -> None:
        """Each in-place operator evaluates to the receiver with new bits."""
        grid = _grid(_row(4, 0, 1))
        original = grid

        grid += _grid(_row(4, 2))
        assert grid is original
        assert _grid_bits(grid) == [[0, 1, 2]]

        grid -= _grid(_row(4, 0))
        assert grid is original
        assert _grid_bits(grid) == [[1, 2]]

        grid &= _grid(_row(4, 2, 3))
        assert grid is original
        assert _grid_bits(grid) == [[2]]

    def test_inplace_ops_do_not_mutate_other(self) -> None:
        """The right operand of an in-place operator is unchanged."""
        grid = _grid(_row(4, 0))
        other = _grid(_row(4, 1))

        grid += other
        grid -= other
        grid &= other

        assert _grid_bits(other) == [[1]]

    @pytest.mark.parametrize("op", ["add", "sub", "and", "iadd", "isub", "iand"])
    def test_shape_mismatch_rejected(self, op: str) -> None:
        """Every operator refuses grids of different shapes."""
        left = _grid(_row(4, 0), _row(4, 0))
        fewer_rows = _grid(_row(4, 0))
        wider = _grid(_row(8, 0), _row(8, 0))

        for other in (fewer_rows, wider):
            with pytest.raises(ValueError):
                if op == "add":
                    left + other
                elif op == "sub":
                    left - other
                elif op == "and":
                    left & other
                elif op == "iadd":
                    left += other
                elif op == "isub":
                    left -= other
                else:
                    left &= other

    def test_ops_on_empty_grids(self) -> None:
        """Empty grids combine to empty grids."""
        empty = Bitmap2D([])

        assert (empty + Bitmap2D([])).size() == (0, 0)
        assert (empty - Bitmap2D([])).size() == (0, 0)
        assert (empty & Bitmap2D([])).size() == (0, 0)


def _memory_config(
    devdax_path: str | None = None, devdax_size_in_bytes: int = 0
) -> L1MemoryManagerConfig:
    """Build a minimal L1 memory config, optionally backed by a devdax path.

    A devdax-backed config must run without shared memory, so the SHM name
    is cleared whenever a path is given.
    """
    return L1MemoryManagerConfig(
        size_in_bytes=1 << 20,
        use_lazy=False,
        shm_name="" if devdax_path is not None else "lmcache_test_l1_pool",
        devdax_path=devdax_path,
        devdax_size_in_bytes=devdax_size_in_bytes,
    )


class TestL1ManagerDescriptor:
    def test_dram_backend_by_default(self) -> None:
        """A plain memory config describes a DRAM-only L1 manager."""
        desc = L1ManagerDescriptor(index=0, config=L1ManagerConfig(_memory_config()))

        assert desc.index == 0
        assert desc.backend_types == {L1BackendType.DRAM}
        assert desc.type_name == "dram"

    def test_pure_devdax_backend(self) -> None:
        """A devdax path without a separate devdax size is Device-DAX only."""
        config = L1ManagerConfig(_memory_config(devdax_path="/dev/dax0.0"))
        desc = L1ManagerDescriptor(index=1, config=config)

        assert desc.backend_types == {L1BackendType.DEVDAX}
        assert desc.type_name == "devdax"

    def test_hybrid_devdax_spans_two_media(self) -> None:
        """A devdax path with its own size keeps a DRAM half as well."""
        config = L1ManagerConfig(
            _memory_config(devdax_path="/dev/dax0.0", devdax_size_in_bytes=1 << 20)
        )
        desc = L1ManagerDescriptor(index=1, config=config)

        assert desc.backend_types == {L1BackendType.DEVDAX, L1BackendType.DRAM}
        assert desc.type_name == "devdax+dram"

    def test_gds_backend_from_gds_config(self) -> None:
        """A GDS L1 config selects the GDS backend regardless of memory config."""
        config = L1ManagerConfig(
            _memory_config(devdax_path="/dev/dax0.0"),
            gds_l1_config=GdsL1Config(file_location="/mnt/nvme", size_in_bytes=1 << 20),
        )
        desc = L1ManagerDescriptor(index=2, config=config)

        assert desc.backend_types == {L1BackendType.GDS}
        assert desc.type_name == "gds"

    def test_descriptor_is_immutable(self) -> None:
        """Descriptors are frozen so they can be shared across controllers."""
        desc = L1ManagerDescriptor(index=0, config=L1ManagerConfig(_memory_config()))

        with pytest.raises(AttributeError):
            desc.index = 5  # type: ignore[misc]
