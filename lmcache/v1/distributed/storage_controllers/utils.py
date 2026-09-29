# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the storage controllers.

Holds the lightweight descriptors the controllers and policies use to tell
L1 managers and L2 adapters apart without touching runtime objects, and the
per-index bitmap map that the prefetch controller keeps for every request.
"""

# Standard
from collections.abc import ItemsView, Iterator
from dataclasses import dataclass

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import L1BackendType
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    get_configured_capacity_bytes,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)


@dataclass(frozen=True)
class L2AdapterDescriptor:
    """
    Lightweight descriptor for an L2 adapter, giving the store policy
    enough information to distinguish adapters without exposing runtime
    objects.
    """

    index: int
    """Position in the L2 adapters list."""

    config: L2AdapterConfigBase
    """The adapter's configuration object."""

    @property
    def type_name(self) -> str:
        """
        Registered adapter type name (e.g., "mock", "disk", "redis").

        Derived from the config's registered type via reverse lookup.

        Returns:
            str: The registered type name.
        """
        return get_type_name_for_config(self.config)


@dataclass(frozen=True)
class L1ManagerDescriptor:
    """
    Lightweight descriptor for an L1 manager, giving the controllers and
    policies enough information to distinguish L1 tiers without exposing
    runtime objects.
    """

    index: int
    """Position in the L1 managers list."""

    config: L1ManagerConfig
    """The manager's configuration object."""

    @property
    def backend_types(self) -> frozenset[L1BackendType]:
        """
        The storage media backing this L1 manager.

        A hybrid Device-DAX tier spans both DRAM and Device-DAX; every other
        tier has a single medium.

        Returns:
            The backend types with a non-zero configured capacity.
        """
        return frozenset(get_configured_capacity_bytes(self.config))

    @property
    def type_name(self) -> str:
        """
        Wire names of the backing media joined with ``+`` in sorted order
        (e.g., "dram", "gds", "devdax+dram").

        Returns:
            str: The combined backend type name.
        """
        return "+".join(sorted(backend.value for backend in self.backend_types))


class Bitmap2D:
    """A rectangular grid of bits stored as one Bitmap per row.

    Row ``i`` of a prefetch request is key group ``i`` and column ``j`` is
    chunk ``j``.

    Note:
        Every row has the same width.
    """

    def __init__(self, rows: list[Bitmap]) -> None:
        """Create a grid from its rows.

        Args:
            rows: One Bitmap per row, all of the same width.

        Raises:
            ValueError: If the rows have different widths.
        """
        cols = {row.size() for row in rows}
        if len(cols) > 1:
            raise ValueError(f"Rows have inconsistent number of columns: {cols}")
        self._rows: list[Bitmap] = list(rows)

    @classmethod
    def zeros(cls, num_rows: int, num_cols: int) -> "Bitmap2D":
        """Return a grid of the given shape with every bit cleared.

        Args:
            num_rows: Number of rows.
            num_cols: Width of every row.

        Raises:
            ValueError: If either dimension is negative.
        """
        if num_rows < 0 or num_cols < 0:
            raise ValueError(f"Negative shape: ({num_rows}, {num_cols})")
        return cls([Bitmap(num_cols) for _ in range(num_rows)])

    def size(self) -> tuple[int, int]:
        """Return the shape as ``(num_rows, num_cols)``; ``(0, 0)`` when empty."""
        if not self._rows:
            return (0, 0)
        return (len(self._rows), self._rows[0].size())

    def popcount(self) -> int:
        """Return the number of set bits over the whole grid."""
        return sum(row.popcount() for row in self._rows)

    def to_list(self) -> list[Bitmap]:
        """Return the rows as a new list holding the stored Bitmap objects."""
        return list(self._rows)

    def copy(self) -> "Bitmap2D":
        """Return a grid with the same bits and no shared Bitmap objects."""
        return Bitmap2D([row.copy() for row in self._rows])

    def zeros_like(self) -> "Bitmap2D":
        """Return a grid of the same shape with every bit cleared."""
        return Bitmap2D([Bitmap(row.size()) for row in self._rows])

    def ones_like(self) -> "Bitmap2D":
        """Return a grid of the same shape with every bit set."""
        return Bitmap2D([Bitmap(row.size(), row.size()) for row in self._rows])

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._rows)

    def __iter__(self) -> Iterator[Bitmap]:
        """Iterate over the rows in order."""
        return iter(self._rows)

    def __getitem__(self, idx: int) -> Bitmap:
        """Return the row at ``idx``."""
        return self._rows[idx]

    def _ensure_same_size(self, other: "Bitmap2D") -> None:
        """Raise ``ValueError`` unless ``other`` has this grid's shape."""
        if self.size() != other.size():
            raise ValueError(
                f"Bitmap2D sizes do not match: {self.size()} vs {other.size()}"
            )

    def __add__(self, other: "Bitmap2D") -> "Bitmap2D":
        """Return the row-wise union of the two grids.

        Raises:
            ValueError: If the shapes differ.
        """
        self._ensure_same_size(other)
        return Bitmap2D(
            [
                row_l | row_r
                for row_l, row_r in zip(self._rows, other._rows, strict=False)
            ]
        )

    def __iadd__(self, other: "Bitmap2D") -> "Bitmap2D":
        """Union ``other`` into this grid and return it.

        Raises:
            ValueError: If the shapes differ.
        """
        self._ensure_same_size(other)
        self._rows = [
            row_l | row_r for row_l, row_r in zip(self._rows, other._rows, strict=False)
        ]
        return self

    def __sub__(self, other: "Bitmap2D") -> "Bitmap2D":
        """Return this grid with every bit set in ``other`` cleared.

        Raises:
            ValueError: If the shapes differ.
        """
        self._ensure_same_size(other)
        return Bitmap2D(
            [
                row_l & ~row_r
                for row_l, row_r in zip(self._rows, other._rows, strict=False)
            ]
        )

    def __isub__(self, other: "Bitmap2D") -> "Bitmap2D":
        """Clear every bit set in ``other`` from this grid and return it.

        Raises:
            ValueError: If the shapes differ.
        """
        self._ensure_same_size(other)
        self._rows = [
            row_l & ~row_r
            for row_l, row_r in zip(self._rows, other._rows, strict=False)
        ]
        return self

    def __and__(self, other: "Bitmap2D") -> "Bitmap2D":
        """Return the row-wise intersection of the two grids.

        Raises:
            ValueError: If the shapes differ.
        """
        self._ensure_same_size(other)
        return Bitmap2D(
            [
                row_l & row_r
                for row_l, row_r in zip(self._rows, other._rows, strict=False)
            ]
        )

    def __iand__(self, other: "Bitmap2D") -> "Bitmap2D":
        """Intersect this grid with ``other`` and return it.

        Raises:
            ValueError: If the shapes differ.
        """
        self._ensure_same_size(other)
        self._rows = [
            row_l & row_r for row_l, row_r in zip(self._rows, other._rows, strict=False)
        ]
        return self


class MapState:
    """Tracks the global bitmap state.

    It maps from L2Adapter/L1Manager index to the global bitmap grid of the
    object key groups. Row ``i`` of every grid is the key group at index
    ``i`` of the prefetch request (see `GroupedObjectKeys`).

    Note:
        Every stored index carries a grid of the same shape. Grids adopted
        from another map are copied, so no two maps ever share a Bitmap
        object.
    """

    def __init__(self) -> None:
        self._state: dict[int, Bitmap2D] = {}
        self._num_rows: int | None = None
        self._num_cols: int | None = None

    def _ensure_layout(self, grid: Bitmap2D) -> None:
        """Ensure that ``grid`` has the shape every stored grid shares.

        Raises:
            ValueError: If ``grid`` has no rows, or its shape differs from
                the shape already established by an earlier store.
        """
        num_rows, num_cols = grid.size()
        if num_rows == 0:
            raise ValueError("Bitmaps cannot be empty")

        if self._num_rows is None:
            self._num_rows = num_rows
            self._num_cols = num_cols
        elif (num_rows, num_cols) != (self._num_rows, self._num_cols):
            raise ValueError(
                f"Shape ({num_rows} rows, {num_cols} cols) does not match "
                f"expected ({self._num_rows} rows, {self._num_cols} cols)"
            )

    def _ensure_compatible(self, other: "MapState") -> None:
        """Ensure that ``other`` can be combined with this map.

        Raises:
            ValueError: If both maps have a layout and the layouts differ.
        """
        if self._num_rows is None or other._num_rows is None:
            return
        if (self._num_rows, self._num_cols) != (other._num_rows, other._num_cols):
            raise ValueError(
                f"Layout ({other._num_rows} rows, {other._num_cols} cols) does "
                f"not match expected ({self._num_rows} rows, "
                f"{self._num_cols} cols)"
            )

    def merge(self) -> Bitmap2D:
        """Union the grids of every adapter index into one grid.

        Returns:
            A grid of the shared shape where a bit is set if any adapter index
            has it set. An empty map yields an empty grid.
        """
        if self._num_cols is None or self._num_rows is None:
            return Bitmap2D([])

        merged = Bitmap2D.zeros(self._num_rows, self._num_cols)
        for grid in self._state.values():
            merged += grid
        return merged

    def copy(self) -> "MapState":
        """Return a map with the same indices and bits and no shared Bitmap
        objects."""
        result = MapState()
        for adapter_idx, grid in self._state.items():
            result[adapter_idx] = grid.copy()
        return result

    def __setitem__(self, adapter_idx: int, grid: Bitmap2D) -> None:
        self._ensure_layout(grid)
        self._state[adapter_idx] = grid

    def __getitem__(self, adapter_idx: int) -> Bitmap2D:
        return self._state[adapter_idx]

    def __contains__(self, adapter_idx: int) -> bool:
        """Return whether ``adapter_idx`` has a grid stored in this map."""
        return adapter_idx in self._state

    def __delitem__(self, adapter_idx: int) -> None:
        """Remove ``adapter_idx`` and its grid from the map.

        Raises:
            KeyError: If ``adapter_idx`` has no grid stored.

        Note:
            The layout established by earlier stores is kept, even when the
            last index is removed.
        """
        del self._state[adapter_idx]

    def items(self) -> ItemsView[int, Bitmap2D]:
        """Return a view of ``(adapter_idx, grid)`` pairs, like ``dict.items``.

        Pairs are yielded in insertion order. The grids are the stored
        objects themselves, not copies.
        """
        return self._state.items()

    def __add__(self, other: "MapState") -> "MapState":
        """Merge two MapState instances by unioning their grids per adapter
        index.

        If an adapter index exists in one instance but not the other, the
        missing grid is treated as all zeros.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        result = MapState()

        for adapter_idx, grid_l in self._state.items():
            grid_r = other._state.get(adapter_idx, None)
            if grid_r is not None:
                result[adapter_idx] = grid_l + grid_r
            else:
                result[adapter_idx] = grid_l.copy()

        for adapter_idx, grid_r in other._state.items():
            if adapter_idx not in self._state:
                result[adapter_idx] = grid_r.copy()

        return result

    def __iadd__(self, other: "MapState") -> "MapState":
        """In-place union of two MapState instances.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        for adapter_idx, grid_r in other._state.items():
            if adapter_idx in self._state:
                self[adapter_idx] = self._state[adapter_idx] + grid_r
            else:
                self[adapter_idx] = grid_r.copy()
        return self

    def __sub__(self, other: "MapState") -> "MapState":
        """Clears the bits in this MapState that are set in the other MapState
        and returns a new MapState instance with the result.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        result = MapState()

        for adapter_idx, grid_l in self._state.items():
            grid_r = other._state.get(adapter_idx, None)
            if grid_r is not None:
                result[adapter_idx] = grid_l - grid_r
            else:
                result[adapter_idx] = grid_l.copy()

        return result

    def __isub__(self, other: "MapState") -> "MapState":
        """In-place clears the bits in this MapState that are set in the
        other MapState.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        for adapter_idx, grid_r in other._state.items():
            if adapter_idx in self._state:
                self[adapter_idx] = self._state[adapter_idx] - grid_r
        return self
