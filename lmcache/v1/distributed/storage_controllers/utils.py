# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the storage controllers.

Holds the lightweight descriptors the controllers and policies use to tell
L1 managers and L2 adapters apart without touching runtime objects, and the
per-index bitmap map that the prefetch controller keeps for every request.
"""

# Standard
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


def _copy_rows(bitmaps: list[Bitmap]) -> list[Bitmap]:
    """Return an independent copy of every bitmap in ``bitmaps``.

    Args:
        bitmaps: The bitmaps to copy.

    Returns:
        New bitmaps with the same sizes and bits, sharing no storage with
        the inputs.
    """
    return [Bitmap(bitmap.size()) | bitmap for bitmap in bitmaps]


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


class MapState:
    """Tracks the global bitmap state.

    It maps from L2Adapter/L1Manager index to the global bitmaps of the
    object key groups.
    The object key groups are represented by the list of Bitmaps. Each
    element corresponds to the "row" in the prefetch request (see
    `GroupedObjectKeys`)

    Note:
        Every stored index carries the same number of rows and the same
        row width. Rows adopted from another map are copied, so no two
        maps ever share a Bitmap object.
    """

    def __init__(self) -> None:
        self._state: dict[int, list[Bitmap]] = {}
        self._num_rows: int | None = None
        self._num_cols: int | None = None

    def _ensure_layout(self, bitmaps: list[Bitmap]) -> None:
        """Ensure that the layout of the bitmaps is consistent across all
        adapters. The number of rows and columns should be the same for all
        adapters.

        Raises:
            ValueError: If the layout of the bitmaps is inconsistent, or
                if the bitmaps are empty.
        """
        num_cols = set(bitmap.size() for bitmap in bitmaps)
        if not num_cols:
            raise ValueError("Bitmaps cannot be empty")

        if len(num_cols) > 1:
            raise ValueError(f"Bitmaps have inconsistent number of columns: {num_cols}")

        if self._num_rows is None:
            self._num_rows = len(bitmaps)
            self._num_cols = num_cols.pop()
        else:
            if len(bitmaps) != self._num_rows:
                raise ValueError(
                    f"Number of rows {len(bitmaps)} does not match "
                    f"expected {self._num_rows}"
                )
            if num_cols.pop() != self._num_cols:
                raise ValueError(
                    f"Number of columns {num_cols} does not match "
                    f"expected {self._num_cols}"
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

    def merge(self) -> list[Bitmap]:
        """Union the rows of every adapter index into a single row list.

        Returns:
            One bitmap per row, where a bit is set if any adapter index has
            it set. An empty map yields an empty list.
        """
        if self._num_cols is None or self._num_rows is None:
            return []

        merged: list[Bitmap] = [Bitmap(self._num_cols) for _ in range(self._num_rows)]
        for bitmaps in self._state.values():
            merged = [m | bitmap for m, bitmap in zip(merged, bitmaps, strict=False)]
        return merged

    def __setitem__(self, adapter_idx: int, bitmaps: list[Bitmap]) -> None:
        self._ensure_layout(bitmaps)
        self._state[adapter_idx] = bitmaps

    def __getitem__(self, adapter_idx: int) -> list[Bitmap]:
        return self._state[adapter_idx]

    def __contains__(self, adapter_idx: int) -> bool:
        """Return whether ``adapter_idx`` has rows stored in this map."""
        return adapter_idx in self._state

    def __add__(self, other: "MapState") -> "MapState":
        """Merge two MapState instances by unioning their bitmaps
        per adapter index.

        If an adapter index exists in one instance but not the other, the
        missing bitmap is treated as an empty bitmap (no keys).

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        result = MapState()

        for adapter_idx, bitmaps_l in self._state.items():
            bitmaps_r = other._state.get(adapter_idx, None)
            if bitmaps_r is not None:
                result[adapter_idx] = [
                    b_l | b_r for b_l, b_r in zip(bitmaps_l, bitmaps_r, strict=False)
                ]
            else:
                result[adapter_idx] = _copy_rows(bitmaps_l)

        for adapter_idx, bitmaps_r in other._state.items():
            if adapter_idx not in self._state:
                result[adapter_idx] = _copy_rows(bitmaps_r)

        return result

    def __iadd__(self, other: "MapState") -> "MapState":
        """In-place union of two MapState instances.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        for adapter_idx, bitmaps_r in other._state.items():
            if adapter_idx in self._state:
                self[adapter_idx] = [
                    b_l | b_r
                    for b_l, b_r in zip(
                        self._state[adapter_idx], bitmaps_r, strict=False
                    )
                ]
            else:
                self[adapter_idx] = _copy_rows(bitmaps_r)
        return self

    def __sub__(self, other: "MapState") -> "MapState":
        """Clears the bits in this MapState that are set in the other MapState
        and returns a new MapState instance with the result.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        result = MapState()

        for adapter_idx, bitmaps_l in self._state.items():
            bitmaps_r = other._state.get(adapter_idx, None)
            if bitmaps_r is not None:
                result[adapter_idx] = [
                    b_l & ~b_r for b_l, b_r in zip(bitmaps_l, bitmaps_r, strict=False)
                ]
            else:
                result[adapter_idx] = _copy_rows(bitmaps_l)

        return result

    def __isub__(self, other: "MapState") -> "MapState":
        """In-place clears the bits in this MapState that are set in the
        other MapState.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        for adapter_idx, bitmaps_r in other._state.items():
            if adapter_idx in self._state:
                self[adapter_idx] = [
                    b_l & ~b_r
                    for b_l, b_r in zip(
                        self._state[adapter_idx], bitmaps_r, strict=False
                    )
                ]
        return self
