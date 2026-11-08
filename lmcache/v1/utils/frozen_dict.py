# SPDX-License-Identifier: Apache-2.0
"""Immutable dictionary helpers."""

# Standard
from collections.abc import Mapping
from typing import Any, NoReturn, TypeVar, cast

K = TypeVar("K")
V = TypeVar("V")

_EMPTY_HASH = hash(frozenset())


class FrozenDict(dict[K, V]):
    """An immutable dictionary snapshot.

    Like other immutable containers, hashing requires every contained key and
    value to be hashable.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def _immutable(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class _EmptyFrozenDict(FrozenDict[K, V]):
    """Snapshot optimized for empty mappings and their reconstructed copies."""

    def __hash__(self) -> int:  # type: ignore[override]
        # dataclasses.asdict reconstructs dict subclasses through their type.
        return hash(frozenset(self.items())) if self else _EMPTY_HASH


def freeze_dict(mapping: Mapping[K, V], *, sort_keys: bool = False) -> FrozenDict[K, V]:
    """Return an independent immutable snapshot, optionally ordered by key.

    Empty snapshots use the same hash as an empty frozenset without allocating
    temporary item sets. Nonempty snapshots retain order-independent hashing.

    Args:
        mapping: Key/value pairs to copy into the snapshot.
        sort_keys: Whether to store entries in sorted key order.

    Returns:
        An immutable dictionary containing a shallow copy of the input entries.

    Raises:
        TypeError: Sorting is requested and the keys cannot be ordered.
    """
    if not mapping:
        return _EmptyFrozenDict()
    if sort_keys:
        return FrozenDict(sorted(mapping.items(), key=lambda item: cast(Any, item[0])))
    return FrozenDict(mapping)
