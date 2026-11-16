# SPDX-License-Identifier: Apache-2.0
"""Immutable dictionary helpers."""

# Standard
from collections.abc import Mapping
from typing import Any, NoReturn, TypeVar, cast

K = TypeVar("K")
V = TypeVar("V")


class FrozenDict(dict[K, V]):
    """An immutable dictionary snapshot.

    Like other immutable containers, hashing requires every contained key and
    value to be hashable.
    """

    _hash: int | None = None
    """Lazily cached hash. Hot paths hash the same snapshot many times
    (every object key of a request shares one tags mapping), and the
    contents cannot change. Process-local: see ``__getstate__``."""

    def __hash__(self) -> int:  # type: ignore[override]
        # Benign race: concurrent first callers compute the same value.
        h = self._hash
        if h is None:
            h = hash(frozenset(self.items()))
            self._hash = h
        return h

    def __getstate__(self) -> None:
        """Keep the hash cache out of pickles and copies.

        String hashes are seeded per process, so a cached value is only
        valid in the process that computed it. Only an empty snapshot
        can be rebuilt on the other side: pickle and ``copy`` restore a
        dict subclass by re-inserting its items through ``__setitem__``,
        which this class blocks, so a snapshot with entries raises
        ``TypeError`` there, with or without the cache.
        """
        return None

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


def freeze_dict(mapping: Mapping[K, V], *, sort_keys: bool = False) -> FrozenDict[K, V]:
    """Return an immutable snapshot, optionally ordered by key."""
    if sort_keys:
        return FrozenDict(sorted(mapping.items(), key=lambda item: cast(Any, item[0])))
    return FrozenDict(mapping)
