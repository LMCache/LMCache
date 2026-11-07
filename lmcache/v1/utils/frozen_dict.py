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

    #: Cached result of ``__hash__``. The mapping is immutable, so the hash
    #: is computed once; object identities are hashed on every dict lookup
    #: that reaches them, and rebuilding a frozenset each time dominates.
    _hash: int | None = None

    def __hash__(self) -> int:  # type: ignore[override]
        cached = self._hash
        if cached is None:
            cached = hash(frozenset(self.items()))
            # Benign race: concurrent writers compute the same value.
            self._hash = cached
        return cached

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
