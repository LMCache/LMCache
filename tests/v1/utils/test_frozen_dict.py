# SPDX-License-Identifier: Apache-2.0
"""Tests for immutable dictionary helpers."""

# Standard
from collections.abc import Callable
from types import ModuleType
import copy

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.utils.frozen_dict import FrozenDict, freeze_dict


def test_freeze_dict_snapshots_and_hashes_independent_of_order() -> None:
    source = {"a": 1, "b": 2}
    frozen = freeze_dict(source)

    source["a"] = 3

    assert frozen == {"a": 1, "b": 2}
    assert hash(frozen) == hash(freeze_dict({"b": 2, "a": 1}))
    with pytest.raises(TypeError, match="immutable"):
        frozen["a"] = 3


def test_freeze_dict_can_sort_keys() -> None:
    frozen = freeze_dict({"b": 2, "a": 1}, sort_keys=True)

    assert list(frozen) == ["a", "b"]


@pytest.mark.parametrize("sort_keys", [False, True])
def test_empty_snapshot_remains_empty_after_source_changes(sort_keys: bool) -> None:
    source: dict[str, int] = {}
    frozen = freeze_dict(source, sort_keys=sort_keys)
    source["a"] = 1

    assert frozen == {}
    assert isinstance(frozen, FrozenDict)
    assert hash(frozen) == hash(FrozenDict()) == hash(frozenset())
    assert {frozen: "found"}[freeze_dict({})] == "found"


@pytest.mark.parametrize("source", [{}, {"a": 1, "b": 2}])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("c", 3),
        lambda value: value.__delitem__("a"),
        lambda value: value.clear(),
        lambda value: value.pop("a"),
        lambda value: value.popitem(),
        lambda value: value.setdefault("c", 3),
        lambda value: value.update({"c": 3}),
        lambda value: value.__ior__({"c": 3}),
    ],
)
def test_snapshot_rejects_mutation(
    source: dict[str, int], mutate: Callable[[FrozenDict[str, int]], object]
) -> None:
    frozen = freeze_dict(source)
    before = hash(frozen)

    with pytest.raises(TypeError, match="immutable"):
        mutate(frozen)

    assert frozen == source
    assert hash(frozen) == before


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_empty_snapshot_wire_roundtrip(codec: ModuleType) -> None:
    frozen: FrozenDict[str, int] = freeze_dict({})
    restored = freeze_dict(codec.decode(codec.encode(frozen)))

    assert restored == frozen
    assert hash(restored) == hash(frozen)
    with pytest.raises(TypeError, match="immutable"):
        restored["a"] = 1


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
def test_empty_snapshot_copy(clone: Callable[[object], object]) -> None:
    frozen: FrozenDict[str, int] = freeze_dict({})
    restored = clone(frozen)

    assert restored == frozen
    assert hash(restored) == hash(frozen)


def test_nonempty_snapshot_hash_requires_hashable_values() -> None:
    frozen: FrozenDict[str, list[int]] = freeze_dict({"a": []})

    with pytest.raises(TypeError, match="unhashable"):
        hash(frozen)


@pytest.mark.parametrize("source", [{}, {"a": 1}])
def test_snapshot_type_reconstruction_preserves_mapping_hash(
    source: dict[str, int],
) -> None:
    frozen: FrozenDict[str, int] = freeze_dict({})
    restored = type(frozen)(source)

    assert restored == freeze_dict(source)
    assert hash(restored) == hash(freeze_dict(source))
