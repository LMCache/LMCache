# SPDX-License-Identifier: Apache-2.0
"""Tests for immutable dictionary helpers."""

# Third Party
import pytest

# First Party
from lmcache.v1.utils.frozen_dict import freeze_dict


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


def test_hash_is_stable_and_stays_out_of_serialized_state() -> None:
    frozen = freeze_dict({"a": 1, "b": 2})
    first = hash(frozen)
    assert hash(frozen) == first
    assert hash(freeze_dict({"b": 2, "a": 1})) == first

    # The cached hash is process-local (string hashes are seeded per
    # process), so the state slot handed to pickle and copy stays empty;
    # the contents travel through the regular dict-items channel.
    assert frozen.__reduce_ex__(2)[2] is None
    assert dict(frozen) == {"a": 1, "b": 2}
