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
