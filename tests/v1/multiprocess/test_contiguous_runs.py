# SPDX-License-Identifier: Apache-2.0

"""Tests for contiguous_runs (store-path batch splitting at skipped chunks)."""

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import contiguous_runs


def test_contiguous_runs():
    a, b, c = object(), object(), object()
    assert contiguous_runs([]) == []
    assert contiguous_runs([None, None]) == []
    assert contiguous_runs([a, b, c]) == [(0, [a, b, c])]
    assert contiguous_runs([a, None, b, c]) == [(0, [a]), (2, [b, c])]
    assert contiguous_runs([None, a, b, None]) == [(1, [a, b])]
