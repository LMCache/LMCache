# SPDX-License-Identifier: Apache-2.0
"""Tests for lmcache.v1.numba_utils."""

# Third Party
import pytest

# First Party
from lmcache.v1 import numba_utils


def test_njit_cached_falls_back_when_disk_cache_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """njit_cached falls back to cache=False and stays functional."""
    real_njit = numba_utils.njit
    requested_cache_flags: list[bool] = []

    def fake_njit(cache: bool):
        requested_cache_flags.append(cache)
        if cache:
            raise RuntimeError(
                "cannot cache function 'f': no locator available for file 'f.py'"
            )
        return real_njit(cache=False)

    monkeypatch.setattr(numba_utils, "njit", fake_njit)

    @numba_utils.njit_cached
    def add_one(x: int) -> int:
        return x + 1

    assert requested_cache_flags == [True, False]
    assert add_one(41) == 42
