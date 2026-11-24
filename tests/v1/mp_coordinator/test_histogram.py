# SPDX-License-Identifier: Apache-2.0
"""Tests for the histogram components keep in their stats()."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.utils.histogram import Histogram


def test_an_observation_goes_in_the_first_bucket_at_or_above_it():
    histogram = Histogram.with_bounds((1, 5, 10))

    for value in (0, 1, 3, 10, 11):
        histogram.observe(value)

    assert histogram.bucket_counts == [2, 1, 1, 1]
    assert (histogram.sum, histogram.count) == (25, 5)


def test_cumulative_buckets_end_with_every_observation():
    histogram = Histogram.with_bounds((1, 2.5))
    for value in (0, 2, 7):
        histogram.observe(value)

    assert histogram.cumulative_buckets() == [("1", 1), ("2.5", 2), ("+Inf", 3)]


def test_a_copy_does_not_follow_later_observations():
    histogram = Histogram.with_bounds((1,))
    snapshot = histogram.copy()

    histogram.observe(0)

    assert (snapshot.bucket_counts, snapshot.count) == ([0, 0], 0)


@pytest.mark.parametrize("bounds", [(), (1, 1), (5, 2)])
def test_bounds_must_be_non_empty_and_ascending(bounds: tuple[float, ...]):
    with pytest.raises(ValueError):
        Histogram.with_bounds(bounds)
