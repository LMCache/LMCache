# SPDX-License-Identifier: Apache-2.0
"""Tests for the registered metric sources."""

# First Party
from lmcache.v1.mp_coordinator.metric_sources import (
    get_metric_sources,
    register_metric_source,
)


class _First:
    def stats(self) -> None:
        return None


class _Second:
    def stats(self) -> None:
        return None


def test_sources_are_taken_in_construction_order_then_forgotten() -> None:
    first, second = _First(), _Second()
    register_metric_source(first)
    register_metric_source(second)

    assert get_metric_sources() == [first, second]
    assert get_metric_sources() == []


def test_a_later_component_of_the_same_class_replaces_the_earlier() -> None:
    register_metric_source(_First())
    latest = _First()
    register_metric_source(latest)

    assert get_metric_sources() == [latest]
