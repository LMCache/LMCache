# SPDX-License-Identifier: Apache-2.0
"""Tests for the registered metric sources."""

# Standard
import threading

# First Party
from lmcache.v1.mp_coordinator.metric_sources import (
    get_and_clear_metric_sources,
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

    assert get_and_clear_metric_sources() == [first, second]
    assert get_and_clear_metric_sources() == []


def test_a_later_component_of_the_same_class_replaces_the_earlier() -> None:
    register_metric_source(_First())
    latest = _First()
    register_metric_source(latest)

    assert get_and_clear_metric_sources() == [latest]


def test_sources_registered_from_many_threads_are_all_taken() -> None:
    # One class per thread, so none replaces another.
    classes = [type(f"_Source{i}", (_First,), {}) for i in range(32)]
    taken: list[object] = []

    def register(cls: type) -> None:
        register_metric_source(cls())

    threads = [threading.Thread(target=register, args=(cls,)) for cls in classes]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        taken += get_and_clear_metric_sources()
    for thread in threads:
        thread.join()
    taken += get_and_clear_metric_sources()

    assert sorted(type(source).__name__ for source in taken) == sorted(
        cls.__name__ for cls in classes
    )
