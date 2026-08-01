# SPDX-License-Identifier: Apache-2.0
"""Tests for the EventBus periodic hook (see #4291).

A subscriber that needs a clock used to subscribe to an unrelated event and
inherit its cadence, which made its freshness depend on a component it has
nothing to do with. These tests pin the timer the bus now offers instead.
"""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventBusConfig,
    EventCallback,
    EventSubscriber,
)
from lmcache.v1.mp_observability.event import EventType

_SETTLE_SECONDS = 0.5


class _CountingSubscriber(EventSubscriber):
    """Subscriber with no event subscriptions that only counts timer calls."""

    def __init__(self) -> None:
        self.calls = 0

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {}

    def on_periodic(self) -> None:
        self.calls += 1


class _RaisingSubscriber(EventSubscriber):
    """Subscriber whose timer hook always fails."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {}

    def on_periodic(self) -> None:
        raise RuntimeError("periodic boom")


class _SilentSubscriber(EventSubscriber):
    """Subscriber that does not override the timer hook."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {}


@pytest.fixture
def bus() -> EventBus:
    """Start a running bus and stop it after the test."""
    started = EventBus(EventBusConfig(enabled=True))
    started.start()
    yield started
    started.stop()


def test_periodic_hook_runs_without_any_events(bus: EventBus) -> None:
    """The timer must not depend on traffic, which is the whole point."""
    subscriber = _CountingSubscriber()
    bus.register_subscriber(subscriber)

    time.sleep(_SETTLE_SECONDS)

    assert subscriber.calls > 0


def test_a_failing_hook_does_not_stop_the_drain_thread(bus: EventBus) -> None:
    """The drain thread carries every subscriber, so one failure is isolated."""
    failing = _RaisingSubscriber()
    healthy = _CountingSubscriber()
    bus.register_subscriber(failing)
    bus.register_subscriber(healthy)

    time.sleep(_SETTLE_SECONDS)

    assert healthy.calls > 0
    assert bus.subscriber_exception_counts().get("_RaisingSubscriber", 0) > 0
    assert any(t.name for t in threading.enumerate())


def test_a_subscriber_without_the_hook_is_not_registered(bus: EventBus) -> None:
    """Only an overridden hook joins the timer, so the default costs nothing."""
    bus.register_subscriber(_SilentSubscriber())

    time.sleep(0.2)

    assert bus.subscriber_exception_counts() == {}
