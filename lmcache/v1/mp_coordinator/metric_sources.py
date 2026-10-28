# SPDX-License-Identifier: Apache-2.0
"""The components whose metrics the coordinator exports.

A component that has metrics calls :func:`register_metric_source` at the end
of its ``__init__``; ``observability.register_coordinator_metrics`` later
takes every registered component and creates its instruments. So the app
names no component for metrics, and no component touches OTel.

This module imports nothing from the coordinator: components import it,
while ``observability`` imports both it and the components.
"""

# Standard
from typing import Protocol
import threading


class MetricSource(Protocol):
    """A component whose metrics are read from its ``stats()`` snapshot."""

    def stats(self) -> object:
        """Return a snapshot of what the component's metrics report."""
        ...


# Keyed by class: a coordinator process builds one of each, so the latest
# one built is the one the app runs. Locked because a component may be built
# on any thread, while one call takes and clears them all.
_metric_sources: dict[type, MetricSource] = {}
_lock = threading.Lock()


def register_metric_source(component: MetricSource) -> None:
    """Record ``component`` so its metrics are registered when metrics are on.

    Call at the end of ``__init__``, from any thread. Replaces an earlier
    component of the same class.

    Args:
        component: A component whose metrics ``observability`` defines.
    """
    with _lock:
        _metric_sources[type(component)] = component


def get_and_clear_metric_sources() -> list[MetricSource]:
    """Return every registered component, in construction order, and clear them.

    Returns:
        The registered components; the next call returns only those built
        after this one.
    """
    with _lock:
        sources = list(_metric_sources.values())
        _metric_sources.clear()
    return sources
