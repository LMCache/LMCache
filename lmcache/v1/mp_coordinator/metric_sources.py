# SPDX-License-Identifier: Apache-2.0
"""The components whose metrics the coordinator exports.

A component that has metrics calls :func:`register_metric_source` at the end
of its ``__init__``; ``observability.register_coordinator_metrics`` later
takes every registered component and creates its instruments. So the app
names no component for metrics, and no component touches OTel.

This module imports nothing from the coordinator: components import it,
while ``observability`` imports both it and the components.
"""

# Keyed by class: a coordinator process builds one of each, so the latest
# one built is the one the app runs.
_metric_sources: dict[type, object] = {}


def register_metric_source(component: object) -> None:
    """Record ``component`` so its metrics are registered when metrics are on.

    Call at the end of ``__init__``. Replaces an earlier component of the
    same class.

    Args:
        component: A component whose metrics ``observability`` defines.
    """
    _metric_sources[type(component)] = component


def take_metric_sources() -> list[object]:
    """Return every registered component, in construction order, and forget them.

    Returns:
        The registered components; the next call returns only those built
        after this one.
    """
    components = list(_metric_sources.values())
    _metric_sources.clear()
    return components
