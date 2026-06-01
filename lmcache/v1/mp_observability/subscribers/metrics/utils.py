# SPDX-License-Identifier: Apache-2.0

"""Shared cache_salt helpers for metrics subscribers."""

# Future
from __future__ import annotations

# Standard
from collections import Counter

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event


def emit_by_salt(counter: metrics.Counter, keys: list) -> None:
    """Add to *counter* once per distinct ``cache_salt`` in *keys*.

    When ``cache_salt`` is missing or empty, the counter increments
    without a ``cache_salt`` attribute (dimensionless), so it behaves
    identically to a plain ``counter.add(len(keys))``.
    """
    for salt, count in Counter(getattr(k, "cache_salt", "") for k in keys).items():
        attrs = {"cache_salt": salt} if salt else {}
        counter.add(count, attributes=attrs)


def emit_salt_counts(counter: metrics.Counter, event: Event) -> None:
    """Add to *counter* using pre-grouped ``key_count_per_salt`` from *event*.

    The emit site computes ``Counter(k.cache_salt for k in keys)`` and
    stores it as ``key_count_per_salt`` in the event metadata so the
    drain thread only iterates over tenants (O(T)), not keys (O(N)).

    When ``key_count_per_salt`` is absent, this is a no-op.
    Empty salt produces a dimensionless increment (no attribute).
    """
    for salt, count in event.metadata.get("key_count_per_salt", {}).items():
        attrs = {"cache_salt": salt} if salt else {}
        counter.add(count, attributes=attrs)
