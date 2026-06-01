# SPDX-License-Identifier: Apache-2.0

"""Shared cache_salt helpers for metrics subscribers."""

# Future
from __future__ import annotations

# Standard
from collections import Counter

# Third Party
from opentelemetry import metrics


def emit_by_salt(counter: metrics.Counter, keys: list) -> None:
    """Add to *counter* once per distinct ``cache_salt`` in *keys*.

    When ``cache_salt`` is missing or empty, the counter increments
    without a ``cache_salt`` attribute (dimensionless), so it behaves
    identically to a plain ``counter.add(len(keys))``.
    """
    for salt, count in Counter(getattr(k, "cache_salt", "") for k in keys).items():
        attrs = {"cache_salt": salt} if salt else {}
        counter.add(count, attributes=attrs)
