# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for emitting OTel metrics tagged with ``cache_salt``.

Metric subscribers that receive ``list[ObjectKey]`` payloads group their
emissions by ``key.cache_salt`` so a single event whose key list spans
multiple tenants produces one datapoint per tenant. These helpers
centralise the grouping + emission so every subscriber stays consistent
and a future cardinality control (hashing, bucketing) has a single edit
site.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from collections.abc import Iterable
from typing import Any

# Third Party
from opentelemetry import metrics


def group_by_salt(keys: Iterable) -> Counter[str]:
    """Return ``{cache_salt: count}`` for *keys*.

    Objects without a ``cache_salt`` attribute fall back to the empty
    string so the helper is safe for any iterable.
    """
    return Counter(getattr(k, "cache_salt", "") for k in keys)


def unique_salts(*key_lists: Iterable) -> set[str]:
    """Return the set of ``cache_salt`` values across all *key_lists*."""
    return {getattr(k, "cache_salt", "") for keys in key_lists for k in keys}


def emit_by_salt(
    counter: metrics.Counter,
    keys: Iterable,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Group *keys* by ``cache_salt`` and emit one ``.add()`` per group.

    Args:
        counter: OTel Counter instrument.
        keys: Iterable of objects carrying ``cache_salt``. An empty
            iterable is a no-op.
        extra_attrs: Additional attributes merged into each datapoint
            (e.g. ``instance_id``, ``model_name`` for lifecycle histograms).
    """
    for salt, count in group_by_salt(keys).items():
        attrs = {"cache_salt": salt}
        if extra_attrs:
            attrs.update(extra_attrs)
        counter.add(count, attrs)


def emit_request_per_salt(
    counter: metrics.Counter,
    *key_lists: Iterable,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Emit one ``+1`` increment per distinct ``cache_salt`` touched.

    When no keys are present (which should not happen for a real call),
    the request is still recorded against the empty-salt tenant so the
    series reflects that the call occurred.
    """
    salts = unique_salts(*key_lists) or {""}
    for salt in salts:
        attrs = {"cache_salt": salt}
        if extra_attrs:
            attrs.update(extra_attrs)
        counter.add(1, attrs)
