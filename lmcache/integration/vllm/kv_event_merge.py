# SPDX-License-Identifier: Apache-2.0
"""Merge per-worker KV event batches (importable without vLLM)."""

# Standard
from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


def merge_worker_kv_events(contributions: Iterable[Sequence[T]]) -> list[T]:
    """Return the order-preserving union of per-worker KV event batches.

    vLLM's ``KVEventAggregator`` keeps only the events every worker reported
    in the same step. LMCache MP workers finish store futures and drain the
    server's event log independently, so the same event usually reaches the
    scheduler from different workers in different steps, and an intersection
    would drop it for good. The union keeps every distinct event once (first
    sighting wins); that is safe because a router applies stores and
    removals idempotently.

    Args:
        contributions: One event sequence per worker, in worker order.

    Returns:
        Every distinct event in first-seen order. Unhashable events are kept
        as they are.
    """
    merged: list[T] = []
    seen: set[T] = set()
    for events in contributions:
        for event in events:
            try:
                if event in seen:
                    continue
                seen.add(event)
            except TypeError:
                pass
            merged.append(event)
    return merged
