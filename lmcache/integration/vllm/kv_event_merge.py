# SPDX-License-Identifier: Apache-2.0
"""Merge per-worker KV event batches (importable without vLLM)."""

# Standard
from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


def merge_worker_kv_events(contributions: Iterable[Sequence[T]]) -> list[T]:
    """Return the order-preserving union of per-worker KV event batches.

    vLLM's ``KVEventAggregator`` counts the ranks that reported something in
    a step and keeps only the events all of them reported, so an event one
    rank reports while another rank reports a different batch in the same
    step is dropped for good. LMCache MP ranks complete their store futures
    independently, which makes that skew normal. The union keeps every
    distinct event once, first sighting wins, and a KV-aware router applies
    a repeated ``BlockStored`` idempotently.

    A repeated ``BlockRemoved`` is not idempotent, so the same record must
    never reach this merge from two ranks: exactly one rank per server reads
    the server's event log, see ``ParallelStrategy.is_kv_event_poller``.

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
