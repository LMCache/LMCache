# SPDX-License-Identifier: Apache-2.0
"""LRU-based prefetch predictor for the engine-driven transfer path.

Tracks recent (key, prefix_hit_count, total_chunks) tuples.  When a new
lookup arrives, ``predict()`` returns the most likely next key to
prefetch based on the access pattern.

Heuristic
---------
For the user's same-prompt benchmark (re-running the same key), the
``predict()`` output is always the same key -- which doesn't help
because the data is already in L1.

For workloads with overlapping prefixes (e.g. chat sessions, doc QA
where each turn extends the prior), the most recent key is the best
predictor for the next one.  We track the last 8 (key, hit_count)
pairs and return the previous key when the current lookup fully
misses the L1 cache.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from typing import Any


class PrefetchPredictor:
    """Tiny LRU predictor for engine-driven prefetch hints.

    Usage::

        predictor = PrefetchPredictor(max_entries=8)
        predictor.record(key, hit_count=0, total_chunks=234)
        # ... later ...
        next_hint = predictor.predict(current_key)
        if next_hint and next_hint != current_key:
            schedule_prefetch(next_hint)
    """

    def __init__(self, max_entries: int = 8) -> None:
        self._max = max_entries
        # Most-recently-used first.  Each entry: (key, hit_count,
        # total_chunks, last_seen).  ``OrderedDict.move_to_end`` is
        # the cache eviction primitive.
        self._entries: "OrderedDict[Any, tuple[int, int]]" = OrderedDict()

    def record(
        self,
        key: Any,
        hit_count: int,
        total_chunks: int,
    ) -> None:
        """Record an observed lookup.  Evicts the oldest entry on overflow."""
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = (hit_count, total_chunks)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def predict(self, current_key: Any) -> Any | None:
        """Return the key most likely to be looked up next.

        Heuristic: the most-recently-used key that is *not* the current
        key.  Returns ``None`` if the predictor has no useful history
        (only the current key is known).
        """
        if not self._entries:
            return None
        for key in reversed(self._entries.keys()):
            if key != current_key:
                return key
        return None

    def stats(self) -> dict[str, int]:
        """Snapshot the predictor state for logging/metrics."""
        return {
            "size": len(self._entries),
            "max": self._max,
        }
