# SPDX-License-Identifier: Apache-2.0
"""
Speculative prefetch predictor.

Predicts which cache keys are likely to be requested *next*, so the storage
layer can prefetch them from a slower L2 tier into L1 before a request actually
asks for them. This is the software analog of the LSTM sequence predictor in
CXL-SpecKV (*A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM
Serving*, arXiv:2512.11920): instead of a neural network, it learns two cheap,
dependency-free signals online from the observed access stream.

  1. **First-order Markov successor model** — decayed transition weights
     ``P(next = B | last = A)``. Captures "a request for chunk ``A`` is usually
     followed by chunk ``B``": multi-turn chat continuation, RAG document
     co-access, and the ordered chunks of a single prompt.

  2. **Popularity prior** — a small-weight global access-frequency term that
     ranks broadly hot keys when the Markov model has no (or weak) evidence for
     the current context.

Both tables are explicitly bounded so memory stays flat in a long-running
server and prediction cost stays constant: the successor table is an LRU over
at most ``max_sources`` source keys, and the popularity table is capped at
``max_pop_keys`` (least-popular entries pruned). This keeps :meth:`observe`
amortized O(1) and :meth:`predict` O(``max_pop_keys`` + out-degree), independent
of the total number of distinct keys ever seen.

The predictor is intentionally generic over a hashable key type and imports
nothing from the rest of LMCache, so it is unit-testable in isolation and can
be reused for any key representation (``ObjectKey``, chunk hashes, etc.). The
:class:`~lmcache.v1.distributed.storage_controllers.prefetch_controller.PrefetchController`
owns an optional instance and feeds it the keys of every prefetch request;
turning the predictions into actual L2->L1 loads goes through the controller's
normal lock-owned prefetch path (see the design doc).

Thread-safety: a single :class:`SpeculativePrefetcher` is **not** internally
synchronized. The ``PrefetchController`` only touches it under its own lock.
Callers sharing one across threads must provide external locking.
"""

# Standard
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

KeyT = TypeVar("KeyT", bound=Hashable)

# Successor weights below this (after decay) are pruned to bound memory.
_PRUNE_EPS = 1e-3


@dataclass(frozen=True)
class PrefetchPrediction(Generic[KeyT]):
    """A single predicted key and its confidence.

    Attributes:
        key: The predicted-to-be-next cache key.
        score: Confidence in ``[0.0, 1.0]``; higher means more likely to be
            requested next. Predictions are returned sorted by descending
            score.
    """

    key: KeyT
    score: float


class SpeculativePrefetcher(Generic[KeyT]):
    """Online predictor of the next likely cache keys.

    The model is updated incrementally via :meth:`observe` /
    :meth:`observe_sequence` and queried via :meth:`predict`. Updates are
    amortized O(1) and predictions are O(``max_pop_keys`` + out-degree of the
    context) — both independent of the total number of distinct keys ever
    seen — so it is cheap enough to call on every request.

    Args:
        max_predictions: Default cap on the number of predictions returned by
            :meth:`predict` when no explicit ``k`` is given.
        min_confidence: Predictions scoring below this are discarded. Raising
            it trades recall for precision (fewer, higher-confidence prefetches).
        decay: Multiplicative recency decay in ``(0.0, 1.0]`` applied to a
            source key's existing successor weights each time a new transition
            from that source is recorded. ``1.0`` disables decay (pure counts);
            lower values make the model adapt faster to changing access
            patterns. The paper's learned predictor adapts continuously; this
            decay is the lightweight stand-in.
        popularity_weight: Weight in ``[0.0, 1.0]`` of the global popularity
            prior relative to the Markov term. ``0.0`` uses the Markov model
            only.
        max_sources: Upper bound on the number of distinct source keys tracked
            in the transition table. It is maintained as an LRU: when exceeded,
            the least-recently-updated source is evicted in O(1).
        max_pop_keys: Upper bound on the number of keys tracked in the
            popularity table. When exceeded, the least-popular half is pruned.
            Bounds both memory and per-prediction cost.

    Raises:
        ValueError: If any argument is outside its valid range.
    """

    def __init__(
        self,
        *,
        max_predictions: int = 8,
        min_confidence: float = 0.1,
        decay: float = 0.95,
        popularity_weight: float = 0.15,
        max_sources: int = 8192,
        max_pop_keys: int = 8192,
    ) -> None:
        if max_predictions < 1:
            raise ValueError(f"max_predictions must be >= 1, got {max_predictions}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {min_confidence}"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0.0, 1.0], got {decay}")
        if not 0.0 <= popularity_weight <= 1.0:
            raise ValueError(
                f"popularity_weight must be in [0.0, 1.0], got {popularity_weight}"
            )
        if max_sources < 1:
            raise ValueError(f"max_sources must be >= 1, got {max_sources}")
        if max_pop_keys < 1:
            raise ValueError(f"max_pop_keys must be >= 1, got {max_pop_keys}")

        self._max_predictions = max_predictions
        self._min_confidence = min_confidence
        self._decay = decay
        self._popularity_weight = popularity_weight
        self._max_sources = max_sources
        self._max_pop_keys = max_pop_keys

        # Transition table: source key -> {successor key -> decayed weight}.
        # An OrderedDict used as an LRU so eviction of the least-recently-used
        # source is O(1) (no scan over all sources).
        self._succ: "OrderedDict[KeyT, dict[KeyT, float]]" = OrderedDict()
        # Bounded global popularity: key -> count. Invariant: ``_total_pop ==
        # sum(_pop.values())`` so popularity fractions are over the retained
        # mass only.
        self._pop: dict[KeyT, float] = {}
        self._total_pop: float = 0.0
        # Last observed key, used as the default prediction context.
        self._last: Optional[KeyT] = None

    # -- model updates -------------------------------------------------------

    def observe(self, key: KeyT) -> None:
        """Record a single key access, updating popularity and the transition
        from the previously observed key.

        Consecutive duplicate observations do not record a self-transition
        (prefetching a key that was just requested is pointless), but they do
        still count toward popularity.

        Args:
            key: The cache key that was just accessed/requested.
        """
        is_new = key not in self._pop
        self._pop[key] = self._pop.get(key, 0.0) + 1.0
        self._total_pop += 1.0
        if is_new and len(self._pop) > self._max_pop_keys:
            self._prune_popularity()
        if self._last is not None and self._last != key:
            self._record_transition(self._last, key)
        self._last = key

    def observe_sequence(self, keys: Iterable[KeyT]) -> None:
        """Record an ordered group of accesses (e.g. the chunks of one prompt
        or one prefetch request), in order.

        Equivalent to calling :meth:`observe` for each key, which both builds
        intra-sequence successor edges (chunk ``i`` -> chunk ``i+1``) and links
        the previous context to this sequence's first key (cross-request
        continuation).

        Args:
            keys: Keys in access order.
        """
        for key in keys:
            self.observe(key)

    def _record_transition(self, src: KeyT, dst: KeyT) -> None:
        """Add a decayed ``src -> dst`` transition, recency-decaying the
        source's existing successors, pruning negligible ones, and maintaining
        the LRU bound on the number of source keys."""
        row = self._succ.get(src)
        if row is None:
            row = {}
            self._succ[src] = row
            # Evict the least-recently-used source (front) in O(1). The newly
            # inserted ``src`` is at the back, so it is never the victim.
            if len(self._succ) > self._max_sources:
                self._succ.popitem(last=False)
        else:
            # Mark this source as most-recently used.
            self._succ.move_to_end(src)
        if self._decay < 1.0:
            for k in list(row):
                w = row[k] * self._decay
                if w < _PRUNE_EPS:
                    del row[k]
                else:
                    row[k] = w
        row[dst] = row.get(dst, 0.0) + 1.0

    def _prune_popularity(self) -> None:
        """Bound the popularity table by keeping only the most-popular half,
        then re-establish the ``_total_pop == sum(_pop.values())`` invariant."""
        keep = max(1, self._max_pop_keys // 2)
        top = sorted(self._pop.items(), key=lambda kv: kv[1], reverse=True)[:keep]
        self._pop = dict(top)
        self._total_pop = sum(self._pop.values())

    # -- queries -------------------------------------------------------------

    def predict(
        self,
        recent: Optional[Sequence[KeyT]] = None,
        k: Optional[int] = None,
    ) -> list[PrefetchPrediction[KeyT]]:
        """Predict the next likely keys given recent context.

        Args:
            recent: Recently accessed keys, most-recent last. The last element
                is the Markov context; all of them are excluded from the
                output (a key just accessed should not be prefetched). If
                ``None``, the single most-recently observed key is used as
                context.
            k: Max number of predictions to return. Defaults to
                ``max_predictions``.

        Returns:
            Predictions sorted by descending score, then by a stable key order,
            filtered to ``score >= min_confidence`` and truncated to ``k``. May
            be empty when there is insufficient evidence.
        """
        limit = self._max_predictions if k is None else k
        if limit < 1:
            return []

        if recent:
            context = recent[-1]
            exclude = set(recent)
        elif self._last is not None:
            context = self._last
            exclude = {self._last}
        else:
            context = None
            exclude = set()

        scores: dict[KeyT, float] = {}

        # Markov successor component: normalized transition probabilities.
        if context is not None:
            row = self._succ.get(context)
            if row:
                total = sum(row.values())
                if total > 0.0:
                    for dst, w in row.items():
                        scores[dst] = scores.get(dst, 0.0) + w / total

        # Popularity prior (small weight): broadly-hot keys as a fallback.
        # Bounded by ``max_pop_keys`` so this loop is constant-cost.
        if self._popularity_weight > 0.0 and self._total_pop > 0.0:
            for key, w in self._pop.items():
                scores[key] = scores.get(key, 0.0) + self._popularity_weight * (
                    w / self._total_pop
                )

        predictions = [
            PrefetchPrediction(key=key, score=min(score, 1.0))
            for key, score in scores.items()
            if key not in exclude and min(score, 1.0) >= self._min_confidence
        ]
        # Descending score; stable, deterministic tie-break by key repr so the
        # output order does not depend on dict insertion order.
        predictions.sort(key=lambda p: (-p.score, repr(p.key)))
        return predictions[:limit]

    def predict_keys(
        self,
        recent: Optional[Sequence[KeyT]] = None,
        k: Optional[int] = None,
    ) -> list[KeyT]:
        """Convenience wrapper over :meth:`predict` returning just the keys."""
        return [p.key for p in self.predict(recent=recent, k=k)]

    def score(self, key: KeyT, recent: Optional[Sequence[KeyT]] = None) -> float:
        """Return the predicted next-access confidence for ``key`` in ``[0, 1]``.

        Useful for ranking or threshold decisions on a specific candidate. A
        key excluded as recent context, or with no evidence, scores ``0.0``.
        """
        for prediction in self.predict(recent=recent, k=None):
            if prediction.key == key:
                return prediction.score
        return 0.0

    def reset(self) -> None:
        """Clear all learned state."""
        self._succ.clear()
        self._pop.clear()
        self._total_pop = 0.0
        self._last = None

    # -- introspection -------------------------------------------------------

    @property
    def num_sources(self) -> int:
        """Number of distinct source keys in the transition table."""
        return len(self._succ)

    @property
    def num_keys_seen(self) -> int:
        """Number of keys currently tracked in the (bounded) popularity table."""
        return len(self._pop)
