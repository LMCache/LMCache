# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SpeculativePrefetcher predictor.

These tests are intentionally dependency-free (the predictor imports nothing
from the rest of LMCache), so they run without torch / native extensions.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.storage_controllers.speculative_prefetcher import (
    PrefetchPrediction,
    SpeculativePrefetcher,
)


class TestConstruction:
    def test_defaults(self):
        sp = SpeculativePrefetcher()
        assert sp.num_sources == 0
        assert sp.num_keys_seen == 0
        assert sp.predict() == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_predictions": 0},
            {"min_confidence": -0.1},
            {"min_confidence": 1.1},
            {"decay": 0.0},
            {"decay": 1.1},
            {"popularity_weight": -0.1},
            {"popularity_weight": 1.1},
            {"max_sources": 0},
            {"max_pop_keys": 0},
        ],
    )
    def test_invalid_args_raise(self, kwargs):
        with pytest.raises(ValueError):
            SpeculativePrefetcher(**kwargs)


class TestMarkovPrediction:
    def test_learns_simple_successor(self):
        """After seeing A->B repeatedly, predicting from A yields B."""
        sp = SpeculativePrefetcher(popularity_weight=0.0)
        for _ in range(5):
            sp.observe("A")
            sp.observe("B")
        preds = sp.predict(recent=["A"])
        assert preds, "expected at least one prediction"
        assert preds[0].key == "B"
        assert preds[0].score == pytest.approx(1.0)

    def test_excludes_context_keys(self):
        """A key in the recent context is never predicted back."""
        sp = SpeculativePrefetcher(popularity_weight=0.0)
        sp.observe_sequence(["A", "B", "A", "B"])
        keys = sp.predict_keys(recent=["A", "B"])
        assert "A" not in keys
        assert "B" not in keys

    def test_no_self_transition(self):
        """Consecutive duplicates do not create an A->A edge."""
        sp = SpeculativePrefetcher(popularity_weight=0.0)
        sp.observe("A")
        sp.observe("A")
        sp.observe("A")
        # No successor evidence for A at all.
        assert sp.predict(recent=["A"]) == []
        assert sp.num_sources == 0

    def test_probabilities_rank_by_frequency(self):
        """B follows A more often than C, so B outranks C."""
        sp = SpeculativePrefetcher(popularity_weight=0.0, decay=1.0)
        seq = ["A", "B", "A", "B", "A", "B", "A", "C"]
        sp.observe_sequence(seq)
        preds = sp.predict(recent=["A"])
        keys = [p.key for p in preds]
        assert keys[0] == "B"
        assert "C" in keys
        assert preds[0].score > preds[1].score

    def test_default_context_is_last_observed(self):
        sp = SpeculativePrefetcher(popularity_weight=0.0)
        sp.observe_sequence(["A", "B", "A", "B"])
        sp.observe("A")  # last == "A"
        assert sp.predict_keys()[0] == "B"

    def test_min_confidence_filters(self):
        sp = SpeculativePrefetcher(popularity_weight=0.0, min_confidence=0.6, decay=1.0)
        # From A: B 3 times (0.75), C 1 time (0.25). Only B clears 0.6.
        sp.observe_sequence(["A", "B", "A", "B", "A", "B", "A", "C"])
        keys = sp.predict_keys(recent=["A"])
        assert keys == ["B"]

    def test_k_truncates(self):
        sp = SpeculativePrefetcher(popularity_weight=0.0, min_confidence=0.0)
        sp.observe_sequence(["A", "B", "A", "C", "A", "D"])
        assert len(sp.predict(recent=["A"], k=1)) == 1
        assert sp.predict(recent=["A"], k=0) == []


class TestPopularity:
    def test_popularity_fallback_when_no_markov(self):
        """With no transition evidence for the context, the popularity prior
        still surfaces broadly-hot keys."""
        sp = SpeculativePrefetcher(popularity_weight=1.0, min_confidence=0.0)
        for _ in range(3):
            sp.observe("HOT")
        sp.observe("X")  # context with no successors
        keys = sp.predict_keys(recent=["UNSEEN"])
        assert "HOT" in keys

    def test_markov_outranks_popularity(self):
        sp = SpeculativePrefetcher(popularity_weight=0.2, min_confidence=0.0)
        # POP is globally frequent, but B specifically follows A.
        for _ in range(10):
            sp.observe("POP")
        sp.observe_sequence(["A", "B", "A", "B"])
        preds = sp.predict(recent=["A"])
        assert preds[0].key == "B"


class TestRecencyDecay:
    def test_decay_favors_recent_transition(self):
        """With decay < 1, a newer successor overtakes an old, stale one."""
        sp = SpeculativePrefetcher(popularity_weight=0.0, decay=0.5)
        # Many old A->OLD, then a few recent A->NEW. Decay erodes OLD.
        for _ in range(5):
            sp.observe("A")
            sp.observe("OLD")
        for _ in range(5):
            sp.observe("A")
            sp.observe("NEW")
        assert sp.predict_keys(recent=["A"])[0] == "NEW"

    def test_no_decay_is_pure_counts(self):
        sp = SpeculativePrefetcher(popularity_weight=0.0, decay=1.0)
        for _ in range(5):
            sp.observe("A")
            sp.observe("OLD")
        for _ in range(3):
            sp.observe("A")
            sp.observe("NEW")
        # Counts: OLD=5 > NEW=3, so OLD still wins without decay.
        assert sp.predict_keys(recent=["A"])[0] == "OLD"


class TestBoundsAndReset:
    def test_max_sources_eviction(self):
        sp = SpeculativePrefetcher(popularity_weight=0.0, max_sources=2)
        sp.observe_sequence(["A", "B"])  # source A
        sp.observe_sequence(["C", "D"])  # source C (B->C also)
        sp.observe_sequence(["E", "F"])  # forces eviction
        assert sp.num_sources <= 2

    def test_max_pop_keys_bound(self):
        """The popularity table stays bounded no matter how many distinct keys
        are observed (no unbounded growth / memory leak)."""
        sp = SpeculativePrefetcher(max_pop_keys=8)
        for i in range(1000):
            sp.observe(f"key-{i}")
        assert sp.num_keys_seen <= 8

    def test_lru_source_eviction_keeps_recent(self):
        """Eviction is least-recently-used: re-recording a source refreshes it
        so a newly-added source evicts the stale one instead.

        Drives the transition recorder directly to isolate the eviction policy
        from the successor edges that ``observe_sequence`` would also create.
        """
        sp = SpeculativePrefetcher(popularity_weight=0.0, max_sources=2)
        sp._record_transition("A", "x")  # sources: [A]
        sp._record_transition("B", "y")  # sources: [A, B]
        sp._record_transition("A", "z")  # refresh A -> most recent: [B, A]
        sp._record_transition("C", "w")  # over cap -> evict LRU (B): [A, C]
        assert sp.num_sources == 2
        assert sp.predict_keys(recent=["A"])  # A survived
        assert sp.predict_keys(recent=["C"])  # C present
        assert sp.predict_keys(recent=["B"]) == []  # B evicted

    def test_reset_clears_state(self):
        sp = SpeculativePrefetcher()
        sp.observe_sequence(["A", "B", "C"])
        sp.reset()
        assert sp.num_sources == 0
        assert sp.num_keys_seen == 0
        assert sp.predict() == []

    def test_deterministic_order_with_ties(self):
        """Equal-score predictions come out in a stable, repeatable order."""
        sp = SpeculativePrefetcher(popularity_weight=0.0, min_confidence=0.0, decay=1.0)
        # From A: B and C each once -> equal 0.5 score.
        sp.observe_sequence(["A", "B", "A", "C"])
        first = sp.predict_keys(recent=["A"])
        second = sp.predict_keys(recent=["A"])
        assert first == second


class TestObjectKeyLikeUsage:
    def test_works_with_tuple_keys(self):
        """Frozen-dataclass-like keys (tuples here) work as keys."""
        sp = SpeculativePrefetcher(popularity_weight=0.0)
        a = ("model", 0, b"hashA")
        b = ("model", 0, b"hashB")
        for _ in range(3):
            sp.observe(a)
            sp.observe(b)
        preds = sp.predict(recent=[a])
        assert isinstance(preds[0], PrefetchPrediction)
        assert preds[0].key == b


if __name__ == "__main__":
    # Standard
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
