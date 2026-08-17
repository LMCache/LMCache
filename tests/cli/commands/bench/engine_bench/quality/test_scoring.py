# SPDX-License-Identifier: Apache-2.0
"""Tests for answer extraction, F1 scoring, and quality aggregation."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.scoring import (
    QualityAggregator,
    SampleScore,
    best_f1,
    extract_final_answer,
    normalize_answer,
    token_f1,
)


class TestExtractFinalAnswer:
    def test_extracts_delimited_answer(self) -> None:
        assert extract_final_answer("<final_answer>Paris</final_answer>") == "Paris"

    def test_strips_surrounding_whitespace(self) -> None:
        assert extract_final_answer("<final_answer>\n Paris \n</final_answer>") == (
            "Paris"
        )

    def test_ignores_text_around_the_region(self) -> None:
        response = "Let me think. The capital is <final_answer>Paris</final_answer>."
        assert extract_final_answer(response) == "Paris"

    def test_takes_the_last_complete_region(self) -> None:
        """A reasoning model may echo an example before its own answer."""
        response = (
            "For example <final_answer>Berlin</final_answer> would be the form. "
            "So: <final_answer>Paris</final_answer>"
        )
        assert extract_final_answer(response) == "Paris"

    def test_unterminated_region_is_not_an_answer(self) -> None:
        """A missing closing tag means generation was cut off."""
        assert extract_final_answer("thinking... <final_answer>Par") == ""

    def test_no_region_at_all(self) -> None:
        assert extract_final_answer("I think the answer is Paris.") == ""

    def test_empty_region(self) -> None:
        assert extract_final_answer("<final_answer></final_answer>") == ""

    def test_is_case_insensitive(self) -> None:
        assert extract_final_answer("<FINAL_ANSWER>Paris</FINAL_ANSWER>") == "Paris"

    def test_spans_newlines(self) -> None:
        assert extract_final_answer("<final_answer>a\nb</final_answer>") == "a\nb"


class TestNormalizeAnswer:
    def test_lowercases(self) -> None:
        assert normalize_answer("PARIS") == "paris"

    def test_strips_punctuation(self) -> None:
        assert normalize_answer("Exeter College, Oxford.") == "exeter college oxford"

    def test_strips_articles(self) -> None:
        assert normalize_answer("the University of a Place") == "university of place"

    def test_collapses_whitespace(self) -> None:
        assert normalize_answer("  a   b  ") == "b"

    def test_empty_string(self) -> None:
        assert normalize_answer("") == ""


class TestTokenF1:
    def test_exact_match(self) -> None:
        assert token_f1("Exeter College", "Exeter College") == 1.0

    def test_match_ignoring_case_and_punctuation(self) -> None:
        assert token_f1("exeter college!", "Exeter College") == 1.0

    def test_partial_overlap(self) -> None:
        # 2 of 3 predicted words overlap 2 of 2 reference words:
        # precision 2/3, recall 1.0, F1 0.8.
        assert token_f1("Exeter College Oxford", "Exeter College") == pytest.approx(0.8)

    def test_no_overlap(self) -> None:
        assert token_f1("Paris", "Exeter College") == 0.0

    def test_both_empty_agree(self) -> None:
        assert token_f1("", "") == 1.0

    def test_one_empty_disagrees(self) -> None:
        assert token_f1("", "Paris") == 0.0
        assert token_f1("Paris", "") == 0.0

    def test_repeated_words_are_clipped(self) -> None:
        """Repeated words are clipped to the reference count."""
        # "paris paris" vs "paris": overlap 1, precision 1/2, recall 1/1.
        assert token_f1("Paris Paris", "Paris") == pytest.approx(2 / 3)


class TestBestF1:
    def test_takes_the_best_reference(self) -> None:
        assert best_f1("Exeter College", ["Nope", "Exeter College"]) == 1.0

    def test_no_references_scores_zero(self) -> None:
        assert best_f1("Exeter College", []) == 0.0

    def test_all_references_wrong(self) -> None:
        assert best_f1("Paris", ["Berlin", "Madrid"]) == 0.0


def _score(sample_id: str, parsed: bool, f1: float) -> SampleScore:
    """Build a score, defaulting the fields a test is not exercising."""
    return SampleScore(
        sample_id=sample_id,
        parsed=parsed,
        f1=f1,
        answer="a" if parsed else "",
        ttft=0.1,
        num_output_tokens=5,
    )


class TestQualityAggregator:
    def test_empty_summary(self) -> None:
        summary = QualityAggregator().summarize()
        assert summary.num_samples == 0
        assert summary.num_parsed == 0
        assert summary.parse_rate == 0.0
        assert summary.f1_mean == 0.0

    def test_f1_mean_covers_parsed_samples_only(self) -> None:
        """An unparsed sample must not be averaged in as a zero."""
        aggregator = QualityAggregator()
        aggregator.record(_score("a", True, 1.0))
        aggregator.record(_score("b", False, 0.0))

        summary = aggregator.summarize()
        assert summary.num_samples == 2
        assert summary.num_parsed == 1
        assert summary.parse_rate == 0.5
        assert summary.f1_mean == 1.0

    def test_f1_mean_averages_over_parsed_samples(self) -> None:
        aggregator = QualityAggregator()
        aggregator.record(_score("a", True, 1.0))
        aggregator.record(_score("b", True, 0.5))
        assert aggregator.summarize().f1_mean == 0.75

    def test_scores_preserve_measurement_order(self) -> None:
        aggregator = QualityAggregator()
        aggregator.record(_score("b", True, 1.0))
        aggregator.record(_score("a", True, 1.0))
        assert [s.sample_id for s in aggregator.scores()] == ["b", "a"]
