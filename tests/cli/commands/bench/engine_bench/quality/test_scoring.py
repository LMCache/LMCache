# SPDX-License-Identifier: Apache-2.0
"""Tests for answer extraction, F1 scoring, and quality aggregation."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.scoring import (
    QualityAggregator,
    SampleScore,
    answer_match,
    best_f1,
    extract_final_answer,
    is_abstention,
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

    @pytest.mark.parametrize("suffix", ["Ber", "", "\n Ber\n", "Berlin</final_ans"])
    def test_truncated_final_region_does_not_fall_back_to_example(
        self, suffix: str
    ) -> None:
        """A truncated final region invalidates an earlier example answer."""
        response = (
            "For example <final_answer>Paris</final_answer>. "
            f"My answer: <FINAL_ANSWER>{suffix}"
        )
        assert extract_final_answer(response) == ""

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


class TestAnswerMatch:
    def test_exact_answer_matches(self) -> None:
        assert answer_match("Bassendean", ["Bassendean"])

    def test_gold_phrase_inside_a_longer_answer_matches(self) -> None:
        assert answer_match("It is based in Bassendean, WA.", ["Bassendean"])

    def test_partial_word_does_not_match(self) -> None:
        assert not answer_match("Bassendeans", ["Bassendean"])

    def test_alternate_phrasing_matches(self) -> None:
        assert answer_match("NYC", ["New York City", "NYC"])

    def test_same_figure_in_another_format_matches(self) -> None:
        """Token F1 scores this 0; the figure is the same."""
        assert answer_match("$11,588 million", ["$11588.00"])

    def test_same_figure_in_another_unit_matches(self) -> None:
        assert answer_match("$11.588 billion", ["$11588.00"])

    def test_rounded_figure_matches(self) -> None:
        assert answer_match("about $8.74 billion", ["$8738.00"])

    def test_neighbouring_figure_does_not_match(self) -> None:
        """The wrong line of the same table must not pass."""
        assert not answer_match("$3,033 million", ["$11588.00"])
        assert not answer_match("$24,873 million", ["$8738.00"])

    def test_every_gold_number_must_appear(self) -> None:
        assert not answer_match("2018", ["2018: $1577"])
        assert answer_match("In 2018 it was 1,577", ["2018: $1577"])

    def test_empty_answer_does_not_match(self) -> None:
        assert not answer_match("", ["Paris"])

    def test_digits_inside_words_are_not_numbers(self) -> None:
        """A shared digit must not make two different names match."""
        assert not answer_match("beta2", ["alpha2"])
        assert not answer_match("Q3 revenue", ["3"])

    def test_figure_with_unit_suffix_still_matches(self) -> None:
        assert answer_match("$1.5B", ["$1.5"])

    def test_rescale_needs_a_named_scale(self) -> None:
        """Without a scale word, a count must not match its thousandfold."""
        assert not answer_match("3,000", ["3"])
        assert not answer_match("about 5", ["5000"])
        assert answer_match("3 thousand", ["3000"])
        assert answer_match("$1.5B", ["$1500"])
        assert answer_match("1,500", ["$1.5 thousand"])

    def test_typographic_punctuation_is_folded(self) -> None:
        """Models emit curly apostrophes, non-breaking hyphens and spaces."""
        assert answer_match(
            "St Patrick\u2019s College in Dublin", ["St Patrick's College"]
        )
        assert answer_match(
            "at home in Goring\u2011on\u2011Thames", ["Goring-on-Thames"]
        )
        assert answer_match("**$1,577\u202fmillion**", ["$1577.00"])

    def test_hyphens_are_word_breaks(self) -> None:
        """Either side may hyphenate a compound the other spells open."""
        answer = "a massively\u2011multiplayer online role-playing game"
        assert answer_match(answer, ["massively multiplayer online role-playing game"])
        assert answer_match("role playing", ["role-playing"])

    def test_years_are_context_not_figures(self) -> None:
        """Naming the years in the question is not stating the answer."""
        assert not answer_match("+1.2% versus FY 2022", ["Flat in FY 2023 vs FY 2022."])
        assert not answer_match("for fiscal year 2023", ["$2,018mn in FY 2023"])
        assert answer_match(
            "Adjusted EBITDA was $2,018 million", ["$2,018mn in FY 2023"]
        )

    def test_sign_may_be_stated_in_words(self) -> None:
        assert answer_match("a \u20110.9% organic decline", ["Shrunk by 0.9%."])

    def test_billions_match_a_figure_in_units(self) -> None:
        assert answer_match("$8.4 billion in total", ["$8,400,000,000"])

    def test_abstention_naming_the_gold_does_not_match(self) -> None:
        answer = (
            "The passages give Robbie Gould's birth date but do not include one "
            "for Chris Gould."
        )
        assert not answer_match(answer, ["Chris Gould"])

    def test_abstention_may_match_a_negative_gold(self) -> None:
        answer = "The passages do not mention any acquisitions; there are none."
        assert answer_match(answer, ["There are none"])

    def test_abstention_like_title_still_matches(self) -> None:
        assert answer_match("Unknown Pleasures", ["Unknown Pleasures"])


class TestIsAbstention:
    @pytest.mark.parametrize(
        "answer",
        [
            "The provided passages do not contain a figure for adjusted EBITDA.",
            "Not provided",
            "UNKNOWN",
            "None",
            "This cannot be determined from the context.",
            "There is insufficient information.",
            "Therefore we cannot determine whether the margin improved.",
            "Without those numbers it is impossible to determine the trend.",
            "The provided passages do not name any companies Pfizer acquired.",
            "None of the excerpts state the county seat.",
            "The information supplied is insufficient to calculate it.",
            "It can\u2019t be determined from the passages.",
            "It isn\u2019t possible to tell which is younger.",
            "The quick ratio cannot be calculated from the passages.",
        ],
    )
    def test_refusals_are_abstentions(self, answer: str) -> None:
        assert is_abstention(answer)

    @pytest.mark.parametrize(
        "answer",
        [
            "Paris",
            "$1,577 million",
            "none of them",
            "The gains do not qualify as high-growth performance.",
            "Its liquid assets do not fully cover current liabilities.",
            "The company cannot reasonably estimate the loss.",
            "The income statement does not list a separate line item, so 0.",
        ],
    )
    def test_answers_are_not_abstentions(self, answer: str) -> None:
        assert not is_abstention(answer)

    def test_empty_answer_is_not_an_abstention(self) -> None:
        """An empty answer is unparsed, a different failure."""
        assert not is_abstention("")
