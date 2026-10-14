# SPDX-License-Identifier: Apache-2.0
"""Answer extraction and scoring: token-overlap F1, answer match, abstention."""

# Standard
from dataclasses import dataclass
import re
import string
import unicodedata

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)

# Lazy match, last region wins: reasoning models may echo an example answer
# before their own. Include an unfinished last region to avoid scoring the example.
_FINAL_ANSWER = re.compile(
    r"<final_answer>\s*(.*?)\s*(</final_answer>|$)",
    re.IGNORECASE | re.DOTALL,
)

# A number with optional thousands separators and decimals ("11,588.00").
# A number must not continue a word ("alpha2", "Q3"); trailing unit letters
# ("5B") are fine.
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])-?\d[\d,]*(?:\.\d+)?")

# A bare four-digit year.  Years in a gold answer are usually context ("in FY
# 2023"), not the figure asked for, and must never match within a tolerance
# ("2,018" vs "2023").
_YEAR = re.compile(r"(?:1[89]|20)\d\d")

# Relative tolerance when comparing numbers: absorbs rounding ("8.74" vs
# "8.738") without letting neighbouring table values ("8.74" vs "8.81") match.
_NUMBER_REL_TOL = 0.005

# Unit rescalings accepted between gold and answer.  Financial golds are often
# stated in one unit and answered in another ("$1577.00" in millions vs
# "$1.577 billion"), which is the same figure.
_UNIT_SCALES = (1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9)

# A rescale is only accepted when the gold or the answer names a scale, so a
# count of "3" does not match "3,000".
_SCALE_WORD = re.compile(
    r"\b(?:thousand|million|billion|trillion)s?\b|(?<=\d)\s?(?:k|m|mn|b|bn)\b",
    re.IGNORECASE,
)

# Typographic punctuation that models emit ("Patrick’s", "Goring‑on‑Thames",
# "1,577\u202fmillion"), mapped to ASCII so it normalizes like the gold.
_TYPOGRAPHIC = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

_DENIAL = r"(?:does|do|did)(?: not|n't)"
_SOURCE = (
    r"(?:passages?|excerpts?|context|documents?|texts?|filings?|sources?|"
    r"materials?|statements?|information|data|report)"
)

# Phrasings with which a model declines to answer from the given context.
# Verbs such as "name" or "identify" count only when the subject is the source
# ("the passages do not name …"), and "list" never does: "the statement does
# not list a separate line item" is an answer, not a refusal.
_ABSTENTION = re.compile(
    r"\b(?:not (?:provided|mentioned|available|stated|specified|found|given|"
    r"disclosed|included|contained|identified)"
    rf"|{_DENIAL} (?:contain|include|provide|mention|state|specify|give|"
    r"disclose|appear)"
    rf"|{_SOURCE}\b[^.;]{{0,60}}?\b{_DENIAL} (?:name|identify|indicate|say|"
    r"supply|offer)"
    r"|(?:cannot|can't|can not|unable to|impossible to"
    r"|(?:not|isn't|wasn't) possible to)"
    r"(?: be)? (?:determine|answer|calculate|compute|identify|confirm|assess|"
    r"establish|tell|find)(?:d|ed)?"
    r"|cannot be (?:made|found|identified)"
    r"|insufficient (?:information|context|data|to (?:answer|determine|"
    r"calculate))"
    r"|(?:information|data|context) (?:is|are) insufficient"
    r"|none of the (?:passages|excerpts|documents|texts|provided|given|sources)"
    r"|no (?:information|data|mention)"
    r"|unknown|unanswerable)\b",
)

# A gold answer that is itself a negative ("There are none", "No, …"), which
# an answer phrased as an absence may legitimately match.
_NEGATIVE = re.compile(r"\b(?:no|none|not|never|nothing|zero)\b")

# Whole answers that mean "no answer" on their own.
_ABSTENTION_ANSWERS = frozenset({"none", "n/a", "na", "null", "unknown"})


def extract_final_answer(output: str) -> str:
    """Extract the model's delimited final answer from a response.

    An unterminated region counts as no answer: generation was cut off, so
    scoring the reasoning before it would report an answer never produced.

    Args:
        output: The model's full response text.

    Returns:
        The extracted answer, or ``""`` when the last region is absent or incomplete.
    """
    matches = _FINAL_ANSWER.findall(output)
    if not matches:
        return ""
    answer, closing_tag = matches[-1]
    return answer.strip() if closing_tag else ""


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace.

    Args:
        text: The raw answer string.

    Returns:
        The normalized string.
    """
    lowered = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def token_f1(prediction: str, reference: str) -> float:
    """Compute the token-overlap F1 of *prediction* against *reference*.

    Args:
        prediction: The model's answer.
        reference: One gold answer.

    Returns:
        A score in ``[0.0, 1.0]``.  Two empty strings score ``1.0``; one
        empty and one not scores ``0.0``.
    """
    predicted_words = normalize_answer(prediction).split()
    reference_words = normalize_answer(reference).split()
    if not predicted_words or not reference_words:
        return float(predicted_words == reference_words)

    overlap = 0
    for word in set(predicted_words):
        overlap += min(predicted_words.count(word), reference_words.count(word))
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_words)
    recall = overlap / len(reference_words)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, references: list[str]) -> float:
    """Return the best token F1 of *prediction* over all *references*.

    Args:
        prediction: The model's answer.
        references: Gold answers, including any alternate phrasings.

    Returns:
        The highest token F1, or ``0.0`` when *references* is empty.
    """
    return max((token_f1(prediction, ref) for ref in references), default=0.0)


def _clean(text: str) -> str:
    """Fold compatibility characters and typographic punctuation to ASCII."""
    return unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHIC)


def _phrase(text: str) -> str:
    """Normalize *text* for phrase matching, treating hyphens as spaces."""
    return normalize_answer(text.replace("-", " "))


def _numbers(text: str) -> list[float]:
    """Return the magnitude of every number in *text* except bare years.

    Thousands separators are removed.  Signs are dropped because answers
    state them in words as often as with a minus ("shrunk by 0.9%" vs
    "-0.9%").
    """
    values: list[float] = []
    for match in _NUMBER.findall(text):
        cleaned = match.replace(",", "")
        if cleaned in ("", "-") or _YEAR.fullmatch(match):
            continue
        values.append(abs(float(cleaned)))
    return values


def _same_number(
    gold: float, candidate: float, scales: tuple[float, ...] = _UNIT_SCALES
) -> bool:
    """Whether *candidate* equals *gold* up to rounding and one of *scales*."""
    for scale in scales:
        scaled = candidate * scale
        if gold == 0.0:
            if scaled == 0.0:
                return True
            continue
        if abs(scaled - gold) <= _NUMBER_REL_TOL * abs(gold):
            return True
    return False


def answer_match(prediction: str, references: list[str]) -> bool:
    """Whether *prediction* states one of the gold *references*.

    A reference matches when, after :func:`normalize_answer`, it equals the
    prediction or appears in it as a whole-word phrase ("Paris" in "the
    capital is Paris").  A reference containing numbers also matches when
    every one of its numbers appears in the prediction, up to 0.5% rounding.
    When either side names a scale ("million", "billion", "5B", …), a rescale
    by a power of 1000 is also accepted, so "$11588.00" (in millions) matches
    "$11,588 million" and "$11.588 billion", but not "$3,033 million"; and a
    bare count of "3" does not match "3,000".  Bare years are context, not
    figures: they are ignored by the numeric check, so "flat in FY 2023 vs FY
    2022" is not matched by any answer that merely names those years.
    Token F1 cannot tell these apart: both "$11,588 million" and
    "$3,033 million" share no word with the gold string and score 0.

    Typographic punctuation is folded to ASCII first, and hyphens count as
    word breaks in the phrase check ("massively-multiplayer" matches
    "massively multiplayer").  An answer that abstains
    (:func:`is_abstention`) never matches, unless the reference is itself a
    negative ("There are none"): an abstention that names the gold entity
    ("gives Robbie Gould's birth date but not Chris Gould's") is not an
    answer.

    This is a lexical screen, not a judge: it misses correct answers that
    paraphrase a long or free-form gold, and an answer naming every
    candidate of a comparison question can match.  Use an LLM judge where
    the size of a quality difference matters.

    Args:
        prediction: The model's extracted answer.
        references: Gold answers, including alternate phrasings.

    Returns:
        ``True`` if any reference matches; ``False`` for an empty prediction.
    """
    prediction = _clean(prediction)
    normalized = _phrase(prediction)
    if not normalized:
        return False
    padded = f" {normalized} "
    predicted_numbers = _numbers(prediction)
    abstains = is_abstention(prediction)

    for reference in references:
        reference = _clean(reference)
        gold = _phrase(reference)
        if abstains and not (_NEGATIVE.search(gold) or is_abstention(reference)):
            continue
        if gold and (gold == normalized or f" {gold} " in padded):
            return True
        gold_numbers = _numbers(reference)
        scales = (
            _UNIT_SCALES
            if _SCALE_WORD.search(prediction) or _SCALE_WORD.search(reference)
            else (1.0,)
        )
        if gold_numbers and all(
            any(_same_number(g, p, scales) for p in predicted_numbers)
            for g in gold_numbers
        ):
            return True
    return False


def is_abstention(prediction: str) -> bool:
    """Whether *prediction* declines to answer instead of stating one.

    Catches the failure mode a lexical gate misses: "the passages do not
    contain X" while X is in the prompt.  An empty prediction is not an
    abstention; it is an unparsed answer.

    The phrase list is calibrated on short QA answers.  Ordinary long-form
    text such as a news summary often contains the same phrases ("details
    were not disclosed"), so this must not be used outside QA.

    Args:
        prediction: The model's extracted answer.

    Returns:
        ``True`` if the answer is a refusal or a "not in the context" claim.
    """
    lowered = _clean(prediction).strip().lower()
    if not lowered:
        return False
    if lowered.rstrip(".") in _ABSTENTION_ANSWERS:
        return True
    return _ABSTENTION.search(lowered) is not None


@dataclass
class SampleScore:
    """One sample's measured quality.

    Attributes:
        sample_id: The dataset's id for this sample.
        parsed: Whether a complete answer region was found.  When ``False``,
            ``f1`` is meaningless and is exported as null.
        f1: Best token F1 against the gold answers; ``0.0`` when unparsed.
        answer: The extracted answer (``""`` when unparsed).
        ttft: Time to first token, in seconds.
        num_output_tokens: Tokens generated.
    """

    sample_id: str
    parsed: bool
    f1: float
    answer: str
    ttft: float
    num_output_tokens: int


@dataclass
class QualitySummary:
    """Aggregate quality over a run.

    ``f1_mean`` covers parsed samples only, so it must be read together with
    ``parse_rate``.
    """

    num_samples: int
    num_parsed: int
    parse_rate: float
    f1_mean: float


class QualityAggregator:
    """Accumulates per-sample scores.  Single-threaded by contract."""

    def __init__(self) -> None:
        self._scores: list[SampleScore] = []

    def record(self, score: SampleScore) -> None:
        """Record one sample's score.

        Args:
            score: The sample's measured quality.
        """
        self._scores.append(score)

    def scores(self) -> list[SampleScore]:
        """Return the recorded scores, in measurement order."""
        return list(self._scores)

    def summarize(self) -> QualitySummary:
        """Compute the aggregate summary over all recorded scores."""
        parsed = [s for s in self._scores if s.parsed]
        num_samples = len(self._scores)
        return QualitySummary(
            num_samples=num_samples,
            num_parsed=len(parsed),
            parse_rate=(len(parsed) / num_samples) if num_samples else 0.0,
            f1_mean=(sum(s.f1 for s in parsed) / len(parsed)) if parsed else 0.0,
        )
