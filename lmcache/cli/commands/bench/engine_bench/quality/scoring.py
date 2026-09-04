# SPDX-License-Identifier: Apache-2.0
"""Answer extraction and SQuAD-style token-overlap F1 scoring."""

# Standard
from dataclasses import dataclass
import re
import string

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)

# Lazy match, last region wins: reasoning models may echo an example answer
# before their own.
_FINAL_ANSWER = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>",
    re.IGNORECASE | re.DOTALL,
)


def extract_final_answer(output: str) -> str:
    """Extract the model's delimited final answer from a response.

    An unterminated region counts as no answer: generation was cut off, so
    scoring the reasoning before it would report an answer never produced.

    Args:
        output: The model's full response text.

    Returns:
        The extracted answer, or ``""`` when no complete region is present.
    """
    matches = _FINAL_ANSWER.findall(output)
    if not matches:
        return ""
    return matches[-1].strip()


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
