# SPDX-License-Identifier: Apache-2.0
"""Tests for the task-type guard."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.quality_bench.core.config import TaskType
from lmcache.cli.commands.bench.quality_bench.core.task import (
    QA_MAX_MEDIAN_REFERENCE_WORDS,
    check_task,
)


def _samples(reference_words: list[int]) -> list[Sample]:
    return [
        Sample(f"s{i}", ["passage"], "question?", [" ".join(["word"] * n)])
        for i, n in enumerate(reference_words)
    ]


class TestCheckTask:
    def test_short_answers_pass(self) -> None:
        check_task(_samples([1, 2, 3, 9, 40]), TaskType.QA)

    def test_a_few_long_answers_pass(self) -> None:
        """FinanceBench mixes figures with sentence-long answers."""
        check_task(_samples([2, 2, 2, 90, 90]), TaskType.QA)

    def test_summaries_are_refused(self) -> None:
        with pytest.raises(ValueError, match="median of 200 words"):
            check_task(_samples([180, 200, 230]), TaskType.QA)

    def test_threshold_is_inclusive(self) -> None:
        check_task(_samples([QA_MAX_MEDIAN_REFERENCE_WORDS]), TaskType.QA)
        with pytest.raises(ValueError):
            check_task(_samples([QA_MAX_MEDIAN_REFERENCE_WORDS + 1]), TaskType.QA)

    def test_empty_is_refused(self) -> None:
        with pytest.raises(ValueError):
            check_task([], TaskType.QA)
