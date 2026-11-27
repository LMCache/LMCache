# SPDX-License-Identifier: Apache-2.0
"""Task-type checks for ``lmcache bench quality``.

The prompt, answer extraction, scorers and loss tags all assume short-answer
QA. A dataset of another kind (summaries, long-form answers) would still
load, but every arm would score near zero against its long references and
the verdict would be a meaningless PASS. :func:`check_task` refuses such a
dataset before any request is sent.
"""

# Standard
import statistics

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.engine_bench.quality.scoring import normalize_answer
from lmcache.cli.commands.bench.quality_bench.core.config import TaskType

# Longest median reference, in words, a QA dataset may have. QA references
# are a few words (HotpotQA, 2WikiMultihopQA and MuSiQue have a median of 2,
# FinanceBench 9) -- multi-document summaries are hundreds (MultiNews: 200).
QA_MAX_MEDIAN_REFERENCE_WORDS = 50


def check_task(samples: list[Sample], task: TaskType) -> None:
    """Refuse a dataset whose shape does not fit *task*.

    Args:
        samples: The questions to be measured.
        task: The task type the run was configured for.

    Raises:
        ValueError: If *samples* is empty, or the references are too long
            for short-answer QA.
    """
    if not samples:
        raise ValueError("no samples to check")
    if task is TaskType.QA:
        lengths = [len(normalize_answer(s.answers[0]).split()) for s in samples]
        median = statistics.median(lengths)
        if median > QA_MAX_MEDIAN_REFERENCE_WORDS:
            raise ValueError(
                f"the dataset's reference answers have a median of {median:g} "
                f"words, but bench quality scores short-answer QA (median at "
                f"most {QA_MAX_MEDIAN_REFERENCE_WORDS}). Long-form tasks such "
                f"as summarization need their own prompt and an LLM-judge "
                f"scorer, which are not supported yet."
            )
