# SPDX-License-Identifier: Apache-2.0
"""Answer-quality measurement helpers for engine benchmark workloads.

* ``dataset`` — loading multi-passage QA datasets into :class:`Sample` objects.
* ``scoring`` — extracting the model's answer and scoring it against gold.
"""

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.dataset import (
    HUB_DATASET_NAMES,
    Sample,
    describe_hub_datasets,
    load_samples,
    resolve_dataset_path,
)
from lmcache.cli.commands.bench.engine_bench.quality.scoring import (
    QualityAggregator,
    QualitySummary,
    SampleScore,
    best_f1,
    extract_final_answer,
    normalize_answer,
    token_f1,
)

__all__ = [
    "HUB_DATASET_NAMES",
    "QualityAggregator",
    "QualitySummary",
    "Sample",
    "SampleScore",
    "best_f1",
    "describe_hub_datasets",
    "extract_final_answer",
    "load_samples",
    "normalize_answer",
    "resolve_dataset_path",
    "token_f1",
]
