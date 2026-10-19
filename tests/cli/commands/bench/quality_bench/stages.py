# SPDX-License-Identifier: Apache-2.0
"""Helpers that drive ``bench quality run`` and ``analyze`` in tests.

Use with the ``dataset``, ``engines`` and ``judge_client`` fixtures from
``conftest.py``, which replace the tokenizer, the engines and the judge.
"""

# Standard
from collections.abc import Callable
import argparse
import json
import os

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.quality_bench.core import analyze_stage, run_stage
from lmcache.cli.commands.bench.quality_bench.core.run_files import SUMMARY_FILE

REF_URL = "http://ref"
BLEND_URL = "http://blend"
APC_URL = "http://apc"
METRICS_URL = "http://blend-server/metrics"

# What the fake engines answer: right or wrong for each dataset question.
RIGHT = {"Capital?": "Paris", "Nordic?": "Oslo", "Swiss?": "Bern"}
WRONG = {"Capital?": "Rome", "Nordic?": "Rome", "Swiss?": "Vienna"}


def parse_run(*argv: str) -> argparse.Namespace:
    """Parse ``bench quality run`` arguments."""
    parser = argparse.ArgumentParser()
    run_stage.add_run_arguments(parser)
    return parser.parse_args(list(argv))


def parse_analyze(*argv: str) -> argparse.Namespace:
    """Parse ``bench quality analyze`` arguments."""
    parser = argparse.ArgumentParser()
    analyze_stage.add_analyze_arguments(parser)
    return parser.parse_args(list(argv))


def run_args(dataset: str, out: str, *arm_flags: str) -> argparse.Namespace:
    """``run`` arguments for the test dataset, with *arm_flags* added."""
    return parse_run(
        "--model", "fake-model",
        "--dataset", dataset,
        "--chunk-size", "32",
        "--output-dir", out,
        *arm_flags,
    )  # fmt: skip


def run(dataset: str, out: str, *arm_flags: str) -> None:
    """Run ``bench quality run`` into *out*."""
    run_stage.run_quality_run(run_args(dataset, out, *arm_flags))


def run_both(dataset: str, out: str, *extra: str) -> None:
    """Run the reference and blend arms into *out* in one invocation."""
    run(dataset, out, "--reference-url", REF_URL, "--blend-url", BLEND_URL, *extra)


def analyze(out: str, *flags: str) -> None:
    """Run ``bench quality analyze`` on *out*."""
    analyze_stage.run_quality_analyze(parse_analyze(out, *flags))


def summary(out: str) -> dict:
    """Load *out*'s ``summary.json``."""
    with open(os.path.join(out, SUMMARY_FILE)) as f:
        return json.load(f)


def expect_setup_error(
    capsys: pytest.CaptureFixture[str], call: Callable[[], None]
) -> str:
    """Run *call*, expect a results-directory error (exit 2), return its message."""
    with pytest.raises(SystemExit) as exc:
        call()
    assert exc.value.code == 2
    return capsys.readouterr().err
