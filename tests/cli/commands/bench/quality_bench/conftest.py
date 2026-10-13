# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for ``bench quality`` tests."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality import alignment
from lmcache.cli.commands.bench.engine_bench.quality.alignment import ChunkAligner
from lmcache.cli.commands.bench.quality_bench import blend_counters, runner

# Local
from ..engine_bench.fake_tokenizer import make_fake_tokenizer

CHUNK = 32


@pytest.fixture
def aligner(monkeypatch) -> ChunkAligner:
    """An aligner over the fake tokenizer with a small vocabulary size."""
    monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
    return ChunkAligner(make_fake_tokenizer(), "fake-model", CHUNK, seed=3)


@pytest.fixture(autouse=True)
def no_counter_waits(monkeypatch) -> None:
    """Skip the waits for late blend-server counters and between retries."""
    monkeypatch.setattr(runner, "COUNTER_SETTLE_S", 0.0)
    monkeypatch.setattr(blend_counters, "FETCH_RETRY_DELAY_S", 0.0)
