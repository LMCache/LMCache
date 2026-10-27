# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for ``bench quality`` tests."""

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality import alignment
from lmcache.cli.commands.bench.engine_bench.quality.alignment import ChunkAligner
from lmcache.cli.commands.bench.quality_bench.core import (
    analyze_stage,
    blend_counters,
    run_stage,
    runner,
)

# Local
from ..engine_bench.fake_tokenizer import make_fake_tokenizer
from .fakes import FakeJudgeClient, FakeSender, answering
from .stages import REF_URL, RIGHT

CHUNK = 32


@pytest.fixture
def aligner(monkeypatch) -> ChunkAligner:
    """An aligner over the fake tokenizer with a small vocabulary size."""
    monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
    return ChunkAligner(
        make_fake_tokenizer(bpe_pieces=True), "fake-model", CHUNK, seed=3
    )


@pytest.fixture(autouse=True)
def no_counter_waits(monkeypatch) -> None:
    """Skip the waits for late blend-server counters and between retries."""
    monkeypatch.setattr(runner, "COUNTER_SETTLE_S", 0.0)
    monkeypatch.setattr(blend_counters, "FETCH_RETRY_DELAY_S", 0.0)


@pytest.fixture
def dataset(tmp_path) -> str:
    """A three-question file in the RAGAS hand-off shape."""
    records = [
        {
            "id": "s0",
            "user_input": "Capital?",
            "retrieved_contexts": ["Paris is in France.", "Rome is in Italy."],
            "reference": "Paris",
        },
        {
            "id": "s1",
            "user_input": "Nordic?",
            "retrieved_contexts": ["Rome is in Italy.", "Oslo is in Norway."],
            "reference": "Oslo",
        },
        {
            "id": "s2",
            "user_input": "Swiss?",
            "retrieved_contexts": ["Bern is Swiss.", "Vienna is Austrian."],
            "reference": "Bern",
        },
    ]
    path = tmp_path / "qa.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


@pytest.fixture
def engines(monkeypatch) -> dict[str, FakeSender]:
    """Replace the tokenizer and the engines; returns senders by URL.

    Tests fill the dict before running; the reference answers correctly.
    Every engine reports version ``fake-1.0``.
    """
    monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
    monkeypatch.setattr(
        run_stage, "try_load_tokenizer", lambda _: make_fake_tokenizer(bpe_pieces=True)
    )
    monkeypatch.setattr(run_stage, "engine_version", lambda _url: "fake-1.0")
    senders: dict[str, FakeSender] = {REF_URL: FakeSender(answering(RIGHT))}

    def make_sender(url: str, model: str, extra_body: dict) -> FakeSender:
        return senders[url]

    monkeypatch.setattr(run_stage, "RequestSender", make_sender)
    return senders


@pytest.fixture
def judge_client(monkeypatch) -> FakeJudgeClient:
    """Replace the judge endpoint; every client created shares one script."""
    client = FakeJudgeClient()
    monkeypatch.setattr(analyze_stage, "OpenAIJudgeClient", lambda _settings: client)
    return client
