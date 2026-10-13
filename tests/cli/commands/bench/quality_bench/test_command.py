# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for ``lmcache bench quality`` with scripted engines."""

# Standard
import argparse
import json
import os

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality import alignment
from lmcache.cli.commands.bench.quality_bench import command
from lmcache.cli.commands.bench.quality_bench.blend_counters import BlendCounters
from lmcache.cli.commands.bench.quality_bench.config import (
    ArmRole,
    ReuseScenario,
    VerdictMetric,
)
from lmcache.cli.commands.bench.quality_bench.report import (
    MANIFEST_FILE,
    ROWS_FILE,
    SUMMARY_FILE,
)

# Local
from ..engine_bench.fake_tokenizer import make_fake_tokenizer
from .fakes import (
    FakeJudgeClient,
    FakeSender,
    SteppingCounters,
    answering,
    rate_exact,
)

REF_URL = "http://ref"
BLEND_URL = "http://blend"
APC_URL = "http://apc"
METRICS_URL = "http://blend-server/metrics"

RIGHT = {"Capital?": "Paris", "Nordic?": "Oslo", "Swiss?": "Bern"}
WRONG = {"Capital?": "Rome", "Nordic?": "Rome", "Swiss?": "Vienna"}


def _parse(*extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    command.add_quality_arguments(parser)
    return parser.parse_args(list(extra))


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
    """Replace the tokenizer and HTTP senders; returns senders by URL.

    Tests fill the dict before running; the reference answers correctly.
    """
    monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
    monkeypatch.setattr(command, "try_load_tokenizer", lambda _: make_fake_tokenizer())
    senders: dict[str, FakeSender] = {REF_URL: FakeSender(answering(RIGHT))}

    def make_sender(url: str, model: str, extra_body: dict) -> FakeSender:
        return senders[url]

    monkeypatch.setattr(command, "RequestSender", make_sender)
    return senders


def _base_args(dataset: str, out: str, *extra: str) -> argparse.Namespace:
    return _parse(
        "--reference-url", REF_URL,
        "--blend-url", BLEND_URL,
        "--model", "fake-model",
        "--dataset", dataset,
        "--chunk-size", "32",
        "--output-dir", out,
        *extra,
    )  # fmt: skip


def _run_dir(out: str) -> str:
    """The one run directory a test's run created under *out*."""
    [name] = [n for n in os.listdir(out) if os.path.isdir(os.path.join(out, n))]
    return os.path.join(out, name)


def _summary(out: str) -> dict:
    with open(os.path.join(_run_dir(out), SUMMARY_FILE)) as f:
        return json.load(f)


class TestConfigFromArgs:
    def test_defaults(self) -> None:
        config = command.config_from_args(
            _parse(
                "--reference-url",
                REF_URL,
                "--blend-url",
                BLEND_URL,
                "--model",
                "m",
                "--dataset",
                "hotpotqa",
            )  # fmt: skip
        )
        assert [a.role for a in config.arms] == [ArmRole.REFERENCE, ArmRole.BLEND]
        assert config.scenario is ReuseScenario.INDEPENDENT
        assert config.verdict_metric is VerdictMetric.MATCH
        assert config.num_samples == 0

    def test_optional_arm_and_options(self) -> None:
        config = command.config_from_args(
            _parse(
                "--reference-url",
                REF_URL,
                "--blend-url",
                BLEND_URL,
                "--apc-url",
                APC_URL,
                "--model",
                "m",
                "--dataset",
                "hotpotqa",
                "--scenario",
                "reordered",
                "--verdict-metric",
                "f1",
                "--limit",
                "5",
                "--template-kwargs",
                "reasoning_effort=low",
            )  # fmt: skip
        )
        assert config.arm(ArmRole.PREFIX_CACHE).url == APC_URL
        assert config.scenario is ReuseScenario.REORDERED
        assert config.verdict_metric is VerdictMetric.F1
        assert config.num_samples == 5
        assert config.template_kwargs

    def test_invalid_value_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            command.config_from_args(
                _parse(
                    "--reference-url",
                    REF_URL,
                    "--blend-url",
                    BLEND_URL,
                    "--model",
                    "m",
                    "--dataset",
                    "hotpotqa",
                    "--chunk-size",
                    "0",
                )  # fmt: skip
            )


class TestRunQualityBench:
    def test_matching_blend_passes_and_writes_results(
        self, dataset, engines, tmp_path
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")

        command.run_quality_bench(_base_args(dataset, out))

        run_dir = _run_dir(out)
        for name in (MANIFEST_FILE, ROWS_FILE, SUMMARY_FILE):
            assert os.path.isfile(os.path.join(run_dir, name))
        for label in ("reference_1", "reference_2", "blend"):
            assert os.path.isfile(os.path.join(run_dir, f"answers_{label}.jsonl"))
        assert _summary(out)["verdict"] == "PASS"
        assert all(s.closed for s in engines.values())

    def test_only_caching_arms_store_first(self, dataset, engines, tmp_path) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        engines[APC_URL] = FakeSender(answering(RIGHT))

        command.run_quality_bench(
            _base_args(dataset, str(tmp_path / "out"), "--apc-url", APC_URL)
        )

        def stores(sender: FakeSender) -> int:
            return sum(rid.startswith("store") for rid, _, _ in sender.sent)

        assert stores(engines[REF_URL]) == 0
        assert stores(engines[BLEND_URL]) > 0
        assert stores(engines[APC_URL]) == stores(engines[BLEND_URL])
        # Two reference passes over three questions.
        assert len(engines[REF_URL].sent) == 6

    def test_quality_loss_exits_non_zero(self, dataset, engines, tmp_path) -> None:
        engines[BLEND_URL] = FakeSender(answering(WRONG))
        out = str(tmp_path / "out")

        with pytest.raises(SystemExit) as exc:
            command.run_quality_bench(_base_args(dataset, out))

        assert exc.value.code == 1
        assert _summary(out)["verdict"] == "FAIL"

    def test_unengaged_blend_is_invalid(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        no_blending = BlendCounters(requested_tokens=100.0, prefix_hit_tokens=100.0)
        monkeypatch.setattr(
            command,
            "PrometheusBlendCounters",
            lambda url: SteppingCounters(no_blending),
        )
        out = str(tmp_path / "out")

        with pytest.raises(SystemExit) as exc:
            command.run_quality_bench(
                _base_args(dataset, out, "--blend-metrics-url", METRICS_URL)
            )

        assert exc.value.code == 1
        assert _summary(out)["verdict"] == "INVALID"

    def test_unreachable_metrics_fail_before_the_arms_run(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        """Without the preflight read, a bad URL surfaces only when the blend
        pass starts, after every reference pass has run."""
        engines[BLEND_URL] = FakeSender(answering(RIGHT))

        class DownCounters(SteppingCounters):
            def read(self) -> BlendCounters:
                raise RuntimeError("Could not read blend-server metrics")

        monkeypatch.setattr(
            command,
            "PrometheusBlendCounters",
            lambda url: DownCounters(BlendCounters()),
        )

        with pytest.raises(RuntimeError, match="blend-server metrics"):
            command.run_quality_bench(
                _base_args(
                    dataset,
                    str(tmp_path / "out"),
                    "--blend-metrics-url",
                    METRICS_URL,
                )
            )
        assert engines[REF_URL].sent == []
        assert engines[BLEND_URL].sent == []

    def test_segmented_prefix_tokens_are_reported(
        self, dataset, engines, tmp_path, monkeypatch, capsys
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        step = BlendCounters(
            requested_tokens=100.0,
            non_prefix_hit_tokens=60.0,
            segmented_prefix_hit_tokens=20.0,
        )
        monkeypatch.setattr(
            command, "PrometheusBlendCounters", lambda url: SteppingCounters(step)
        )
        out = str(tmp_path / "out")

        command.run_quality_bench(
            _base_args(dataset, out, "--blend-metrics-url", METRICS_URL)
        )

        engagement = _summary(out)["engagement"]
        assert engagement["segmented_prefix_hit_tokens"] > 0
        assert engagement["blend_share"] == pytest.approx(0.6)
        assert "segmented-prefix tokens:" in capsys.readouterr().out

    def test_unreachable_engines_are_invalid(self, dataset, engines, tmp_path) -> None:
        """Every request failing on both arms scores 0 = 0, which is no PASS."""
        # Three questions over five distinct passages.
        down = frozenset([f"q{i}" for i in range(3)] + [f"store{i}" for i in range(5)])
        engines[REF_URL] = FakeSender(answering(RIGHT), failing=down)
        engines[BLEND_URL] = FakeSender(answering(RIGHT), failing=down)
        out = str(tmp_path / "out")

        with pytest.raises(SystemExit) as exc:
            command.run_quality_bench(_base_args(dataset, out))

        assert exc.value.code == 1
        summary = _summary(out)
        assert summary["verdict"] == "INVALID"
        assert "blend: no request succeeded" in summary["invalid_reasons"]
        assert "reference_1: no request succeeded" in summary["invalid_reasons"]

    def test_missing_tokenizer_is_reported(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(command, "try_load_tokenizer", lambda _: None)
        with pytest.raises(RuntimeError, match="tokenizer"):
            command.run_quality_bench(_base_args(dataset, str(tmp_path / "out")))


@pytest.fixture
def judge_client(monkeypatch) -> FakeJudgeClient:
    """Replace the judge endpoint; every client created shares one script."""
    client = FakeJudgeClient()
    monkeypatch.setattr(command, "OpenAIJudgeClient", lambda _settings: client)
    return client


class TestJudge:
    def test_judge_url_makes_judge_the_default_metric(self) -> None:
        config = command.config_from_args(
            _parse(
                "--reference-url",
                REF_URL,
                "--blend-url",
                BLEND_URL,
                "--model",
                "m",
                "--dataset",
                "hotpotqa",
                "--judge-url",
                "http://judge",
            )  # fmt: skip
        )
        assert config.verdict_metric is VerdictMetric.JUDGE
        assert config.judge is not None
        assert config.judge.model == "m"

    def test_explicit_metric_wins_over_the_judge(self) -> None:
        config = command.config_from_args(
            _parse(
                "--reference-url",
                REF_URL,
                "--blend-url",
                BLEND_URL,
                "--model",
                "m",
                "--dataset",
                "hotpotqa",
                "--judge-url",
                "http://judge",
                "--verdict-metric",
                "match",
            )  # fmt: skip
        )
        assert config.verdict_metric is VerdictMetric.MATCH

    def test_judged_run_fails_on_wrong_answers(
        self, dataset, engines, judge_client, tmp_path
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(WRONG))
        out = str(tmp_path / "out")

        with pytest.raises(SystemExit):
            command.run_quality_bench(
                _base_args(dataset, out, "--judge-url", "http://judge")
            )

        summary = _summary(out)
        assert summary["verdict"] == "FAIL"
        assert summary["judge"]["model"] == "fake-model"
        assert summary["settings"]["verdict_metric"] == "judge"
        assert judge_client.closed
        assert os.path.isfile(os.path.join(out, command.JUDGE_CACHE_FILE))
        with open(os.path.join(_run_dir(out), ROWS_FILE)) as f:
            row = json.loads(f.readline())
        assert "match" in row

    def test_reruns_get_their_own_directory_and_share_the_judge_cache(
        self, dataset, engines, judge_client, tmp_path
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        engines[APC_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        judged = ("--judge-url", "http://judge")

        command.run_quality_bench(
            _base_args(dataset, out, *judged, "--apc-url", APC_URL)
        )
        calls_after_first = len(judge_client.prompts)
        # The rerun drops the apc arm.
        command.run_quality_bench(_base_args(dataset, out, *judged))

        names = sorted(n for n in os.listdir(out) if os.path.isdir(f"{out}/{n}"))
        first, second = (os.path.join(out, n) for n in names)
        assert os.path.isfile(os.path.join(first, "answers_apc.jsonl"))
        assert not os.path.exists(os.path.join(second, "answers_apc.jsonl"))
        assert os.path.isfile(os.path.join(second, SUMMARY_FILE))
        # One cache at the top level; the rerun only re-ran the judge sanity check.
        assert os.path.isfile(os.path.join(out, command.JUDGE_CACHE_FILE))
        assert not os.path.exists(os.path.join(second, command.JUDGE_CACHE_FILE))
        assert len(judge_client.prompts) - calls_after_first == 4

    def test_unreachable_judge_fails_before_the_arms_run(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        monkeypatch.setattr(
            command, "OpenAIJudgeClient", lambda _s: FakeJudgeClient(failing=1)
        )
        with pytest.raises(RuntimeError, match="judge down"):
            command.run_quality_bench(
                _base_args(dataset, str(tmp_path / "out"), "--judge-url", "http://j")
            )
        assert engines[REF_URL].sent == []

    def test_judge_that_misreads_its_reasoning_fails_before_the_arms_run(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        """A thinking judge without a reasoning parser would zero every arm."""
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        thinking = FakeJudgeClient(lambda _p: "<think>0 doubts.</think> 4")
        monkeypatch.setattr(command, "OpenAIJudgeClient", lambda _s: thinking)
        with pytest.raises(RuntimeError, match="sanity-check"):
            command.run_quality_bench(
                _base_args(dataset, str(tmp_path / "out"), "--judge-url", "http://j")
            )
        assert engines[REF_URL].sent == []

    def test_judge_failing_mid_run_keeps_the_answers(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))

        def reply(prompt: str) -> str:
            # The sanity check passes; the run's own questions hit a dead judge.
            if "Nordic?" in prompt:
                raise RuntimeError("judge down")
            return rate_exact(prompt)

        client = FakeJudgeClient(reply)
        monkeypatch.setattr(command, "OpenAIJudgeClient", lambda _s: client)
        out = str(tmp_path / "out")

        with pytest.raises(RuntimeError, match="failed mid-run"):
            command.run_quality_bench(
                _base_args(dataset, out, "--judge-url", "http://j")
            )

        run_dir = _run_dir(out)
        assert os.path.isfile(os.path.join(run_dir, MANIFEST_FILE))
        assert os.path.isfile(os.path.join(run_dir, "answers_blend.jsonl"))
        assert not os.path.exists(os.path.join(run_dir, SUMMARY_FILE))
        assert client.closed


class TestTaskGuard:
    def test_long_references_are_refused(self, engines, tmp_path) -> None:
        summary = " ".join(["word"] * 200)
        path = tmp_path / "news.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"n{i}",
                        "user_input": "Summarize the articles.",
                        "retrieved_contexts": ["Article one.", "Article two."],
                        "reference": summary,
                    }
                )
                + "\n"
                for i in range(3)
            )
        )
        with pytest.raises(ValueError, match="short-answer QA"):
            command.run_quality_bench(_base_args(str(path), str(tmp_path / "out")))
        assert engines[REF_URL].sent == []

    def test_summary_reports_evidence_sources(self, dataset, engines, tmp_path) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        command.run_quality_bench(_base_args(dataset, out))
        sources = _summary(out)["evidence_sources"]
        assert sum(sources.values()) == 3
        assert set(sources) == {"dataset", "answer_string", "none"}

    def test_summary_reports_passages_from_earlier_questions(
        self, dataset, engines, tmp_path, capsys
    ) -> None:
        engines[BLEND_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        command.run_quality_bench(_base_args(dataset, out))
        # "Rome is in Italy." recurs: one of six measured passages.
        assert _summary(out)["measured_earlier_share"] == pytest.approx(1 / 6)
        assert "passages also in an earlier question: 16.7%" in capsys.readouterr().out
