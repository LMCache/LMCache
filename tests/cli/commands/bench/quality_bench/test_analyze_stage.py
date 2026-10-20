# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache bench quality analyze`` on saved arm runs."""

# Standard
import json
import os
import shutil

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.quality_bench.core import analyze_stage, run_stage
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import BlendCounters
from lmcache.cli.commands.bench.quality_bench.core.config import VerdictMetric
from lmcache.cli.commands.bench.quality_bench.core.run_files import (
    JUDGE_CACHE_FILE,
    PLAN_FILE,
    ROWS_FILE,
    SUMMARY_FILE,
)

# Local
from .fakes import FakeJudgeClient, FakeSender, SteppingCounters, answering, rate_exact
from .stages import (
    CANDIDATE_URL,
    CONTROL_URL,
    METRICS_URL,
    REF_URL,
    RIGHT,
    WRONG,
    analyze,
    expect_setup_error,
    parse_analyze,
    run,
    run_both,
    summary,
)

# The candidate answers one of three questions wrongly.
ONE_WRONG = {**RIGHT, "Swiss?": "Vienna"}


def _sent(engines: dict[str, FakeSender]) -> int:
    return sum(len(sender.sent) for sender in engines.values())


class TestVerdict:
    def test_matching_candidate_passes(self, dataset, engines, tmp_path) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)

        analyze(out)

        result = summary(out)
        assert result["verdict"] == "PASS"
        assert os.path.isfile(os.path.join(out, ROWS_FILE))
        assert result["arms"]["candidate"]["url"] == CANDIDATE_URL
        assert result["arms"]["reference_1"]["engine_version"] == "fake-1.0"
        assert result["settings"]["reference_repeats"] == 2

    def test_arms_run_one_at_a_time_are_analyzed_together(
        self, dataset, engines, tmp_path
    ) -> None:
        """The serialized CI flow: one engine per job, then analyze."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        engines[CONTROL_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run(dataset, out, "--reference-url", REF_URL)
        run(dataset, out, "--control-url", CONTROL_URL)
        run(dataset, out, "--candidate-url", CANDIDATE_URL)

        analyze(out)

        result = summary(out)
        assert result["verdict"] == "PASS"
        assert set(result["arms"]) == {
            "reference_1",
            "reference_2",
            "control",
            "candidate",
        }

    def test_quality_loss_exits_non_zero(self, dataset, engines, tmp_path) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(WRONG))
        out = str(tmp_path / "out")
        run_both(dataset, out)

        with pytest.raises(SystemExit) as exc:
            analyze(out)

        assert exc.value.code == 1
        assert summary(out)["verdict"] == "FAIL"

    def test_unengaged_candidate_is_invalid(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        no_blending = BlendCounters(requested_tokens=100.0, prefix_hit_tokens=100.0)
        monkeypatch.setattr(
            run_stage,
            "PrometheusBlendCounters",
            lambda url: SteppingCounters(no_blending),
        )
        out = str(tmp_path / "out")
        run_both(dataset, out, "--capture-blend-metrics", METRICS_URL)

        with pytest.raises(SystemExit) as exc:
            analyze(out)

        assert exc.value.code == 1
        result = summary(out)
        assert result["verdict"] == "INVALID"
        assert result["settings"]["blend_metrics_url"] == METRICS_URL

    def test_unreachable_engines_are_invalid(self, dataset, engines, tmp_path) -> None:
        """Every request failing on both arms scores 0 = 0, which is no PASS."""
        down = frozenset([f"q{i}" for i in range(3)] + [f"store{i}" for i in range(5)])
        engines[REF_URL] = FakeSender(answering(RIGHT), failing=down)
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT), failing=down)
        out = str(tmp_path / "out")
        with pytest.raises(SystemExit):
            run_both(dataset, out)

        with pytest.raises(SystemExit):
            analyze(out)

        result = summary(out)
        assert result["verdict"] == "INVALID"
        assert "candidate: no request succeeded" in result["invalid_reasons"]
        assert "reference_1: no request succeeded" in result["invalid_reasons"]

    def test_segmented_prefix_tokens_are_reported(
        self, dataset, engines, tmp_path, monkeypatch, capsys
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        step = BlendCounters(
            requested_tokens=100.0,
            non_prefix_hit_tokens=60.0,
            segmented_prefix_hit_tokens=20.0,
        )
        monkeypatch.setattr(
            run_stage, "PrometheusBlendCounters", lambda url: SteppingCounters(step)
        )
        out = str(tmp_path / "out")
        run_both(dataset, out, "--capture-blend-metrics", METRICS_URL)

        analyze(out)

        engagement = summary(out)["engagement"]
        assert engagement["segmented_prefix_hit_tokens"] > 0
        assert engagement["blend_share"] == pytest.approx(0.6)
        assert "segmented-prefix tokens:" in capsys.readouterr().out

    def test_results_are_strict_json(self, dataset, engines, tmp_path) -> None:
        """One question leaves the standard error undefined: null, not Infinity."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out, "--limit", "1")

        analyze(out)

        def refuse(constant: str) -> float:
            raise ValueError(f"non-standard JSON constant {constant}")

        with open(os.path.join(out, SUMMARY_FILE)) as f:
            result = json.loads(f.read(), parse_constant=refuse)
        assert result["candidate"]["standard_error"] is None
        with open(os.path.join(out, ROWS_FILE)) as f:
            for line in f:
                json.loads(line, parse_constant=refuse)

    def test_summary_reports_the_plan(self, dataset, engines, tmp_path, capsys) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)

        analyze(out)

        result = summary(out)
        assert sum(result["evidence_sources"].values()) == 3
        # "Rome is in Italy." recurs: one of six measured passages.
        assert result["measured_earlier_share"] == pytest.approx(1 / 6)
        assert "passages also in an earlier question: 16.7%" in capsys.readouterr().out


class TestSavedRuns:
    def test_analyze_needs_no_dataset_or_tokenizer(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        os.remove(dataset)

        def unavailable(_: str) -> None:
            raise AssertionError("analyze must not load a tokenizer")

        monkeypatch.setattr(run_stage, "try_load_tokenizer", unavailable)

        analyze(out)

        assert summary(out)["verdict"] == "PASS"

    def test_reanalyzing_with_another_tolerance_sends_nothing(
        self, dataset, engines, tmp_path
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(ONE_WRONG))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        sent = _sent(engines)

        analyze(out)
        # A 1-in-3 loss on three questions is beyond tolerance but not 2 SE.
        strict = summary(out)["verdict"]
        analyze(out, "--tolerance", "0.5")

        assert strict == "UNDERPOWERED"
        assert summary(out)["verdict"] == "PASS"
        assert _sent(engines) == sent

    @pytest.mark.parametrize(
        ("flag", "missing"),
        [("--reference-url", "candidate"), ("--candidate-url", "reference")],
    )
    def test_a_missing_arm_names_the_run_to_do(
        self, dataset, engines, tmp_path, capsys, flag: str, missing: str
    ) -> None:
        """A setup problem, not a verdict: exit 2, no summary written."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run(dataset, out, flag, REF_URL if flag == "--reference-url" else CANDIDATE_URL)

        error = expect_setup_error(capsys, lambda: analyze(out))

        assert f"has no {missing} arm:" in error
        assert not os.path.exists(os.path.join(out, SUMMARY_FILE))

    def test_every_missing_arm_is_named_at_once(
        self, dataset, engines, tmp_path, capsys
    ) -> None:
        engines[CONTROL_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run(dataset, out, "--control-url", CONTROL_URL)

        error = expect_setup_error(capsys, lambda: analyze(out))

        assert "has no reference and candidate arms" in error
        assert "--reference-url … --candidate-url …" in error

    def test_missing_plan_names_the_run_to_do(self, tmp_path, capsys) -> None:
        error = expect_setup_error(capsys, lambda: analyze(str(tmp_path / "empty")))
        assert "bench quality run" in error

    def test_an_arm_from_another_plan_is_refused(
        self, dataset, engines, tmp_path, capsys
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        here, there = str(tmp_path / "here"), str(tmp_path / "there")
        run_both(dataset, here)
        run(dataset, there, "--candidate-url", CANDIDATE_URL, "--seed", "7")
        for name in ("arm_candidate.json", "answers_candidate.jsonl"):
            shutil.copy(os.path.join(there, name), os.path.join(here, name))

        assert "another plan" in expect_setup_error(capsys, lambda: analyze(here))

    def test_another_schema_version_is_refused(
        self, dataset, engines, tmp_path, capsys
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        path = os.path.join(out, PLAN_FILE)
        with open(path) as f:
            plan = json.load(f)
        plan["schema_version"] = 99
        with open(path, "w") as f:
            json.dump(plan, f)

        assert "schema version 99" in expect_setup_error(capsys, lambda: analyze(out))

    def test_an_unfinished_pass_is_ignored(self, dataset, engines, tmp_path) -> None:
        """Answers without their arm file are from a pass that never ended."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        shutil.copy(
            os.path.join(out, "answers_candidate.jsonl"),
            os.path.join(out, "answers_control.jsonl"),
        )

        analyze(out)

        assert "control" not in summary(out)["arms"]

    def test_a_saved_reference_can_be_reused(self, dataset, engines, tmp_path) -> None:
        """CI copies a saved reference in and runs only the candidate arm."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        saved, fresh = str(tmp_path / "saved"), str(tmp_path / "fresh")
        run(dataset, saved, "--reference-url", REF_URL)
        os.makedirs(fresh)
        for name in os.listdir(saved):
            shutil.copy(os.path.join(saved, name), os.path.join(fresh, name))
        reference_requests = len(engines[REF_URL].sent)

        run(dataset, fresh, "--candidate-url", CANDIDATE_URL)
        analyze(fresh)

        assert summary(fresh)["verdict"] == "PASS"
        assert len(engines[REF_URL].sent) == reference_requests


class TestJudge:
    def test_judge_url_makes_judge_the_default_metric(self) -> None:
        config = analyze_stage.analyze_config_from_args(
            parse_analyze("out", "--judge-url", "http://judge"), served_model="m"
        )
        assert config.verdict_metric is VerdictMetric.JUDGE
        assert config.judge is not None
        assert config.judge.model == "m"

    def test_explicit_metric_wins_over_the_judge(self) -> None:
        config = analyze_stage.analyze_config_from_args(
            parse_analyze(
                "out", "--judge-url", "http://judge", "--verdict-metric", "match"
            ),
            served_model="m",
        )
        assert config.verdict_metric is VerdictMetric.MATCH

    def test_judged_analysis_fails_on_wrong_answers(
        self, dataset, engines, judge_client, tmp_path
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(WRONG))
        out = str(tmp_path / "out")
        run_both(dataset, out)

        with pytest.raises(SystemExit):
            analyze(out, "--judge-url", "http://judge")

        result = summary(out)
        assert result["verdict"] == "FAIL"
        assert result["judge"]["model"] == "fake-model"
        assert result["settings"]["verdict_metric"] == "judge"
        assert judge_client.closed
        assert os.path.isfile(os.path.join(out, JUDGE_CACHE_FILE))
        with open(os.path.join(out, ROWS_FILE)) as f:
            assert "match" in json.loads(f.readline())

    def test_reanalysis_reuses_the_judge_cache(
        self, dataset, engines, judge_client, tmp_path
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        analyze(out, "--judge-url", "http://judge")
        calls = len(judge_client.prompts)

        analyze(out, "--judge-url", "http://judge", "--tolerance", "0.1")

        # Only the sanity check is judged again.
        assert len(judge_client.prompts) - calls == 4

    def test_unreachable_judge_fails_before_scoring(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        monkeypatch.setattr(
            analyze_stage, "OpenAIJudgeClient", lambda _s: FakeJudgeClient(failing=1)
        )

        with pytest.raises(RuntimeError, match="judge down"):
            analyze(out, "--judge-url", "http://j")
        assert not os.path.exists(os.path.join(out, SUMMARY_FILE))

    def test_judge_that_misreads_its_reasoning_is_refused(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        """A thinking judge without a reasoning parser would zero every arm."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        thinking = FakeJudgeClient(lambda _p: "<think>0 doubts.</think> 4")
        monkeypatch.setattr(analyze_stage, "OpenAIJudgeClient", lambda _s: thinking)

        with pytest.raises(RuntimeError, match="sanity-check"):
            analyze(out, "--judge-url", "http://j")

    def test_judge_failing_mid_scoring_keeps_its_progress(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)

        def reply(prompt: str) -> str:
            # The sanity check passes; the run's own questions hit a dead judge.
            if "Nordic?" in prompt:
                raise RuntimeError("judge down")
            return rate_exact(prompt)

        client = FakeJudgeClient(reply)
        monkeypatch.setattr(analyze_stage, "OpenAIJudgeClient", lambda _s: client)

        with pytest.raises(RuntimeError, match="rerun `lmcache bench quality analyze`"):
            analyze(out, "--judge-url", "http://j")

        assert os.path.isfile(os.path.join(out, JUDGE_CACHE_FILE))
        assert not os.path.exists(os.path.join(out, SUMMARY_FILE))
        assert client.closed
