# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache bench quality run`` with scripted engines."""

# Standard
import json
import os

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality import alignment
from lmcache.cli.commands.bench.engine_bench.quality.alignment import Filler
from lmcache.cli.commands.bench.quality_bench.core import run_stage
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import BlendCounters
from lmcache.cli.commands.bench.quality_bench.core.config import (
    ArmRole,
    ReuseScenario,
)
from lmcache.cli.commands.bench.quality_bench.core.run_files import (
    MANIFEST_FILE,
    PLAN_FILE,
    SUMMARY_FILE,
)

# Local
from .fakes import FakeSender, SteppingCounters, answering
from .stages import (
    CANDIDATE_URL,
    CONTROL_URL,
    METRICS_URL,
    REF_URL,
    RIGHT,
    analyze,
    expect_setup_error,
    parse_run,
    run,
    run_both,
    summary,
)


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _files(out: str) -> set[str]:
    return set(os.listdir(out))


class TestRunConfigFromArgs:
    def test_one_arm_with_defaults(self) -> None:
        config = run_stage.run_config_from_args(
            parse_run("--reference-url", REF_URL, "--model", "m", "--dataset", "d")
        )
        assert [a.role for a in config.arms] == [ArmRole.REFERENCE]
        assert config.reference_repeats == 2
        assert config.scenario is ReuseScenario.INDEPENDENT
        assert config.output_dir == ""

    def test_every_arm_and_options(self) -> None:
        config = run_stage.run_config_from_args(
            parse_run(
                "--reference-url",
                REF_URL,
                "--candidate-url",
                CANDIDATE_URL,
                "--control-url",
                CONTROL_URL,
                "--model",
                "m",
                "--dataset",
                "d",
                "--scenario",
                "reordered",
                "--limit",
                "5",
                "--template-kwargs",
                "reasoning_effort=low",
                "--reference-repeats",
                "3",
            )  # fmt: skip
        )
        assert [a.role for a in config.arms] == [
            ArmRole.REFERENCE,
            ArmRole.CONTROL,
            ArmRole.CANDIDATE,
        ]
        assert config.scenario is ReuseScenario.REORDERED
        assert config.num_samples == 5
        assert config.reference_repeats == 3
        assert config.template_kwargs

    def test_an_arm_is_required(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            run_stage.run_config_from_args(parse_run("--model", "m", "--dataset", "d"))

    def test_reference_repeats_need_the_reference(self) -> None:
        """Otherwise the flag would be silently ignored."""
        with pytest.raises(ValueError, match="--reference-url"):
            run_stage.run_config_from_args(
                parse_run(
                    "--candidate-url",
                    CANDIDATE_URL,
                    "--model",
                    "m",
                    "--dataset",
                    "d",
                    "--reference-repeats",
                    "3",
                )  # fmt: skip
            )

    def test_metrics_need_the_candidate_arm(self) -> None:
        with pytest.raises(ValueError, match="--candidate-url"):
            run_stage.run_config_from_args(
                parse_run(
                    "--reference-url",
                    REF_URL,
                    "--model",
                    "m",
                    "--dataset",
                    "d",
                    "--capture-blend-metrics",
                    METRICS_URL,
                )  # fmt: skip
            )


class TestRunFiles:
    def test_reference_run_saves_plan_and_passes(
        self, dataset, engines, tmp_path
    ) -> None:
        out = str(tmp_path / "out")
        run(dataset, out, "--reference-url", REF_URL)

        assert _files(out) == {
            PLAN_FILE,
            MANIFEST_FILE,
            "arm_reference_1.json",
            "answers_reference_1.jsonl",
            "arm_reference_2.json",
            "answers_reference_2.jsonl",
        }
        meta = _read(os.path.join(out, "arm_reference_1.json"))
        assert meta["url"] == REF_URL
        assert meta["engine_version"] == "fake-1.0"
        assert (
            meta["plan_fingerprint"]
            == _read(os.path.join(out, PLAN_FILE))["plan_fingerprint"]
        )
        assert engines[REF_URL].closed

    def test_another_arm_joins_the_same_directory(
        self, dataset, engines, tmp_path
    ) -> None:
        """Arms served one at a time share one results directory."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run(dataset, out, "--reference-url", REF_URL)
        plan_before = _read(os.path.join(out, PLAN_FILE))

        run(dataset, out, "--candidate-url", CANDIDATE_URL)

        assert "arm_candidate.json" in _files(out)
        assert "arm_reference_1.json" in _files(out)
        assert _read(os.path.join(out, PLAN_FILE)) == plan_before

    def test_other_settings_are_refused_before_any_request(
        self, dataset, engines, tmp_path, capsys
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run(dataset, out, "--reference-url", REF_URL)

        error = expect_setup_error(
            capsys,
            lambda: run(dataset, out, "--candidate-url", CANDIDATE_URL, "--seed", "7"),
        )
        assert "seed: 42 -> 7" in error
        assert engines[CANDIDATE_URL].sent == []

    def test_drifted_prompts_are_refused(
        self, dataset, engines, tmp_path, monkeypatch, capsys
    ) -> None:
        """Same flags, different prompts: e.g. the tokenizer changed."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        # The words filler draws from a pool sized by the tokenizer; a
        # smaller pool stands in for a changed tokenizer.
        run(dataset, out, "--reference-url", REF_URL, "--filler", "words")
        monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 150)

        error = expect_setup_error(
            capsys,
            lambda: run(
                dataset, out, "--candidate-url", CANDIDATE_URL, "--filler", "words"
            ),
        )
        assert "prompts differ" in error
        assert engines[CANDIDATE_URL].sent == []

    def test_rerunning_an_arm_replaces_its_passes(
        self, dataset, engines, tmp_path
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run(dataset, out, "--reference-url", REF_URL, "--reference-repeats", "3")
        run(dataset, out, "--candidate-url", CANDIDATE_URL)
        analyze(out)
        assert SUMMARY_FILE in _files(out)

        run(dataset, out, "--reference-url", REF_URL, "--reference-repeats", "2")

        files = _files(out)
        assert "arm_reference_2.json" in files
        assert "arm_reference_3.json" not in files
        assert "answers_reference_3.jsonl" not in files
        # The candidate pass stays; the summary described the old passes.
        assert "arm_candidate.json" in files
        assert SUMMARY_FILE not in files

    def test_without_output_dir_a_new_directory_is_created(
        self, dataset, engines, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.chdir(tmp_path)
        run_stage.run_quality_run(
            parse_run(
                "--reference-url",
                REF_URL,
                "--model",
                "fake-model",
                "--dataset",
                dataset,
                "--chunk-size",
                "32",
            )  # fmt: skip
        )
        [created] = os.listdir(tmp_path / run_stage.DEFAULT_RESULTS_ROOT)
        results_dir = os.path.join(run_stage.DEFAULT_RESULTS_ROOT, created)
        assert os.path.isfile(tmp_path / results_dir / PLAN_FILE)
        assert f"analyze {results_dir}" in capsys.readouterr().out

    def test_a_crash_in_a_later_pass_keeps_the_earlier_ones(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        """Each pass is saved as soon as it finishes."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))

        class FailAfterPreflight(SteppingCounters):
            reads = 0

            def read(self) -> BlendCounters:
                self.reads += 1
                if self.reads > 1:
                    raise RuntimeError("Could not read blend-server metrics")
                return super().read()

        step = BlendCounters(requested_tokens=1.0)
        monkeypatch.setattr(
            run_stage, "PrometheusBlendCounters", lambda url: FailAfterPreflight(step)
        )
        out = str(tmp_path / "out")

        with pytest.raises(RuntimeError, match="blend-server metrics"):
            run_both(dataset, out, "--capture-blend-metrics", METRICS_URL)

        files = _files(out)
        assert {"arm_reference_1.json", "arm_reference_2.json"} <= files
        assert "arm_candidate.json" not in files


class TestRunOutcome:
    def test_only_caching_arms_store_first(self, dataset, engines, tmp_path) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        engines[CONTROL_URL] = FakeSender(answering(RIGHT))

        run_both(dataset, str(tmp_path / "out"), "--control-url", CONTROL_URL)

        def stores(sender: FakeSender) -> int:
            return sum(rid.startswith("store") for rid, _, _ in sender.sent)

        assert stores(engines[REF_URL]) == 0
        assert stores(engines[CANDIDATE_URL]) > 0
        assert stores(engines[CONTROL_URL]) == stores(engines[CANDIDATE_URL])
        # Two reference passes over three questions.
        assert len(engines[REF_URL].sent) == 6

    def test_an_unreachable_engine_saves_its_pass_and_exits_non_zero(
        self, dataset, engines, tmp_path
    ) -> None:
        """CI flags the broken job; analyze still reports the INVALID."""
        # Three questions over five distinct passages.
        down = frozenset([f"q{i}" for i in range(3)] + [f"store{i}" for i in range(5)])
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT), failing=down)
        out = str(tmp_path / "out")

        with pytest.raises(SystemExit) as exc:
            run(dataset, out, "--candidate-url", CANDIDATE_URL)

        assert exc.value.code == 1
        assert "arm_candidate.json" in _files(out)

    def test_unreachable_metrics_fail_before_the_arms_run(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        """Without the preflight read, a bad URL surfaces only when the candidate
        pass starts, after every other pass has run."""
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))

        class DownCounters(SteppingCounters):
            def read(self) -> BlendCounters:
                raise RuntimeError("Could not read blend-server metrics")

        monkeypatch.setattr(
            run_stage,
            "PrometheusBlendCounters",
            lambda url: DownCounters(BlendCounters()),
        )

        with pytest.raises(RuntimeError, match="blend-server metrics"):
            run_both(
                dataset,
                str(tmp_path / "out"),
                "--capture-blend-metrics",
                METRICS_URL,
            )
        assert engines[REF_URL].sent == []
        assert engines[CANDIDATE_URL].sent == []

    def test_missing_tokenizer_is_reported(
        self, dataset, engines, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(run_stage, "try_load_tokenizer", lambda _: None)
        with pytest.raises(RuntimeError, match="tokenizer"):
            run(dataset, str(tmp_path / "out"), "--reference-url", REF_URL)

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
            run(str(path), str(tmp_path / "out"), "--reference-url", REF_URL)
        assert engines[REF_URL].sent == []


class TestEngineVersion:
    def test_unreachable_engine_reports_none(self) -> None:
        assert run_stage.engine_version("http://127.0.0.1:9/v1") is None


class TestFiller:
    def test_dots_is_the_default(self) -> None:
        config = run_stage.run_config_from_args(
            parse_run("--reference-url", REF_URL, "--model", "m", "--dataset", "d")
        )
        assert config.filler is Filler.DOTS
        assert config.plan_settings()["filler"] == "dots"

    def test_words_is_recorded_too(self) -> None:
        config = run_stage.run_config_from_args(
            parse_run(
                "--reference-url",
                REF_URL,
                "--model",
                "m",
                "--dataset",
                "d",
                "--filler",
                "words",
            )  # fmt: skip
        )
        assert config.plan_settings()["filler"] == "words"

    def test_another_filler_is_refused_in_the_same_directory(
        self, dataset, engines, tmp_path, capsys
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run(dataset, out, "--reference-url", REF_URL)

        error = expect_setup_error(
            capsys,
            lambda: run(
                dataset, out, "--candidate-url", CANDIDATE_URL, "--filler", "dashes"
            ),
        )
        assert "filler: 'dots' -> 'dashes'" in error
        assert engines[CANDIDATE_URL].sent == []

    def test_padding_is_reported_in_the_summary(
        self, dataset, engines, tmp_path, judge_client
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        analyze(out)
        padding = summary(out)["padding"]
        assert padding["filler"] == "dots"
        assert 0 < padding["share"] < 1

    def test_a_misplaced_passage_exits_2_before_any_request(
        self, dataset, engines, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.setattr(
            alignment.ChunkAligner, "_chat_prefix_tokens", lambda _self: 1
        )
        out = str(tmp_path / "out")
        error = expect_setup_error(
            capsys, lambda: run(dataset, out, "--reference-url", REF_URL)
        )
        assert "would lose cache reuse" in error
        assert engines[REF_URL].sent == []
        assert not os.path.exists(out)


class TestConcatenatedRun:
    def test_filler_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--filler"):
            run_stage.run_config_from_args(
                parse_run(
                    "--reference-url",
                    REF_URL,
                    "--model",
                    "m",
                    "--dataset",
                    "d",
                    "--scenario",
                    "concatenated",
                    "--filler",
                    "dots",
                )  # fmt: skip
            )

    def test_settings_record_the_unit_size_and_no_filler(self) -> None:
        config = run_stage.run_config_from_args(
            parse_run(
                "--reference-url",
                REF_URL,
                "--model",
                "m",
                "--dataset",
                "d",
                "--scenario",
                "concatenated",
                "--min-unit-tokens",
                "4096",
            )  # fmt: skip
        )
        settings = config.plan_settings()
        assert settings["filler"] == "none"
        assert settings["min_unit_tokens"] == 4096

    def test_other_scenarios_do_not_record_the_unit_size(self) -> None:
        config = run_stage.run_config_from_args(
            parse_run("--reference-url", REF_URL, "--model", "m", "--dataset", "d")
        )
        assert "min_unit_tokens" not in config.plan_settings()

    def test_reuse_is_reported_by_kind(
        self, dataset, engines, tmp_path, monkeypatch, judge_client, capsys
    ) -> None:
        blending = BlendCounters(requested_tokens=100.0, non_prefix_hit_tokens=60.0)
        monkeypatch.setattr(
            run_stage, "PrometheusBlendCounters", lambda url: SteppingCounters(blending)
        )
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(
            dataset, out,
            "--scenario", "concatenated", "--min-unit-tokens", "64",
            "--capture-blend-metrics", METRICS_URL,
        )  # fmt: skip
        analyze(out)

        assert "reuse: expected" in capsys.readouterr().out
        reuse = summary(out)["reuse"]
        assert reuse["expected_tokens"] > 0
        assert reuse["observed_blended_tokens"] > 0
        assert reuse["observed_prefix_tokens"] == 0
        assert [g["kind"] for g in reuse["groups"]] == ["blended"]
        with open(os.path.join(out, "rows.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        assert {row["reuse"]["kind"] for row in rows} == {"blended"}
        assert all(row["reuse"]["expected_tokens"] is not None for row in rows)
        manifest = [json.loads(line) for line in open(os.path.join(out, MANIFEST_FILE))]
        assert all(entry["unit_position"] is not None for entry in manifest)

    def test_too_little_reuse_is_invalid_whatever_the_blend_share(
        self, dataset, engines, tmp_path, monkeypatch, judge_client, capsys
    ) -> None:
        """All-blended counters pass the blend share but reuse 1 token a read."""
        trickle = BlendCounters(requested_tokens=1.0, non_prefix_hit_tokens=1.0)
        monkeypatch.setattr(
            run_stage, "PrometheusBlendCounters", lambda url: SteppingCounters(trickle)
        )
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(
            dataset, out,
            "--scenario", "concatenated", "--min-unit-tokens", "64",
            "--capture-blend-metrics", METRICS_URL,
        )  # fmt: skip
        with pytest.raises(SystemExit) as exc:
            analyze(out)
        assert exc.value.code == 1
        assert "blend share (not gated): 1.000" in capsys.readouterr().out
        result = summary(out)
        assert result["verdict"] == "INVALID"
        [reason] = result["invalid_reasons"]
        assert reason.startswith("observed reuse ")
        assert result["engagement"]["gate"] == "reuse"
        assert result["settings"]["min_reuse_ratio"] == 0.8

        analyze(out, "--min-reuse-ratio", "0")
        assert summary(out)["verdict"] == "PASS"

    def test_without_counters_only_the_expectation_is_reported(
        self, dataset, engines, tmp_path, judge_client
    ) -> None:
        engines[CANDIDATE_URL] = FakeSender(answering(RIGHT))
        out = str(tmp_path / "out")
        run_both(dataset, out)
        analyze(out)
        assert summary(out)["reuse"] is None
        with open(os.path.join(out, "rows.jsonl")) as f:
            row = json.loads(f.readline())
        assert row["reuse"]["expected_tokens"] > 0
        assert row["reuse"]["kind"] is None
