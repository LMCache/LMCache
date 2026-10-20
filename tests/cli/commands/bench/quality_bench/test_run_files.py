# SPDX-License-Identifier: Apache-2.0
"""Tests for the results directory shared by ``run`` and ``analyze``."""

# Standard
from datetime import datetime
import json
import os

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import BlendCounters
from lmcache.cli.commands.bench.quality_bench.core.config import (
    ArmRole,
    ReuseScenario,
)
from lmcache.cli.commands.bench.quality_bench.core.plan import QualityPlan, build_plan
from lmcache.cli.commands.bench.quality_bench.core.run_files import (
    PLAN_FILE,
    SUMMARY_FILE,
    create_run_dir,
    ensure_plan,
    load_arms,
    load_plan,
    plan_fingerprint,
    remove_role,
    write_arm,
)
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmAnswer, ArmRun

# Local
from .conftest import CHUNK
from .fakes import samples

_STARTED = datetime(2026, 10, 12, 18, 30, 5)
_SETTINGS = {"model": "m", "seed": 1}


@pytest.fixture
def plan(aligner) -> QualityPlan:
    return build_plan(samples(), aligner, ReuseScenario.REORDERED, CHUNK, 1)


def _run(plan: QualityPlan, role: ArmRole, run_index: int = 0) -> ArmRun:
    measured = role is ArmRole.CANDIDATE
    counters = BlendCounters(requested_tokens=64.0, non_prefix_hit_tokens=32.0)
    answers = [
        ArmAnswer(
            sample_id=s.sample.sample_id,
            request_id=s.request_id,
            successful=True,
            response=f"<final_answer>{s.sample.answers[0]}</final_answer>",
            answer=s.sample.answers[0],
            ttft=0.125,
            num_output_tokens=7,
            counters=counters if measured else BlendCounters(),
        )
        for s in plan.samples
    ]
    return ArmRun(
        role=role,
        run_index=run_index,
        answers=answers,
        counters=counters if measured else BlendCounters(),
        engagement_measured=measured,
        store_failures=0,
    )


def _save(out: str, plan: QualityPlan, run: ArmRun) -> None:
    fingerprint = ensure_plan(out, plan, _SETTINGS)
    write_arm(out, run, fingerprint, "http://x", "", "v1", _STARTED, _STARTED)


class TestCreateRunDir:
    def test_named_for_the_start_time(self, tmp_path) -> None:
        path = create_run_dir(str(tmp_path), _STARTED)
        assert path == os.path.join(tmp_path, "20261012-183005")
        assert os.path.isdir(path)

    def test_creates_a_missing_parent(self, tmp_path) -> None:
        root = tmp_path / "nested" / "results"
        assert os.path.isdir(create_run_dir(str(root), _STARTED))

    def test_same_second_runs_get_their_own_directories(self, tmp_path) -> None:
        paths = [create_run_dir(str(tmp_path), _STARTED) for _ in range(3)]
        assert [os.path.basename(p) for p in paths] == [
            "20261012-183005",
            "20261012-183005-2",
            "20261012-183005-3",
        ]


class TestPlanRecord:
    def test_round_trip(self, plan) -> None:
        record = json.loads(json.dumps(plan.to_record()))
        assert QualityPlan.from_record(record) == plan

    def test_edited_prompt_is_refused(self, plan) -> None:
        record = plan.to_record()
        record["samples"][0]["messages"][0]["content"] += " edited"  # type: ignore[index]
        with pytest.raises(ValueError, match="prompt_hash"):
            QualityPlan.from_record(record)

    def test_malformed_record_is_refused(self, plan) -> None:
        record = plan.to_record()
        del record["samples"][0]["question"]  # type: ignore[index]
        with pytest.raises(ValueError, match="malformed"):
            QualityPlan.from_record(record)

    def test_fingerprint_ignores_key_order(self) -> None:
        assert plan_fingerprint({"a": 1, "b": 2}, "p") == plan_fingerprint(
            {"b": 2, "a": 1}, "p"
        )


class TestEnsurePlan:
    def test_first_run_records_the_plan(self, plan, tmp_path) -> None:
        out = str(tmp_path / "out")
        fingerprint = ensure_plan(out, plan, _SETTINGS)
        record = load_plan(out)
        assert record.plan == plan
        assert record.fingerprint == fingerprint
        assert dict(record.settings) == _SETTINGS

    def test_same_plan_is_accepted(self, plan, tmp_path) -> None:
        out = str(tmp_path)
        assert ensure_plan(out, plan, _SETTINGS) == ensure_plan(out, plan, _SETTINGS)

    def test_other_settings_name_each_change(self, plan, tmp_path) -> None:
        ensure_plan(str(tmp_path), plan, _SETTINGS)
        with pytest.raises(ValueError, match="seed: 1 -> 2"):
            ensure_plan(str(tmp_path), plan, {**_SETTINGS, "seed": 2})

    def test_other_prompts_with_the_same_settings_are_refused(
        self, plan, aligner, tmp_path
    ) -> None:
        ensure_plan(str(tmp_path), plan, _SETTINGS)
        other = build_plan(samples()[:2], aligner, ReuseScenario.REORDERED, CHUNK, 1)
        with pytest.raises(ValueError, match="prompts differ"):
            ensure_plan(str(tmp_path), other, _SETTINGS)

    def test_edited_plan_file_is_refused(self, plan, tmp_path) -> None:
        ensure_plan(str(tmp_path), plan, _SETTINGS)
        path = os.path.join(tmp_path, PLAN_FILE)
        with open(path) as f:
            record = json.load(f)
        record["settings"]["seed"] = 99
        with open(path, "w") as f:
            json.dump(record, f)
        with pytest.raises(ValueError, match="digests"):
            load_plan(str(tmp_path))


class TestArms:
    def test_round_trip_in_verdict_order(self, plan, tmp_path) -> None:
        out = str(tmp_path)
        written = [
            _run(plan, ArmRole.CANDIDATE),
            _run(plan, ArmRole.REFERENCE, 1),
            _run(plan, ArmRole.CONTROL),
            _run(plan, ArmRole.REFERENCE, 0),
        ]
        for run in written:
            _save(out, plan, run)

        arms = load_arms(out, load_plan(out))

        assert [a.run.label for a in arms] == [
            "reference_1",
            "reference_2",
            "control",
            "candidate",
        ]
        assert arms[-1].run == written[0]
        assert arms[0].url == "http://x"
        assert arms[0].engine_version == "v1"

    def test_missing_reference_repeat_is_refused(self, plan, tmp_path) -> None:
        out = str(tmp_path)
        _save(out, plan, _run(plan, ArmRole.REFERENCE, 1))
        _save(out, plan, _run(plan, ArmRole.CANDIDATE))
        with pytest.raises(ValueError, match="consecutive"):
            load_arms(out, load_plan(out))

    def test_remove_role_leaves_other_roles(self, plan, tmp_path) -> None:
        out = str(tmp_path)
        for run in (_run(plan, ArmRole.REFERENCE), _run(plan, ArmRole.CANDIDATE)):
            _save(out, plan, run)
        (tmp_path / SUMMARY_FILE).write_text("{}")

        removed = remove_role(out, ArmRole.REFERENCE)

        assert {os.path.basename(p) for p in removed} == {
            "arm_reference_1.json",
            "answers_reference_1.jsonl",
            SUMMARY_FILE,
        }
        assert os.path.exists(tmp_path / "arm_candidate.json")
