# SPDX-License-Identifier: Apache-2.0
"""Tests for running a plan against one arm."""

# Standard
import asyncio

# First Party
from lmcache.cli.commands.bench.quality_bench.core import runner
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import (
    BlendCounters,
    BlendCounterSource,
    UnmeasuredBlendCounters,
)
from lmcache.cli.commands.bench.quality_bench.core.config import ArmRole, ReuseScenario
from lmcache.cli.commands.bench.quality_bench.core.plan import build_plan
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmRunner, StorePhase

# Local
from .conftest import CHUNK
from .fakes import FakeSender, SteppingCounters, answering, samples

_ANSWERS = {"Capital?": "Paris", "Nordic?": "Oslo", "Swiss?": "Bern"}


class LaggingCounters(BlendCounterSource):
    """Counters a request can bump now or after a delay, e.g. a server that
    records a lookup after its HTTP response has returned."""

    def __init__(self) -> None:
        self._current = BlendCounters()

    @property
    def measures_engagement(self) -> bool:
        return True

    def read(self) -> BlendCounters:
        return self._current

    def add_now(self, delta: BlendCounters) -> None:
        self._current = self._current + delta

    def add_later(self, delta: BlendCounters, delay_s: float) -> None:
        asyncio.get_running_loop().call_later(delay_s, self.add_now, delta)


def _run(aligner, sender, store_phase, counters=None, role=ArmRole.BLEND):
    plan = build_plan(samples(), aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
    runner = ArmRunner(sender, 64, counters or UnmeasuredBlendCounters())
    return plan, asyncio.run(runner.run(plan, role, 0, store_phase))


class TestArmRunner:
    def test_store_first_sends_store_requests_before_questions(self, aligner) -> None:
        sender = FakeSender(answering(_ANSWERS))
        plan, _ = _run(aligner, sender, StorePhase.STORE_FIRST)
        ids = [request_id for request_id, _, _ in sender.sent]
        stores = [s.request_id for s in plan.store_requests]
        questions = [s.request_id for s in plan.samples]
        assert ids == stores + questions

    def test_store_requests_generate_one_token(self, aligner) -> None:
        sender = FakeSender(answering(_ANSWERS))
        plan, _ = _run(aligner, sender, StorePhase.STORE_FIRST)
        store_budgets = {m for _, _, m in sender.sent[: len(plan.store_requests)]}
        assert store_budgets == {1}

    def test_skip_store_measures_immediately(self, aligner) -> None:
        sender = FakeSender(answering(_ANSWERS))
        plan, _ = _run(aligner, sender, StorePhase.SKIP_STORE, role=ArmRole.REFERENCE)
        assert [r for r, _, _ in sender.sent] == [s.request_id for s in plan.samples]

    def test_questions_use_the_answer_budget(self, aligner) -> None:
        sender = FakeSender(answering(_ANSWERS))
        _run(aligner, sender, StorePhase.SKIP_STORE)
        assert {m for _, _, m in sender.sent} == {64}

    def test_answers_are_extracted_in_plan_order(self, aligner) -> None:
        sender = FakeSender(answering(_ANSWERS))
        _, run = _run(aligner, sender, StorePhase.STORE_FIRST)
        assert [a.answer for a in run.answers] == ["Paris", "Oslo", "Bern"]
        assert all(a.parsed for a in run.answers)

    def test_missing_answer_region_is_unparsed(self, aligner) -> None:
        sender = FakeSender(answering({"Capital?": "Paris"}))
        _, run = _run(aligner, sender, StorePhase.SKIP_STORE)
        assert [a.parsed for a in run.answers] == [True, False, False]

    def test_failed_request_is_unparsed(self, aligner) -> None:
        sender = FakeSender(answering(_ANSWERS), failing=frozenset({"q1"}))
        _, run = _run(aligner, sender, StorePhase.SKIP_STORE)
        assert not run.answers[1].successful
        assert not run.answers[1].parsed

    def test_store_failures_are_counted(self, aligner) -> None:
        sender = FakeSender(answering(_ANSWERS), failing=frozenset({"store0"}))
        _, run = _run(aligner, sender, StorePhase.STORE_FIRST)
        assert run.store_failures == 1

    def test_unmeasured_counters_stay_zero(self, aligner) -> None:
        _, run = _run(aligner, FakeSender(answering(_ANSWERS)), StorePhase.SKIP_STORE)
        assert not run.engagement_measured
        assert run.counters == BlendCounters()

    def test_per_question_and_run_counters(self, aligner) -> None:
        step = BlendCounters(requested_tokens=100, non_prefix_hit_tokens=60)
        _, run = _run(
            aligner,
            FakeSender(answering(_ANSWERS)),
            StorePhase.STORE_FIRST,
            counters=SteppingCounters(step),
        )
        assert run.engagement_measured
        assert all(a.counters == step for a in run.answers)
        assert run.counters.blend_share == 0.6

    def test_late_store_lookups_stay_out_of_the_pass(
        self, aligner, monkeypatch
    ) -> None:
        """Store lookups recorded after their response must not dilute the
        measured blend share: the opening read waits for them to arrive."""
        monkeypatch.setattr(runner, "COUNTER_SETTLE_S", 0.05)
        counters = LaggingCounters()
        store_miss = BlendCounters(requested_tokens=100)
        question_hit = BlendCounters(requested_tokens=100, non_prefix_hit_tokens=60)
        answer = answering(_ANSWERS)

        def respond(request_id: str, messages: list[dict[str, str]]) -> str:
            if request_id.startswith("store"):
                counters.add_later(store_miss, delay_s=0.01)
            else:
                counters.add_now(question_hit)
            return answer(request_id, messages)

        _, run = _run(
            aligner, FakeSender(respond), StorePhase.STORE_FIRST, counters=counters
        )
        assert run.counters.requested_tokens == 300
        assert run.counters.blend_share == 0.6
        assert all(a.counters == question_hit for a in run.answers)

    def test_labels(self, aligner) -> None:
        _, reference = _run(
            aligner,
            FakeSender(answering(_ANSWERS)),
            StorePhase.SKIP_STORE,
            role=ArmRole.REFERENCE,
        )
        assert reference.label == "reference_1"
