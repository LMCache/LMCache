# SPDX-License-Identifier: Apache-2.0
"""Tests for the RAGAS AnswerAccuracy judge port."""

# Standard
import asyncio
import hashlib
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.quality_bench.core import judge as judge_module
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import BlendCounters
from lmcache.cli.commands.bench.quality_bench.core.config import ArmRole, ReuseScenario
from lmcache.cli.commands.bench.quality_bench.core.judge import (
    JUDGE_RETRIES,
    AnswerAccuracyJudge,
    combine_ratings,
    judge_prompts,
    parse_rating,
)
from lmcache.cli.commands.bench.quality_bench.core.plan import build_plan
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmAnswer, ArmRun

# Local
from .conftest import CHUNK
from .fakes import FakeJudgeClient, samples

# sha256 of ragas 0.4.3 AnswerAccuracy.template_accuracy1 / template_accuracy2
# (ragas/metrics/_nv_metrics.py). A mismatch means the port drifted from the
# prompts its scores are compared with.
_RAGAS_TEMPLATE_SHA256 = (
    "6a57033664632416f00db87e6406ead3a43c6f426d59992abfc7dd71da971903",
    "f91f7db28b947f8a324dfc335dcf1166e709e6ae6ec7aeff402a5acacd5fd85e",
)


def _judge(client: FakeJudgeClient, cache_path: str = "") -> AnswerAccuracyJudge:
    return AnswerAccuracyJudge(client, "judge-model", 4, cache_path=cache_path)


class TestPrompts:
    def test_templates_match_ragas_byte_for_byte(self) -> None:
        templates = (
            judge_module._TEMPLATE_ANSWER_FIRST,
            judge_module._TEMPLATE_REFERENCE_FIRST,
        )
        digests = tuple(hashlib.sha256(t.encode()).hexdigest() for t in templates)
        assert digests == _RAGAS_TEMPLATE_SHA256

    def test_second_prompt_swaps_the_roles(self) -> None:
        first, second = judge_prompts("Q?", "mine", "gold")
        assert "### User Answer: mine\n### Reference Answer: gold" in first
        assert "Reference Answer: gold\n\nUser Answer: mine" in second
        assert first.endswith("The rating is:\n")
        assert second.endswith("Rating: ")


class TestParseRating:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("4", 1.0),
            ("Rating: 2", 0.5),
            ("0", 0.0),
            # ragas takes the smallest digit 0-4 anywhere in the reply.
            ("4.0", 0.0),
            ("I'd say 4, not 2", 0.5),
            ("excellent", None),
            ("", None),
        ],
    )
    def test_ragas_parsing(self, reply: str, expected: float | None) -> None:
        assert parse_rating(reply) == expected

    def test_combine(self) -> None:
        assert combine_ratings(1.0, 0.5) == 0.75
        assert combine_ratings(None, 0.5) == 0.5
        assert combine_ratings(1.0, None) == 1.0
        assert combine_ratings(None, None) is None


class TestAnswerAccuracyJudge:
    def test_both_prompts_are_asked(self) -> None:
        client = FakeJudgeClient()
        result = asyncio.run(_judge(client).score("Q?", "Paris", "Paris"))
        assert result.score == 1.0
        assert result.ratings == (1.0, 1.0)
        assert len(client.prompts) == 2

    def test_split_ratings_average(self) -> None:
        def reply(prompt: str) -> str:
            return "4" if prompt.startswith("Instruction") else "2"

        result = asyncio.run(_judge(FakeJudgeClient(reply)).score("Q?", "a", "b"))
        assert result.score == 0.75

    def test_unparseable_reply_is_retried(self) -> None:
        replies = iter(["no idea", "still no idea", "4", "4"])
        client = FakeJudgeClient(lambda _p: next(replies))
        result = asyncio.run(_judge(client).score("Q?", "a", "a"))
        assert result.score == 1.0
        assert len(client.prompts) == 4

    def test_gives_up_after_the_retry_limit(self) -> None:
        client = FakeJudgeClient(lambda _p: "no rating")
        result = asyncio.run(_judge(client).score("Q?", "a", "a"))
        assert result.score is None
        assert len(client.prompts) == 2 * JUDGE_RETRIES

    def test_failed_request_raises_instead_of_being_retried(self) -> None:
        """Transport retries belong to the client; a failure that reaches the
        judge means the endpoint is down, not that the reply lacked a rating."""
        client = FakeJudgeClient(failing=1)
        with pytest.raises(RuntimeError, match="judge down"):
            asyncio.run(_judge(client).score("Q?", "a", "a"))
        assert len(client.prompts) == 1

    def test_identical_answers_are_judged_once(self) -> None:
        client = FakeJudgeClient()
        judge = _judge(client)

        async def twice() -> None:
            await asyncio.gather(
                judge.score("Q?", "a", "a"), judge.score("Q?", "a", "a")
            )

        asyncio.run(twice())
        assert len(client.prompts) == 2

    def test_cache_file_is_reused(self, tmp_path) -> None:
        path = str(tmp_path / "judge_cache.jsonl")
        asyncio.run(_judge(FakeJudgeClient(), path).score("Q?", "a", "a"))
        again = FakeJudgeClient()
        result = asyncio.run(_judge(again, path).score("Q?", "a", "a"))
        assert result.score == 1.0
        assert again.prompts == []
        [record] = [json.loads(line) for line in open(path)]
        assert record["score"] == 1.0

    def test_truncated_cache_line_is_skipped_and_rejudged(self, tmp_path) -> None:
        """A run killed mid-append leaves a partial last line."""
        path = tmp_path / "judge_cache.jsonl"
        asyncio.run(_judge(FakeJudgeClient(), str(path)).score("Q?", "a", "a"))
        with open(path, "a") as f:
            f.write('{"key": "abc", "sco')

        again = FakeJudgeClient()
        judge = _judge(again, str(path))
        assert asyncio.run(judge.score("Q?", "a", "a")).score == 1.0
        assert again.prompts == []  # the intact line was reused
        assert asyncio.run(judge.score("Q?", "b", "b")).score == 1.0

        # The judgment appended after the partial line survives a reload.
        third = FakeJudgeClient()
        asyncio.run(_judge(third, str(path)).score("Q?", "b", "b"))
        assert third.prompts == []

    def test_unrated_answers_are_not_cached(self, tmp_path) -> None:
        path = str(tmp_path / "judge_cache.jsonl")
        client = FakeJudgeClient(lambda _p: "no rating")
        asyncio.run(_judge(client, path).score("Q?", "a", "a"))
        assert not (tmp_path / "judge_cache.jsonl").exists()

    def test_cache_key_includes_the_judge_model(self, tmp_path) -> None:
        path = str(tmp_path / "judge_cache.jsonl")
        asyncio.run(_judge(FakeJudgeClient(), path).score("Q?", "a", "a"))
        other = FakeJudgeClient()
        judge = AnswerAccuracyJudge(other, "another-model", 4, cache_path=path)
        asyncio.run(judge.score("Q?", "a", "a"))
        assert len(other.prompts) == 2

    def test_rejects_non_positive_concurrency(self) -> None:
        with pytest.raises(ValueError):
            AnswerAccuracyJudge(FakeJudgeClient(), "m", 0)


def _thinking_in_content(prompt: str) -> str:
    """A thinking judge served without a reasoning parser: the verdict is
    right, but the reasoning before it holds smaller digits."""
    return "<think>Paris is correct; I see 0 errors and 1 match.</think>\n4"


class TestCheckRatings:
    def test_a_working_judge_passes(self) -> None:
        client = FakeJudgeClient()
        asyncio.run(_judge(client).check_ratings())
        # A right and a wrong answer, each on both prompts.
        assert len(client.prompts) == 4

    def test_reasoning_in_the_reply_is_refused(self) -> None:
        client = FakeJudgeClient(_thinking_in_content)
        with pytest.raises(RuntimeError, match="reasoning-parser"):
            asyncio.run(_judge(client).check_ratings())

    def test_a_judge_that_rates_everything_4_is_refused(self) -> None:
        client = FakeJudgeClient(lambda _p: "4")
        with pytest.raises(RuntimeError, match="'Berlin'.* 4 instead of 0"):
            asyncio.run(_judge(client).check_ratings())

    def test_no_rating_is_refused(self) -> None:
        client = FakeJudgeClient(lambda _p: "")
        with pytest.raises(RuntimeError, match="no rating"):
            asyncio.run(_judge(client).check_ratings())
        assert len(client.prompts) == JUDGE_RETRIES

    def test_unreachable_judge_raises(self) -> None:
        with pytest.raises(RuntimeError, match="judge down"):
            asyncio.run(_judge(FakeJudgeClient(failing=1)).check_ratings())

    def test_sanity_check_is_never_cached(self, tmp_path) -> None:
        """A cached sanity check would skip the check on the next run."""
        path = tmp_path / "judge_cache.jsonl"
        asyncio.run(_judge(FakeJudgeClient(), str(path)).check_ratings())
        assert not path.exists()


class TestScoreRuns:
    def _run(self, plan, texts: list[str], role: ArmRole) -> ArmRun:
        answers = [
            ArmAnswer(
                sample_id=s.sample.sample_id,
                request_id=s.request_id,
                successful=True,
                response=f"<final_answer>{t}</final_answer>" if t else "thinking",
                answer=t,
                ttft=0.1,
                num_output_tokens=4,
                counters=BlendCounters(),
            )
            for s, t in zip(plan.samples, texts, strict=True)
        ]
        return ArmRun(
            role=role,
            run_index=0,
            answers=answers,
            counters=BlendCounters(),
            engagement_measured=False,
            store_failures=0,
        )

    def test_scores_every_run_and_skips_unparsed(self, aligner) -> None:
        plan = build_plan(samples(), aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
        golds = [s.sample.answers[0] for s in plan.samples]
        reference = self._run(plan, golds, ArmRole.REFERENCE)
        candidate = self._run(plan, [golds[0], "wrong", ""], ArmRole.CANDIDATE)
        client = FakeJudgeClient()
        table = asyncio.run(_judge(client).score_runs(plan, [reference, candidate]))
        assert table[reference.label] == [1.0, 1.0, 1.0]
        assert table[candidate.label] == [1.0, 0.0, 0.0]
        # Reference answers 0 and the candidate's identical answer 0 share one
        # judgment; the unparsed answer is never sent.
        assert len(client.prompts) == 2 * 4

    def test_failed_request_aborts_scoring(self, aligner) -> None:
        plan = build_plan(samples(), aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
        run = self._run(plan, ["a", "b", "c"], ArmRole.REFERENCE)

        def reply(prompt: str) -> str:
            if "User Answer: b" in prompt:
                raise RuntimeError("judge down")
            return "4"

        with pytest.raises(RuntimeError, match="judge down"):
            asyncio.run(_judge(FakeJudgeClient(reply)).score_runs(plan, [run]))

    def test_question_and_first_gold_are_sent(self, aligner) -> None:
        plan = build_plan(samples(), aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
        run = self._run(plan, ["x", "x", "x"], ArmRole.REFERENCE)
        client = FakeJudgeClient()
        asyncio.run(_judge(client).score_runs(plan, [run]))
        first = plan.samples[0].sample
        assert any(
            f"### Question: {first.question}" in p
            and f"### Reference Answer: {first.answers[0]}" in p
            for p in client.prompts
        )
