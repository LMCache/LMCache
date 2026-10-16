# SPDX-License-Identifier: Apache-2.0
"""LLM-judge scoring for ``lmcache bench quality``.

:class:`AnswerAccuracyJudge` reproduces RAGAS ``AnswerAccuracy``
(``nv_accuracy``) as of ragas 0.4.3, so its scores are comparable with
RAGAS runs on the same answers without depending on ragas and langchain.
The two prompt templates, the rating parser, the per-call temperature, the
retry count and the way the two ratings are combined are copied from
``ragas/metrics/_nv_metrics.py`` (Apache-2.0, Copyright the Ragas authors).
Note that, in following the Ragas implementation: the parser takes the
smallest digit 0-4 found anywhere in the reply + the templates keep their
original wording and spacing.

The judge rates each answer twice, once per template (the second swaps the
roles of answer and reference), on a 0/2/4 scale. The score is the mean of
the two ratings divided by 4, so 0 to 1 in steps of 0.25.
"""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import asyncio
import hashlib
import json
import os

# First Party
from lmcache.cli.commands.bench.quality_bench.config import JudgeSettings
from lmcache.cli.commands.bench.quality_bench.plan import QualityPlan
from lmcache.cli.commands.bench.quality_bench.runner import ArmRun
from lmcache.cli.commands.bench.quality_bench.verdict import ScoreTable
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Identifies the prompts and scoring rules. Recorded with every result.
JUDGE_METRIC = "ragas-0.4.3/nv_accuracy"

# ragas sends every judge call at this temperature, whatever the judge
# client was configured with.
JUDGE_TEMPERATURE = 0.10

# ragas retries a prompt this many times while its reply has no rating. Each
# retry is a fresh sample.
JUDGE_RETRIES = 5

# Transient request failures (connection errors, 408/409/429, 5xx) are
# retried by the OpenAI client itself with exponential backoff. A request
# that still fails means the judge is down, which aborts scoring (rather than
# leaving answers unrated).
JUDGE_HTTP_RETRIES = 5

# Per-request timeout. The client's default (600 s) would let a hung judge
# stall a run for ten minutes per call.
JUDGE_TIMEOUT_S = 120.0

# A question any working judge rates unambiguously, judged before the arms
# run (see ``AnswerAccuracyJudge.check_ratings``). The right answer must rate
# 4 and the wrong one 0 on both prompts.
_SANITY_QUESTION = "What is the capital of France?"
_SANITY_REFERENCE = "Paris"
_SANITY_ANSWERS = (("Paris", 1.0), ("Berlin", 0.0))

# Verbatim from ragas 0.4.3 AnswerAccuracy.template_accuracy1 and
# template_accuracy2 (includes some typos).
_TEMPLATE_ANSWER_FIRST = (
    "Instruction: You are a world class state of the art assistant for "
    "rating a User Answer given a Question. The Question is completely "
    "answered by the Reference Answer.\n"
    "Say 4, if User Answer is full contained and equivalent to Reference "
    "Answerin all terms, "  # codespell:ignore
    "topics, numbers, metrics, dates and units.\n"
    "Say 2, if User Answer is partially contained and almost equivalent to "
    "Reference Answerin all "  # codespell:ignore
    "terms, topics, numbers, metrics, dates and units.\n"
    "Say 0, if User Answer is not contained in Reference Answer or not "
    "accurate in all terms, topics,numbers, metrics, dates and units or the "
    "User Answer do not answer the question.\n"
    "Do not explain or justify your rating. Your rating must be only 4, 2 or "
    "0 according to the instructions above.\n"
    "### Question: {query}\n"
    "### {answer0}: {sentence_inference}\n"
    "### {answer1}: {sentence_true}\n"
    "The rating is:\n"
)
_TEMPLATE_REFERENCE_FIRST = (
    "I will rate the User Answer in comparison to the Reference Answer for a "
    "given Question.\n"
    "A rating of 4 indicates that the User Answer is entirely consistent "
    "with the Reference Answer, covering all aspects, topics, numbers, "
    "metrics, dates, and units.\n"
    "A rating of 2 signifies that the User Answer is mostly aligned with the "
    "Reference Answer, with minor discrepancies in some areas.\n"
    "A rating of 0 means that the User Answer is either inaccurate, "
    "incomplete, or unrelated to the Reference Answer, or it fails to "
    "address the Question.\n"
    "I will provide the rating without any explanation or justification, "
    "adhering to the following scale: 0 (no match), 2 (partial match), 4 "
    "(exact match).\n"
    "Do not explain or justify my rating. My rating must be only 4, 2 or 0 "
    "only.\n\n"
    "Question: {query}\n\n"
    "{answer0}: {sentence_inference}\n\n"
    "{answer1}: {sentence_true}\n\n"
    "Rating: "
)


def judge_prompts(question: str, answer: str, reference: str) -> tuple[str, str]:
    """Render the two judge prompts for one answer, as ragas does.

    Args:
        question: The question asked.
        answer: The answer to rate.
        reference: The gold answer.

    Returns:
        The answer-first prompt and the reference-first prompt.
    """
    first = _TEMPLATE_ANSWER_FIRST.format(
        query=question,
        answer0="User Answer",
        answer1="Reference Answer",
        sentence_inference=answer,
        sentence_true=reference,
    )
    second = _TEMPLATE_REFERENCE_FIRST.format(
        query=question,
        answer0="Reference Answer",
        answer1="User Answer",
        sentence_inference=reference,
        sentence_true=answer,
    )
    return first, second


def parse_rating(reply: str) -> float | None:
    """Read a rating from a judge reply, as ragas does.

    The smallest digit 0-4 found anywhere in the reply wins, so "Rating: 4"
    is 1.0 but "4.0" is 0.0. That is ragas's behavior, kept for parity.

    Args:
        reply: The judge's text.

    Returns:
        The rating divided by 4, or ``None`` if the reply has no digit 0-4.
    """
    for digit in range(5):
        if str(digit) in reply:
            return digit / 4
    return None


def combine_ratings(first: float | None, second: float | None) -> float | None:
    """Combine the two prompts' ratings, as ragas does.

    Args:
        first: Rating from the answer-first prompt.
        second: Rating from the reference-first prompt.

    Returns:
        Their mean / the one available rating if the other is missing /
        ``None`` if both are.
    """
    if first is not None and second is not None:
        return (first + second) / 2
    if first is not None:
        return first
    return second


class JudgeClient(ABC):
    """Sends one prompt to the judge model and returns its reply."""

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Return the judge's reply to *prompt* (empty if it gave none).

        Raises:
            RuntimeError: If the request fails, after any retries the client
                makes itself.
        """

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release the client's connections."""


class OpenAIJudgeClient(JudgeClient):
    """Judge client for an OpenAI-compatible chat completions endpoint.

    Each prompt is sent as a single user message, as ragas's langchain
    wrapper does, at :data:`JUDGE_TEMPERATURE`. Transient failures are
    retried :data:`JUDGE_HTTP_RETRIES` times with backoff, each request
    bounded by :data:`JUDGE_TIMEOUT_S`.
    """

    def __init__(self, settings: JudgeSettings) -> None:
        """Create the client.

        Args:
            settings: The judge's endpoint, model and limits.
        """
        # Third Party
        import openai

        base_url = settings.url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=settings.api_key,
            max_retries=JUDGE_HTTP_RETRIES,
            timeout=JUDGE_TIMEOUT_S,
        )
        self._model = settings.model
        self._max_tokens = settings.max_tokens

    async def complete(self, prompt: str) -> str:
        """Return the judge's reply to *prompt*.

        Raises:
            RuntimeError: If the request fails.
        """
        # Third Party
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=self._max_tokens,
            )
        except openai.OpenAIError as e:
            # Connection errors, timeouts and HTTP errors all arrive as
            # OpenAIError subclasses, after the client's own retries.
            raise RuntimeError(f"judge request failed: {e}") from e
        return response.choices[0].message.content or ""

    async def close(self) -> None:
        """Release the client's connections."""
        await self._client.close()


@dataclass(frozen=True)
class JudgeScore:
    """One answer's judge result.

    Attributes:
        score: Combined score in [0, 1], or ``None`` if neither prompt
            yielded a rating.
        ratings: The answer-first and reference-first ratings.
    """

    score: float | None
    ratings: tuple[float | None, float | None]


class AnswerAccuracyJudge:
    """Scores answers with the RAGAS ``AnswerAccuracy`` judge prompts.

    Identical (question, answer, reference) triples are judged once, both
    within a run and, when a cache file is given, across runs. Answers
    the judge could not rate are not cached.
    """

    def __init__(
        self, client: JudgeClient, model: str, concurrency: int, cache_path: str = ""
    ) -> None:
        """Create the judge.

        Args:
            client: Sends prompts to the judge model.
            model: The judge model's name (part of every cache key).
            concurrency: Answers judged at once.
            cache_path: JSONL file of earlier judgments to reuse and extend.
                If empty, disables the file cache.

        Raises:
            ValueError: If *concurrency* is not positive.
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self._client = client
        self._model = model
        self._semaphore = asyncio.Semaphore(concurrency)
        self._cache_path = cache_path
        self._cache: dict[str, JudgeScore] = self._load_cache()
        self._in_flight: dict[str, asyncio.Future[JudgeScore]] = {}
        self.calls = 0

    async def score(self, question: str, answer: str, reference: str) -> JudgeScore:
        """Judge one answer.

        Args:
            question: The question asked.
            answer: The answer to rate.
            reference: The gold answer.

        Returns:
            The judgment. A prompt whose replies never contain a rating
            after :data:`JUDGE_RETRIES` attempts contributes no rating.

        Raises:
            RuntimeError: If a judge request fails.
        """
        key = self._key(question, answer, reference)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        pending = self._in_flight.get(key)
        if pending is None:
            pending = asyncio.ensure_future(
                self._judge(key, judge_prompts(question, answer, reference))
            )
            self._in_flight[key] = pending
        return await pending

    async def score_runs(self, plan: QualityPlan, runs: Sequence[ArmRun]) -> ScoreTable:
        """Judge every answer of every run.

        An unparsed answer scores ``0.0`` without a judge call, as it does
        under the lexical scores. The reference passed to the judge is the
        question's first gold answer, as in a RAGAS hand-off.

        Args:
            plan: The plan the runs measured.
            runs: The arm runs, each answering every plan sample in order.

        Returns:
            Per run label, one score per question. ``None`` where the judge
            gave no rating.

        Raises:
            RuntimeError: If a judge request fails. Judgments already made
                stay in the cache file.
        """

        async def judge(run: ArmRun) -> list[float | None]:
            async def one(index: int) -> float | None:
                answer = run.answers[index]
                if not answer.parsed:
                    return 0.0
                sample = plan.samples[index].sample
                result = await self.score(
                    sample.question, answer.answer, sample.answers[0]
                )
                return result.score

            tasks = [asyncio.ensure_future(one(i)) for i in range(len(run.answers))]
            try:
                return list(await asyncio.gather(*tasks))
            finally:
                # When one judgment fails, gather leaves the others running --
                # stop them before the caller closes the client under them.
                # A no-op once every task has finished.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        table: ScoreTable = {}
        for run in runs:
            table[run.label] = await judge(run)
        missing = sum(score is None for scores in table.values() for score in scores)
        logger.info(
            "Judged %d answers with %s (%d judge calls, %d unrated)",
            sum(len(scores) for scores in table.values()),
            self._model,
            self.calls,
            missing,
        )
        return table

    async def check_ratings(self) -> None:
        """Judge a sanity-check question and refuse a judge that rates it incorrectly.

        Reaching the endpoint is not enough. Ratings are read as ragas reads
        them, the smallest digit 0-4 anywhere in the reply, so a reply that
        carries the judge's reasoning (a thinking model served without a
        reasoning parser, e.g. vLLM without ``--reasoning-parser``) reads as
        0 almost every time. That zeroes every arm, so the blend gap vanishes
        and the verdict is a silent PASS. An obviously right and an obviously
        wrong answer are judged on both prompts, through the same retries and
        parser as real answers. Nothing is cached, so every runs fresh checks
        on the judge.

        Raises:
            RuntimeError: If a request fails, or a sanity-check rating is missing
                or wrong.
        """
        for answer, expected in _SANITY_ANSWERS:
            prompts = judge_prompts(_SANITY_QUESTION, answer, _SANITY_REFERENCE)
            for prompt in prompts:
                rating = await self._rate(prompt)
                if rating is None:
                    raise RuntimeError(
                        f"The judge gave no rating for a sanity-check answer after "
                        f"{JUDGE_RETRIES} attempts. A reasoning judge may be "
                        f"running out of --judge-max-tokens before it answers."
                    )
                if rating != expected:
                    raise RuntimeError(
                        f"The judge rated the sanity-check answer {answer!r} "
                        f"(reference {_SANITY_REFERENCE!r}) {rating * 4:g} instead of "
                        f"{expected * 4:g}. Ratings are read as the smallest "
                        f"digit 0-4 in the reply, so a reply that includes the "
                        f"judge's reasoning reads as 0: serve a reasoning judge "
                        f"with its reasoning split from the reply (e.g. vLLM "
                        f"--reasoning-parser), or use a non-reasoning judge."
                    )

    async def _judge(self, key: str, prompts: tuple[str, str]) -> JudgeScore:
        """Rate one answer with both prompts and record the result."""
        try:
            async with self._semaphore:
                ratings = (await self._rate(prompts[0]), await self._rate(prompts[1]))
            result = JudgeScore(score=combine_ratings(*ratings), ratings=ratings)
            if result.score is not None:
                # A missing rating is not cached, so a re-run retries it.
                self._cache[key] = result
                self._append_cache(key, result)
            return result
        finally:
            self._in_flight.pop(key, None)

    async def _rate(self, prompt: str) -> float | None:
        """Send *prompt* until the reply carries a rating, up to the limit.

        Only a reply without a rating is retried. A failed request raises
        (the client has already retried it).
        """
        for attempt in range(1, JUDGE_RETRIES + 1):
            self.calls += 1
            reply = await self._client.complete(prompt)
            rating = parse_rating(reply)
            if rating is not None:
                return rating
            logger.warning("Judge reply %d had no rating: %.80r", attempt, reply)
        return None

    def _key(self, question: str, answer: str, reference: str) -> str:
        """Return the cache key for one judgment."""
        material = json.dumps([JUDGE_METRIC, self._model, question, answer, reference])
        return hashlib.sha256(material.encode()).hexdigest()

    def _load_cache(self) -> dict[str, JudgeScore]:
        """Read earlier judgments from the cache file, if any.

        A line that is not valid JSON is skipped with a warning: a run killed
        mid-append (SIGKILL, out of memory, full disk) leaves a partial last
        line, and failing on it would discard the arm runs that just
        finished. A skipped judgment is a cache miss and is made again. A
        partial last line is also terminated, so the next appended judgment
        starts on a line of its own instead of joining the broken one.
        """
        cache: dict[str, JudgeScore] = {}
        if not self._cache_path or not os.path.isfile(self._cache_path):
            return cache
        line = ""
        with open(self._cache_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping unreadable line %d of %s; it will be re-judged",
                        line_number,
                        self._cache_path,
                    )
                    continue
                first, second = record["ratings"]
                cache[record["key"]] = JudgeScore(
                    score=record["score"], ratings=(first, second)
                )
        if line and not line.endswith("\n"):
            with open(self._cache_path, "a", encoding="utf-8") as f:
                f.write("\n")
        logger.info("Loaded %d cached judgments from %s", len(cache), self._cache_path)
        return cache

    def _append_cache(self, key: str, result: JudgeScore) -> None:
        """Record one judgment in the cache file."""
        if not self._cache_path:
            return
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
        with open(self._cache_path, "a", encoding="utf-8") as f:
            record = {"key": key, "score": result.score, "ratings": result.ratings}
            f.write(json.dumps(record) + "\n")
