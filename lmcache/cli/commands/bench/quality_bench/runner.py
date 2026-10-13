# SPDX-License-Identifier: Apache-2.0
"""Runs one pass of a :class:`QualityPlan` against one arm."""

# Standard
from dataclasses import dataclass, field
import asyncio
import enum

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.scoring import (
    extract_final_answer,
)
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.quality_bench.blend_counters import (
    BlendCounters,
    BlendCounterSource,
)
from lmcache.cli.commands.bench.quality_bench.config import ArmRole
from lmcache.cli.commands.bench.quality_bench.plan import QualityPlan
from lmcache.logging import init_logger

logger = init_logger(__name__)

# The blend server may record a lookup after the HTTP response has returned,
# so the run's opening and closing counter reads wait this long for
# stragglers: the opening read so late store-phase lookups (mostly misses)
# stay out of the measured pass. Per-row deltas are read without waiting and
# can lag by up to one question. Public so tests can shorten the wait.
COUNTER_SETTLE_S = 2.0


class StorePhase(enum.Enum):
    """Whether a pass sends the plan's store requests before measuring."""

    STORE_FIRST = "store_first"
    """Send every store request, then measure. For arms that cache."""

    SKIP_STORE = "skip_store"
    """Measure immediately. For the recompute reference, where storing
    cannot change the answers and only costs time."""


@dataclass(frozen=True)
class ArmAnswer:
    """One measured question's outcome on one arm.

    Attributes:
        sample_id: The dataset's id for the question.
        request_id: The measured request's id (same on every arm).
        successful: Whether the request completed.
        response: The full response text, reasoning included.
        answer: The extracted ``<final_answer>``. ``""`` when absent or cut
            off.
        ttft: Time to first token in seconds. On failure: ``-1.0``.
        num_output_tokens: Tokens generated.
        counters: Blend-server counters accumulated while this question ran.
            All zero when engagement is not measured. Approximate: a lookup
            recorded after its response lands in the next question's row.
    """

    sample_id: str
    request_id: str
    successful: bool
    response: str
    answer: str
    ttft: float
    num_output_tokens: int
    counters: BlendCounters = field(default_factory=BlendCounters)

    @property
    def parsed(self) -> bool:
        """Whether a complete answer was produced."""
        return self.successful and bool(self.answer)


@dataclass(frozen=True)
class ArmRun:
    """All measured answers from one pass over the plan on one arm.

    Attributes:
        role: The arm's role.
        run_index: ``0`` for the first pass on this arm, ``1`` for the
            second (the reference runs more than once).
        answers: One per plan sample, in plan order.
        counters: Blend-server counters accumulated over the measured pass,
            each end read after a settle delay. All zero when not measured.
        engagement_measured: Whether ``counters`` are real readings.
        store_failures: Store requests that did not complete.
    """

    role: ArmRole
    run_index: int
    answers: list[ArmAnswer]
    counters: BlendCounters
    engagement_measured: bool
    store_failures: int

    @property
    def label(self) -> str:
        """File-name-safe label, e.g. ``reference_1`` or ``blend``."""
        if self.role is ArmRole.REFERENCE:
            return f"{self.role.value}_{self.run_index + 1}"
        return self.role.value


class ArmRunner:
    """Sends a plan to one arm, sequentially and greedily.

    Requests are sent one at a time so per-question counter deltas are
    attributable and no request's cache state depends on a concurrent one.
    Decoding is greedy (the sender always sends ``temperature=0``).
    """

    def __init__(
        self,
        request_sender: RequestSender,
        max_output_tokens: int,
        counter_source: BlendCounterSource,
    ) -> None:
        """Bind the runner to an arm.

        Args:
            request_sender: Sender pointed at the arm's URL. The runner
                registers a completion callback on it.
            max_output_tokens: Token budget per measured answer.
            counter_source: Engagement counters for this arm. Pass an
                unmeasured source for arms without a blend server.
        """
        self._sender = request_sender
        self._max_output_tokens = max_output_tokens
        self._counters = counter_source
        self._responses: dict[str, str] = {}
        self._sender.add_on_finished_callback(self._record_response)

    async def run(
        self,
        plan: QualityPlan,
        role: ArmRole,
        run_index: int,
        store_phase: StorePhase,
    ) -> ArmRun:
        """Run one pass: optionally store, then measure every question.

        Args:
            plan: The prompts to send.
            role: The arm's role, recorded in the result.
            run_index: Which pass this is on the arm, from ``0``.
            store_phase: Whether to send the store requests first.

        Returns:
            The pass's answers and counters.

        Raises:
            RuntimeError: If engagement counters are configured but cannot be
                read.
        """
        store_failures = 0
        if store_phase is StorePhase.STORE_FIRST:
            for store in plan.store_requests:
                result = await self._sender.send_warmup_request(
                    store.request_id, store.messages
                )
                if not result.successful:
                    store_failures += 1
                    logger.warning(
                        "Store request %s failed: %s", store.request_id, result.error
                    )
            logger.info(
                "Stored %d prompts (%d failed)",
                len(plan.store_requests),
                store_failures,
            )
            # Store responses are never scored.
            self._responses.clear()

        start = await self._settled_read_counters()
        before = start
        answers: list[ArmAnswer] = []
        for index, sample_plan in enumerate(plan.samples):
            result = await self._sender.send_request(
                sample_plan.request_id,
                sample_plan.messages,
                max_tokens=self._max_output_tokens,
            )
            after = await self._read_counters()
            answers.append(
                self._answer(sample_plan.sample.sample_id, result, after - before)
            )
            before = after
            logger.info(
                "%s pass %d: %d/%d answered",
                role.value,
                run_index + 1,
                index + 1,
                len(plan.samples),
            )

        end = await self._settled_read_counters()
        return ArmRun(
            role=role,
            run_index=run_index,
            answers=answers,
            counters=end - start,
            engagement_measured=self._counters.measures_engagement,
            store_failures=store_failures,
        )

    def _answer(
        self, sample_id: str, result: RequestResult, counters: BlendCounters
    ) -> ArmAnswer:
        """Build one question's outcome from its request result."""
        response = self._responses.pop(result.request_id, "")
        return ArmAnswer(
            sample_id=sample_id,
            request_id=result.request_id,
            successful=result.successful,
            response=response,
            answer=extract_final_answer(response) if result.successful else "",
            ttft=result.ttft,
            num_output_tokens=result.num_output_tokens,
            counters=counters,
        )

    async def _read_counters(self) -> BlendCounters:
        """Read the counter source off the event loop."""
        return await asyncio.to_thread(self._counters.read)

    async def _settled_read_counters(self) -> BlendCounters:
        """Read the counters once lookups still being recorded have landed."""
        if self._counters.measures_engagement:
            await asyncio.sleep(COUNTER_SETTLE_S)
        return await self._read_counters()

    def _record_response(self, result: RequestResult, text: str) -> None:
        """Hold a response's text until the step that sent it collects it."""
        self._responses[result.request_id] = text
