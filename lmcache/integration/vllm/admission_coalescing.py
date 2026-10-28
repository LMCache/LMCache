# SPDX-License-Identifier: Apache-2.0
"""Worker-side admission coalescing for the MP adapter.

Holds completed KV retrieves for one engine step so that several
prefill tails are admitted on the same step boundary and the per-step
token budget is filled. See docs/design/v1/multiprocess/admission_coalescing.md.
"""

# Standard
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Fallback step budget in tokens; the connector supplies vLLM's real
# max_num_batched_tokens, so this only applies to direct construction.
DEFAULT_BUDGET_TOKENS: int = 8192

# Hold deadline in seconds, measured from retrieve submission. This is
# deliberately generous: it must comfortably exceed the typical sibling
# retrieve duration (d), otherwise a sibling submitted shortly after the
# ready request can never arrive inside the window and every hold is
# wasted latency. On our benchmark d converges to ~260 ms and the
# measured hold durations stay well under 1 s, so 3 s gives the ETA
# condition room to admit real siblings. An adaptive variant
# (factor * d EWMA) was tested and rejected: it collapsed the window to
# ~0.6 s on fast deployments and the coalescing never fired.
_DEADLINE_S: float = 3.0
# Fraction of the step budget that counts as a full step.
_FILL_ALPHA: float = 0.75
# Number of ready retrieves to coalesce.
_K: int = 3
# Enable the sibling-ETA release condition.
_SIBLING_ETA: bool = True
# Log a decision summary every N gate calls (0 = never). The summary
# (hold rate, duration EWMA) is what the deadline and alpha were tuned
# from; keep it cheap enough to leave on in production.
_LOG_EVERY: int = 2000


class AdmissionCoalescer:
    """Per-step gate deciding which completed retrieves to report.

    Knows nothing about futures or the MP adapter: each engine step the
    caller passes the ready and pending request-id sets. Per-request
    state (submission timestamp, prefill tail) lives here and is dropped
    when the request is reported or drained.
    """

    def __init__(self, enabled: bool, budget_tokens: int) -> None:
        self._enabled = enabled
        self._budget = budget_tokens
        # request_id -> retrieve submission timestamp.
        self._submit_ts: dict[str, float] = {}
        # request_id -> prefill tail in tokens (prompt length minus cache hit).
        self._tail: dict[str, int] = {}
        # EWMA of retrieve durations (submission to report); None until
        # the first observation.
        self._duration_ewma: float | None = None
        # Periodic decision summary (every _LOG_EVERY calls).
        self._log_calls = 0
        self._log_holds = 0
        # request_id -> timestamp of the first hold decision, used to
        # subtract hold time from duration samples (the EWMA feeds the
        # adaptive deadline, so uncorrected samples would form a
        # self-amplifying loop: longer holds -> larger EWMA -> longer
        # deadline -> longer holds).
        self._first_hold: dict[str, float] = {}

    def on_submit(self, request_id: str, tail_tokens: int) -> None:
        """Record submission time and prefill tail for a retrieve.

        The tail is computed by the adapter from the op itself
        (packed token count minus the cached range) because new metadata
        fields do not survive SchedulerOutput serialization.
        """
        self._submit_ts[request_id] = time.monotonic()
        self._tail[request_id] = tail_tokens

    def drain(self) -> None:
        """Drop all per-request state (unhealthy path reports everything)."""
        self._submit_ts.clear()
        self._tail.clear()
        self._first_hold.clear()

    def select_to_report(self, ready: set[str], pending: set[str]) -> set[str]:
        """Return the subset of *ready* to report this engine step.

        Held ids keep their state and are re-evaluated on the next call,
        so a completed retrieve is reported exactly once. Reported
        requests update the duration EWMA and drop their state.
        """
        reported = self._hold(ready, pending)
        now = time.monotonic()
        if len(ready) > len(reported):
            self._log_holds += len(ready) - len(reported)
        if _LOG_EVERY and self._enabled:
            self._log_calls += 1
            if self._log_calls % _LOG_EVERY == 0:
                logger.info(
                    "admit_coalesce: calls=%d holds=%d ewma_ms=%s "
                    "deadline_ms=%d",
                    self._log_calls,
                    self._log_holds,
                    f"{self._duration_ewma * 1000:.0f}"
                    if self._duration_ewma is not None else "n/a",
                    int(_DEADLINE_S * 1000),
                )
        for request_id in ready - reported:
            self._first_hold.setdefault(request_id, now)
        self._note_durations(reported)
        for request_id in reported:
            self._submit_ts.pop(request_id, None)
            self._tail.pop(request_id, None)
            self._first_hold.pop(request_id, None)
        return reported

    def _hold(self, ready: set[str], pending: set[str]) -> set[str]:
        """Hold completed retrieves for one step so that several prefill
        tails are admitted on the same step boundary.

        Release is implicit when any condition stops holding: K or more
        ready, no sibling retrieve in flight, a ready tail alone fills a
        step, the deadline (from submit) expired, or the earliest
        in-flight sibling is estimated to arrive after it.
        """
        if not self._enabled or not pending:
            return ready
        if not 0 < len(ready) < _K:
            return ready
        inflight = pending - ready
        if not inflight:
            return ready

        # A ready request whose tail alone fills a step gains nothing
        # from waiting: a sibling cannot join an already-full step. And
        # if even all ready + in-flight tails together cannot fill a
        # step, waiting cannot produce a full step either (K counts
        # requests, not tokens -- K small tails still batch to a thin
        # step).
        fill_floor = _FILL_ALPHA * self._budget
        tails = self._tail
        if any(tails.get(r, 0) >= fill_floor for r in ready):
            return ready
        cohort = ready | inflight
        if sum(tails.get(r, 0) for r in cohort) < fill_floor:
            return ready

        now = time.monotonic()
        oldest_submit = min(self._submit_ts.get(r, 0.0) for r in ready)
        deadline_left = _DEADLINE_S - (now - oldest_submit)
        if deadline_left <= 0:
            return ready

        # Hold only while the earliest in-flight sibling is estimated to
        # finish before the deadline: its remaining time is the typical
        # retrieve duration minus how long it has already waited.
        # Without duration observations, release instead of holding.
        if _SIBLING_ETA:
            if self._duration_ewma is None:
                return ready
            eta_min = min(
                max(0.0, self._duration_ewma - (now - self._submit_ts.get(r, now)))
                for r in inflight
            )
            if eta_min > deadline_left:
                return ready

        return set()

    def _note_durations(self, reported: set[str]) -> None:
        """Update the retrieve-duration EWMA (submission to report).

        Sampling happens at report time. A request that was held across
        steps contributes a duration that includes its hold time; hold
        time is subtracted so the EWMA tracks the retrieve pipeline
        rather than our own waiting.
        """
        now = time.monotonic()
        for request_id in reported:
            t0 = self._submit_ts.get(request_id)
            if t0 is None:
                continue
            duration = now - t0
            first_hold = self._first_hold.get(request_id)
            if first_hold is not None:
                duration = max(0.0, first_hold - t0)
            self._duration_ewma = (
                duration
                if self._duration_ewma is None
                else 0.8 * self._duration_ewma + 0.2 * duration
            )
