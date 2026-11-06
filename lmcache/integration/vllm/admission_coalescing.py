# SPDX-License-Identifier: Apache-2.0
"""Worker-side admission coalescing for the MP adapter.

Holds completed KV retrieves for one engine step so that several
prefill tails are admitted on the same step boundary and the per-step
token budget is filled. See docs/design/integration/vllm/admission_coalescing.md.
"""

# Standard
import math
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
# wasted latency.
_DEADLINE_S: float = 3.0
# Fraction of the step budget that counts as a full step.
_FILL_ALPHA: float = 0.75
# Number of ready retrieves to coalesce.
_K: int = 3
# Log a decision summary every N gate calls (0 = never). The summary
# (hold rate, duration EWMA) is what the deadline and alpha were tuned
# from; keep it cheap enough to leave on in production.
_LOG_EVERY: int = 2000
# Duration-estimator bucket boundaries, in log2(tail_tokens). Each
# bucket holds an independent EWMA of retrieve durations, so short and
# long retrieves no longer share one estimate. Buckets are spaced 2
# powers of two apart: ~256, ~1024, ~4096, ~16K, ~64K tokens.
_DURATION_BUCKETS: tuple[int, ...] = (8, 10, 12, 14, 16)
# EWMA decay for per-bucket duration estimates.
_EWMA_ALPHA: float = 0.2


def _bucket_for(tail_tokens: int) -> int:
    """Return the duration-bucket key for a tail length.

    Args:
        tail_tokens: Prefill tail in tokens.

    Returns:
        The nearest bucket boundary from ``_DURATION_BUCKETS`` in
        ``log2(tail_tokens)`` space.
    """
    if tail_tokens <= 0:
        return _DURATION_BUCKETS[0]
    log2 = math.log2(tail_tokens)
    return min(_DURATION_BUCKETS, key=lambda b: abs(log2 - b))


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
        # Per-bucket EWMA of retrieve durations (submission to first
        # hold, or to report if never held); None until the bucket's
        # first observation. A missing bucket estimate falls back to the
        # global EWMA, which itself falls back to release (no hold).
        self._bucket_ewma: dict[int, float | None] = {
            b: None for b in _DURATION_BUCKETS
        }
        # Global (all-length) EWMA, used as a fallback for buckets that
        # have no observations yet.
        self._duration_ewma: float | None = None
        # Periodic decision summary (every _LOG_EVERY calls).
        self._log_calls = 0
        self._log_holds = 0
        # request_id -> timestamp of the first hold decision, used to
        # subtract hold time from duration samples (the EWMA feeds the
        # sibling ETA, so uncorrected samples would conflate the gate's
        # own waiting with the pipeline latency).
        self._first_hold: dict[str, float] = {}

    def on_submit(self, request_id: str, tail_tokens: int) -> None:
        """Record submission time and prefill tail for a retrieve.

        The tail is computed by the adapter from the op itself
        (packed token count minus the cached range) because new metadata
        fields do not survive SchedulerOutput serialization.

        No-op when the gate is disabled: the off path keeps no
        per-request state, so it cannot grow without bound.

        Args:
            request_id: The request identifier.
            tail_tokens: Prefill tail in tokens.
        """
        if not self._enabled:
            return
        self._submit_ts[request_id] = time.monotonic()
        self._tail[request_id] = tail_tokens

    def forget(self, request_ids: set[str]) -> None:
        """Drop per-request state without reporting.

        Used for requests that leave ``retrieve_futures`` without going
        through the gate (preempted or cancelled retrieves). Without
        this, a preempted request's entry survives until an unrelated
        ``drain()`` clears everything, and a resumed request would see
        a stale ``_first_hold`` timestamp from its previous life.

        Args:
            request_ids: Request ids whose state should be dropped.
        """
        for request_id in request_ids:
            self._submit_ts.pop(request_id, None)
            self._tail.pop(request_id, None)
            self._first_hold.pop(request_id, None)

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

        Args:
            ready: Request ids whose retrieve completed this call.
            pending: All in-flight retrieve ids (superset of *ready*).

        Returns:
            The subset of *ready* to report this engine step; an empty
            set means hold everything for one more step.
        """
        if not self._enabled:
            return ready
        reported = self._hold(ready, pending)
        now = time.monotonic()
        if len(ready) > len(reported):
            self._log_holds += len(ready) - len(reported)
        if _LOG_EVERY and self._enabled:
            self._log_calls += 1
            if self._log_calls % _LOG_EVERY == 0:
                logger.debug(
                    "admit_coalesce: calls=%d holds=%d ewma_ms=%s deadline_ms=%d",
                    self._log_calls,
                    self._log_holds,
                    f"{self._duration_ewma * 1000:.0f}"
                    if self._duration_ewma is not None
                    else "n/a",
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

    def _estimate_duration(self, tail_tokens: int) -> float | None:
        """Estimate retrieve duration for a tail of the given length.

        Uses the per-bucket EWMA when the bucket has observations;
        falls back to the global EWMA; returns None if neither exists
        (the caller releases rather than holding blind).

        Args:
            tail_tokens: Prefill tail in tokens.

        Returns:
            Estimated retrieve duration in seconds, or None.
        """
        bucket = _bucket_for(tail_tokens)
        est = self._bucket_ewma.get(bucket)
        if est is not None:
            return est
        return self._duration_ewma

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
        # retrieve duration for ITS tail length minus how long it has
        # already waited. Without a duration estimate for that length,
        # release instead of holding.
        eta_min = None
        for r in inflight:
            est = self._estimate_duration(tails.get(r, 0))
            if est is None:
                continue
            waited = now - self._submit_ts.get(r, now)
            eta = max(0.0, est - waited)
            if eta_min is None or eta < eta_min:
                eta_min = eta
        if eta_min is None or eta_min > deadline_left:
            return ready

        return set()

    def _note_durations(self, reported: set[str]) -> None:
        """Update the retrieve-duration EWMA (submission to report).

        Sampling happens at report time. A request that was held across
        steps contributes a duration that includes its hold time; hold
        time is subtracted so the EWMA tracks the retrieve pipeline
        rather than our own waiting. Both the per-bucket estimate for
        the request's tail length and the global EWMA are updated.
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
                else (1 - _EWMA_ALPHA) * self._duration_ewma + _EWMA_ALPHA * duration
            )
            bucket = _bucket_for(self._tail.get(request_id, 0))
            prev = self._bucket_ewma.get(bucket)
            self._bucket_ewma[bucket] = (
                duration
                if prev is None
                else (1 - _EWMA_ALPHA) * prev + _EWMA_ALPHA * duration
            )
