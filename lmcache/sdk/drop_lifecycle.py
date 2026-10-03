# SPDX-License-Identifier: Apache-2.0
"""Generation-fenced lifecycle primitives for decode-time cache edits."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, replace
from enum import Enum
import threading


class DropLifecycleError(RuntimeError):
    """Raised when a local caller attempts an invalid lifecycle transition."""


class DropOperationState(str, Enum):
    """State of one decode-time token-selection and cache-edit operation."""

    Q_CAPTURED = "q_captured"
    PLAN_COMPUTING = "plan_computing"
    PLAN_READY = "plan_ready"
    PLAN_VALIDATED = "plan_validated"
    DEACTIVATION_SUBMITTED = "deactivation_submitted"
    REMOTE_COMPLETE = "remote_complete"
    LOCAL_VISIBLE = "local_visible"
    PLAN_CONSUMED = "plan_consumed"
    ABORTED = "aborted"
    STALE = "stale"
    FAILED = "failed"
    RECOMPUTE_REQUIRED = "recompute_required"


class DropCompletionDisposition(str, Enum):
    """Outcome of a lifecycle event without exposing mutable tracker state."""

    RECORDED = "recorded"
    APPLIED = "applied"
    DUPLICATE_NOOP = "duplicate_noop"
    STALE_NOOP = "stale_noop"
    RECOMPUTE_REQUIRED = "recompute_required"


@dataclass(frozen=True)
class DropOperationKey:
    """Identity of one asynchronous decode-time cache edit.

    Args:
        request_id: Serving-engine request identifier.
        request_generation: Monotonic lifecycle generation for that identifier.
        operation_id: Monotonic operation number within a generation.
        drop_round: Decode-time token-dropping round.
        source_kv_revision: KV representation revision used to build the plan.

    Raises:
        ValueError: If the request id is empty or a numeric field is negative.
    """

    request_id: str
    request_generation: int
    operation_id: int
    drop_round: int
    source_kv_revision: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        for name in (
            "request_generation",
            "operation_id",
            "drop_round",
            "source_kv_revision",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class DropOperationCompletion:
    """Authoritative completion metadata returned by an asynchronous backend.

    Args:
        key: Operation identity copied from submission.
        topology_fingerprint: Cache geometry used by the backend operation.
        policy_revision: Token-selection policy revision used by the operation.
        accepted_seq_len: Accepted, committed sequence length for the plan.

    Raises:
        ValueError: If a revision is empty or accepted sequence length is negative.
    """

    key: DropOperationKey
    topology_fingerprint: str
    policy_revision: str
    accepted_seq_len: int

    def __post_init__(self) -> None:
        if not self.topology_fingerprint:
            raise ValueError("topology_fingerprint must not be empty")
        if not self.policy_revision:
            raise ValueError("policy_revision must not be empty")
        if self.accepted_seq_len < 0:
            raise ValueError("accepted_seq_len must be non-negative")


@dataclass(frozen=True)
class DropOperationSnapshot:
    """Immutable public view of an operation's lifecycle state."""

    key: DropOperationKey
    state: DropOperationState
    topology_fingerprint: str
    policy_revision: str
    accepted_seq_len: int
    source_reusable: bool
    effect_applied: bool


@dataclass(frozen=True)
class DropLifecycleMetrics:
    """Monotonic counters describing lifecycle and fail-closed outcomes."""

    operations_started: int = 0
    remote_completions: int = 0
    effects_applied: int = 0
    duplicate_events: int = 0
    stale_events: int = 0
    recompute_required: int = 0
    aborted_operations: int = 0
    invalidated_operations: int = 0
    failed_operations: int = 0
    tombstones_pruned: int = 0


@dataclass
class _DropOperationRecord:
    key: DropOperationKey
    topology_fingerprint: str
    policy_revision: str
    accepted_seq_len: int
    state: DropOperationState = DropOperationState.Q_CAPTURED
    source_reusable: bool = False
    effect_applied: bool = False

    def snapshot(self) -> DropOperationSnapshot:
        return DropOperationSnapshot(
            key=self.key,
            state=self.state,
            topology_fingerprint=self.topology_fingerprint,
            policy_revision=self.policy_revision,
            accepted_seq_len=self.accepted_seq_len,
            source_reusable=self.source_reusable,
            effect_applied=self.effect_applied,
        )


_FORWARD_TRANSITIONS = {
    DropOperationState.Q_CAPTURED: DropOperationState.PLAN_COMPUTING,
    DropOperationState.PLAN_COMPUTING: DropOperationState.PLAN_READY,
    DropOperationState.PLAN_READY: DropOperationState.PLAN_VALIDATED,
    DropOperationState.PLAN_VALIDATED: DropOperationState.DEACTIVATION_SUBMITTED,
}

_TERMINAL_STATES = {
    DropOperationState.PLAN_CONSUMED,
    DropOperationState.ABORTED,
    DropOperationState.STALE,
    DropOperationState.FAILED,
    DropOperationState.RECOMPUTE_REQUIRED,
}


class GenerationFencedDropLifecycle:
    """Track asynchronous decode-time cache edits for one reusable request id.

    The tracker keeps a bounded set of old operation records as diagnostic
    tombstones. A delayed completion from an older request generation is a
    side-effect-free stale event even after its tombstone is pruned, while a
    duplicate completion in the live generation is an idempotent no-op.
    Topology, policy, source-KV, and accepted-sequence mismatches fail closed to
    recomputation.

    Args:
        request_id: Serving-engine request identifier.
        request_generation: Initial monotonic lifecycle generation.
        topology_fingerprint: Cache topology expected by new operations.
        policy_revision: Token-selection policy expected by new operations.
        max_tombstones: Maximum old-generation records retained for diagnostics.

    Raises:
        ValueError: If identifiers are empty or the generation is negative.

    Notes:
        ``source_reusable`` becomes true only after an authoritative remote
        completion. ``effect_applied`` becomes true only after local visibility,
        and can become true at most once.
    """

    def __init__(
        self,
        request_id: str,
        request_generation: int,
        topology_fingerprint: str,
        policy_revision: str,
        max_tombstones: int = 1024,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_generation < 0:
            raise ValueError("request_generation must be non-negative")
        if not topology_fingerprint:
            raise ValueError("topology_fingerprint must not be empty")
        if not policy_revision:
            raise ValueError("policy_revision must not be empty")
        if max_tombstones < 0:
            raise ValueError("max_tombstones must be non-negative")
        self._request_id = request_id
        self._request_generation = request_generation
        self._topology_fingerprint = topology_fingerprint
        self._policy_revision = policy_revision
        self._max_tombstones = max_tombstones
        self._next_operation_id = 0
        self._records: dict[DropOperationKey, _DropOperationRecord] = {}
        self._records_by_slot: dict[
            tuple[str, int, int, int], _DropOperationRecord
        ] = {}
        self._metrics = DropLifecycleMetrics()
        self._lock = threading.RLock()

    @property
    def request_id(self) -> str:
        """Return the request identifier owned by this tracker."""
        return self._request_id

    @property
    def request_generation(self) -> int:
        """Return the currently live request generation."""
        with self._lock:
            return self._request_generation

    @property
    def metrics(self) -> DropLifecycleMetrics:
        """Return an immutable snapshot of the lifecycle counters."""
        with self._lock:
            return self._metrics

    def begin_operation(
        self,
        *,
        drop_round: int,
        source_kv_revision: int,
        accepted_seq_len: int,
    ) -> DropOperationKey:
        """Create a Q-captured operation in the live request generation.

        Args:
            drop_round: Decode-time token-dropping round.
            source_kv_revision: KV revision the plan will inspect.
            accepted_seq_len: Committed sequence length the plan may modify.

        Returns:
            A unique immutable operation key for backend submission/completion.

        Raises:
            ValueError: If a supplied numeric value is negative.
        """
        if accepted_seq_len < 0:
            raise ValueError("accepted_seq_len must be non-negative")
        with self._lock:
            key = DropOperationKey(
                request_id=self._request_id,
                request_generation=self._request_generation,
                operation_id=self._next_operation_id,
                drop_round=drop_round,
                source_kv_revision=source_kv_revision,
            )
            self._next_operation_id += 1
            record = _DropOperationRecord(
                key=key,
                topology_fingerprint=self._topology_fingerprint,
                policy_revision=self._policy_revision,
                accepted_seq_len=accepted_seq_len,
            )
            self._records[key] = record
            self._records_by_slot[self._slot(key)] = record
            self._bump_metrics(operations_started=1)
            return key

    def advance(
        self, key: DropOperationKey, state: DropOperationState
    ) -> DropCompletionDisposition:
        """Advance a local operation through the pre-submission state machine.

        Args:
            key: Key returned by ``begin_operation``.
            state: The next expected pre-submission state.

        Returns:
            ``RECORDED``, ``DUPLICATE_NOOP``, or ``STALE_NOOP``.

        Raises:
            DropLifecycleError: If a live operation skips or reverses a phase.
        """
        with self._lock:
            record = self._records.get(key)
            if record is None or key.request_generation != self._request_generation:
                self._bump_metrics(stale_events=1)
                return DropCompletionDisposition.STALE_NOOP
            if record.state == state:
                self._bump_metrics(duplicate_events=1)
                return DropCompletionDisposition.DUPLICATE_NOOP
            expected = _FORWARD_TRANSITIONS.get(record.state)
            if state != expected:
                raise DropLifecycleError(
                    f"invalid drop lifecycle transition {record.state.value} "
                    f"-> {state.value} for {key}"
                )
            record.state = state
            return DropCompletionDisposition.RECORDED

    def record_remote_completion(
        self, completion: DropOperationCompletion
    ) -> DropCompletionDisposition:
        """Validate and record an authoritative asynchronous completion.

        Args:
            completion: Backend completion with all revision fences attached.

        Returns:
            ``RECORDED`` for the first valid completion, an idempotent no-op for
            a duplicate/stale completion, or ``RECOMPUTE_REQUIRED`` on any
            live-generation revision or accepted-length mismatch.
        """
        with self._lock:
            record, disposition = self._resolve_completion(completion)
            if disposition is not None:
                return disposition
            assert record is not None
            if record.state in {
                DropOperationState.REMOTE_COMPLETE,
                DropOperationState.LOCAL_VISIBLE,
                DropOperationState.PLAN_CONSUMED,
            }:
                self._bump_metrics(duplicate_events=1)
                return DropCompletionDisposition.DUPLICATE_NOOP
            if record.state in _TERMINAL_STATES:
                self._bump_metrics(stale_events=1)
                return DropCompletionDisposition.STALE_NOOP
            if record.state != DropOperationState.DEACTIVATION_SUBMITTED:
                return self._require_recompute(record)
            record.state = DropOperationState.REMOTE_COMPLETE
            record.source_reusable = True
            self._bump_metrics(remote_completions=1)
            return DropCompletionDisposition.RECORDED

    def mark_local_visible(
        self, completion: DropOperationCompletion
    ) -> DropCompletionDisposition:
        """Make a remotely completed edit visible exactly once.

        Args:
            completion: The same fenced completion recorded for remote finish.

        Returns:
            ``APPLIED`` exactly once, an idempotent no-op after application, or
            a fail-closed disposition if the operation is stale or mismatched.
        """
        with self._lock:
            record, disposition = self._resolve_completion(completion)
            if disposition is not None:
                return disposition
            assert record is not None
            if record.state in {
                DropOperationState.LOCAL_VISIBLE,
                DropOperationState.PLAN_CONSUMED,
            }:
                self._bump_metrics(duplicate_events=1)
                return DropCompletionDisposition.DUPLICATE_NOOP
            if record.state in _TERMINAL_STATES:
                self._bump_metrics(stale_events=1)
                return DropCompletionDisposition.STALE_NOOP
            if record.state != DropOperationState.REMOTE_COMPLETE:
                return self._require_recompute(record)
            record.state = DropOperationState.LOCAL_VISIBLE
            record.effect_applied = True
            self._bump_metrics(effects_applied=1)
            return DropCompletionDisposition.APPLIED

    def consume(self, key: DropOperationKey) -> DropCompletionDisposition:
        """Mark a locally visible plan as consumed by the decode loop.

        Args:
            key: Key returned by ``begin_operation``.

        Returns:
            ``RECORDED`` once, an idempotent no-op for a consumed plan, or a
            stale/recompute disposition when the plan cannot be consumed.
        """
        with self._lock:
            record = self._records.get(key)
            if record is None or key.request_generation != self._request_generation:
                self._bump_metrics(stale_events=1)
                return DropCompletionDisposition.STALE_NOOP
            if record.state == DropOperationState.PLAN_CONSUMED:
                self._bump_metrics(duplicate_events=1)
                return DropCompletionDisposition.DUPLICATE_NOOP
            if record.state in _TERMINAL_STATES:
                self._bump_metrics(stale_events=1)
                return DropCompletionDisposition.STALE_NOOP
            if record.state != DropOperationState.LOCAL_VISIBLE:
                return self._require_recompute(record)
            record.state = DropOperationState.PLAN_CONSUMED
            return DropCompletionDisposition.RECORDED

    def abort(self) -> int:
        """Abort every non-terminal operation in the live generation.

        Returns:
            Number of operations newly marked aborted. Calling ``abort`` again
            is a no-op and returns zero.
        """
        with self._lock:
            aborted = 0
            for record in self._records.values():
                if record.key.request_generation != self._request_generation:
                    continue
                if record.state in _TERMINAL_STATES:
                    continue
                record.state = DropOperationState.ABORTED
                aborted += 1
            self._bump_metrics(aborted_operations=aborted)
            return aborted

    def fail(self, key: DropOperationKey) -> DropCompletionDisposition:
        """Record a local or backend failure without applying an edit.

        Args:
            key: Key returned by ``begin_operation``.

        Returns:
            ``RECORDED`` once, or an idempotent stale/duplicate no-op.
        """
        with self._lock:
            record = self._records.get(key)
            if record is None or key.request_generation != self._request_generation:
                self._bump_metrics(stale_events=1)
                return DropCompletionDisposition.STALE_NOOP
            if record.state == DropOperationState.FAILED:
                self._bump_metrics(duplicate_events=1)
                return DropCompletionDisposition.DUPLICATE_NOOP
            if record.state in _TERMINAL_STATES:
                self._bump_metrics(stale_events=1)
                return DropCompletionDisposition.STALE_NOOP
            record.state = DropOperationState.FAILED
            self._bump_metrics(failed_operations=1)
            return DropCompletionDisposition.RECORDED

    def advance_generation(
        self,
        request_generation: int,
        *,
        topology_fingerprint: str,
        policy_revision: str,
    ) -> int:
        """Fence old operations and bind the reusable id to a new generation.

        Args:
            request_generation: New generation, strictly greater than current.
            topology_fingerprint: Topology expected by new operations.
            policy_revision: Selection policy expected by new operations.

        Returns:
            Number of old non-terminal operations invalidated as stale.

        Raises:
            ValueError: If the generation does not advance or a revision is empty.
        """
        if not topology_fingerprint:
            raise ValueError("topology_fingerprint must not be empty")
        if not policy_revision:
            raise ValueError("policy_revision must not be empty")
        with self._lock:
            if request_generation <= self._request_generation:
                raise ValueError(
                    "request_generation must be greater than the live generation"
                )
            invalidated = 0
            for record in self._records.values():
                if record.key.request_generation != self._request_generation:
                    continue
                if record.state in _TERMINAL_STATES:
                    continue
                record.state = DropOperationState.STALE
                invalidated += 1
            self._request_generation = request_generation
            self._topology_fingerprint = topology_fingerprint
            self._policy_revision = policy_revision
            self._next_operation_id = 0
            self._bump_metrics(invalidated_operations=invalidated)
            self._prune_tombstones()
            return invalidated

    def snapshot(self, key: DropOperationKey) -> DropOperationSnapshot | None:
        """Return an immutable operation snapshot, or ``None`` if unknown.

        Args:
            key: Operation key to inspect.

        Returns:
            A snapshot of the operation record, or ``None``.
        """
        with self._lock:
            record = self._records.get(key)
            return record.snapshot() if record is not None else None

    def _resolve_completion(
        self, completion: DropOperationCompletion
    ) -> tuple[_DropOperationRecord | None, DropCompletionDisposition | None]:
        key = completion.key
        if (
            key.request_id != self._request_id
            or key.request_generation != self._request_generation
        ):
            record = self._records.get(key)
            if record is not None and record.state not in _TERMINAL_STATES:
                record.state = DropOperationState.STALE
            self._bump_metrics(stale_events=1)
            return None, DropCompletionDisposition.STALE_NOOP

        record = self._records_by_slot.get(self._slot(key))
        if record is None:
            self._bump_metrics(stale_events=1)
            return None, DropCompletionDisposition.STALE_NOOP
        mismatched = key.source_kv_revision != record.key.source_kv_revision or (
            completion.topology_fingerprint != record.topology_fingerprint
            or completion.policy_revision != record.policy_revision
            or completion.accepted_seq_len != record.accepted_seq_len
        )
        if mismatched and record.state == DropOperationState.RECOMPUTE_REQUIRED:
            self._bump_metrics(duplicate_events=1)
            return record, DropCompletionDisposition.RECOMPUTE_REQUIRED
        if mismatched and record.state in _TERMINAL_STATES:
            self._bump_metrics(stale_events=1)
            return record, DropCompletionDisposition.STALE_NOOP
        if mismatched:
            return record, self._require_recompute(record)
        return record, None

    def _require_recompute(
        self, record: _DropOperationRecord
    ) -> DropCompletionDisposition:
        if record.state != DropOperationState.RECOMPUTE_REQUIRED:
            record.state = DropOperationState.RECOMPUTE_REQUIRED
            self._bump_metrics(recompute_required=1)
        else:
            self._bump_metrics(duplicate_events=1)
        return DropCompletionDisposition.RECOMPUTE_REQUIRED

    def _bump_metrics(self, **changes: int) -> None:
        values = {
            name: getattr(self._metrics, name) + amount
            for name, amount in changes.items()
        }
        self._metrics = replace(self._metrics, **values)

    def _prune_tombstones(self) -> None:
        tombstones = sorted(
            (
                record
                for record in self._records.values()
                if record.key.request_generation < self._request_generation
            ),
            key=lambda record: (
                record.key.request_generation,
                record.key.operation_id,
                record.key.drop_round,
            ),
        )
        prune_count = max(0, len(tombstones) - self._max_tombstones)
        for record in tombstones[:prune_count]:
            del self._records[record.key]
            del self._records_by_slot[self._slot(record.key)]
        self._bump_metrics(tombstones_pruned=prune_count)

    @staticmethod
    def _slot(key: DropOperationKey) -> tuple[str, int, int, int]:
        return (
            key.request_id,
            key.request_generation,
            key.operation_id,
            key.drop_round,
        )
