# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for generation-fenced cache-edit lifecycles."""

# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest

# First Party
from lmcache.sdk.drop_lifecycle import (
    DropCompletionDisposition,
    DropLifecycleError,
    DropOperationCompletion,
    DropOperationKey,
    DropOperationSnapshot,
    DropOperationState,
    GenerationFencedDropLifecycle,
)


def _lifecycle() -> GenerationFencedDropLifecycle:
    return GenerationFencedDropLifecycle(
        request_id="request-1",
        request_generation=3,
        topology_fingerprint="topology-v1",
        policy_revision="policy-v2",
    )


def _submitted_operation(
    lifecycle: GenerationFencedDropLifecycle,
    *,
    drop_round: int = 7,
    source_kv_revision: int = 11,
    accepted_seq_len: int = 128,
) -> tuple[DropOperationKey, DropOperationCompletion]:
    key = lifecycle.begin_operation(
        drop_round=drop_round,
        source_kv_revision=source_kv_revision,
        accepted_seq_len=accepted_seq_len,
    )
    for state in (
        DropOperationState.PLAN_COMPUTING,
        DropOperationState.PLAN_READY,
        DropOperationState.PLAN_VALIDATED,
        DropOperationState.DEACTIVATION_SUBMITTED,
    ):
        assert lifecycle.advance(key, state) == DropCompletionDisposition.RECORDED
    completion = DropOperationCompletion(
        key=key,
        topology_fingerprint="topology-v1",
        policy_revision="policy-v2",
        accepted_seq_len=accepted_seq_len,
    )
    return key, completion


def _snapshot(
    lifecycle: GenerationFencedDropLifecycle, key: DropOperationKey
) -> DropOperationSnapshot:
    snapshot = lifecycle.snapshot(key)
    assert snapshot is not None
    return snapshot


def test_happy_path_separates_completion_visibility_and_consumption() -> None:
    """A valid operation exposes source reuse before exactly-once visibility."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)

    assert lifecycle.record_remote_completion(completion) == (
        DropCompletionDisposition.RECORDED
    )
    remote = lifecycle.snapshot(key)
    assert remote is not None
    assert remote.state == DropOperationState.REMOTE_COMPLETE
    assert remote.source_reusable
    assert not remote.effect_applied

    assert lifecycle.mark_local_visible(completion) == DropCompletionDisposition.APPLIED
    visible = lifecycle.snapshot(key)
    assert visible is not None
    assert visible.state == DropOperationState.LOCAL_VISIBLE
    assert visible.source_reusable
    assert visible.effect_applied

    assert lifecycle.consume(key) == DropCompletionDisposition.RECORDED
    assert _snapshot(lifecycle, key).state == DropOperationState.PLAN_CONSUMED
    assert lifecycle.metrics.remote_completions == 1
    assert lifecycle.metrics.effects_applied == 1


def test_duplicate_completion_and_visibility_are_idempotent() -> None:
    """Repeated backend notifications cannot apply one edit twice."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)

    assert lifecycle.record_remote_completion(completion) == (
        DropCompletionDisposition.RECORDED
    )
    assert lifecycle.record_remote_completion(completion) == (
        DropCompletionDisposition.DUPLICATE_NOOP
    )
    assert lifecycle.mark_local_visible(completion) == DropCompletionDisposition.APPLIED
    assert lifecycle.mark_local_visible(completion) == (
        DropCompletionDisposition.DUPLICATE_NOOP
    )
    assert lifecycle.consume(key) == DropCompletionDisposition.RECORDED
    assert lifecycle.consume(key) == DropCompletionDisposition.DUPLICATE_NOOP

    assert lifecycle.metrics.remote_completions == 1
    assert lifecycle.metrics.effects_applied == 1
    assert lifecycle.metrics.duplicate_events == 3


def test_abort_is_idempotent_and_completion_is_side_effect_free() -> None:
    """Repeated aborts are no-ops and a delayed completion stays unapplied."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)
    lifecycle.begin_operation(
        drop_round=8,
        source_kv_revision=12,
        accepted_seq_len=129,
    )

    assert lifecycle.abort() == 2
    assert lifecycle.abort() == 0
    assert lifecycle.record_remote_completion(completion) == (
        DropCompletionDisposition.STALE_NOOP
    )
    snapshot = lifecycle.snapshot(key)
    assert snapshot is not None
    assert snapshot.state == DropOperationState.ABORTED
    assert not snapshot.source_reusable
    assert not snapshot.effect_applied
    assert lifecycle.metrics.aborted_operations == 2


def test_mismatched_completion_after_abort_cannot_rewrite_terminal_state() -> None:
    """Even a malformed delayed completion leaves an aborted operation unchanged."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)
    mismatched = DropOperationCompletion(
        key=completion.key,
        topology_fingerprint="unexpected-topology",
        policy_revision=completion.policy_revision,
        accepted_seq_len=completion.accepted_seq_len,
    )

    assert lifecycle.abort() == 1
    assert lifecycle.record_remote_completion(mismatched) == (
        DropCompletionDisposition.STALE_NOOP
    )
    snapshot = _snapshot(lifecycle, key)
    assert snapshot.state == DropOperationState.ABORTED
    assert not snapshot.source_reusable
    assert not snapshot.effect_applied
    assert lifecycle.metrics.recompute_required == 0


def test_old_generation_completion_does_not_touch_reused_request() -> None:
    """A late completion is fenced after the same request id is reused."""
    lifecycle = _lifecycle()
    old_key, old_completion = _submitted_operation(lifecycle)

    assert (
        lifecycle.advance_generation(
            4,
            topology_fingerprint="topology-v2",
            policy_revision="policy-v3",
        )
        == 1
    )
    new_key = lifecycle.begin_operation(
        drop_round=0,
        source_kv_revision=20,
        accepted_seq_len=1,
    )
    assert new_key.request_generation == 4
    assert new_key.operation_id == 0

    assert lifecycle.record_remote_completion(old_completion) == (
        DropCompletionDisposition.STALE_NOOP
    )
    assert _snapshot(lifecycle, old_key).state == DropOperationState.STALE
    assert _snapshot(lifecycle, new_key).state == DropOperationState.Q_CAPTURED
    assert lifecycle.metrics.invalidated_operations == 1
    assert lifecycle.metrics.stale_events == 1


def test_old_generation_tombstones_are_bounded_without_weakening_fence() -> None:
    """Pruned old records remain fenced by request generation."""
    lifecycle = GenerationFencedDropLifecycle(
        request_id="request-1",
        request_generation=3,
        topology_fingerprint="topology-v1",
        policy_revision="policy-v2",
        max_tombstones=2,
    )
    operations = [_submitted_operation(lifecycle, drop_round=i) for i in range(3)]

    assert (
        lifecycle.advance_generation(
            4,
            topology_fingerprint="topology-v2",
            policy_revision="policy-v3",
        )
        == 3
    )
    oldest_key, oldest_completion = operations[0]
    assert lifecycle.snapshot(oldest_key) is None
    assert lifecycle.snapshot(operations[1][0]) is not None
    assert lifecycle.snapshot(operations[2][0]) is not None
    assert lifecycle.record_remote_completion(oldest_completion) == (
        DropCompletionDisposition.STALE_NOOP
    )
    assert lifecycle.metrics.tombstones_pruned == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("topology_fingerprint", "other-topology"),
        ("policy_revision", "other-policy"),
        ("accepted_seq_len", 127),
    ],
)
def test_live_completion_mismatch_fails_closed(field: str, value: str | int) -> None:
    """Topology, policy, and accepted-length mismatches require recomputation."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)
    mismatched = DropOperationCompletion(
        key=completion.key,
        topology_fingerprint=(
            str(value)
            if field == "topology_fingerprint"
            else completion.topology_fingerprint
        ),
        policy_revision=(
            str(value) if field == "policy_revision" else completion.policy_revision
        ),
        accepted_seq_len=(
            int(value) if field == "accepted_seq_len" else completion.accepted_seq_len
        ),
    )

    assert lifecycle.record_remote_completion(mismatched) == (
        DropCompletionDisposition.RECOMPUTE_REQUIRED
    )
    snapshot = lifecycle.snapshot(key)
    assert snapshot is not None
    assert snapshot.state == DropOperationState.RECOMPUTE_REQUIRED
    assert not snapshot.source_reusable
    assert not snapshot.effect_applied
    assert lifecycle.metrics.recompute_required == 1


def test_source_kv_revision_mismatch_fails_closed() -> None:
    """A completion for the wrong source KV cannot publish the edit."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)
    wrong_key = DropOperationKey(
        request_id=key.request_id,
        request_generation=key.request_generation,
        operation_id=key.operation_id,
        drop_round=key.drop_round,
        source_kv_revision=key.source_kv_revision + 1,
    )
    mismatched = DropOperationCompletion(
        key=wrong_key,
        topology_fingerprint=completion.topology_fingerprint,
        policy_revision=completion.policy_revision,
        accepted_seq_len=completion.accepted_seq_len,
    )

    assert lifecycle.record_remote_completion(mismatched) == (
        DropCompletionDisposition.RECOMPUTE_REQUIRED
    )
    assert _snapshot(lifecycle, key).state == DropOperationState.RECOMPUTE_REQUIRED


def test_mismatch_after_visibility_preserves_audit_facts() -> None:
    """Fail-closed invalidation does not erase an effect that already happened."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)
    assert lifecycle.record_remote_completion(completion) == (
        DropCompletionDisposition.RECORDED
    )
    assert lifecycle.mark_local_visible(completion) == DropCompletionDisposition.APPLIED
    mismatched = DropOperationCompletion(
        key=completion.key,
        topology_fingerprint=completion.topology_fingerprint,
        policy_revision="unexpected-policy",
        accepted_seq_len=completion.accepted_seq_len,
    )

    assert lifecycle.record_remote_completion(mismatched) == (
        DropCompletionDisposition.RECOMPUTE_REQUIRED
    )
    snapshot = lifecycle.snapshot(key)
    assert snapshot is not None
    assert snapshot.state == DropOperationState.RECOMPUTE_REQUIRED
    assert snapshot.source_reusable
    assert snapshot.effect_applied


def test_unknown_and_foreign_completions_are_stale_noops() -> None:
    """Unknown operation ids and another request id never alter live state."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)
    unknown_key = DropOperationKey(
        request_id=key.request_id,
        request_generation=key.request_generation,
        operation_id=999,
        drop_round=key.drop_round,
        source_kv_revision=key.source_kv_revision,
    )
    foreign_key = DropOperationKey(
        request_id="other-request",
        request_generation=key.request_generation,
        operation_id=key.operation_id,
        drop_round=key.drop_round,
        source_kv_revision=key.source_kv_revision,
    )
    for stale_key in (unknown_key, foreign_key):
        stale = DropOperationCompletion(
            key=stale_key,
            topology_fingerprint=completion.topology_fingerprint,
            policy_revision=completion.policy_revision,
            accepted_seq_len=completion.accepted_seq_len,
        )
        assert lifecycle.record_remote_completion(stale) == (
            DropCompletionDisposition.STALE_NOOP
        )

    assert _snapshot(lifecycle, key).state == (
        DropOperationState.DEACTIVATION_SUBMITTED
    )
    assert lifecycle.metrics.stale_events == 2


def test_completion_before_submission_requires_recompute() -> None:
    """An out-of-phase completion is not treated as a successful operation."""
    lifecycle = _lifecycle()
    key = lifecycle.begin_operation(
        drop_round=0,
        source_kv_revision=1,
        accepted_seq_len=32,
    )
    completion = DropOperationCompletion(
        key=key,
        topology_fingerprint="topology-v1",
        policy_revision="policy-v2",
        accepted_seq_len=32,
    )

    assert lifecycle.record_remote_completion(completion) == (
        DropCompletionDisposition.RECOMPUTE_REQUIRED
    )
    assert _snapshot(lifecycle, key).state == DropOperationState.RECOMPUTE_REQUIRED


def test_local_visibility_before_remote_completion_requires_recompute() -> None:
    """Source ownership cannot be released by a local-only notification."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)

    assert lifecycle.mark_local_visible(completion) == (
        DropCompletionDisposition.RECOMPUTE_REQUIRED
    )
    snapshot = lifecycle.snapshot(key)
    assert snapshot is not None
    assert not snapshot.source_reusable
    assert not snapshot.effect_applied


def test_invalid_local_transition_raises_without_mutating_state() -> None:
    """Local phase skips remain visible programming errors."""
    lifecycle = _lifecycle()
    key = lifecycle.begin_operation(
        drop_round=0,
        source_kv_revision=1,
        accepted_seq_len=32,
    )

    with pytest.raises(DropLifecycleError, match="invalid drop lifecycle transition"):
        lifecycle.advance(key, DropOperationState.PLAN_READY)
    assert _snapshot(lifecycle, key).state == DropOperationState.Q_CAPTURED


def test_failure_is_recorded_once() -> None:
    """Failure notification is idempotent and never makes the source reusable."""
    lifecycle = _lifecycle()
    key = lifecycle.begin_operation(
        drop_round=0,
        source_kv_revision=1,
        accepted_seq_len=32,
    )

    assert lifecycle.fail(key) == DropCompletionDisposition.RECORDED
    assert lifecycle.fail(key) == DropCompletionDisposition.DUPLICATE_NOOP
    snapshot = lifecycle.snapshot(key)
    assert snapshot is not None
    assert snapshot.state == DropOperationState.FAILED
    assert not snapshot.source_reusable
    assert lifecycle.metrics.failed_operations == 1


def test_generation_must_advance_monotonically() -> None:
    """Generation rollback cannot re-enable stale operation identities."""
    lifecycle = _lifecycle()

    for generation in (2, 3):
        with pytest.raises(ValueError, match="greater than the live generation"):
            lifecycle.advance_generation(
                generation,
                topology_fingerprint="topology-v1",
                policy_revision="policy-v2",
            )


def test_concurrent_duplicate_completions_apply_exactly_once() -> None:
    """Concurrent duplicate notifications preserve exactly-once visibility."""
    lifecycle = _lifecycle()
    key, completion = _submitted_operation(lifecycle)

    with ThreadPoolExecutor(max_workers=8) as pool:
        remote_results = list(
            pool.map(
                lambda _: lifecycle.record_remote_completion(completion), range(32)
            )
        )
    assert remote_results.count(DropCompletionDisposition.RECORDED) == 1
    assert remote_results.count(DropCompletionDisposition.DUPLICATE_NOOP) == 31

    with ThreadPoolExecutor(max_workers=8) as pool:
        visible_results = list(
            pool.map(lambda _: lifecycle.mark_local_visible(completion), range(32))
        )
    assert visible_results.count(DropCompletionDisposition.APPLIED) == 1
    assert visible_results.count(DropCompletionDisposition.DUPLICATE_NOOP) == 31
    assert _snapshot(lifecycle, key).effect_applied
    assert lifecycle.metrics.remote_completions == 1
    assert lifecycle.metrics.effects_applied == 1


def test_operation_key_rejects_invalid_identity() -> None:
    """Invalid operation identities fail before they can enter a tracker."""
    with pytest.raises(ValueError, match="request_id"):
        DropOperationKey("", 0, 0, 0, 0)
    with pytest.raises(ValueError, match="drop_round"):
        DropOperationKey("request", 0, 0, -1, 0)


def test_snapshot_returns_none_for_unknown_key() -> None:
    """Unknown keys are observable without exposing internal dictionaries."""
    lifecycle = _lifecycle()
    unknown = DropOperationKey("request-1", 3, 99, 0, 0)
    assert lifecycle.snapshot(unknown) is None
