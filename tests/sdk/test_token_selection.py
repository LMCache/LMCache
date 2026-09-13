# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for topology-aware token-selection plans."""

# Standard
from collections.abc import Callable
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.sdk.token_selection import (
    CacheGroupGeometry,
    CacheSemanticKind,
    CacheTopologyDescriptor,
    GroupSelection,
    LogicalSpan,
    PlanExpansion,
    PlanValidationCode,
    RetentionAction,
    TokenSelectionPlan,
    expand_plan,
)


def _geometry(
    group_id: str = "main",
    *,
    semantic_kind: CacheSemanticKind = CacheSemanticKind.DENSE_ATTENTION,
    compression_ratio: int = 1,
    siblings: tuple[str, ...] = (),
    window_size_tokens: int | None = None,
) -> CacheGroupGeometry:
    return CacheGroupGeometry(
        group_id=group_id,
        semantic_kind=semantic_kind,
        logical_tokens_per_block=compression_ratio,
        physical_entries_per_block=1,
        compression_ratio=compression_ratio,
        rank_sharding="tp-contiguous",
        page_stride_bytes=256,
        alignment_bytes=128,
        sibling_group_ids=siblings,
        window_size_tokens=window_size_tokens,
    )


def _topology(*groups: CacheGroupGeometry) -> CacheTopologyDescriptor:
    return CacheTopologyDescriptor.create(
        model_architecture="HybridForCausalLM",
        backend_name="paged-backend",
        topology_version="v1",
        groups=tuple(groups) or (_geometry(),),
    )


def _selection(
    geometry: CacheGroupGeometry,
    *,
    span: LogicalSpan | None = None,
    action: RetentionAction = RetentionAction.INVALIDATE,
    valid_until_step: int | None = None,
) -> GroupSelection:
    return GroupSelection(
        group_id=geometry.group_id,
        semantic_kind=geometry.semantic_kind,
        logical_span=span or LogicalSpan(0, 256),
        required_siblings=geometry.sibling_group_ids,
        action=action,
        valid_until_step=valid_until_step,
    )


def _plan(
    topology: CacheTopologyDescriptor,
    *groups: GroupSelection,
    accepted_seq_len: int = 256,
) -> TokenSelectionPlan:
    return TokenSelectionPlan.create(
        request_id="request-1",
        request_generation=3,
        decode_round=7,
        accepted_seq_len=accepted_seq_len,
        source_kv_revision=11,
        topology_fingerprint=topology.fingerprint,
        policy_revision="policy-v2",
        groups=tuple(groups),
    )


def _expand(
    plan: TokenSelectionPlan,
    topology: CacheTopologyDescriptor,
    **overrides: object,
) -> PlanExpansion:
    context: dict[str, object] = {
        "request_id": "request-1",
        "request_generation": 3,
        "decode_round": 7,
        "accepted_seq_len": 256,
        "source_kv_revision": 11,
        "policy_revision": "policy-v2",
        "current_step": 8,
    }
    context.update(overrides)
    return expand_plan(plan, topology, **context)  # type: ignore[arg-type]


def _codes(expansion: PlanExpansion) -> set[PlanValidationCode]:
    return {issue.code for issue in expansion.validation.issues}


@pytest.mark.parametrize(
    ("semantic_kind", "compression_ratio", "expected_entries"),
    [
        (CacheSemanticKind.DENSE_ATTENTION, 1, 256),
        (CacheSemanticKind.COMPRESSED_DENSE, 4, 64),
        (CacheSemanticKind.COMPRESSED_SPARSE, 128, 2),
    ],
)
def test_aligned_logical_span_maps_to_expected_physical_entries(
    semantic_kind: CacheSemanticKind,
    compression_ratio: int,
    expected_entries: int,
) -> None:
    """Dense, C4, and C128 mappings use topology rather than model names."""
    geometry = _geometry(
        semantic_kind=semantic_kind,
        compression_ratio=compression_ratio,
    )
    topology = _topology(geometry)
    expansion = _expand(_plan(topology, _selection(geometry)), topology)

    assert expansion.validation.valid
    assert len(expansion.operations) == 1
    operation = expansion.operations[0]
    assert operation.physical_entry_start == 0
    assert operation.physical_entry_end == expected_entries
    assert operation.physical_entry_count == expected_entries
    assert operation.byte_end == expected_entries * geometry.page_stride_bytes
    assert operation.full_block_logical_span == LogicalSpan(0, 256)
    assert not operation.requires_residual_handling


def test_partial_compression_maps_only_full_blocks_and_exposes_residuals() -> None:
    """Partial compressed blocks are never silently widened for invalidation."""
    geometry = _geometry(
        semantic_kind=CacheSemanticKind.COMPRESSED_DENSE,
        compression_ratio=4,
    )
    topology = _topology(geometry)
    selection = _selection(geometry, span=LogicalSpan(3, 10))
    plan = _plan(topology, selection, accepted_seq_len=10)
    expansion = _expand(plan, topology, accepted_seq_len=10)

    assert expansion.validation.valid
    operation = expansion.operations[0]
    assert operation.full_block_logical_span == LogicalSpan(4, 8)
    assert operation.physical_entry_start == 1
    assert operation.physical_entry_end == 2
    assert operation.residual_spans == (LogicalSpan(3, 4), LogicalSpan(8, 10))
    assert operation.requires_residual_handling


def test_span_without_a_full_compressed_block_is_residual_only() -> None:
    """A sub-block edit produces no destructive physical range."""
    geometry = _geometry(
        semantic_kind=CacheSemanticKind.COMPRESSED_DENSE,
        compression_ratio=128,
    )
    topology = _topology(geometry)
    selection = _selection(geometry, span=LogicalSpan(3, 10))
    expansion = _expand(
        _plan(topology, selection, accepted_seq_len=10),
        topology,
        accepted_seq_len=10,
    )

    operation = expansion.operations[0]
    assert operation.full_block_logical_span is None
    assert operation.physical_entry_count == 0
    assert operation.byte_start == operation.byte_end == 256
    assert operation.residual_spans == (LogicalSpan(3, 10),)


def test_sliding_window_enforces_strict_active_boundary() -> None:
    """Selections cannot claim tokens already outside the active window."""
    geometry = _geometry(
        semantic_kind=CacheSemanticKind.SLIDING_WINDOW,
        window_size_tokens=128,
    )
    topology = _topology(geometry)

    valid = _expand(
        _plan(topology, _selection(geometry, span=LogicalSpan(128, 256))),
        topology,
    )
    invalid = _expand(
        _plan(topology, _selection(geometry, span=LogicalSpan(127, 256))),
        topology,
    )

    assert valid.validation.valid
    assert _codes(invalid) == {PlanValidationCode.SLIDING_WINDOW_BOUNDARY}
    assert not invalid.operations


def test_complete_hybrid_family_expands_atomically() -> None:
    """All members of a hybrid family expand together."""
    main = _geometry(
        "compressed-main",
        semantic_kind=CacheSemanticKind.COMPRESSED_DENSE,
        compression_ratio=4,
        siblings=("indexer",),
    )
    indexer = _geometry(
        "indexer",
        semantic_kind=CacheSemanticKind.INDEXER,
        siblings=("compressed-main",),
    )
    topology = _topology(indexer, main)
    plan = _plan(topology, _selection(main), _selection(indexer))

    expansion = _expand(plan, topology)

    assert expansion.validation.valid
    assert [operation.group_id for operation in expansion.operations] == [
        "compressed-main",
        "indexer",
    ]
    assert [operation.physical_entry_count for operation in expansion.operations] == [
        64,
        256,
    ]


def test_missing_hybrid_sibling_fails_closed_without_operations() -> None:
    """A partial family hit becomes recompute rather than partial reuse."""
    main = _geometry(
        "compressed-main",
        semantic_kind=CacheSemanticKind.COMPRESSED_DENSE,
        compression_ratio=4,
        siblings=("indexer",),
    )
    indexer = _geometry(
        "indexer",
        semantic_kind=CacheSemanticKind.INDEXER,
        siblings=("compressed-main",),
    )
    topology = _topology(main, indexer)

    expansion = _expand(_plan(topology, _selection(main)), topology)

    assert PlanValidationCode.MISSING_FAMILY_MEMBER in _codes(expansion)
    assert not expansion.operations


@pytest.mark.parametrize("difference", ["action", "span"])
def test_hybrid_family_rejects_non_atomic_decisions(difference: str) -> None:
    """Family members must share one action, logical span, and validity."""
    main = _geometry("main", siblings=("indexer",))
    indexer = _geometry(
        "indexer",
        semantic_kind=CacheSemanticKind.INDEXER,
        siblings=("main",),
    )
    topology = _topology(main, indexer)
    indexer_selection = _selection(indexer)
    if difference == "action":
        indexer_selection = replace(indexer_selection, action=RetentionAction.KEEP)
        expected = PlanValidationCode.INCONSISTENT_FAMILY_ACTION
    else:
        indexer_selection = replace(
            indexer_selection,
            logical_span=LogicalSpan(1, 256),
        )
        expected = PlanValidationCode.INCONSISTENT_FAMILY_SPAN

    expansion = _expand(
        _plan(topology, _selection(main), indexer_selection),
        topology,
    )

    assert expected in _codes(expansion)
    assert not expansion.operations


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("request_id", "other", PlanValidationCode.REQUEST_ID_MISMATCH),
        (
            "request_generation",
            4,
            PlanValidationCode.REQUEST_GENERATION_MISMATCH,
        ),
        ("decode_round", 8, PlanValidationCode.DECODE_ROUND_MISMATCH),
        ("accepted_seq_len", 255, PlanValidationCode.ACCEPTED_SEQ_LEN_MISMATCH),
        (
            "source_kv_revision",
            12,
            PlanValidationCode.SOURCE_KV_REVISION_MISMATCH,
        ),
        (
            "policy_revision",
            "policy-v3",
            PlanValidationCode.POLICY_REVISION_MISMATCH,
        ),
    ],
)
def test_live_request_revision_mismatches_fail_closed(
    field: str,
    value: object,
    expected: PlanValidationCode,
) -> None:
    """Every live request fence is checked before physical expansion."""
    topology = _topology(_geometry())
    plan = _plan(topology, _selection(topology.groups[0]))

    expansion = _expand(plan, topology, **{field: value})

    assert expected in _codes(expansion)
    assert not expansion.operations


def test_topology_fingerprint_mismatch_fails_closed() -> None:
    """A plan cannot be expanded under different cache geometry."""
    original = _topology(_geometry())
    changed = _topology(
        _geometry(
            semantic_kind=CacheSemanticKind.COMPRESSED_DENSE,
            compression_ratio=4,
        )
    )
    plan = _plan(original, _selection(original.groups[0]))

    expansion = _expand(plan, changed)

    assert PlanValidationCode.TOPOLOGY_MISMATCH in _codes(expansion)
    assert PlanValidationCode.SEMANTIC_KIND_MISMATCH in _codes(expansion)
    assert not expansion.operations


def test_unknown_and_duplicate_groups_fail_closed() -> None:
    """Unknown group ids and ambiguous duplicate selections never expand."""
    geometry = _geometry()
    topology = _topology(geometry)
    unknown = GroupSelection(
        group_id="unknown",
        semantic_kind=CacheSemanticKind.DENSE_ATTENTION,
        logical_span=LogicalSpan(0, 256),
        required_siblings=(),
        action=RetentionAction.INVALIDATE,
    )

    unknown_expansion = _expand(_plan(topology, unknown), topology)
    duplicate_expansion = _expand(
        _plan(topology, _selection(geometry), _selection(geometry)),
        topology,
    )

    assert _codes(unknown_expansion) == {PlanValidationCode.UNKNOWN_GROUP}
    assert _codes(duplicate_expansion) == {PlanValidationCode.DUPLICATE_GROUP}
    assert unknown_expansion.validation.recompute_required
    assert duplicate_expansion.validation.recompute_required
    assert not unknown_expansion.operations
    assert not duplicate_expansion.operations


def test_span_past_accepted_tokens_fails_closed() -> None:
    """Speculative or rejected decode tokens cannot enter physical edits."""
    topology = _topology(_geometry())
    plan = _plan(
        topology,
        _selection(topology.groups[0], span=LogicalSpan(0, 257)),
    )

    expansion = _expand(plan, topology)

    assert _codes(expansion) == {PlanValidationCode.SPAN_AFTER_ACCEPTED_SEQUENCE}
    assert not expansion.operations


def test_expired_or_explicit_recompute_plan_never_expands() -> None:
    """Expired decisions and explicit recompute actions are fail-closed."""
    topology = _topology(_geometry())
    selection = _selection(
        topology.groups[0],
        action=RetentionAction.RECOMPUTE_REQUIRED,
        valid_until_step=7,
    )

    expansion = _expand(_plan(topology, selection), topology, current_step=8)

    assert _codes(expansion) == {
        PlanValidationCode.EXPIRED_SELECTION,
        PlanValidationCode.EXPLICIT_RECOMPUTE,
    }
    assert not expansion.operations


def test_negative_live_step_is_rejected_as_caller_error() -> None:
    """A nonsensical live step does not participate in plan validation."""
    topology = _topology(_geometry())
    plan = _plan(topology, _selection(topology.groups[0]))

    with pytest.raises(ValueError, match="current_step"):
        _expand(plan, topology, current_step=-1)


def test_selection_siblings_must_match_topology() -> None:
    """The algorithm cannot silently omit topology-declared family members."""
    main = _geometry("main", siblings=("indexer",))
    indexer = _geometry(
        "indexer",
        semantic_kind=CacheSemanticKind.INDEXER,
        siblings=("main",),
    )
    topology = _topology(main, indexer)
    wrong = replace(_selection(main), required_siblings=())

    expansion = _expand(_plan(topology, wrong, _selection(indexer)), topology)

    assert PlanValidationCode.SIBLING_DECLARATION_MISMATCH in _codes(expansion)
    assert not expansion.operations


def test_topology_requires_symmetric_existing_siblings() -> None:
    """Malformed family graphs are rejected when the descriptor is built."""
    missing = _geometry("main", siblings=("missing",))
    with pytest.raises(ValueError, match="unknown sibling"):
        _topology(missing)

    main = _geometry("main", siblings=("indexer",))
    indexer = _geometry("indexer", semantic_kind=CacheSemanticKind.INDEXER)
    with pytest.raises(ValueError, match="must be symmetric"):
        _topology(main, indexer)


def test_topology_rejects_overlapping_incomplete_families() -> None:
    """Pairwise-symmetric edges must still form one complete family clique."""
    first = _geometry("first", siblings=("second",))
    second = _geometry("second", siblings=("first", "third"))
    third = _geometry("third", siblings=("second",))

    with pytest.raises(ValueError, match="complete, non-overlapping"):
        _topology(first, second, third)


def test_descriptor_and_plan_digests_are_order_independent() -> None:
    """Equivalent tuple order produces stable topology and plan identities."""
    first = _geometry("first", siblings=("second",))
    second = _geometry("second", siblings=("first",))
    forward = _topology(first, second)
    reverse = _topology(second, first)

    assert forward.fingerprint == reverse.fingerprint
    forward_plan = _plan(forward, _selection(first), _selection(second))
    reverse_plan = _plan(reverse, _selection(second), _selection(first))
    assert forward_plan.plan_digest == reverse_plan.plan_digest


def test_duplicate_group_digest_is_stable_before_fail_closed_validation() -> None:
    """Malformed duplicate selections still have one canonical wire identity."""
    geometry = _geometry()
    topology = _topology(geometry)
    keep = _selection(geometry, action=RetentionAction.KEEP)
    invalidate = _selection(geometry, action=RetentionAction.INVALIDATE)

    forward = _plan(topology, keep, invalidate)
    reverse = _plan(topology, invalidate, keep)

    assert forward.plan_digest == reverse.plan_digest
    assert _codes(_expand(forward, topology)) == {PlanValidationCode.DUPLICATE_GROUP}


def test_tampered_digest_is_rejected_before_expansion() -> None:
    """Serialized plan content cannot silently diverge from its identity."""
    topology = _topology(_geometry())
    plan = _plan(topology, _selection(topology.groups[0]))

    with pytest.raises(ValueError, match="plan digest"):
        replace(plan, plan_digest="sha256:" + "0" * 64)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda geometry: replace(
            geometry,
            logical_tokens_per_block=4,
            compression_ratio=2,
        ),
        lambda geometry: replace(
            geometry,
            page_stride_bytes=192,
            alignment_bytes=128,
        ),
        lambda geometry: replace(geometry, window_size_tokens=128),
    ],
)
def test_invalid_geometry_is_rejected(
    mutate: Callable[[CacheGroupGeometry], CacheGroupGeometry],
) -> None:
    """Inconsistent compression, alignment, or window metadata is invalid."""
    with pytest.raises(ValueError):
        mutate(_geometry())
