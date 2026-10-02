# SPDX-License-Identifier: Apache-2.0
"""Topology-aware, model-agnostic token-selection plans."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
import hashlib
import json


def _selection_sort_key(
    selection: GroupSelection,
) -> tuple[str, str, int, int, tuple[str, ...], str, int]:
    """Return a total ordering for canonical plan serialization."""
    return (
        selection.group_id,
        selection.semantic_kind.value,
        selection.logical_span.start_token,
        selection.logical_span.end_token,
        selection.required_siblings,
        selection.action.value,
        -1 if selection.valid_until_step is None else selection.valid_until_step,
    )


class CacheSemanticKind(str, Enum):
    """Semantic role of a cache group."""

    DENSE_ATTENTION = "dense_attention"
    SLIDING_WINDOW = "sliding_window"
    COMPRESSED_SPARSE = "compressed_sparse"
    COMPRESSED_DENSE = "compressed_dense"
    INDEXER = "indexer"
    COMPRESSOR_STATE = "compressor_state"
    RECURRENT_STATE = "recurrent_state"
    SPECULATIVE_STATE = "speculative_state"


class RetentionAction(str, Enum):
    """Semantic action requested for a logical token span."""

    KEEP = "keep"
    DEMOTE_TO_L1 = "demote_to_l1"
    DEMOTE_TO_L2 = "demote_to_l2"
    INVALIDATE = "invalidate"
    RECOMPUTE_REQUIRED = "recompute_required"


class PlanValidationCode(str, Enum):
    """Stable reason codes for fail-closed plan validation."""

    REQUEST_ID_MISMATCH = "request_id_mismatch"
    REQUEST_GENERATION_MISMATCH = "request_generation_mismatch"
    DECODE_ROUND_MISMATCH = "decode_round_mismatch"
    ACCEPTED_SEQ_LEN_MISMATCH = "accepted_seq_len_mismatch"
    SOURCE_KV_REVISION_MISMATCH = "source_kv_revision_mismatch"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    POLICY_REVISION_MISMATCH = "policy_revision_mismatch"
    DUPLICATE_GROUP = "duplicate_group"
    UNKNOWN_GROUP = "unknown_group"
    SEMANTIC_KIND_MISMATCH = "semantic_kind_mismatch"
    SPAN_AFTER_ACCEPTED_SEQUENCE = "span_after_accepted_sequence"
    SLIDING_WINDOW_BOUNDARY = "sliding_window_boundary"
    SIBLING_DECLARATION_MISMATCH = "sibling_declaration_mismatch"
    MISSING_FAMILY_MEMBER = "missing_family_member"
    INCONSISTENT_FAMILY_ACTION = "inconsistent_family_action"
    INCONSISTENT_FAMILY_SPAN = "inconsistent_family_span"
    EXPIRED_SELECTION = "expired_selection"
    EXPLICIT_RECOMPUTE = "explicit_recompute"


@dataclass(frozen=True, order=True)
class LogicalSpan:
    """Half-open logical token range ``[start_token, end_token)``."""

    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token <= self.start_token:
            raise ValueError("end_token must be greater than start_token")

    @property
    def token_count(self) -> int:
        """Return the number of logical tokens in the span."""
        return self.end_token - self.start_token


@dataclass(frozen=True)
class CacheGroupGeometry:
    """Logical-to-physical geometry for one independently addressable group."""

    group_id: str
    semantic_kind: CacheSemanticKind
    logical_tokens_per_block: int
    physical_entries_per_block: int
    compression_ratio: int
    rank_sharding: str
    page_stride_bytes: int
    alignment_bytes: int
    sibling_group_ids: tuple[str, ...] = ()
    window_size_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        for name in (
            "logical_tokens_per_block",
            "physical_entries_per_block",
            "compression_ratio",
            "page_stride_bytes",
            "alignment_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.rank_sharding:
            raise ValueError("rank_sharding must not be empty")
        if (
            self.logical_tokens_per_block
            != self.physical_entries_per_block * self.compression_ratio
        ):
            raise ValueError(
                "logical_tokens_per_block must equal "
                "physical_entries_per_block * compression_ratio"
            )
        if self.page_stride_bytes % self.alignment_bytes:
            raise ValueError("page_stride_bytes must be alignment_bytes aligned")
        siblings = tuple(sorted(self.sibling_group_ids))
        if len(siblings) != len(set(siblings)):
            raise ValueError("sibling_group_ids must be unique")
        if self.group_id in siblings:
            raise ValueError("a cache group cannot be its own sibling")
        object.__setattr__(self, "sibling_group_ids", siblings)
        if self.semantic_kind == CacheSemanticKind.SLIDING_WINDOW:
            if self.window_size_tokens is None or self.window_size_tokens <= 0:
                raise ValueError("sliding-window groups require a positive window size")
        elif self.window_size_tokens is not None:
            raise ValueError("only sliding-window groups may define window_size_tokens")


@dataclass(frozen=True)
class CacheTopologyDescriptor:
    """Versioned cache geometry exposed by a model/backend adapter."""

    model_architecture: str
    backend_name: str
    topology_version: str
    groups: tuple[CacheGroupGeometry, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        for name in ("model_architecture", "backend_name", "topology_version"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        groups = tuple(sorted(self.groups, key=lambda group: group.group_id))
        if not groups:
            raise ValueError("a topology must contain at least one cache group")
        if len({group.group_id for group in groups}) != len(groups):
            raise ValueError("cache group ids must be unique")
        object.__setattr__(self, "groups", groups)
        by_id = {group.group_id: group for group in groups}
        for group in groups:
            for sibling_id in group.sibling_group_ids:
                sibling = by_id.get(sibling_id)
                if sibling is None:
                    raise ValueError(
                        f"group {group.group_id} references unknown sibling "
                        f"{sibling_id}"
                    )
                if group.group_id not in sibling.sibling_group_ids:
                    raise ValueError(
                        f"sibling relationship {group.group_id}<->{sibling_id} "
                        "must be symmetric"
                    )
            family = frozenset((group.group_id, *group.sibling_group_ids))
            for member_id in sorted(family):
                member = by_id[member_id]
                member_family = frozenset((member.group_id, *member.sibling_group_ids))
                if member_family != family:
                    raise ValueError(
                        "sibling relationships must describe complete, "
                        f"non-overlapping cache families; {group.group_id} "
                        f"and {member_id} disagree"
                    )
        expected = _topology_fingerprint(
            self.model_architecture,
            self.backend_name,
            self.topology_version,
            groups,
        )
        if self.fingerprint != expected:
            raise ValueError("topology fingerprint does not match descriptor content")

    @classmethod
    def create(
        cls,
        *,
        model_architecture: str,
        backend_name: str,
        topology_version: str,
        groups: tuple[CacheGroupGeometry, ...],
    ) -> CacheTopologyDescriptor:
        """Build a descriptor with a deterministic content fingerprint."""
        normalized = tuple(sorted(groups, key=lambda group: group.group_id))
        return cls(
            model_architecture=model_architecture,
            backend_name=backend_name,
            topology_version=topology_version,
            groups=normalized,
            fingerprint=_topology_fingerprint(
                model_architecture,
                backend_name,
                topology_version,
                normalized,
            ),
        )

    def group(self, group_id: str) -> CacheGroupGeometry | None:
        """Return a group by id without exposing a mutable mapping."""
        return next(
            (group for group in self.groups if group.group_id == group_id), None
        )


@dataclass(frozen=True)
class GroupSelection:
    """A semantic retention decision for one cache group."""

    group_id: str
    semantic_kind: CacheSemanticKind
    logical_span: LogicalSpan
    required_siblings: tuple[str, ...]
    action: RetentionAction
    valid_until_step: int | None = None

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        siblings = tuple(sorted(self.required_siblings))
        if len(siblings) != len(set(siblings)):
            raise ValueError("required_siblings must be unique")
        if self.group_id in siblings:
            raise ValueError("a selection cannot require itself as a sibling")
        object.__setattr__(self, "required_siblings", siblings)
        if self.valid_until_step is not None and self.valid_until_step < 0:
            raise ValueError("valid_until_step must be non-negative")


@dataclass(frozen=True)
class TokenSelectionPlan:
    """Immutable semantic plan, independent of page tables and rank addresses."""

    request_id: str
    request_generation: int
    decode_round: int
    accepted_seq_len: int
    source_kv_revision: int
    topology_fingerprint: str
    policy_revision: str
    groups: tuple[GroupSelection, ...]
    plan_digest: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        for name in (
            "request_generation",
            "decode_round",
            "accepted_seq_len",
            "source_kv_revision",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not self.topology_fingerprint:
            raise ValueError("topology_fingerprint must not be empty")
        if not self.policy_revision:
            raise ValueError("policy_revision must not be empty")
        groups = tuple(sorted(self.groups, key=_selection_sort_key))
        if not groups:
            raise ValueError("a token-selection plan must contain at least one group")
        object.__setattr__(self, "groups", groups)
        expected = _plan_digest(
            request_id=self.request_id,
            request_generation=self.request_generation,
            decode_round=self.decode_round,
            accepted_seq_len=self.accepted_seq_len,
            source_kv_revision=self.source_kv_revision,
            topology_fingerprint=self.topology_fingerprint,
            policy_revision=self.policy_revision,
            groups=groups,
        )
        if self.plan_digest != expected:
            raise ValueError("plan digest does not match plan content")

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        request_generation: int,
        decode_round: int,
        accepted_seq_len: int,
        source_kv_revision: int,
        topology_fingerprint: str,
        policy_revision: str,
        groups: tuple[GroupSelection, ...],
    ) -> TokenSelectionPlan:
        """Build an immutable plan with a deterministic content digest."""
        normalized = tuple(sorted(groups, key=_selection_sort_key))
        return cls(
            request_id=request_id,
            request_generation=request_generation,
            decode_round=decode_round,
            accepted_seq_len=accepted_seq_len,
            source_kv_revision=source_kv_revision,
            topology_fingerprint=topology_fingerprint,
            policy_revision=policy_revision,
            groups=normalized,
            plan_digest=_plan_digest(
                request_id=request_id,
                request_generation=request_generation,
                decode_round=decode_round,
                accepted_seq_len=accepted_seq_len,
                source_kv_revision=source_kv_revision,
                topology_fingerprint=topology_fingerprint,
                policy_revision=policy_revision,
                groups=normalized,
            ),
        )


@dataclass(frozen=True)
class PlanValidationIssue:
    """One stable fail-closed validation finding."""

    code: PlanValidationCode
    message: str
    group_id: str | None = None


@dataclass(frozen=True)
class PlanValidationResult:
    """Validation result with all detected issues."""

    issues: tuple[PlanValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether physical expansion is safe."""
        return not self.issues

    @property
    def recompute_required(self) -> bool:
        """Return whether callers must fall back to cache miss and recompute."""
        return bool(self.issues)


@dataclass(frozen=True)
class PhysicalGroupOperation:
    """Topology-expanded full-block operation plus explicit residual spans."""

    group_id: str
    semantic_kind: CacheSemanticKind
    action: RetentionAction
    logical_span: LogicalSpan
    full_block_logical_span: LogicalSpan | None
    physical_entry_start: int
    physical_entry_end: int
    byte_start: int
    byte_end: int
    residual_spans: tuple[LogicalSpan, ...]
    rank_sharding: str

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if self.physical_entry_start < 0:
            raise ValueError("physical_entry_start must be non-negative")
        if self.physical_entry_end < self.physical_entry_start:
            raise ValueError("physical entry range must not be negative")
        if self.byte_start < 0 or self.byte_end < self.byte_start:
            raise ValueError("byte range must not be negative")
        if not self.rank_sharding:
            raise ValueError("rank_sharding must not be empty")

    @property
    def physical_entry_count(self) -> int:
        """Return the number of full-block physical entries."""
        return self.physical_entry_end - self.physical_entry_start

    @property
    def requires_residual_handling(self) -> bool:
        """Return whether partial logical blocks need adapter handling."""
        return bool(self.residual_spans)


@dataclass(frozen=True)
class PlanExpansion:
    """Fail-closed validation and physical expansion result."""

    validation: PlanValidationResult
    operations: tuple[PhysicalGroupOperation, ...] = ()

    def __post_init__(self) -> None:
        if self.validation.recompute_required and self.operations:
            raise ValueError("invalid plans cannot expose physical operations")


def validate_plan(
    plan: TokenSelectionPlan,
    topology: CacheTopologyDescriptor,
    *,
    request_id: str,
    request_generation: int,
    decode_round: int,
    accepted_seq_len: int,
    source_kv_revision: int,
    policy_revision: str,
    current_step: int,
) -> PlanValidationResult:
    """Validate a plan against live request and topology revisions."""
    if current_step < 0:
        raise ValueError("current_step must be non-negative")
    issues: list[PlanValidationIssue] = []

    def add(
        code: PlanValidationCode, message: str, group_id: str | None = None
    ) -> None:
        issues.append(
            PlanValidationIssue(code=code, message=message, group_id=group_id)
        )

    expected_values = (
        (plan.request_id, request_id, PlanValidationCode.REQUEST_ID_MISMATCH),
        (
            plan.request_generation,
            request_generation,
            PlanValidationCode.REQUEST_GENERATION_MISMATCH,
        ),
        (plan.decode_round, decode_round, PlanValidationCode.DECODE_ROUND_MISMATCH),
        (
            plan.accepted_seq_len,
            accepted_seq_len,
            PlanValidationCode.ACCEPTED_SEQ_LEN_MISMATCH,
        ),
        (
            plan.source_kv_revision,
            source_kv_revision,
            PlanValidationCode.SOURCE_KV_REVISION_MISMATCH,
        ),
        (
            plan.topology_fingerprint,
            topology.fingerprint,
            PlanValidationCode.TOPOLOGY_MISMATCH,
        ),
        (
            plan.policy_revision,
            policy_revision,
            PlanValidationCode.POLICY_REVISION_MISMATCH,
        ),
    )
    for actual, expected, code in expected_values:
        if actual != expected:
            add(code, f"plan value {actual!r} does not match live value {expected!r}")

    selections: dict[str, GroupSelection] = {}
    for selection in plan.groups:
        if selection.group_id in selections:
            add(
                PlanValidationCode.DUPLICATE_GROUP,
                "plan contains more than one selection for the group",
                selection.group_id,
            )
            continue
        selections[selection.group_id] = selection
        geometry = topology.group(selection.group_id)
        if geometry is None:
            add(
                PlanValidationCode.UNKNOWN_GROUP,
                "selection does not exist in the topology",
                selection.group_id,
            )
            continue
        if selection.semantic_kind != geometry.semantic_kind:
            add(
                PlanValidationCode.SEMANTIC_KIND_MISMATCH,
                "selection semantic kind differs from the topology",
                selection.group_id,
            )
        if selection.logical_span.end_token > plan.accepted_seq_len:
            add(
                PlanValidationCode.SPAN_AFTER_ACCEPTED_SEQUENCE,
                "selection extends past the accepted sequence boundary",
                selection.group_id,
            )
        if geometry.window_size_tokens is not None:
            window_start = max(0, plan.accepted_seq_len - geometry.window_size_tokens)
            if selection.logical_span.start_token < window_start:
                add(
                    PlanValidationCode.SLIDING_WINDOW_BOUNDARY,
                    f"selection begins before active window token {window_start}",
                    selection.group_id,
                )
        if set(selection.required_siblings) != set(geometry.sibling_group_ids):
            add(
                PlanValidationCode.SIBLING_DECLARATION_MISMATCH,
                "selection siblings differ from topology family membership",
                selection.group_id,
            )
        if (
            selection.valid_until_step is not None
            and current_step > selection.valid_until_step
        ):
            add(
                PlanValidationCode.EXPIRED_SELECTION,
                "selection validity has expired",
                selection.group_id,
            )
        if selection.action == RetentionAction.RECOMPUTE_REQUIRED:
            add(
                PlanValidationCode.EXPLICIT_RECOMPUTE,
                "selection explicitly requires recomputation",
                selection.group_id,
            )

    checked_families: set[frozenset[str]] = set()
    for group_id, selection in selections.items():
        geometry = topology.group(group_id)
        if geometry is None:
            continue
        family = frozenset((group_id, *geometry.sibling_group_ids))
        if family in checked_families:
            continue
        checked_families.add(family)
        missing = sorted(family - selections.keys())
        for missing_id in missing:
            add(
                PlanValidationCode.MISSING_FAMILY_MEMBER,
                f"cache family is missing required group {missing_id}",
                group_id,
            )
        present = [selections[item] for item in family if item in selections]
        if not present:
            continue
        first = present[0]
        if any(item.action != first.action for item in present[1:]):
            add(
                PlanValidationCode.INCONSISTENT_FAMILY_ACTION,
                "cache family selections must use one atomic action",
                group_id,
            )
        if any(
            item.logical_span != first.logical_span
            or item.valid_until_step != first.valid_until_step
            for item in present[1:]
        ):
            add(
                PlanValidationCode.INCONSISTENT_FAMILY_SPAN,
                "cache family selections must share span and validity",
                group_id,
            )

    return PlanValidationResult(issues=tuple(issues))


def expand_plan(
    plan: TokenSelectionPlan,
    topology: CacheTopologyDescriptor,
    *,
    request_id: str,
    request_generation: int,
    decode_round: int,
    accepted_seq_len: int,
    source_kv_revision: int,
    policy_revision: str,
    current_step: int,
) -> PlanExpansion:
    """Validate then map a semantic plan to safe full-block operations."""
    validation = validate_plan(
        plan,
        topology,
        request_id=request_id,
        request_generation=request_generation,
        decode_round=decode_round,
        accepted_seq_len=accepted_seq_len,
        source_kv_revision=source_kv_revision,
        policy_revision=policy_revision,
        current_step=current_step,
    )
    if not validation.valid:
        return PlanExpansion(validation=validation)
    operations = tuple(
        _expand_selection(selection, topology.group(selection.group_id))
        for selection in plan.groups
    )
    return PlanExpansion(validation=validation, operations=operations)


def _expand_selection(
    selection: GroupSelection,
    geometry: CacheGroupGeometry | None,
) -> PhysicalGroupOperation:
    assert geometry is not None
    span = selection.logical_span
    tokens_per_block = geometry.logical_tokens_per_block
    first_full_block = (span.start_token + tokens_per_block - 1) // tokens_per_block
    end_full_block = max(first_full_block, span.end_token // tokens_per_block)
    full_start_token = first_full_block * tokens_per_block
    full_end_token = end_full_block * tokens_per_block

    residuals: list[LogicalSpan] = []
    prefix_end = min(span.end_token, full_start_token)
    if span.start_token < prefix_end:
        residuals.append(LogicalSpan(span.start_token, prefix_end))
    suffix_start = max(prefix_end, full_end_token)
    if suffix_start < span.end_token:
        residuals.append(LogicalSpan(suffix_start, span.end_token))

    full_span = (
        LogicalSpan(full_start_token, full_end_token)
        if full_start_token < full_end_token
        else None
    )
    entry_start = first_full_block * geometry.physical_entries_per_block
    entry_end = end_full_block * geometry.physical_entries_per_block
    return PhysicalGroupOperation(
        group_id=selection.group_id,
        semantic_kind=selection.semantic_kind,
        action=selection.action,
        logical_span=span,
        full_block_logical_span=full_span,
        physical_entry_start=entry_start,
        physical_entry_end=entry_end,
        byte_start=entry_start * geometry.page_stride_bytes,
        byte_end=entry_end * geometry.page_stride_bytes,
        residual_spans=tuple(residuals),
        rank_sharding=geometry.rank_sharding,
    )


def _topology_fingerprint(
    model_architecture: str,
    backend_name: str,
    topology_version: str,
    groups: tuple[CacheGroupGeometry, ...],
) -> str:
    payload = {
        "backend_name": backend_name,
        "groups": [
            {
                "alignment_bytes": group.alignment_bytes,
                "compression_ratio": group.compression_ratio,
                "group_id": group.group_id,
                "logical_tokens_per_block": group.logical_tokens_per_block,
                "page_stride_bytes": group.page_stride_bytes,
                "physical_entries_per_block": group.physical_entries_per_block,
                "rank_sharding": group.rank_sharding,
                "semantic_kind": group.semantic_kind.value,
                "sibling_group_ids": list(group.sibling_group_ids),
                "window_size_tokens": group.window_size_tokens,
            }
            for group in groups
        ],
        "model_architecture": model_architecture,
        "topology_version": topology_version,
    }
    return _sha256(payload)


def _plan_digest(
    *,
    request_id: str,
    request_generation: int,
    decode_round: int,
    accepted_seq_len: int,
    source_kv_revision: int,
    topology_fingerprint: str,
    policy_revision: str,
    groups: tuple[GroupSelection, ...],
) -> str:
    payload = {
        "accepted_seq_len": accepted_seq_len,
        "decode_round": decode_round,
        "groups": [
            {
                "action": selection.action.value,
                "group_id": selection.group_id,
                "logical_span": {
                    "end_token": selection.logical_span.end_token,
                    "start_token": selection.logical_span.start_token,
                },
                "required_siblings": list(selection.required_siblings),
                "semantic_kind": selection.semantic_kind.value,
                "valid_until_step": selection.valid_until_step,
            }
            for selection in groups
        ],
        "policy_revision": policy_revision,
        "request_generation": request_generation,
        "request_id": request_id,
        "source_kv_revision": source_kv_revision,
        "topology_fingerprint": topology_fingerprint,
    }
    return _sha256(payload)


def _sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
