# SPDX-License-Identifier: Apache-2.0
"""Domain types for runtime management-policy updates in MP mode."""

# Standard
from dataclasses import dataclass
from typing import Literal
import math

RuntimePolicyErrorCode = Literal[
    "invalid_policy",
    "invalid_value",
    "version_conflict",
    "restart_required",
    "state_migration_required",
    "unknown_l2_adapter",
    "eviction_not_configured",
    "duplicate_l2_adapter",
    "invalid_update",
    "unsupported_field",
]

_VALID_EVICTION_POLICIES = frozenset({"LRU", "IsolatedLRU", "noop"})


@dataclass(frozen=True)
class RuntimePolicyError(Exception):
    """A structured validation failure for a runtime policy request."""

    code: RuntimePolicyErrorCode
    message: str
    status_code: int = 400
    field: str | None = None
    current: object | None = None
    requested: object | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON representation used by the HTTP API."""
        result: dict[str, object] = {
            "error": self.code,
            "message": self.message,
        }
        if self.field is not None:
            result["field"] = self.field
        if self.current is not None:
            result["current"] = self.current
        if self.requested is not None:
            result["requested"] = self.requested
        return result


@dataclass(frozen=True)
class RuntimePolicyTunables:
    """Mutable numeric knobs accepted by the Phase 1 runtime API."""

    policy: str | None = None
    trigger_watermark: float | None = None
    eviction_ratio: float | None = None

    def has_value(self) -> bool:
        """Return whether the request carries at least one field."""
        return any(
            value is not None
            for value in (
                self.policy,
                self.trigger_watermark,
                self.eviction_ratio,
            )
        )


@dataclass(frozen=True)
class RuntimeL2EvictionUpdate:
    """Runtime update addressed to one live L2 adapter."""

    adapter_id: int
    tunables: RuntimePolicyTunables


@dataclass(frozen=True)
class RuntimePolicyUpdate:
    """A fully parsed node-local runtime policy update."""

    expected_version: int | None = None
    store_policy: str | None = None
    prefetch_policy: str | None = None
    l1_eviction: RuntimePolicyTunables | None = None
    l2_eviction: tuple[RuntimeL2EvictionUpdate, ...] = ()
    restart_required_fields: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimePolicyValidation:
    """Result of validating a policy update without applying it."""

    version: int
    changed_fields: tuple[str, ...]
    effective_on: dict[str, str]


@dataclass(frozen=True)
class RuntimePolicyUpdateResult:
    """Result of applying a validated runtime policy update."""

    status: Literal["updated", "unchanged"]
    version: int
    applied: tuple[str, ...]
    effective_on: dict[str, str]


def validate_eviction_value(field: str, value: float | None) -> None:
    """Validate a Phase 1 eviction value.

    Args:
        field: Fully qualified field name used in an error response.
        value: Candidate value, or ``None`` when the field was omitted.

    Raises:
        RuntimePolicyError: If *value* is not finite or outside ``[0, 1]``.
    """
    if value is None:
        return
    if not math.isfinite(value):
        raise RuntimePolicyError(
            code="invalid_value",
            field=field,
            requested=value,
            message=f"{field} must be a finite number between 0 and 1",
        )
    if not 0.0 <= value <= 1.0:
        raise RuntimePolicyError(
            code="invalid_value",
            field=field,
            requested=value,
            message=f"{field} must be between 0 and 1",
        )


def validate_eviction_policy(field: str, policy: str | None) -> None:
    """Validate an eviction policy selector without changing state."""
    if policy is None or policy in _VALID_EVICTION_POLICIES:
        return
    raise RuntimePolicyError(
        code="invalid_policy",
        field=field,
        requested=policy,
        message=(
            f"{field} must be one of {', '.join(sorted(_VALID_EVICTION_POLICIES))}"
        ),
    )
