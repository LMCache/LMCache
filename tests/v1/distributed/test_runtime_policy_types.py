# SPDX-License-Identifier: Apache-2.0
"""Tests for transport-independent runtime policy validation types."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.runtime_policy import (
    RuntimePolicyError,
    validate_eviction_policy,
    validate_eviction_value,
)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_eviction_value_must_be_finite_and_bounded(value: float) -> None:
    with pytest.raises(RuntimePolicyError) as exc_info:
        validate_eviction_value("l1_eviction.eviction_ratio", value)

    assert exc_info.value.code == "invalid_value"
    assert exc_info.value.field == "l1_eviction.eviction_ratio"


def test_error_serialization_is_stable() -> None:
    error = RuntimePolicyError(
        code="version_conflict",
        status_code=409,
        field="expected_version",
        current=4,
        requested=3,
        message="stale version",
    )

    assert error.as_dict() == {
        "error": "version_conflict",
        "message": "stale version",
        "field": "expected_version",
        "current": 4,
        "requested": 3,
    }


def test_eviction_policy_must_be_registered() -> None:
    with pytest.raises(RuntimePolicyError, match="must be one of") as exc_info:
        validate_eviction_policy("l1_eviction.policy", "unknown")

    assert exc_info.value.code == "invalid_policy"
