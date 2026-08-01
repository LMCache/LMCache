# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP runtime management-policy HTTP API."""

# Standard
from dataclasses import dataclass, field

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.distributed.runtime_policy import (
    RuntimePolicyError,
    RuntimePolicyUpdate,
    RuntimePolicyUpdateResult,
    RuntimePolicyValidation,
)
from lmcache.v1.multiprocess.http_apis.policy_api import router


@dataclass
class _FakeStorageManager:
    calls: list[RuntimePolicyUpdate] = field(default_factory=list)
    runtime_policy: dict[str, object] = field(
        default_factory=lambda: {"version": 3, "capabilities": {}}
    )
    validation_error: RuntimePolicyError | None = None

    def get_runtime_policy(self) -> dict[str, object]:
        return self.runtime_policy

    def validate_runtime_policy_update(
        self,
        update: RuntimePolicyUpdate,
    ) -> RuntimePolicyValidation:
        if self.validation_error is not None:
            raise self.validation_error
        self.calls.append(update)
        return RuntimePolicyValidation(
            version=3,
            changed_fields=("store_policy",),
            effective_on={"store_policy": "next_store_plan"},
        )

    def update_runtime_policy(
        self,
        update: RuntimePolicyUpdate,
    ) -> RuntimePolicyUpdateResult:
        if self.validation_error is not None:
            raise self.validation_error
        self.calls.append(update)
        return RuntimePolicyUpdateResult(
            status="updated",
            version=4,
            applied=("store_policy",),
            effective_on={"store_policy": "next_store_plan"},
        )


@dataclass
class _FakeEngine:
    storage_manager: _FakeStorageManager


def _client(storage_manager: _FakeStorageManager | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if storage_manager is not None:
        app.state.engine = _FakeEngine(storage_manager)
    return TestClient(app)


def test_get_policies_returns_runtime_capabilities() -> None:
    storage_manager = _FakeStorageManager()

    response = _client(storage_manager).get("/config/policies")

    assert response.status_code == 200
    assert response.json() == {"version": 3, "capabilities": {}}


def test_validate_does_not_apply_and_converts_nested_update() -> None:
    storage_manager = _FakeStorageManager()

    response = _client(storage_manager).post(
        "/config/policies/validate",
        json={
            "expected_version": 3,
            "store_policy": "skip_l1",
            "l1_eviction": {
                "tunables": {
                    "trigger_watermark": 0.9,
                    "eviction_ratio": 0.1,
                }
            },
            "l2_eviction": [
                {
                    "adapter_id": 7,
                    "tunables": {"eviction_ratio": 0.2},
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "valid",
        "version": 3,
        "changed": ["store_policy"],
        "effective_on": {"store_policy": "next_store_plan"},
    }
    assert len(storage_manager.calls) == 1
    update = storage_manager.calls[0]
    assert update.expected_version == 3
    assert update.store_policy == "skip_l1"
    assert update.l1_eviction is not None
    assert update.l1_eviction.trigger_watermark == 0.9
    assert update.l2_eviction[0].adapter_id == 7


def test_patch_returns_new_version_and_applied_fields() -> None:
    storage_manager = _FakeStorageManager()

    response = _client(storage_manager).patch(
        "/config/policies",
        json={"store_policy": "skip_l1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "updated",
        "version": 4,
        "applied": ["store_policy"],
        "effective_on": {"store_policy": "next_store_plan"},
    }


def test_restart_only_fields_have_structured_error() -> None:
    storage_manager = _FakeStorageManager(
        validation_error=RuntimePolicyError(
            code="restart_required",
            field="chunk_size",
            message="chunk_size affects startup-only storage semantics",
        )
    )

    response = _client(storage_manager).patch(
        "/config/policies",
        json={"chunk_size": 512},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "restart_required",
        "message": "chunk_size affects startup-only storage semantics",
        "field": "chunk_size",
    }


def test_invalid_numeric_shape_is_rejected_by_http_schema() -> None:
    response = _client(_FakeStorageManager()).patch(
        "/config/policies",
        json={"l1_eviction": {"tunables": {"eviction_ratio": 2.0}}},
    )

    assert response.status_code == 422


def test_uninitialized_engine_returns_503() -> None:
    response = _client().get("/config/policies")

    assert response.status_code == 503
    assert response.json() == {"error": "engine not initialized"}
