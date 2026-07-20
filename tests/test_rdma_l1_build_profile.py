# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock

# First Party
from setup_extensions import policy
from setup_extensions.storage_backend_profiles.rdma_l1 import RdmaL1Profile


def test_rdma_l1_build_profile_is_explicit_only(monkeypatch):
    profile = RdmaL1Profile()

    monkeypatch.delenv("BUILD_WITH_RDMA_L1", raising=False)
    assert not profile.is_explicitly_requested()
    assert not profile.detect()

    monkeypatch.setenv("BUILD_WITH_RDMA_L1", "1")
    assert profile.is_explicitly_requested()


def test_build_policy_only_selects_rdma_l1_when_requested(monkeypatch):
    profile = RdmaL1Profile()
    profile.build = MagicMock(return_value=["rdma-extension"])
    monkeypatch.setattr(policy, "_discover_storage_backends", lambda: [profile])

    monkeypatch.delenv("BUILD_WITH_RDMA_L1", raising=False)
    assert policy.BuildPolicy.collect_storage_backends([]) == []
    profile.build.assert_not_called()

    monkeypatch.setenv("BUILD_WITH_RDMA_L1", "1")
    assert policy.BuildPolicy.collect_storage_backends([]) == ["rdma-extension"]
    profile.build.assert_called_once_with([])
