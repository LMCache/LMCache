# SPDX-License-Identifier: Apache-2.0
"""Tests for the platform IPC policy."""

# Third Party
import pytest

# First Party
from lmcache.v1.platform.ipc_policy import (
    current_ipc_policy,
    get_ipc_policy,
    is_isolated_ipc,
    is_use_vmm_api,
    set_ipc_policy,
    set_isolated_ipc,
    set_use_vmm_api,
)


@pytest.fixture(autouse=True)
def restore_ipc_policy():
    """Restore the process-global IPC policy after each test."""
    previous = get_ipc_policy()
    previous_isolated_ipc = previous.isolated_ipc
    previous_use_vmm_api = previous.use_vmm_api
    yield
    set_ipc_policy(
        isolated_ipc=previous_isolated_ipc,
        use_vmm_api=previous_use_vmm_api,
    )


def test_individual_switches_share_the_platform_policy() -> None:
    policy = set_ipc_policy(isolated_ipc=False, use_vmm_api=False)

    assert current_ipc_policy is policy
    assert policy is get_ipc_policy()
    assert set_isolated_ipc(True) is policy
    assert get_ipc_policy().isolated_ipc is True
    assert is_isolated_ipc() is True
    assert is_use_vmm_api() is False

    assert set_use_vmm_api(True) is policy
    assert get_ipc_policy().isolated_ipc is True
    assert get_ipc_policy().use_vmm_api is True
    assert is_use_vmm_api() is True


def test_partial_policy_update_preserves_other_settings() -> None:
    policy = set_ipc_policy(isolated_ipc=True, use_vmm_api=False)
    assert set_ipc_policy(use_vmm_api=True) is policy

    assert get_ipc_policy().isolated_ipc is True
    assert get_ipc_policy().use_vmm_api is True


def test_policy_can_be_chained_in_place() -> None:
    policy = get_ipc_policy()
    result = policy.set(isolated_ipc=False, use_vmm_api=False).with_isolated_ipc()
    result = result.with_use_vmm_api(True)

    assert result is policy
    assert get_ipc_policy() is policy
    assert policy.isolated_ipc is True
    assert policy.use_vmm_api is True


def test_vmm_wrapper_selection_follows_policy_only() -> None:
    # First Party
    from lmcache.v1.platform.cuda.ipc_wrapper import VmmCudaIPCWrapper
    from lmcache.v1.platform.rocm import RocmDeviceSpec

    set_ipc_policy(use_vmm_api=True)

    assert RocmDeviceSpec().ipc_wrapper_cls is VmmCudaIPCWrapper
