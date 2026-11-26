# SPDX-License-Identifier: Apache-2.0
"""The write policy registry and the default placement."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.storage_controllers.write_policy import (
    DefaultWritePolicy,
    WritePolicy,
    create_write_policy,
    get_registered_write_policies,
)


def _manager(l1_manager_id: int) -> L1ManagerInterface:
    return cast(L1ManagerInterface, SimpleNamespace(l1_manager_id=l1_manager_id))


def _key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "policy-test", 0)


def test_default_policy_places_every_key_in_the_first_manager() -> None:
    keys = [_key(1), _key(2), _key(3)]
    managers = [_manager(7), _manager(3)]
    assert DefaultWritePolicy().select_write_targets(keys, managers) == {7: keys}


def test_registry_creates_known_policies_and_rejects_unknown_names() -> None:
    assert "default" in get_registered_write_policies()
    policy = create_write_policy("default")
    assert isinstance(policy, WritePolicy)
    assert isinstance(policy, DefaultWritePolicy)
    with pytest.raises(ValueError, match="Unknown write policy 'nope'"):
        create_write_policy("nope")
