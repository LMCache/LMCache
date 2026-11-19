# SPDX-License-Identifier: Apache-2.0
"""
Write policy interface and default implementation for L1 placement decisions.

The write policy decides, for every key an engine store reserves, which L1
manager holds the object. ``StorageManager.reserve_write`` asks it once per
batch, before any allocation.
"""

# Standard
from abc import ABC, abstractmethod

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ManagerInterface


class WritePolicy(ABC):
    """
    Abstract interface for L1 write placement.

    The write policy is called by the StorageManager to decide which L1
    manager each key is reserved in (select_write_targets).
    """

    @abstractmethod
    def select_write_targets(
        self,
        keys: list[ObjectKey],
        l1_managers: list[L1ManagerInterface],
    ) -> dict[int, list[ObjectKey]]:
        """
        Decide which L1 manager reserves each key.

        Args:
            keys: Keys an engine store is about to write.
            l1_managers: The L1 managers in placement order. A policy may
                read their memory tags, memory usage or configuration to
                decide.

        Returns:
            Mapping from ``l1_manager_id`` to the keys to reserve there.
            Every key appears exactly once across the mapping.
        """


# -----------------------------------------------------------------------------
# Registry: write policy name -> policy class
# -----------------------------------------------------------------------------

_WRITE_POLICY_REGISTRY: dict[str, type[WritePolicy]] = {}


def register_write_policy(
    name: str,
    policy_cls: type[WritePolicy],
) -> None:
    """
    Register a write policy class under a name.

    Each policy module should call this at import time.

    Args:
        name: Policy name (e.g. "default").
        policy_cls: A concrete WritePolicy subclass.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _WRITE_POLICY_REGISTRY:
        raise ValueError(f"Write policy already registered: {name!r}")
    _WRITE_POLICY_REGISTRY[name] = policy_cls


def get_registered_write_policies() -> list[str]:
    """Return the list of registered write policy names."""
    return list(_WRITE_POLICY_REGISTRY)


def create_write_policy(name: str) -> WritePolicy:
    """
    Create a write policy instance by name.

    Args:
        name: Registered policy name.

    Returns:
        A new WritePolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _WRITE_POLICY_REGISTRY:
        known = ", ".join(sorted(_WRITE_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown write policy {name!r}. Known: {known}")
    return _WRITE_POLICY_REGISTRY[name]()


class DefaultWritePolicy(WritePolicy):
    """
    Default write policy: every key goes to the first L1 in placement order.
    """

    def select_write_targets(
        self,
        keys: list[ObjectKey],
        l1_managers: list[L1ManagerInterface],
    ) -> dict[int, list[ObjectKey]]:
        """Reserve all keys in the first L1 manager."""
        return {l1_managers[0].l1_manager_id: list(keys)}


register_write_policy("default", DefaultWritePolicy)
