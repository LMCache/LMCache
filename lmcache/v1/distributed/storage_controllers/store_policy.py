# SPDX-License-Identifier: Apache-2.0
"""Store policy interface and built-in policies for L1-to-L2 storage.

After keys are written to L1, the policy returns the store actions for them:
which L2 adapter stores which keys, and whether each key leaves L1."""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass

# First Party
from lmcache.v1.distributed.api import ObjectKey, is_l1_only
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor


@dataclass(frozen=True)
class StoreAction:
    """One thing the store controller does with keys just written to L1."""

    keys: list[ObjectKey]
    """The keys this action applies to."""

    destination: int | None
    """Index of the L2 adapter that stores ``keys``, or None for no L2 store.
    The store controller never sends a key marked ``L1_ONLY_ATTRIBUTE`` to an
    adapter (see ``StorePolicy.get_store_actions``)."""

    l1_delete: bool
    """Whether ``keys`` leave L1. With a destination, each key is deleted once
    that adapter stored it successfully; without one, right away. A key a
    reader still holds is deleted once the reader is done."""


class StorePolicy(ABC):
    """
    Abstract interface for store decisions.

    The StoreController calls ``get_store_actions`` with keys that finished
    writing to L1 and carries out the returned actions. A key in no action
    stays in L1 and is not stored.

    Args:
        l1_managers: Every L1 manager of the storage manager, in
            write-placement order. A policy may read their memory tags,
            memory usage or configuration to decide.
    """

    def __init__(self, l1_managers: list[L1ManagerInterface]) -> None:
        self._l1_managers = l1_managers

    @abstractmethod
    def get_store_actions(
        self,
        keys: list[ObjectKey],
        adapters: list[L2AdapterDescriptor],
    ) -> list[StoreAction]:
        """
        Decide where ``keys`` are stored and whether they leave L1.

        Called only from the store controller's loop thread, so a policy may
        keep plain (unlocked) state.

        Args:
            keys: Keys that were just written to L1. Keys marked
                ``L1_ONLY_ATTRIBUTE`` come in a call of their own,
                with no adapters; the controller ignores any action with a
                destination for them.
            adapters: Descriptors of the live (not draining) L2 adapters.

        Returns:
            The actions to carry out, in any order. A key may appear in
            several actions (for example one per adapter); a key in none
            stays in L1 and is not stored.
        """


# -----------------------------------------------------------------------------
# Registry: store policy name -> policy class
# -----------------------------------------------------------------------------

_STORE_POLICY_REGISTRY: dict[str, type[StorePolicy]] = {}


def register_store_policy(
    name: str,
    policy_cls: type[StorePolicy],
) -> None:
    """
    Register a store policy class under a name.

    Each policy module should call this at import time.

    Args:
        name: Policy name (e.g. "default").
        policy_cls: A concrete StorePolicy subclass.
    """
    if name in _STORE_POLICY_REGISTRY:
        raise ValueError(f"Store policy already registered: {name!r}")
    _STORE_POLICY_REGISTRY[name] = policy_cls


def get_registered_store_policies() -> list[str]:
    """Return the list of registered store policy names."""
    return list(_STORE_POLICY_REGISTRY)


def create_store_policy(
    name: str, l1_managers: list[L1ManagerInterface]
) -> StorePolicy:
    """
    Create a store policy instance by name.

    Args:
        name: Registered policy name.
        l1_managers: Every L1 manager of the storage manager, in
            write-placement order.

    Returns:
        A new StorePolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _STORE_POLICY_REGISTRY:
        known = ", ".join(sorted(_STORE_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown store policy {name!r}. Known: {known}")
    return _STORE_POLICY_REGISTRY[name](l1_managers)


class DefaultStorePolicy(StorePolicy):
    """
    Default store policy: store all keys to all adapters,
    never delete from L1.
    """

    def get_store_actions(
        self,
        keys: list[ObjectKey],
        adapters: list[L2AdapterDescriptor],
    ) -> list[StoreAction]:
        """
        Store all keys to every adapter and keep them in L1.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of the live L2 adapters.

        Returns:
            One action per adapter, with every key and ``l1_delete=False``.
        """
        return [
            StoreAction(keys=list(keys), destination=ad.index, l1_delete=False)
            for ad in adapters
        ]


class BufferOnlyStorePolicy(StorePolicy):
    """
    Buffer-only store policy: store all keys to all adapters,
    then delete them from L1.

    Use this with NoOpEvictionPolicy to avoid useless LRU
    tracking overhead when L1 is a pure write buffer. Since nothing else
    frees L1 then, keys marked ``L1_ONLY_ATTRIBUTE`` leave L1 right away.
    """

    def get_store_actions(
        self,
        keys: list[ObjectKey],
        adapters: list[L2AdapterDescriptor],
    ) -> list[StoreAction]:
        """
        Store all keys to every adapter, deleting each from L1 once stored.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of the live L2 adapters.

        Returns:
            One action per adapter with ``l1_delete=True`` for the keys not
            marked ``L1_ONLY_ATTRIBUTE``, plus one action without a
            destination that deletes the marked keys from L1.
        """
        l1_only = [key for key in keys if is_l1_only(key)]
        stored = [key for key in keys if not is_l1_only(key)]
        actions = [
            StoreAction(keys=list(stored), destination=ad.index, l1_delete=True)
            for ad in adapters
            if stored
        ]
        if l1_only:
            actions.append(StoreAction(keys=l1_only, destination=None, l1_delete=True))
        return actions


register_store_policy("default", DefaultStorePolicy)
register_store_policy("skip_l1", BufferOnlyStorePolicy)
