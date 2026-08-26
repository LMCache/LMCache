# SPDX-License-Identifier: Apache-2.0
"""
Store policy interface and default implementation for L1-to-L2 storage decisions.

The store policy makes two decisions after data is written to L1:
1. Which L2 adapter(s) should each key be stored to?
2. After a successful L2 store, should the key be deleted from L1?
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Third Party
import blake3

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)


@dataclass(frozen=True)
class AdapterDescriptor:
    """
    Lightweight descriptor for an L2 adapter, giving the store policy
    enough information to distinguish adapters without exposing runtime
    objects.
    """

    index: int
    """Position in the L2 adapters list."""

    config: L2AdapterConfigBase
    """The adapter's configuration object."""

    @property
    def type_name(self) -> str:
        """
        Registered adapter type name (e.g., "mock", "disk", "redis").

        Derived from the config's registered type via reverse lookup.

        Returns:
            str: The registered type name.
        """
        return get_type_name_for_config(self.config)

    @property
    def placement_id(self) -> str | None:
        """Return the adapter's restart-stable placement identifier.

        Returns:
            The stable identifier supplied by the adapter config, or ``None``
            when the backend does not provide one.
        """
        return self.config.placement_id


class StorePolicy(ABC):
    """
    Abstract interface for store decisions.

    The store policy is called by the StoreController to decide:
    1. Which adapter(s) to store each key to (select_store_targets).
    2. Which keys to delete from L1 after successful L2 store
       (select_l1_deletions).
    """

    def validate_adapters(self, adapters: list[AdapterDescriptor]) -> None:
        """Validate an adapter set before the controller starts.

        Args:
            adapters: Configured L2 adapter descriptors.
        """
        return None

    @abstractmethod
    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Decide which keys to store to which L2 adapters.

        Args:
            keys: Keys that were just written to L1 and are
                candidates for L2 storage.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to list of keys to store
            to that adapter. Keys absent from all lists are NOT
            stored to L2.
        """

    @abstractmethod
    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Decide which keys to delete from L1 after successful L2 store.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Keys to delete from L1. Empty list means keep all.
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


def create_store_policy(name: str) -> StorePolicy:
    """
    Create a store policy instance by name.

    Args:
        name: Registered policy name.

    Returns:
        A new StorePolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _STORE_POLICY_REGISTRY:
        known = ", ".join(sorted(_STORE_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown store policy {name!r}. Known: {known}")
    return _STORE_POLICY_REGISTRY[name]()


def _routing_key_bytes(key: ObjectKey) -> bytes:
    """Encode every identity field of an object key without ambiguous joins."""
    fields = (
        key.model_name.encode(),
        str(key.kv_rank).encode(),
        str(key.object_group_id).encode(),
        key.chunk_hash,
        key.cache_salt.encode(),
    )
    return b"".join(len(field).to_bytes(8, byteorder="big") + field for field in fields)


def _validate_stable_adapters(
    adapters: list[AdapterDescriptor],
) -> list[tuple[int, str, bytes]]:
    """Validate and deterministically order adapters for stable placement."""
    missing = [adapter.index for adapter in adapters if not adapter.placement_id]
    if missing:
        raise ValueError(
            "striped policy requires a stable placement_id for every adapter; "
            f"missing on adapter indices {missing}"
        )

    placement_ids = [adapter.placement_id for adapter in adapters]
    if len(set(placement_ids)) != len(placement_ids):
        raise ValueError("striped policy requires unique adapter placement_id values")

    stable_adapters: list[tuple[int, str, bytes]] = []
    for adapter in sorted(adapters, key=lambda value: value.placement_id or ""):
        placement_id = adapter.placement_id
        assert placement_id is not None
        placement_bytes = placement_id.encode()
        framed_placement = (
            len(placement_bytes).to_bytes(8, byteorder="big") + placement_bytes
        )
        stable_adapters.append((adapter.index, placement_id, framed_placement))
    return stable_adapters


def rendezvous_adapter_index_for_key(
    key: ObjectKey,
    adapters: list[AdapterDescriptor],
) -> int:
    """Select one adapter with highest-random-weight rendezvous hashing.

    Args:
        key: Object key to route.
        adapters: Candidate adapters. Every adapter must expose a unique,
            restart-stable ``placement_id``.

    Returns:
        Runtime index of the selected adapter.

    Raises:
        ValueError: If no adapters are supplied or placement identifiers are
            missing or duplicated.
    """
    stable_adapters = _validate_stable_adapters(adapters)
    if not stable_adapters:
        raise ValueError("rendezvous hashing requires at least one adapter")

    return _rendezvous_adapter_index_for_key(key, stable_adapters)


def rendezvous_adapter_indices_for_keys(
    keys: list[ObjectKey],
    adapters: list[AdapterDescriptor],
) -> list[int]:
    """Route a batch of keys while validating the adapter set only once.

    Args:
        keys: Object keys to route.
        adapters: Candidate adapters with unique stable placement identifiers.

    Returns:
        Adapter indices in the same order as ``keys``.

    Raises:
        ValueError: If the adapter set is empty or lacks unique stable ids.
    """
    stable_adapters = _validate_stable_adapters(adapters)
    if not stable_adapters:
        raise ValueError("rendezvous hashing requires at least one adapter")
    return [_rendezvous_adapter_index_for_key(key, stable_adapters) for key in keys]


def _rendezvous_adapter_index_for_key(
    key: ObjectKey,
    stable_adapters: list[tuple[int, str, bytes]],
) -> int:
    """Route a key over an already validated, non-empty adapter list."""
    key_bytes = _routing_key_bytes(key)
    winner_index = -1
    winner: tuple[bytes, str] | None = None
    for adapter_index, placement_id, placement_bytes in stable_adapters:
        score = blake3.blake3(key_bytes + placement_bytes).digest(length=16)
        candidate = (score, placement_id)
        # The stable id is a deterministic tie-breaker for the vanishingly
        # unlikely case of equal 128-bit scores.
        if winner is None or candidate > winner:
            winner = candidate
            winner_index = adapter_index
    return winner_index


class DefaultStorePolicy(StorePolicy):
    """
    Default store policy: store all keys to all adapters,
    never delete from L1.
    """

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Store all keys to all adapters.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from every adapter index to the full list of keys.
        """
        return {ad.index: list(keys) for ad in adapters}

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Never delete from L1.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Empty list (keep all keys in L1).
        """
        return []


class BufferOnlyStorePolicy(DefaultStorePolicy):
    """
    Buffer-only store policy: store all keys to all adapters,
    then delete them from L1 immediately.

    Use this with NoOpEvictionPolicy to avoid useless LRU
    tracking overhead when L1 is a pure write buffer.

    Inherits ``select_store_targets`` from ``DefaultStorePolicy``
    (store all keys to all adapters) and only overrides the L1
    deletion decision.
    """

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Delete all keys from L1 after successful L2 store.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            All keys (remove everything from L1).
        """
        return list(keys)


class StripedStorePolicy(StorePolicy):
    """Store every key on one stable rendezvous-hashed adapter.

    The adapter's persistent placement identifier, rather than its runtime
    list position, participates in the hash. Adding or removing a disk thus
    remaps only keys owned by the changed disk set while preserving the
    single-copy capacity and bandwidth benefits of striping.
    """

    def validate_adapters(self, adapters: list[AdapterDescriptor]) -> None:
        """Require unique, persistent placement identifiers.

        Args:
            adapters: Configured L2 adapter descriptors.
        """
        _validate_stable_adapters(adapters)

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """Assign every key to exactly one stable adapter.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to its assigned keys. With no adapters,
            returns an empty mapping.
        """
        if not adapters:
            return {}
        stable_adapters = _validate_stable_adapters(adapters)
        targets: dict[int, list[ObjectKey]] = {
            adapter_index: []
            for adapter_index, _placement_id, _bytes in stable_adapters
        }
        adapter_indices = [
            _rendezvous_adapter_index_for_key(key, stable_adapters) for key in keys
        ]
        for key, adapter_index in zip(keys, adapter_indices, strict=True):
            targets[adapter_index].append(key)
        return targets

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """Keep successfully stored keys in L1.

        Args:
            keys: Keys successfully stored in L2.

        Returns:
            An empty list.
        """
        return []


register_store_policy("default", DefaultStorePolicy)
register_store_policy("skip_l1", BufferOnlyStorePolicy)
register_store_policy("striped", StripedStorePolicy)
