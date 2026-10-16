# SPDX-License-Identifier: Apache-2.0
"""Committed CPU residency, independent of event transport and publication."""

# Standard
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import EventBinding, KVEventProfile

ChunkIdentity = tuple[str, int, str, bytes]
PendingStore = tuple[str, str]


@dataclass(frozen=True)
class KVResidencyTransition:
    """One readiness change with the exact immutable native descriptor."""

    kind: Literal["stored", "removed"]
    identity: ChunkIdentity
    binding: EventBinding


@dataclass
class _BindingState:
    descriptor: EventBinding
    pending: set[PendingStore]
    advertised: bool = False


class CommittedKVResidency:
    """Join native bindings with every required rank/group and readable L1 copy.

    The caller serializes all calls and supplies committed storage changes in
    their original order. ``bind`` only retains metadata; ``refresh`` joins it
    when its position in that order is reached. ``finish`` releases in-flight
    references after the corresponding commit notifications. Live objects keep
    their bindings independently of request lifetime.
    """

    def __init__(self) -> None:
        self._profiles: dict[
            tuple[str, int], tuple[KVEventProfile, frozenset[tuple[int, int]]]
        ] = {}
        self._bindings: dict[ChunkIdentity, _BindingState] = {}
        self._owners: dict[ObjectKey, set[str]] = {}
        self._pending: dict[PendingStore, set[ChunkIdentity]] = {}

    def register_profile(
        self,
        model_name: str,
        world_size: int,
        profile: KVEventProfile,
        required: frozenset[tuple[int, int]],
    ) -> None:
        """Pin a validated profile and its required ``(kv_rank, group)`` pairs.

        Args:
            model_name: Registered model namespace.
            world_size: Registered KV world size.
            profile: Engine-native identity contract.
            required: Complete geometry supplied by registration.

        Raises:
            ValueError: This context already has a different profile or geometry.
        """
        candidate = (profile, required)
        if self._profiles.setdefault((model_name, world_size), candidate) != candidate:
            raise ValueError("native KV event profile or registered layout mismatch")

    def bind(
        self,
        bindings: Sequence[tuple[ChunkIdentity, EventBinding]],
        pending: PendingStore,
    ) -> None:
        """Retain validated descriptors for a ``(request_id, transfer_id)`` store.

        ``bindings`` must pass the caller's token, profile and geometry checks.
        Conflicts with existing descriptors raise ValueError before any mutation.
        This operation never proves residency or emits a transition.
        """
        for identity, binding in bindings:
            current = self._bindings.get(identity)
            if current is not None and current.descriptor != binding:
                raise ValueError("conflicting native KV event binding")
        for identity, binding in bindings:
            state = self._bindings.setdefault(identity, _BindingState(binding, set()))
            state.pending.add(pending)
            self._pending.setdefault(pending, set()).add(identity)

    def refresh(
        self, identities: Iterable[ChunkIdentity]
    ) -> list[KVResidencyTransition]:
        """Return ordered readiness transitions for the supplied chunk identities."""
        transitions = []
        for identity in identities:
            if transition := self._update_chunk(identity):
                transitions.append(transition)
        return transitions

    def finish(self, pending: PendingStore) -> list[KVResidencyTransition]:
        """Release one completed/failed store and return any readiness transitions.

        Args:
            pending: The request and transfer IDs previously supplied to bind.

        Duplicate completions and late completions from before reset are harmless.
        """
        identities = self._pending.pop(pending, ())
        for identity in identities:
            self._bindings[identity].pending.discard(pending)
        return self.refresh(identities)

    def commit(
        self, keys: Iterable[ObjectKey], owner: str
    ) -> list[KVResidencyTransition]:
        """Record a committed readable copy per key and return new transitions.

        Args:
            keys: Objects whose actual L1 writes finished successfully.
            owner: Stable L1 tag identifying the committed copies.
        """
        return self._update_owners(keys, owner, removed=False)

    def evict(
        self, keys: Iterable[ObjectKey], owner: str
    ) -> list[KVResidencyTransition]:
        """Remove one owner's copies and return lost-readiness transitions.

        Args:
            keys: Objects actually removed from this L1 owner.
            owner: Stable L1 tag; other owners' copies remain readable.
        """
        return self._update_owners(keys, owner, removed=True)

    def reset(self) -> None:
        """Discard all residency and pending evidence, retaining registered profiles."""
        self._owners.clear()
        self._bindings.clear()
        self._pending.clear()

    def report_status(self) -> dict[str, int]:
        """Return binding, live-object and pending-store counts for the owner."""
        return {
            "bindings": len(self._bindings),
            "live_objects": len(self._owners),
            "pending_stores": len(self._pending),
        }

    def _update_owners(
        self, keys: Iterable[ObjectKey], owner: str, *, removed: bool
    ) -> list[KVResidencyTransition]:
        transitions = []
        for key in keys:
            if removed:
                owners = self._owners.get(key)
                if owners is not None:
                    owners.discard(owner)
                    if not owners:
                        del self._owners[key]
            else:
                self._owners.setdefault(key, set()).add(owner)
            identity = (
                key.model_name,
                ObjectKey.WorldSizeFromKVRank(key.kv_rank),
                key.cache_salt,
                key.chunk_hash,
            )
            if transition := self._update_chunk(identity):
                transitions.append(transition)
        return transitions

    def _update_chunk(self, identity: ChunkIdentity) -> KVResidencyTransition | None:
        state = self._bindings.get(identity)
        if state is None:
            return None
        model, world_size, salt, chunk_hash = identity
        _, required = self._profiles[(model, world_size)]
        keys = [
            ObjectKey(chunk_hash, model, rank, group, salt) for rank, group in required
        ]
        ready = all(self._owners.get(key) for key in keys)
        transition = None
        if ready != state.advertised:
            transition = KVResidencyTransition(
                "stored" if ready else "removed", identity, state.descriptor
            )
            state.advertised = ready
        if not state.pending and not any(self._owners.get(key) for key in keys):
            del self._bindings[identity]
        return transition
