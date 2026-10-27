# SPDX-License-Identifier: Apache-2.0
"""Per-request ownership record for the read locks a lookup acquired."""

# Standard
from dataclasses import dataclass, field, replace
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

logger = init_logger(__name__)


@dataclass
class LookupLockState:
    """Read-lock bookkeeping for one request's most recent lookup.

    Attributes:
        key: The lookup's worker-less cache key, or ``None`` when only a
            store resolved object keys for the request.
        hit_chunks: Model-wide hit length in chunks; ``-1`` until the
            prefetch result is recorded.
        locked_gids: Object group ids the prefetch read-locked.
        group_windows: Sliding-window size (in chunks) per object group.
        owners: L1 owners whose read locks the lookup retained, for
            multi-L1 deployments.
        generation: Incremented by every :meth:`LookupLockLedger.begin_lookup`,
            so a stale response cannot claim a newer lookup's release.
        failed_releases: Instances whose failed RETRIEVE already released its
            share, as ``(generation, instance_id, worker_id, start, end)``.
    """

    key: IPCCacheServerKey | None = None
    hit_chunks: int = -1
    locked_gids: tuple[int, ...] = ()
    group_windows: tuple[int, ...] = ()
    owners: dict[ObjectKey, int] = field(default_factory=dict)
    generation: int = 0
    failed_releases: set[tuple[int, int, int, int, int]] = field(default_factory=set)


class LookupLockLedger:
    """Thread-safe per-request record of what a lookup read-locked.

    A scheduler lookup reserves one read lock per KV worker (or one reader
    count per TP worker for MLA) on every key it prefetched. Those locks are
    anonymous refcounts in L1, so releasing them requires knowing exactly
    which keys the lookup locked: the hit length, the locked object groups
    and each group's sliding window. This ledger is that record, keyed by
    request id, and it is the only place a release may be derived from.

    Entries live until :meth:`release` drops them: a request that ends
    normally is released by ``end_session``, and one that is abandoned is
    released when its session is destroyed, so the ledger cannot outlive the
    request it describes.
    """

    def __init__(self) -> None:
        self._states: dict[str, LookupLockState] = {}
        self._lock = threading.Lock()

    def ensure(self, request_id: str, key: IPCCacheServerKey) -> None:
        """Record the lookup key of a request that has no ledger entry yet.

        Used when object keys are resolved without a lookup (a store), so
        ``end_session`` can still touch what the request wrote.

        Args:
            request_id: Unique request identifier.
            key: Cache key describing model/tokens/range; stored without its
                worker id.
        """
        with self._lock:
            if request_id not in self._states:
                self._states[request_id] = LookupLockState(
                    key=key.no_worker_id_version()
                )

    def begin_lookup(
        self,
        request_id: str,
        key: IPCCacheServerKey,
        group_windows: tuple[int, ...],
    ) -> None:
        """Record a new lookup and reset its per-lookup release state.

        Args:
            request_id: Unique request identifier.
            key: The lookup's cache key; stored without its worker id.
            group_windows: Sliding-window size (in chunks) per object group.
        """
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                self._states[request_id] = LookupLockState(
                    key=key.no_worker_id_version(),
                    group_windows=group_windows,
                    generation=1,
                )
                return
            state.key = key.no_worker_id_version()
            state.hit_chunks = -1
            state.locked_gids = ()
            state.group_windows = group_windows
            state.owners = {}
            state.generation += 1
            state.failed_releases.clear()

    def record_prefetch_result(
        self,
        request_id: str,
        hit_chunks: int,
        locked_gids: tuple[int, ...],
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> None:
        """Record the lock set acquired by the request's current lookup.

        Args:
            request_id: Unique request identifier.
            hit_chunks: Model-wide hit length in chunks.
            locked_gids: Object group ids the prefetch read-locked.
            l1_owners: L1 owners whose read locks the lookup retained.
        """
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                logger.warning(
                    "Prefetch result for request %s recorded without a lookup",
                    request_id,
                )
                return
            state.hit_chunks = hit_chunks
            state.locked_gids = locked_gids
            state.owners = dict(l1_owners) if l1_owners is not None else {}

    def lock_state(self, request_id: str) -> LookupLockState | None:
        """Return a snapshot of a request's lock state, or ``None``.

        Args:
            request_id: Unique request identifier.

        Returns:
            A copy of the recorded state, so callers cannot mutate the
            ledger's own bookkeeping.
        """
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return None
            return replace(state, owners=dict(state.owners))

    def get_prefetch_owners(self, request_id: str) -> dict[ObjectKey, int]:
        """Return the L1 owners whose read locks a lookup retained.

        Args:
            request_id: Unique request identifier.

        Returns:
            A copy of the retained owners; empty when the request has no
            recorded lookup.
        """
        with self._lock:
            state = self._states.get(request_id)
            return dict(state.owners) if state is not None else {}

    def release(self, request_id: str) -> LookupLockState | None:
        """Drop a request's lock state and return it.

        Args:
            request_id: Unique request identifier.

        Returns:
            The removed state, or ``None`` when nothing was recorded.
        """
        with self._lock:
            return self._states.pop(request_id, None)

    def clear(self) -> None:
        """Drop every recorded request; used when the server rebinds state."""
        with self._lock:
            self._states.clear()

    def prepare_failed_retrieve_release(
        self,
        request_id: str,
        key: IPCCacheServerKey,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
        """Return a stable snapshot for a failed worker's lock release.

        Args:
            request_id: Unique request identifier.
            key: The failing RETRIEVE's cache key.

        Returns:
            The hit chunks, locked group ids, group windows and lookup
            generation, or ``None`` when the request has no lookup that
            covers ``key``'s range.
        """
        if key.worker_id is None:
            return None

        with self._lock:
            state = self._states.get(request_id)
            if state is None or state.key is None or state.hit_chunks < 0:
                return None
            if not self._covers(state.key, key):
                return None
            return (
                state.hit_chunks,
                state.locked_gids,
                state.group_windows,
                state.generation,
            )

    def claim_failed_retrieve_release(
        self,
        request_id: str,
        instance_id: int,
        key: IPCCacheServerKey,
        lookup_generation: int,
    ) -> bool:
        """Atomically claim one failed worker's prepared lock release.

        A scheduler lookup acquires read locks for every KV worker (or one
        reader count per TP worker for MLA), while RETRIEVE responses are per
        worker instance. When a worker has lost its GPU registration, only
        that instance's share may be released, and only once.

        Args:
            request_id: Unique request identifier.
            instance_id: Worker instance whose RETRIEVE failed.
            key: The failing RETRIEVE's cache key.
            lookup_generation: Generation returned by
                :meth:`prepare_failed_retrieve_release`.

        Returns:
            ``True`` when this instance's release was claimed, ``False`` when
            it was already claimed or the ledger cannot prove that the
            request owns ``key``'s range. Leaving the lock to its TTL is
            safer than decrementing a concurrent request's anonymous L1 count.
        """
        if key.worker_id is None:
            return False

        with self._lock:
            state = self._states.get(request_id)
            if state is None or state.key is None:
                return False
            if lookup_generation != state.generation:
                return False
            if not self._covers(state.key, key):
                return False

            owner = (
                lookup_generation,
                instance_id,
                key.worker_id,
                key.start,
                key.end,
            )
            if owner in state.failed_releases:
                return False
            state.failed_releases.add(owner)
            return True

    @staticmethod
    def _covers(lookup_key: IPCCacheServerKey, key: IPCCacheServerKey) -> bool:
        """Return whether ``key``'s range lies inside the lookup's range."""
        return (
            key.model_name == lookup_key.model_name
            and key.world_size == lookup_key.world_size
            and key.token_bytes == lookup_key.token_bytes
            and key.cache_salt == lookup_key.cache_salt
            and key.start >= lookup_key.start
            and key.end <= lookup_key.end
        )
