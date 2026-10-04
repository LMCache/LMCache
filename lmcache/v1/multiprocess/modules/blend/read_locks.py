# SPDX-License-Identifier: Apache-2.0
"""The sparse lookup's read-lock reservation and its per-request ledger."""

# Standard
from collections import Counter
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import CBMatchResult, IPCCacheServerKey


@dataclass
class ReadLockReservation:
    """The read locks one request's sparse lookup took, and which it still holds.

    The lookup read-locks every object key of every match it finds,
    ``read_locks`` times per key (one per reader). A request may then be
    retrieved over several calls, one per engine prefill chunk, so the
    reservation lives on the session. ``held`` counts, per key, the locks the
    request still holds and nobody has claimed. Every lock leaves it exactly
    once: a retrieve claims one per key it reads (and then owns its release),
    a sweep releases a match no later call can send, and session end releases
    the rest. A key whose count is 0 is never read or released again.

    Not thread-safe: the owner (``BlendModule``) guards every call with
    ``_cb_retain_lock``.
    """

    #: Locks the lookup took per key (``IPCCacheServerKey.num_kv_readers``).
    read_locks: int
    #: Every rank's object keys per match hash, group-major and rank-minor.
    per_hash: dict[bytes, list[ObjectKey]]
    #: Where each match ends in the prompt (``CBMatchResult.cur_ed``).
    ends: dict[bytes, int]
    #: The L1 instance holding each key (multi-L1), or None.
    l1_owners: dict[ObjectKey, int] | None = None
    #: Locks still held per key, unclaimed.
    held: dict[ObjectKey, int] = field(init=False)

    def __post_init__(self) -> None:
        self.held = {
            k: self.read_locks for keys in self.per_hash.values() for k in keys
        }

    def rank_keys(
        self,
        cb_match_result: list[CBMatchResult],
        key: IPCCacheServerKey,
        n_read: int,
    ) -> list[list[ObjectKey] | None]:
        """This rank's stashed keys for each match, or None if not stashed.

        :param cb_match_result: The matches a retrieve consumes.
        :param key: The retrieve's cache key (``worker_id`` / ``world_size``).
        :param n_read: Read groups per hash.
        :return: Per match, ``n_read`` keys, or None when the reservation has
            no keys for this rank (the caller derives them).
        """
        if key.worker_id is not None and key.world_size > 1:
            ws, rank = key.world_size, key.worker_id
        else:
            ws, rank = 1, 0
        out: list[list[ObjectKey] | None] = []
        for r in cb_match_result:
            keys = self.per_hash.get(r.hash)
            if keys is None:
                out.append(None)
                continue
            idx = (
                [g * ws + rank for g in range(n_read)]
                if ws > 1
                else list(range(len(keys)))
            )
            out.append(
                [keys[i] for i in idx] if all(i < len(keys) for i in idx) else None
            )
        return out

    def claim(self, keys: list[ObjectKey]) -> bool:
        """Claim one held lock per occurrence of each key, all or nothing.

        :param keys: The keys one retrieve reads.
        :return: True if every key had a lock left for each of its
            occurrences; the caller then owns those locks and must release or
            :meth:`unclaim` them. False (and nothing claimed) otherwise.
        """
        need = Counter(keys)
        if any(self.held.get(k, 0) < n for k, n in need.items()):
            return False
        for k, n in need.items():
            self.held[k] -= n
        return True

    def unclaim(self, keys: list[ObjectKey]) -> None:
        """Hand back claimed locks a retrieve did not use (still held)."""
        for k in keys:
            self.held[k] = self.held.get(k, 0) + 1

    def sweep(
        self, sent: set[bytes], upto: int, final: bool
    ) -> dict[int, list[ObjectKey]]:
        """Release the matches no later retrieve can send.

        A client sends each match at most once, in the call whose window
        holds it whole, and calls for a request in increasing window order.
        So a match this call was not sent is unreachable once it ends at or
        before the last match this call was sent (``upto``), and every unsent
        match is unreachable on the final call (``final``).

        :param sent: Hashes this retrieve was sent.
        :param upto: The largest ``cur_ed`` among them.
        :param final: Whether the engine has allocated the whole prompt.
        :return: The locks to release, grouped by count per key.
        """
        out: dict[int, list[ObjectKey]] = {}
        for h in [h for h in self.per_hash if h not in sent]:
            if not final and self.ends[h] > upto:
                continue
            for k in self.per_hash.pop(h):
                n = self.held.pop(k, 0)
                if n:
                    out.setdefault(n, []).append(k)
        return out

    def release_all(self) -> dict[int, list[ObjectKey]]:
        """Give up every lock still held (session end, superseded lookup).

        :return: The locks to release, grouped by count per key.
        """
        out: dict[int, list[ObjectKey]] = {}
        for k, n in self.held.items():
            if n:
                out.setdefault(n, []).append(k)
        self.held.clear()
        self.per_hash.clear()
        return out
