# SPDX-License-Identifier: Apache-2.0
"""CacheBlend reorder planner.

Step 1 rebuilds the longest possible beginning of one cached prompt ``H`` out
of the incoming prompt ``P``'s own tokens.
Step 2 serves that copy followed by the rest of ``P`` in its original order.
The copy is an exact-prefix hit for the KV cache; the planner needs only token
ids (no document boundaries).

Pure Python + numpy.
TODO(Jiayi): Needs to improve speed with numba or cpp extension.
"""

# Standard
from collections import Counter, OrderedDict
from typing import Sequence
import threading
import time

# Third Party
import numpy as np

# Shortest run of P that is copied as one piece. P's last LMIN tokens are never
# copied, so the end of the prompt (the question, the assistant header) stays
# last.
LMIN = 16
# Longest short run (e.g. a separator) copied between two pieces when H has it
# there but P has it elsewhere.
GLUE = 3


class PlanBudgetExceeded(Exception):
    pass


# Window hash: sum of x[j] * _MUL**(LMIN-1-j), mod 2**64.
_MUL = 0x9E3779B97F4A7C15
_POW = np.array([pow(_MUL, LMIN - 1 - j, 1 << 64) for j in range(LMIN)], np.uint64)


def _window_hashes(x: np.ndarray) -> np.ndarray:
    """The hash of every LMIN-token window of ``x``."""
    m = max(len(x) - LMIN + 1, 0)
    xu = x.astype(np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    for j in range(LMIN):
        h = h * np.uint64(_MUL) + xu[j : j + m]
    return h


class _Copier:
    """Step 1 for one prompt ``P`` against any number of candidates ``H``."""

    def __init__(self, P: Sequence[int], deadline: float):
        self.Pn = np.asarray(P, dtype=np.int64)
        self.deadline = deadline
        # P's LMIN-token windows sorted by hash; a piece starts at one. A hash
        # collision is harmless: every run is checked token by token.
        h = _window_hashes(self.Pn)
        self._order = np.argsort(h)
        self._sorted = h[self._order]
        self._check()

    def _starts(self, Hn: np.ndarray, k: int) -> list[int]:
        """Ascending positions of P whose LMIN-token window has the hash of
        H[k : k + LMIN]."""
        if k + LMIN > len(Hn):
            return []
        key = (Hn[k : k + LMIN].astype(np.uint64) * _POW).sum()
        lo = np.searchsorted(self._sorted, key, "left")
        hi = np.searchsorted(self._sorted, key, "right")
        return np.sort(self._order[lo:hi]).tolist()

    def _check(self) -> None:
        if time.monotonic() > self.deadline:
            raise PlanBudgetExceeded

    def _longest(
        self, Hn: np.ndarray, k: int, used: np.ndarray
    ) -> tuple[int, int] | None:
        """(start, length) of the longest unused run of P equal to H[k:], at
        least LMIN tokens; ties keep the leftmost."""
        best = None
        for i in self._starts(Hn, k):
            if used[i]:
                continue
            n = min(len(self.Pn) - i, len(Hn) - k)  # never grows with i
            if best is not None and n <= best[1]:
                break  # later windows can only tie
            self._check()
            stop = np.flatnonzero(
                (self.Pn[i : i + n] != Hn[k : k + n]) | used[i : i + n]
            )
            m = int(stop[0]) if stop.size else n
            if m >= LMIN and (best is None or m > best[1]):
                best = (i, m)
        return best

    def copy(self, Hn: np.ndarray, keep_prefix: int) -> list[tuple[int, int]]:
        """Pieces ``(i, j)`` of P that, laid end to end, equal the beginning
        of H. The first piece is P's first ``keep_prefix`` tokens, in place."""
        used = np.zeros(len(self.Pn), dtype=bool)
        used[:keep_prefix] = True
        used[max(len(self.Pn) - LMIN, 0) :] = True  # the prompt's end stays last
        pieces = [(0, keep_prefix)] if keep_prefix else []
        k = keep_prefix
        while k < len(Hn):
            piece = self._longest(Hn, k, used)
            if piece is None:
                glued = self._glue(Hn, k, used, pieces)  # glue + the piece after it
                if glued is None:
                    break
                (i, j), piece = glued
                pieces.append((i, j))
                used[i:j] = True
                k += j - i
            i, m = piece
            pieces.append((i, i + m))
            used[i : i + m] = True
            k += m
        return pieces

    def _glue(
        self, Hn: np.ndarray, k: int, used: np.ndarray, pieces: list[tuple[int, int]]
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Up to GLUE tokens of P equal to H[k:], then a piece for what follows.
        Glue next to copied text comes first, then the leftmost; it is never
        taken from beyond the copy (P's final run, where the question is)."""
        L = len(self.Pn)
        for gl in range(1, GLUE + 1):
            if k + gl >= len(Hn):
                break
            if not self._starts(Hn, k + gl):
                continue  # no piece can follow
            cands = [
                int(i)
                for i in np.flatnonzero((self.Pn == Hn[k]) & ~used)
                if i + gl <= L
                and not used[i : i + gl].any()
                and np.array_equal(self.Pn[i : i + gl], Hn[k : k + gl])
            ]
            cands.sort(
                key=lambda i: (
                    not ((i > 0 and used[i - 1]) or (i + gl < L and used[i + gl])),
                    i,
                )
            )
            for i in cands:
                self._check()
                used[i : i + gl] = True  # the next piece must not reuse the glue
                nxt = self._longest(Hn, k + gl, used)
                used[i : i + gl] = False
                if nxt and i + gl <= max([j for _, j in pieces] + [nxt[0] + nxt[1]]):
                    return (i, i + gl), nxt
        return None


def plan(
    P: Sequence[int],
    candidates: Sequence[Sequence[int]],
    chunk_size: int,
    baseline_chunks: int,
    deadline: float,
    keep_prefix: int = 0,
) -> list[int] | None:
    """Order P so that it starts with the longest copy of one candidate's
    beginning, then the rest of P in its original order.

    Candidates come newest first. The best copies the most whole chunks, then
    the most tokens, with the fewest pieces; ties keep the newest. Only
    candidates that share P's first ``keep_prefix`` tokens are copied, and
    those tokens stay in place. A plan must copy more whole chunks than
    ``baseline_chunks``, the exact prefix P already has; a tie keeps P.

    Returns:
        The order to serve P in (``[P[i] for i in perm]``), or None to keep
        P as it is.
    """
    best: tuple[tuple[int, int, int], list[tuple[int, int]], np.ndarray] | None = None
    try:
        copier = _Copier(P, deadline)
        seen: set[bytes] = set()
        for H in candidates:
            Hn = np.asarray(H)  # the store keeps uint32; numpy compares across types
            key = Hn.tobytes()
            if key in seen or not np.array_equal(
                Hn[:keep_prefix], copier.Pn[:keep_prefix]
            ):
                continue
            seen.add(key)
            pieces = copier.copy(Hn, keep_prefix)
            k = sum(j - i for i, j in pieces)
            score = (k // chunk_size, k, -len(pieces))
            if best is None or score > best[0]:
                best = (score, pieces, Hn)
    except PlanBudgetExceeded:
        pass  # out of time: a finished plan still counts
    if best is None or best[0][0] <= baseline_chunks:
        return None
    (_, k, _), pieces, Hn = best
    used = np.zeros(len(P), dtype=bool)
    perm: list[int] = []
    for i, j in pieces:
        perm.extend(range(i, j))
        used[i:j] = True
    # Then H's next token (often a separator), if P has it beside a cut before
    # the copy's end: served prompts keep that junction for later copies.
    if k < len(Hn):
        for i in range(max(j for _, j in pieces)):
            if (
                not used[i]
                and P[i] == Hn[k]
                and ((i > 0 and used[i - 1]) or (i + 1 < len(P) and used[i + 1]))
            ):
                perm.append(i)
                used[i] = True
                break
    perm += np.flatnonzero(~used).tolist()
    return perm if perm != list(range(len(P))) else None


NS = tuple[str, int, str]  # namespace: (model_name, world_size, cache_salt)


class PromptStore:
    """Token ids of stored prompts per namespace, LRU-bounded by their total
    tokens, with an owner map from each chunk's chain hash to the newest
    prompt that recorded it."""

    def __init__(self, max_tokens: int = 1 << 24):
        self._lock = threading.Lock()
        self._max_tokens, self._tokens, self._seq = max_tokens, 0, 0
        # (ns, request_id) -> [seq, token ids, chain hashes]
        self._entries: "OrderedDict[tuple, list]" = OrderedDict()
        self._owner: dict[tuple, tuple] = {}  # (ns, chain hash) -> (ns, request_id)
        # cache_salt -> (model_name, world_size) -> [its number of prompts, its
        # newest (ns, request_id)]; a namespace goes with its last prompt.
        self._ns: dict[str, dict[tuple[str, int], list]] = {}

    def record(
        self,
        ns: NS,
        request_id: str,
        token_ids: Sequence[int],
        chain_hashes: Sequence[bytes],
    ) -> None:
        """Called on every store that committed KV. The first call's
        ``token_ids`` are the prompt (later calls may carry decode tokens);
        ``chain_hashes`` must be the request's chain from chunk 0."""
        key = (ns, request_id)
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                self._seq += 1
                ids = np.asarray(token_ids, dtype=np.uint32)
                e = self._entries[key] = [self._seq, ids, set()]
                self._tokens += len(ids)
                n = self._ns.setdefault(ns[2], {}).setdefault(ns[:2], [0, key])
                n[0] += 1
                n[1] = key
            self._entries.move_to_end(key)
            for h in chain_hashes:
                e[2].add(h)
                self._owner[(ns, h)] = key
            while len(self._entries) > 1 and self._tokens > self._max_tokens:
                old, (_, ids, hashes) = self._entries.popitem(last=False)
                self._tokens -= len(ids)
                for h in hashes:
                    if self._owner.get((old[0], h)) == old:
                        del self._owner[(old[0], h)]
                salt, name = old[0][2], old[0][:2]
                n = self._ns[salt][name]
                n[0] -= 1
                if not n[0]:  # the namespace's last prompt
                    del self._ns[salt][name]
                    if not self._ns[salt]:
                        del self._ns[salt]

    def resolve(self, model_name: str, world_size: int, cache_salt: str) -> NS | None:
        """The one recorded namespace (model_name, world_size, cache_salt) that
        matches the caller: the salt must be equal; an empty model_name or a
        zero world_size match any. None if zero or several match."""
        with self._lock:
            hits = [
                (m, w, cache_salt)
                for m, w in self._ns.get(cache_salt, ())
                if (not model_name or m == model_name)
                and (not world_size or w == world_size)
            ]
        return hits[0] if len(hits) == 1 else None

    def candidates(
        self,
        ns: NS,
        hit_hashes: Sequence[bytes],
        prompt_chain: Sequence[bytes],
        top_k: int = 4,
    ) -> tuple[list[np.ndarray], int]:
        """Candidates for prompt P, newest first: the ``top_k`` prompts owning
        the most fingerprint hits, the prompt holding P's own exact prefix, and
        the newest recorded prompt (its fingerprints may still be draining).
        Also returns P's own exact prefix in chunks (the baseline)."""
        with self._lock:
            votes = Counter(
                self._owner[(ns, h)] for h in hit_hashes if (ns, h) in self._owner
            )
            keys = set(
                sorted(votes, key=lambda x: (-votes[x], -self._entries[x][0]))[:top_k]
            )
            base = 0
            while base < len(prompt_chain) and (ns, prompt_chain[base]) in self._owner:
                base += 1
            if base:
                keys.add(self._owner[(ns, prompt_chain[base - 1])])
            newest = self._ns.get(ns[2], {}).get(ns[:2])
            if newest and newest[1] in self._entries:
                keys.add(newest[1])
            entries = sorted((self._entries[x] for x in keys), key=lambda e: -e[0])
            return [e[1] for e in entries], base
