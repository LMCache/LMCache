# SPDX-License-Identifier: Apache-2.0
"""CacheBlend reorder planner.

Step 1 rebuilds the longest possible beginning of one cached prompt ``H`` out
of the incoming prompt ``P``'s own tokens.
Step 2 serves that copy followed by
the rest of ``P`` in its original order. The copy is an exact-prefix hit for
the KV cache; the planner needs only token ids (no document boundaries).

Pure Python + numpy.
"""

# Standard
from collections import Counter, OrderedDict
from functools import cached_property
from typing import Sequence
import threading
import time

# Third Party
import numpy as np

LMIN = 16  # shortest long piece (also the anchor length)
GLUE = 3  # longest glue run (e.g. a separator) between two long pieces
MAX_GLUE_CANDIDATES = 8  # glue positions tried per (k, glue length)
MAX_PLAN_TOKENS = 32768  # longer prompts are returned unchanged ("too_long")


class PlanBudgetExceeded(Exception):
    pass


class _Copier:
    """Step 1 for one prompt ``P`` against any number of candidates ``H``."""

    def __init__(self, P: Sequence[int], deadline: float):
        self.P = list(P)
        self.Pn = np.asarray(self.P, dtype=np.int64)
        self.deadline = deadline
        # LMIN-gram anchors (keyed by raw bytes): a long piece starts at one.
        self.anchors: dict[bytes, list[int]] = {}
        pb, w = self.Pn.tobytes(), self.Pn.itemsize
        for i in range(len(self.P) - LMIN + 1):
            if not i % 8192:
                self._check()
            self.anchors.setdefault(pb[w * i : w * (i + LMIN)], []).append(i)

    @cached_property
    def tpos(self) -> dict[int, list[int]]:
        """Positions of each token; only the glue path needs it."""
        tpos: dict[int, list[int]] = {}
        for i, t in enumerate(self.P):
            tpos.setdefault(t, []).append(i)
        return tpos

    def _check(self) -> None:
        if time.monotonic() > self.deadline:
            raise PlanBudgetExceeded

    def _longest(self, H, Hn, k, used, exclude=None):
        """Longest unused run of P equal to H[k:] with >= LMIN tokens."""
        best = None
        for i in self.anchors.get(Hn[k : k + LMIN].tobytes(), ()):
            if used[i] or (exclude and exclude[0] <= i < exclude[1]):
                continue
            n = min(len(self.P) - i, len(H) - k)  # never grows with i
            if best is not None and n <= best[1]:
                break  # later anchors can only tie, and ties keep the leftmost
            self._check()
            if exclude and i < exclude[0]:
                n = min(n, exclude[0] - i)
            stop = np.flatnonzero(
                (self.Pn[i : i + n] != Hn[k : k + n]) | used[i : i + n]
            )
            m = int(stop[0]) if stop.size else n
            if m >= LMIN and (best is None or m > best[1]):
                best = (i, m)
        return best

    def copy(self, Hn: np.ndarray, seam_rule: bool = False):
        """Returns (k, pieces): pieces (i, j, h) with P[i:j] == H[h:h+j-i],
        laid end to end from h = 0, reproduce H[:k]."""
        P, L = self.P, len(self.P)
        H = Hn.tolist()
        used = np.zeros(L, dtype=bool)
        k = 0
        pieces: list[tuple[int, int, int]] = []

        def add_long(i, m, h):
            if seam_rule and pieces:
                # Tokens just before the new piece that also end the copy so far
                # (a label, a separator) go to the new piece.
                pi, pj, ph = pieces[-1]
                t = 0
                while (
                    t < pj - pi - 1
                    and i - t - 1 >= 0
                    and not used[i - t - 1]
                    and P[i - t - 1] == H[h - t - 1]
                ):
                    t += 1
                if t:
                    pieces[-1] = (pi, pj - t, ph)
                    used[pj - t : pj] = False
                    i, m, h = i - t, m + t, h - t
            pieces.append((i, i + m, h))
            used[i : i + m] = True
            return h + m

        while k < len(H):
            self._check()
            b = self._longest(H, Hn, k, used)
            if b:
                k = add_long(b[0], b[1], k)
                continue
            took = False
            for gl in range(1, GLUE + 1):  # glue only if a long piece follows
                if k + gl >= len(H):
                    break
                if Hn[k + gl : k + gl + LMIN].tobytes() not in self.anchors:
                    continue  # no long piece can follow this glue
                cands = [
                    i
                    for i in self.tpos.get(H[k], ())
                    if i + gl <= L
                    and not used[i : i + gl].any()
                    and P[i : i + gl] == H[k : k + gl]
                ]
                cands.sort(
                    key=lambda i: (
                        not ((i > 0 and used[i - 1]) or (i + gl < L and used[i + gl])),
                        i,
                    )
                )
                for i in cands[:MAX_GLUE_CANDIDATES]:
                    self._check()
                    b = self._longest(H, Hn, k + gl, used, exclude=(i, i + gl))
                    # never borrow glue from P's final run (where the question is)
                    if b and i + gl <= max([j for _, j, _ in pieces] + [b[0] + b[1]]):
                        pieces.append((i, i + gl, k))
                        used[i : i + gl] = True
                        k = add_long(b[0], b[1], k + gl)
                        took = True
                        break
                if took:
                    break
            if not took:
                break
        return k, pieces


def build_order(P: Sequence[int], H: Sequence[int], pieces) -> list[int]:
    """Step 3: the copied pieces, H's next token if it sits beside a cut,
    then every unused position of P in original order (a permutation)."""
    L = len(P)
    used = np.zeros(L, dtype=bool)
    perm: list[int] = []
    for i, j, _ in pieces:
        perm.extend(range(i, j))
        used[i:j] = True
    k = len(perm)
    if pieces and k < len(H):
        fr = max(j for _, j, _ in pieces)  # P[fr:] is P's final run
        for i in range(fr):
            if (
                not used[i]
                and P[i] == H[k]
                and ((i > 0 and used[i - 1]) or (i + 1 < L and used[i + 1]))
            ):
                perm.append(i)
                used[i] = True
                break
    perm.extend(int(i) for i in np.flatnonzero(~used))
    return perm


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if diff.size else n


def plan(
    P: Sequence[int],
    candidates: Sequence[Sequence[int]],
    chunk_size: int,
    baseline_chunks: int = 0,
    gain_only: bool = False,
    seam_rule: bool = False,
    budget_s: float = 0.2,
    keep_prefix: int = 0,
) -> tuple[list[int], dict]:
    """Choose the candidate (NEWEST FIRST) whose beginning P can rebuild
    furthest; score = (whole chunks copied, tokens copied, -pieces), ties keep
    the newest. Returns (perm, info); served = [P[i] for i in perm].

    ``baseline_chunks`` is the exact prefix P already has: a plan never serves
    fewer whole chunks than that. The first ``keep_prefix`` tokens never move
    (e.g. vLLM's own prefix-cache hit), so only candidates that share them can
    be copied."""
    ident = list(range(len(P)))
    info = {
        "copy_tokens": 0,
        "exact_chunks": 0,
        "baseline_chunks": baseline_chunks,
        "candidates": len(candidates),
        "reason": "",
    }
    if not candidates:
        return ident, dict(info, reason="no_candidates")
    if len(P) > MAX_PLAN_TOKENS:
        return ident, dict(info, reason="too_long")
    best, seen, expired = None, set(), False
    try:
        copier = _Copier(P, time.monotonic() + budget_s)
        for H in candidates:
            Hn = np.asarray(H, dtype=np.int64)
            key = Hn.tobytes()
            if key in seen or _lcp(copier.Pn, Hn) < keep_prefix:
                continue
            seen.add(key)
            k, pieces = copier.copy(Hn, seam_rule)
            score = (k // chunk_size, k, -len(pieces))
            if best is None or score > best[0]:
                best = (score, Hn, k, pieces)
    except PlanBudgetExceeded:
        # Keep a finished plan only if it beats P's own prefix: the candidate
        # holding that prefix may not have been scored yet.
        expired = True
        if best is None or best[2] // chunk_size <= baseline_chunks:
            return ident, dict(info, reason="budget")
    if best is None:  # every candidate was filtered out (keep_prefix)
        return ident, dict(info, reason="no_candidates")
    _, Hn, k, pieces = best
    info.update(copy_tokens=k, exact_chunks=k // chunk_size, budget_hit=expired)
    if k == 0 or k // chunk_size < baseline_chunks:
        return ident, dict(info, reason="no_gain")
    if gain_only and k // chunk_size <= baseline_chunks:
        return ident, dict(info, reason="no_gain")
    perm = build_order(P, Hn, pieces)
    if sorted(perm) != ident or perm[:keep_prefix] != ident[:keep_prefix]:
        return ident, dict(info, reason="error:invalid_perm")
    return perm, info


NS = tuple[str, int, str]  # namespace: (model_name, world_size, cache_salt)


class PromptStore:
    """Token ids of stored prompts, per namespace, LRU-bounded by entries and
    by total tokens, with an owner map from each chunk's chain hash to the
    newest prompt that recorded it."""

    def __init__(self, max_prompts: int = 65536, max_tokens: int = 1 << 24):
        self._lock = threading.Lock()
        self._max, self._max_tokens, self._tokens = max_prompts, max_tokens, 0
        self._seq = 0
        # (ns, request_id) -> [seq, token ids, chain hashes]
        self._entries: "OrderedDict[tuple, list]" = OrderedDict()
        self._owner: dict[tuple, tuple] = {}  # (ns, chain hash) -> (ns, request_id)
        self._newest: dict[NS, tuple] = {}  # ns -> newest (ns, request_id)

    def record(
        self,
        ns: NS,
        request_id: str,
        token_ids: Sequence[int],
        chain_hashes: Sequence[bytes],
    ) -> None:
        """Called on every successful store of a request. The first call's
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
                self._newest[ns] = key
            self._entries.move_to_end(key)
            for h in chain_hashes:
                e[2].add(h)
                self._owner[(ns, h)] = key
            while len(self._entries) > 1 and (
                len(self._entries) > self._max or self._tokens > self._max_tokens
            ):
                old, (_, ids, hashes) = self._entries.popitem(last=False)
                self._tokens -= len(ids)
                for h in hashes:
                    if self._owner.get((old[0], h)) == old:
                        del self._owner[(old[0], h)]
                if self._newest.get(old[0]) == old:
                    del self._newest[old[0]]

    def resolve(self, model_name: str, world_size: int, cache_salt: str) -> NS | None:
        """The one recorded namespace (model_name, world_size, cache_salt) that
        matches the caller: the salt must be equal; an empty model_name or a
        zero world_size match any. None if zero or several match."""
        with self._lock:
            hits = [
                ns
                for ns in self._newest
                if ns[2] == cache_salt
                and (not model_name or ns[0] == model_name)
                and (not world_size or ns[1] == world_size)
            ]
        return hits[0] if len(hits) == 1 else None

    def candidates(
        self,
        ns: NS,
        hit_hashes: Sequence[bytes],
        prompt_chain: Sequence[bytes],
        top_k: int,
    ) -> tuple[list[np.ndarray], int]:
        """Candidates for prompt P, newest first: the top_k prompts owning the
        most fingerprint hits, the prompt holding P's own exact prefix, and the
        newest recorded prompt (its fingerprints may still be draining).
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
            if ns in self._newest:
                keys.add(self._newest[ns])
            entries = sorted((self._entries[x] for x in keys), key=lambda e: -e[0])
            return [e[1] for e in entries], base

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._owner.clear()
            self._newest.clear()
            self._tokens = 0
