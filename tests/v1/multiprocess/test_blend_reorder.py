# SPDX-License-Identifier: Apache-2.0
"""Reorder planner ("copy one cached prompt"): pure-Python unit tests."""

# Standard
import random
import time

# First Party
from lmcache.v1.multiprocess.modules.blend import reorder as br

C = 32  # chunk size for these tests
SYS = list(range(1, C + 1))  # a one-chunk "system prompt"
SEP = [7]
Q = [9] * br.LMIN  # the question: the prompt's last LMIN tokens


def _doc(seed: int, n: int = 40) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 50000) for _ in range(n)]


def _prompt(*docs: list[int], question: list[int] = Q) -> list[int]:
    out = list(SYS)
    for i, d in enumerate(docs):
        out += d + (SEP if i + 1 < len(docs) else [])
    return out + question


def _serve(
    P: list[int], cached: list[list[int]], baseline_chunks: int = 0, **kw
) -> list[int]:
    kw.setdefault("deadline", time.monotonic() + 5)
    perm = br.plan(P, cached, C, baseline_chunks, **kw)
    if perm is None:
        return P
    assert sorted(perm) == list(range(len(P)))
    return [P[i] for i in perm]


def test_copies_cached_prompt_from_token_0():
    a, b, x, y = _doc(1), _doc(2), _doc(3), _doc(4)
    cached = _prompt(a, b, x)
    P = _prompt(y, b, a, question=[8] * br.LMIN)
    served = _serve(P, [cached])
    # [y][b][a] -> [a][b][y]: the start of `cached`, then the rest in order
    assert served[: len(SYS) + 2 * 41] == cached[: len(SYS) + 2 * 41]
    assert served[-br.LMIN :] == P[-br.LMIN :]  # the question stays last


def test_no_move_when_the_first_slot_differs():
    # cached [x][d][x'], new [y][y'][d]: moving d alone cannot extend the
    # exact prefix, so the prompt is served unchanged.
    d, x, x2, y, y2 = (_doc(s) for s in range(10, 15))
    P = _prompt(y, y2, d)
    served = _serve(P, [_prompt(x, d, x2)])
    assert served == P


def test_moves_to_slot_2_when_slot_1_matches():
    d, x, x2, y = (_doc(s) for s in range(20, 24))
    served = _serve(_prompt(x, y, d), [_prompt(x, d, x2)])
    n = len(SYS) + len(x) + 1 + len(d)  # [sys][x][sep][d]
    assert served[:n] == _prompt(x, d, x2)[:n]


def test_identity_cases():
    P = _prompt(_doc(30), _doc(31))
    assert _serve(P, []) == P
    assert _serve(P, [_prompt(_doc(32))]) == P  # nothing in common


def test_the_prompt_end_stays_last():
    # The cached prompt ends with the same text as P (same question, more
    # documents): copying it whole would move P's end into the middle.
    a, b, tail = _doc(35), _doc(36), _doc(37, n=24)
    P = SYS + a + b + tail
    served = _serve(P, [SYS + a + tail])
    assert served[-len(tail) :] == tail


def test_budget_returns_identity():
    a, b = _doc(40), _doc(41)
    P = _prompt(b, a)
    assert _serve(P, [_prompt(a, b)], deadline=time.monotonic() - 1) == P


def test_a_plan_must_gain_a_whole_chunk():
    a, b = _doc(60), _doc(61)
    P, cached = _prompt(b, a), _prompt(a, b, _doc(62))
    served = _serve(P, [cached])
    same = [x == y for x, y in zip(served, cached, strict=False)]
    n = same.index(False) // C  # the whole chunks the plan copies
    assert n > 0
    assert _serve(P, [cached], baseline_chunks=n - 1) == served
    assert _serve(P, [cached], baseline_chunks=n) == P  # a tie keeps P


def test_budget_expiry_keeps_a_finished_plan_only_if_it_gains(monkeypatch):
    a, b, x = _doc(80), _doc(81), _doc(82)
    P = _prompt(b, a)
    good, other = _prompt(a, b), _prompt(x, a)
    real_copy, calls = br._Copier.copy, []

    def copy_then_expire(self, Hn, keep_prefix):
        calls.append(1)
        if len(calls) > 1:
            raise br.PlanBudgetExceeded
        return real_copy(self, Hn, keep_prefix)

    monkeypatch.setattr(br._Copier, "copy", copy_then_expire)
    served = _serve(P, [good, other])  # newest (good) finished first
    assert served[: len(SYS) + len(a)] == good[: len(SYS) + len(a)]
    calls.clear()
    assert _serve(P, [good, other], baseline_chunks=10**6) == P


def test_keep_prefix_never_moves_the_kept_tokens():
    a, b, x, y = _doc(90), _doc(91), _doc(92), _doc(93)
    P = _prompt(x, b, a)
    # copying `cached` would keep [sys][x] and move a before b
    cached = _prompt(x, a, b)
    keep = len(SYS) + len(x)
    served = _serve(P, [cached], keep_prefix=keep)
    assert (
        served[:keep] == P[:keep]
        and served[: keep + 1 + len(a)] == cached[: keep + 1 + len(a)]
    )
    # a candidate that differs inside the kept prefix is never used
    served = _serve(P, [_prompt(y, a, b)], keep_prefix=keep)
    assert served == P


def test_keep_prefix_with_a_repeated_block():
    # P repeats its kept prefix later, followed by the cached prompt's next
    # text: the kept tokens stay in place and the copy continues after them.
    s, x, a, b = SYS + _doc(94), _doc(95), _doc(96), _doc(97)
    P = s + x + s + a + Q
    served = _serve(P, [s + a + b], keep_prefix=len(s))
    assert served[: len(s) + len(a)] == s + a


def test_prompt_store_candidates():
    store = br.PromptStore(max_tokens=6)
    store.record(("m", 1, ""), "r1", [1, 2, 3], [b"h0", b"h1"])
    store.record(("m", 1, ""), "r2", [4, 5, 6], [b"h0", b"g1", b"g2"])
    # a later store of r2 carries decode tokens: its ids stay the prompt's
    store.record(("m", 1, ""), "r2", [4, 5, 6, 7], [b"g3"])
    ns = ("m", 1, "")
    cands, base = store.candidates(ns, [b"g2"], [b"h0", b"h1", b"zz"], top_k=1)
    # r1 holds P's own 2-chunk prefix, r2 owns the hit (and is the newest)
    assert [list(c) for c in cands] == [[4, 5, 6], [1, 2, 3]] and base == 2
    store.record(ns, "r3", [8], [b"h1"])  # 7 tokens > 6: evicts r1 (LRU)
    cands, base = store.candidates(ns, [], [b"h0", b"h1"])
    assert [list(c) for c in cands] == [[8]] and base == 2  # holder r3 is also newest
    assert store.candidates(("other", 1, ""), [b"g2"], []) == ([], 0)


def test_prompt_store_namespaces():
    store = br.PromptStore(max_tokens=10)
    ns, salted = ("m", 1, ""), ("m", 1, "salt")
    store.record(ns, "r1", [1] * 3, [b"a"])
    store.record(ns, "r2", [2] * 3, [b"b"])  # the namespace's newest
    store.record(salted, "r3", [3] * 3, [b"c"])
    assert store.resolve("", 0, "") == ns  # salt must match exactly
    assert store.resolve("", 0, "salt") == salted
    assert store.resolve("m", 2, "") is None and store.resolve("x", 0, "") is None
    store.record(ns, "r1", [1] * 4, [b"d"])  # touch r1: r2 is now the LRU
    store.record(salted, "r4", [4] * 3, [b"e"])  # 12 > 10: evicts r2
    # the namespace stays while it holds a prompt (r1); its newest is gone
    assert store.resolve("", 0, "") == ns
    assert [list(c) for c in store.candidates(ns, [b"a"], [])[0]] == [[1] * 3]
    store.record(salted, "r5", [5] * 6, [b"f"])  # evicts r3, then r1: ns's last
    assert store.resolve("", 0, "") is None
    assert store.candidates(ns, [b"a"], []) == ([], 0)
    assert store._ns == {"salt": {("m", 1): [2, (salted, "r5")]}}  # nothing left
