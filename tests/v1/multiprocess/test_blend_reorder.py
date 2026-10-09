# SPDX-License-Identifier: Apache-2.0
"""Reorder planner ("copy one cached prompt"): pure-Python unit tests."""

# Standard
import random

# First Party
from lmcache.v1.multiprocess.modules.blend import reorder as br

C = 32  # chunk size for these tests
SYS = list(range(1, C + 1))  # a one-chunk "system prompt"
SEP = [7]


def _doc(seed: int, n: int = 40) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 50000) for _ in range(n)]


def _prompt(*docs: list[int], question: int = 9) -> list[int]:
    out = list(SYS)
    for i, d in enumerate(docs):
        out += d + (SEP if i + 1 < len(docs) else [])
    return out + [question] * 5


def _serve(P: list[int], cached: list[list[int]], **kw) -> tuple[list[int], dict]:
    perm, info = br.plan(P, cached, C, **kw)
    assert sorted(perm) == list(range(len(P)))
    return [P[i] for i in perm], info


def test_copies_cached_prompt_from_token_0():
    a, b, x, y = _doc(1), _doc(2), _doc(3), _doc(4)
    cached = _prompt(a, b, x)
    served, info = _serve(_prompt(y, b, a, question=8), [cached])
    # [y][b][a] -> [a][b][y]: the start of `cached`, then the rest in order
    assert served[: len(SYS) + 2 * 41] == cached[: len(SYS) + 2 * 41]
    assert served[-5:] == [8] * 5  # the question stays last
    assert info["copy_tokens"] >= len(SYS) + 2 * 41


def test_no_move_when_the_first_slot_differs():
    # cached [x][d][x'], new [y][y'][d]: moving d alone cannot extend the
    # exact prefix, so the prompt is served unchanged.
    d, x, x2, y, y2 = (_doc(s) for s in range(10, 15))
    P = _prompt(y, y2, d)
    served, _ = _serve(P, [_prompt(x, d, x2)])
    assert served == P


def test_moves_to_slot_2_when_slot_1_matches():
    d, x, x2, y = (_doc(s) for s in range(20, 24))
    served, _ = _serve(_prompt(x, y, d), [_prompt(x, d, x2)])
    assert served[: len(SYS) + 2 * 41] == _prompt(x, d, x2)[: len(SYS) + 2 * 41]


def test_identity_cases():
    P = _prompt(_doc(30), _doc(31))
    assert _serve(P, [])[1]["reason"] == "no_candidates"
    assert _serve(P, [_prompt(_doc(32))])[0] == P  # nothing in common
    # gain_only: a copy that adds no whole chunk over P's own prefix is refused
    a, b = _doc(33), _doc(34)
    cached = _prompt(a, b)
    served, info = _serve(
        _prompt(a, b, question=5), [cached], baseline_chunks=10**6, gain_only=True
    )
    assert info["reason"] == "no_gain"


def test_budget_returns_identity():
    a, b = _doc(40), _doc(41)
    P = _prompt(b, a)
    served, info = _serve(P, [_prompt(a, b)], budget_s=-1.0)
    assert served == P and info["reason"] == "budget"


def test_seam_rule_keeps_each_label_with_its_document():
    label = [5, 6]  # e.g. "Document:"
    a, b, x = ([*label, *_doc(s)] for s in (50, 51, 52))
    cached = _prompt(a, x)
    P = _prompt(b, a)
    plain, _ = _serve(P, [cached])
    seam, _ = _serve(P, [cached], seam_rule=True)
    assert (
        plain[: len(SYS) + len(a)]
        == seam[: len(SYS) + len(a)]
        == cached[: len(SYS) + len(a)]
    )
    # with the rule, `a` brings its own label, so `b` keeps its label intact
    assert seam[len(SYS) + len(a) + 1 :][: len(b)] == b


def test_prompt_store_candidates():
    store = br.PromptStore(max_prompts=2)
    store.record("ns", "r1", [1, 2, 3], [b"h0", b"h1"])
    store.record("ns", "r2", [4, 5, 6], [b"h0", b"g1", b"g2"])
    store.record(
        "ns", "r2", [4, 5, 6, 7], [b"g3"]
    )  # later range: ids stay the prompt's
    cands, base = store.candidates("ns", [b"g2"], [b"h0", b"h1", b"zz"], top_k=1)
    # r1 holds P's own 2-chunk prefix, r2 owns the hit (and is the newest)
    assert [list(c) for c in cands] == [[4, 5, 6], [1, 2, 3]] and base == 2
    store.record("ns", "r3", [8], [b"h1"])  # evicts r1 (LRU, max 2)
    cands, base = store.candidates("ns", [], [b"h0", b"h1"], top_k=4)
    assert [list(c) for c in cands] == [[8]] and base == 2  # holder r3 is also newest
    assert store.candidates("other", [b"g2"], [], top_k=4) == ([], 0)


def test_never_serves_fewer_chunks_than_the_prompt_already_has():
    a, b = _doc(60), _doc(61)
    P = _prompt(b, a)
    served, info = _serve(P, [_prompt(a, b)], baseline_chunks=10**6)
    assert served == P and info["reason"] == "no_gain"


def test_budget_expiry_keeps_a_finished_plan_only_if_it_gains(monkeypatch):
    a, b, x = _doc(80), _doc(81), _doc(82)
    P = _prompt(b, a)
    good, other = _prompt(a, b), _prompt(x, a)
    real_copy, calls = br._Copier.copy, []

    def copy_then_expire(self, Hn, seam_rule=False):
        calls.append(1)
        if len(calls) > 1:
            raise br.PlanBudgetExceeded
        return real_copy(self, Hn, seam_rule)

    monkeypatch.setattr(br._Copier, "copy", copy_then_expire)
    served, info = _serve(P, [good, other])  # newest (good) finished first
    assert served[: len(SYS) + len(a)] == good[: len(SYS) + len(a)]
    assert info["budget_hit"]
    calls.clear()
    served, info = _serve(P, [good, other], baseline_chunks=10**6)
    assert served == P and info["reason"] == "budget"


def test_too_long_prompts_are_returned_unchanged():
    P = list(range(br.MAX_PLAN_TOKENS + 1))
    assert _serve(P, [P[:100]])[1]["reason"] == "too_long"


def test_prompt_store_namespace_and_token_cap():
    store = br.PromptStore(max_tokens=10)
    store.record(("m", 1, ""), "r1", [1] * 6, [b"a"])
    store.record(("m", 1, "salt"), "r2", [2] * 3, [b"b"])
    assert store.resolve("", 0, "") == ("m", 1, "")  # salt must match exactly
    assert store.resolve("", 0, "salt") == ("m", 1, "salt")
    assert store.resolve("m", 2, "") is None and store.resolve("x", 0, "") is None
    store.record(("m", 1, ""), "r3", [3] * 6, [b"c"])  # 15 tokens > 10: evicts r1
    assert [
        list(c) for c in store.candidates(("m", 1, ""), [b"a", b"c"], [], 4)[0]
    ] == [[3] * 6]


def test_keep_prefix_never_moves_the_kept_tokens():
    a, b, x, y = _doc(90), _doc(91), _doc(92), _doc(93)
    P = _prompt(x, b, a)
    # copying `cached` would keep [sys][x] and move a before b
    cached = _prompt(x, a, b)
    keep = len(SYS) + len(x)
    served, _ = _serve(P, [cached], keep_prefix=keep)
    assert (
        served[:keep] == P[:keep]
        and served[: keep + 1 + len(a)] == cached[: keep + 1 + len(a)]
    )
    # a candidate that differs inside the kept prefix is never used
    served, info = _serve(P, [_prompt(y, a, b)], keep_prefix=keep)
    assert served == P
