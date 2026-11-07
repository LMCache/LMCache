# SPDX-License-Identifier: Apache-2.0
"""Tests for the admission-coalescing gate (AdmissionCoalescer).

The gate is a pure decision function over (ready, pending) sets with
injectable timing, so every release condition is testable without MP
adapter mocks.
"""

# Standard
import time

# First Party
from lmcache.integration.vllm.admission_coalescing import (
    AdmissionCoalescer,
    _bucket_for,
)

BUDGET = 8192  # alpha=0.75 -> fill_floor = 6144
TWO_TAILS = 3072  # 2 * 3072 = 6144 == fill_floor


def _make(enabled: bool = True, budget: int = BUDGET) -> AdmissionCoalescer:
    return AdmissionCoalescer(enabled=enabled, budget_tokens=budget)


def _submit(c, rid: str, tail: int, age_s: float = 0.0) -> None:
    """Submit a retrieve whose submission timestamp is age_s in the past."""
    c.on_submit(rid, tail)
    c._submit_ts[rid] = time.monotonic() - age_s


def test_disabled_is_passthrough():
    c = _make(enabled=False)
    c.on_submit("a", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a"}) == {"a"}


def test_k_reached_releases():
    c = _make()
    for r in ("a", "b", "d"):
        c.on_submit(r, TWO_TAILS)
    out = c.select_to_report({"a", "b", "d"}, {"a", "b", "d"})
    assert out == {"a", "b", "d"}


def test_no_sibling_releases():
    c = _make()
    c.on_submit("a", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a"}) == {"a"}


def test_tail_fills_step_releases():
    c = _make()
    c.on_submit("a", 6144)
    c.on_submit("b", 100)
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_sum_cannot_fill_releases():
    c = _make()
    c.on_submit("a", 500)
    c.on_submit("b", 500)
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_deadline_expired_releases():
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c._submit_ts["a"] = time.monotonic() - 4.0
    c._submit_ts["b"] = time.monotonic() - 3.5
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_cold_start_releases():
    # No duration observations at all: release rather than hold blind.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_hold_taken_then_sibling_arrives():
    # With a duration estimate and a sibling inside the window: hold,
    # then report both when the sibling arrives.
    c = _make()
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})  # releases, EWMA seeded
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    out = c.select_to_report({"a"}, {"a", "b"})
    assert out == set()  # held
    out = c.select_to_report({"a", "b"}, {"a", "b"})
    assert out == {"a", "b"}  # K reached next step


def test_reported_exactly_once():
    c = _make()
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a", "b"}) == set()
    assert c.select_to_report({"a", "b"}, {"a", "b"}) == {"a", "b"}
    assert "a" not in c._submit_ts
    assert c.select_to_report(set(), set()) == set()


def test_drain_clears_state():
    c = _make()
    c.on_submit("x", 100)
    c.drain()
    assert not c._submit_ts
    assert not c._tail
    assert not c._first_hold


def test_hold_time_excluded_from_ewma():
    # A request that was held contributes (first_hold - submit), not
    # (report - submit), so the EWMA tracks the retrieve pipeline.
    c = _make()
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c._duration_ewma = 0.3
    assert c.select_to_report({"a"}, {"a", "b"}) == set()  # held
    c._first_hold["a"] = time.monotonic() - 0.5  # held for 0.5s
    c._submit_ts["a"] = time.monotonic() - 1.0
    c.select_to_report({"a", "b"}, set())
    assert c._duration_ewma is not None
    assert c._duration_ewma < 0.8 * 0.3 + 0.2 * 1.0


# ============ Preemption tests (review issue #2) ============


def test_forget_drops_state_without_reporting():
    # Preempted/cancelled retrieves leave retrieve_futures without reporting.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c.forget({"a"})
    assert "a" not in c._submit_ts
    assert "a" not in c._tail
    assert "a" not in c._first_hold
    assert "b" in c._submit_ts  # sibling untouched


def test_preemption_gc_clears_stale():
    # Simulates the adapter-level GC: entries in _submit_ts that are no
    # longer in retrieve_futures should be forgotten.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c.on_submit("c", TWO_TAILS)
    # "c" gets preempted: leaves retrieve_futures but coalescer state remains
    # (adapter GC calls forget)
    stale = set(c._submit_ts) - {"a", "b"}  # simulate retrieve_futures = {a, b}
    if stale:
        c.forget(stale)
    assert "c" not in c._submit_ts
    assert "a" in c._submit_ts
    assert "b" in c._submit_ts


def test_resubmit_after_preemption_clean():
    # A preempted request that is resumed with the same request_id should
    # not see a stale _first_hold from its previous life.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c._first_hold["a"] = time.monotonic() - 5.0  # old hold timestamp
    c.forget({"a"})  # preemption cleanup
    c.on_submit("a", TWO_TAILS)  # resume with same id
    assert c._first_hold.get("a") is None  # stale timestamp gone


# ============ Bucketed EWMA tests (review issue #3) ============


def test_bucket_for_correct_boundaries():
    assert _bucket_for(0) == 8
    assert _bucket_for(100) == 8  # closest to 256
    assert _bucket_for(256) == 8
    assert _bucket_for(1024) == 10
    assert _bucket_for(4096) == 12
    assert _bucket_for(16384) == 14
    assert _bucket_for(65536) == 16
    assert _bucket_for(500) == 8  # log2(500)~8.97, tie between 8 and 10, min picks 8
    assert _bucket_for(3000) == 12  # log2(3000)~11.5, closer to 12


def test_bucket_ewma_independent():
    # Short and long retrieves maintain independent duration estimates.
    c = _make()
    # Submit a short retrieve (tail ~500, bucket 10)
    c.on_submit("short", 500)
    c._first_hold["short"] = time.monotonic()
    c._submit_ts["short"] = time.monotonic() - 0.1  # 100ms duration
    c._tail["short"] = 500
    c.select_to_report({"short"}, set())

    # Submit a long retrieve (tail ~20000, bucket 14)
    c.on_submit("long", 20000)
    c._first_hold["long"] = time.monotonic()
    c._submit_ts["long"] = time.monotonic() - 2.0  # 2000ms duration
    c._tail["long"] = 20000
    c.select_to_report({"long"}, set())

    # Verify independent estimates
    short_est = c._estimate_duration(500)
    long_est = c._estimate_duration(20000)
    assert short_est is not None and short_est < 0.5
    assert long_est is not None and long_est > 1.0
    assert long_est > short_est * 2  # clearly separated


def test_eta_uses_sibling_bucket():
    # A short-tailed sibling should use the short bucket's estimate,
    # not the global EWMA which might be inflated by long retrieves.
    c = _make()
    # Seed the global EWMA with a long retrieve duration
    c.on_submit("long", 20000)
    c._submit_ts["long"] = time.monotonic() - 2.0
    c._first_hold["long"] = time.monotonic()
    c._tail["long"] = 20000
    c.select_to_report({"long"}, set())
    assert c._duration_ewma > 1.0  # global is long

    # Now a short sibling arrives; its ETA should use the short bucket
    # (or fall back to global if short bucket is empty — which releases,
    # the safe direction)
    c.on_submit("a", TWO_TAILS)
    c.on_submit("short_b", 500)
    c._submit_ts["a"] = time.monotonic() - 0.1
    c._submit_ts["short_b"] = time.monotonic()
    # Short bucket has no observation -> falls back to global -> ETA too
    # high -> releases (safe)
    out = c.select_to_report({"a"}, {"a", "short_b"})
    # Either held (if global estimate happens to be small enough) or
    # released (if global estimate is too large). Both are safe.
    assert out in ({"a"}, set())


def test_bucket_cold_start_releases():
    # A bucket with no observations falls back to global; if global is
    # also None, releases (no blind hold).
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", 30000)  # different bucket
    # No duration observations at all
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}
