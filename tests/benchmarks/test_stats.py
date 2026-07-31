# SPDX-License-Identifier: Apache-2.0
"""Tests for the dependency-free bootstrap-CI helpers in
benchmarks/cache_policy/stats.py."""

# Third Party
import pytest

# First Party
from benchmarks.cache_policy.stats import (
    bootstrap_ci,
    paired_bootstrap_ci_diff,
    paired_sign_test,
)


def test_bootstrap_ci_single_value_has_zero_width_interval():
    mean, lo, hi = bootstrap_ci([0.5])
    assert mean == lo == hi == 0.5


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_ci_diff_detects_consistent_advantage():
    # values_a consistently beats values_b at every paired repeat -- the
    # mean difference's CI must be strictly positive (exclude zero).
    values_a = [0.10, 0.12, 0.11, 0.13, 0.14, 0.12]
    values_b = [0.09, 0.09, 0.10, 0.10, 0.11, 0.10]
    mean_diff, lo, hi = paired_bootstrap_ci_diff(values_a, values_b)
    assert mean_diff > 0
    assert lo > 0, "CI should exclude zero given a's consistent advantage"


def test_paired_bootstrap_ci_diff_identical_sequences_is_zero():
    values = [0.10, 0.20, 0.30]
    mean_diff, lo, hi = paired_bootstrap_ci_diff(values, values)
    assert (mean_diff, lo, hi) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_ci_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci_diff([0.1, 0.2], [0.1])


def test_paired_sign_test_all_positive_pairs_is_significant():
    values_a = [0.10, 0.12, 0.11, 0.13, 0.14, 0.12]
    values_b = [0.09, 0.09, 0.10, 0.10, 0.11, 0.10]
    p = paired_sign_test(values_a, values_b)
    # Exact two-sided sign test, 6/6 positive: 2 * C(6,0) / 2**6.
    assert p == pytest.approx(2 / 64)


def test_paired_sign_test_identical_sequences_is_one():
    values = [0.10, 0.20, 0.30]
    assert paired_sign_test(values, values) == 1.0


def test_paired_sign_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_sign_test([0.1, 0.2], [0.1])


def test_paired_sign_test_mixed_signs_not_significant():
    # Alternating advantage -- no consistent winner, p should be large.
    values_a = [0.10, 0.09, 0.10, 0.09]
    values_b = [0.09, 0.10, 0.09, 0.10]
    p = paired_sign_test(values_a, values_b)
    assert p == 1.0
