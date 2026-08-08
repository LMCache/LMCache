# SPDX-License-Identifier: Apache-2.0
"""
Dependency-free percentile-bootstrap confidence interval helpers.

Used to turn repeated-run benchmark readings into mean +/- CI instead of
trusting a single run -- see ``real_dataset_eval.py`` and
``docs/design/v1/storage_backend/cache_policy/cost-aware-policy-eval.md``
for why this matters.

Two distinct comparisons are supported, and they are not interchangeable:

- :func:`bootstrap_ci` -- an independent-samples CI for one policy's own
  mean metric across repeats. Comparing two policies by checking whether
  their independently computed CIs overlap is a common but weaker,
  overly conservative test, and is *invalid* outright when the repeats
  being compared are not independent across policies -- see the next
  point.
- :func:`paired_bootstrap_ci_diff` -- for exactly this codebase's
  real-data evaluation, where every policy in a given (scale,
  cache-size) cell is replayed against the *same* ``n_repeats`` corpus
  subsamples (``real_dataset_eval.py`` calls
  ``requests_from_conversations(..., seed=repeat)`` identically for every
  policy at a given ``repeat`` index), so repeat ``i``'s reading for
  policy A and repeat ``i``'s reading for policy B are not independent
  observations -- they share the exact same subsampled requests. Treating
  them as independent (bootstrapping each policy's values separately and
  eyeballing CI overlap) throws away that paired structure and understates
  the evidence for a real difference. The correct comparison is the
  per-repeat *difference* (A's reading minus B's reading at each shared
  repeat index), bootstrapped as its own single sample.
"""

# Standard
import random


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Compute a percentile-bootstrap confidence interval for the mean.

    Args:
        values: Sample of per-repeat metric readings (e.g. one
            ``token_hit_rate`` per repeated benchmark run).
        n_boot: Number of bootstrap resamples to draw.
        alpha: Significance level; the returned interval covers
            ``1 - alpha`` (e.g. ``alpha=0.05`` -> 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``(mean, lower, upper)``. If ``values`` has fewer than 2 entries,
        ``lower`` and ``upper`` both equal the mean (no spread to
        resample).

    Raises:
        ValueError: If ``values`` is empty, ``n_boot`` is non-positive, or
            ``alpha`` is not in ``(0, 1)``.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    mean = sum(values) / len(values)
    if len(values) < 2:
        return (mean, mean, mean)

    rng = random.Random(seed)
    n = len(values)
    boot_means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    ]
    boot_means.sort()

    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (mean, boot_means[lo_idx], boot_means[hi_idx])


def paired_bootstrap_ci_diff(
    values_a: list[float],
    values_b: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Percentile-bootstrap confidence interval for the mean *paired*
    difference ``values_a[i] - values_b[i]``.

    Use this instead of comparing two independent :func:`bootstrap_ci`
    intervals for overlap whenever ``values_a`` and ``values_b`` were
    produced from the same underlying repeat/seed at each index (e.g.
    two policies replayed against the same corpus subsample at repeat
    ``i``) -- see the module docstring for why treating paired
    observations as independent is invalid.

    Args:
        values_a: Per-repeat readings for the treatment (e.g. the policy
            being evaluated).
        values_b: Per-repeat readings for the baseline (e.g. LRU), in the
            same repeat order as ``values_a`` -- ``values_a[i]`` and
            ``values_b[i]`` must come from the same repeat/seed.
        n_boot: Number of bootstrap resamples to draw.
        alpha: Significance level; the returned interval covers
            ``1 - alpha``.
        seed: RNG seed for reproducibility.

    Returns:
        ``(mean_diff, lower, upper)`` for ``values_a - values_b``. An
        interval that excludes 0 is evidence the paired difference is
        unlikely to be due to chance at the corresponding significance
        level -- this bootstrap-of-differences procedure is itself a
        valid paired significance test, not just a descriptive interval.

    Raises:
        ValueError: If ``values_a`` and ``values_b`` have different
            lengths, or (via :func:`bootstrap_ci`) if empty, ``n_boot``
            is non-positive, or ``alpha`` is not in ``(0, 1)``.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"values_a and values_b must be the same length (paired), "
            f"got {len(values_a)} vs {len(values_b)}"
        )
    diffs = [a - b for a, b in zip(values_a, values_b, strict=True)]
    return bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)


def paired_sign_test(values_a: list[float], values_b: list[float]) -> float:
    """
    Exact two-sided sign test p-value for whether ``values_a`` tends to
    exceed ``values_b`` in paired observations.

    A minimal, dependency-free complement to
    :func:`paired_bootstrap_ci_diff`: nonparametric, makes no assumption
    about the distribution of the differences (useful validation given
    how few repeats this suite typically runs), at the cost of being
    less powerful than a test that uses the differences' magnitudes.
    Exact ties (``values_a[i] == values_b[i]``) are dropped before
    testing, per standard sign-test practice.

    Args:
        values_a: Per-repeat readings for the treatment.
        values_b: Per-repeat readings for the baseline, paired with
            ``values_a`` by index (same repeat/seed at each position).

    Returns:
        Two-sided p-value in ``[0, 1]``. ``1.0`` if every pair is an
        exact tie (nothing to test).

    Raises:
        ValueError: If ``values_a`` and ``values_b`` have different
            lengths.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"values_a and values_b must be the same length (paired), "
            f"got {len(values_a)} vs {len(values_b)}"
        )
    signs = [a - b for a, b in zip(values_a, values_b, strict=True) if a != b]
    n = len(signs)
    if n == 0:
        return 1.0
    positives = sum(1 for d in signs if d > 0)

    def _binom_pmf(k: int, n: int) -> float:
        from math import comb  # noqa: PLC0415

        return comb(n, k) / (2**n)

    def _binom_cdf_le(k: int, n: int) -> float:
        return sum(_binom_pmf(i, n) for i in range(0, k + 1))

    tail_lo = _binom_cdf_le(min(positives, n - positives), n)
    return min(1.0, 2.0 * tail_lo)
