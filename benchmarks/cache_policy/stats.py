# SPDX-License-Identifier: Apache-2.0
"""
Dependency-free percentile-bootstrap confidence interval helper.

Used to turn repeated-run benchmark readings into mean +/- CI instead of
trusting a single run -- see ``real_dataset_eval.py`` and
``docs/design/v1/storage_backend/cache_policy/cost-aware-policy-eval.md``
for why this matters (``CostAwareEvictionPolicy`` uses real wall-clock time
for recency decay, which makes single-run readings noisy on
low-eviction-count workloads).
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
