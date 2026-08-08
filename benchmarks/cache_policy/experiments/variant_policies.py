# SPDX-License-Identifier: Apache-2.0
"""
Score-rebalancing variants of ``CostAwareEvictionPolicy``.

Direction 2 of the direction-finding experiment: the real-data evaluation
(Finding 5) diagnosed that cost-density, as the *dominant* term in the
production score, actively misleads eviction under real, high-fan-out
traffic (it protects expensive-to-recompute chunks that are poor
predictors of imminent reuse). These variants only override
``_score_for_meta`` -- the shared scoring hook the production class
already factors its two call sites through -- to test whether
de-emphasizing cost recovers real-data hit rate without losing the
`multi_round_chat` win that depends on cost varying by position.

These are experimental, not shipped: real deployment would mean either
parameterizing the production formula or picking one variant as the new
default, a decision this experiment's results are meant to inform, not
preempt.
"""

# Standard
from typing import Any
import math

# First Party
from lmcache.v1.storage_backend.cache_policy.cost_aware_policy import (
    CostAwareEvictionPolicy,
)


class FrequencyFirstPolicy(CostAwareEvictionPolicy):
    """Cost dropped entirely: pure recency + frequency, like a decayed LFU."""

    def _score_for_meta(self, meta: Any, current_time: float) -> float:
        age_seconds = max(0.0, current_time - meta.last_access_time)
        frequency_weight = 1.0 + math.log1p(meta.hit_count)
        time_decay = 1.0 + (age_seconds / self.half_life_seconds)
        return frequency_weight / time_decay


class BlendedPolicy(CostAwareEvictionPolicy):
    """Cost still contributes, but damped to a secondary factor (** 0.25)
    instead of the production formula's full linear weight."""

    _COST_EXPONENT = 0.25

    def _score_for_meta(self, meta: Any, current_time: float) -> float:
        age_seconds = max(0.0, current_time - meta.last_access_time)
        cost_density = meta.estimated_recompute_tokens / meta.memory_size_bytes
        damped_cost = cost_density**self._COST_EXPONENT
        frequency_weight = 1.0 + math.log1p(meta.hit_count)
        time_decay = 1.0 + (age_seconds / self.half_life_seconds)
        return (damped_cost * frequency_weight) / time_decay
