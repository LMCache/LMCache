# SPDX-License-Identifier: Apache-2.0
"""
Latency cost model used by the cache-policy simulator.

There is no GPU/model inference in this benchmark suite (see
``docs/design/v1/storage_backend/cache_policy/cost-aware-policy-eval.md``
for rationale), so "latency" here is a **modeled** quantity, not a measured
wall-clock one: a cache hit costs a fixed KV-transfer constant per chunk,
and a cache miss costs a per-token prefill constant times the number of
tokens that must be recomputed. This is deliberately the same
``recompute_tokens`` quantity that ``CostAwareEvictionPolicy`` itself scores
on, so the model stays consistent with what the policy is optimizing for.
"""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModelConfig:
    """Constants for the modeled latency cost function.

    Attributes:
        retrieval_cost_per_chunk_seconds: Modeled time to fetch one cached
            chunk from the backend (KV transfer, not recomputed).
        prefill_cost_per_token_seconds: Modeled time to recompute (prefill)
            one token that missed the cache.
    """

    retrieval_cost_per_chunk_seconds: float = 0.0005
    prefill_cost_per_token_seconds: float = 0.00002


class CostModel:
    """Computes modeled per-request latency from hit/miss counts."""

    def __init__(self, config: CostModelConfig) -> None:
        """Initialize the cost model.

        Args:
            config: Latency constants to use.
        """
        self.config = config

    def hit_latency(self, hit_chunks: int) -> float:
        """Modeled latency contribution from cache-hit chunks.

        Args:
            hit_chunks: Number of chunks served from cache.

        Returns:
            Modeled latency in seconds.
        """
        return hit_chunks * self.config.retrieval_cost_per_chunk_seconds

    def miss_latency(self, recompute_tokens: int) -> float:
        """Modeled latency contribution from recomputed (missed) tokens.

        Args:
            recompute_tokens: Number of tokens that must be recomputed.

        Returns:
            Modeled latency in seconds.
        """
        return max(0, recompute_tokens) * self.config.prefill_cost_per_token_seconds

    def total_latency(self, hit_chunks: int, recompute_tokens: int) -> float:
        """Modeled end-to-end latency for one request.

        Args:
            hit_chunks: Number of chunks served from cache.
            recompute_tokens: Number of tokens that must be recomputed.

        Returns:
            Modeled latency in seconds.
        """
        return self.hit_latency(hit_chunks) + self.miss_latency(recompute_tokens)
