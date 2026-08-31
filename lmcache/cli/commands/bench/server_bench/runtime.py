# SPDX-License-Identifier: Apache-2.0
"""Operation result models for the server benchmark runtime."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class LookupResult:
    """Result of one LOOKUP, including its hit range and latency."""

    hit_chunks: int
    total_chunks: int
    latency_ms: float
    error: str | None = None

    def __post_init__(self) -> None:
        if self.total_chunks < 0:
            raise ValueError(
                f"total_chunks must be non-negative, got {self.total_chunks}"
            )
        if not 0 <= self.hit_chunks <= self.total_chunks:
            raise ValueError(
                "hit_chunks must be between zero and total_chunks, got "
                f"{self.hit_chunks}/{self.total_chunks}"
            )
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")

    @property
    def succeeded(self) -> bool:
        """Return whether LOOKUP completed without an error."""
        return self.error is None

    @property
    def is_full_hit(self) -> bool:
        """Return whether every requested chunk was found."""
        return (
            self.succeeded
            and self.total_chunks > 0
            and (self.hit_chunks == self.total_chunks)
        )

    @property
    def is_full_miss(self) -> bool:
        """Return whether no requested chunk was found."""
        return self.succeeded and self.total_chunks > 0 and self.hit_chunks == 0

    @property
    def is_partial_hit(self) -> bool:
        """Return whether LOOKUP found some but not all requested chunks."""
        return self.succeeded and 0 < self.hit_chunks < self.total_chunks


@dataclass(frozen=True)
class TransferResult:
    """Aggregate STORE or RETRIEVE result for its target Worker ranks."""

    operation: str
    token_count: int
    latency_ms: float
    attempted_worker_ranks: tuple[int, ...]
    successful_worker_ranks: tuple[int, ...]
    failed_worker_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError(
                f"token_count must be non-negative, got {self.token_count}"
            )
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")

        attempted = set(self.attempted_worker_ranks)
        successful = set(self.successful_worker_ranks)
        failed = set(self.failed_worker_ranks)
        if len(attempted) != len(self.attempted_worker_ranks):
            raise ValueError("attempted_worker_ranks must not contain duplicates")
        if len(successful) != len(self.successful_worker_ranks):
            raise ValueError("successful_worker_ranks must not contain duplicates")
        if len(failed) != len(self.failed_worker_ranks):
            raise ValueError("failed_worker_ranks must not contain duplicates")
        if successful & failed:
            raise ValueError("successful and failed Worker ranks must be disjoint")
        if attempted != successful | failed:
            raise ValueError(
                "successful and failed Worker ranks must exactly partition "
                "attempted_worker_ranks"
            )

    @property
    def succeeded(self) -> bool:
        """Return whether every attempted Worker completed successfully."""
        return not self.failed_worker_ranks
