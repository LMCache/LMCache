# SPDX-License-Identifier: Apache-2.0
"""
Synthetic request-workload generators for cache-policy benchmarking.

Each generator produces a list of :class:`Request` objects describing a
sequence of prefill requests, in the same shape the LMCache lookup path
consumes: a total token count plus an ordered list of per-chunk hash keys.
Chunk hashes are deterministic given the (prompt, chunk-index) pair, so
identical prompts across requests share hashes and therefore hit the cache,
while distinct prompts never collide.

These are pure-Python and require no GPU or running model, so the same
workloads drive both local development and CI.
"""

# Standard
from dataclasses import dataclass, field
import random


@dataclass(frozen=True)
class Request:
    """A single simulated prefill request.

    Attributes:
        total_tokens: Total number of tokens in the request (including any
            trailing partial-chunk tokens, which are always a cache miss).
        chunk_hashes: Ordered list of per-chunk cache keys covering the
            leading ``len(chunk_hashes) * chunk_size`` tokens of the request.
        chunk_size: Number of tokens per chunk for this request.
    """

    total_tokens: int
    chunk_hashes: list[str] = field(default_factory=list)
    chunk_size: int = 256


def _chunk_hashes(prompt_id: str, num_chunks: int) -> list[str]:
    return [f"{prompt_id}:c{i}" for i in range(num_chunks)]


def repetitive_short(
    n_requests: int,
    vocab_size: int = 20,
    chunk_size: int = 256,
    min_chunks: int = 1,
    max_chunks: int = 4,
    seed: int = 0,
) -> list[Request]:
    """Generate short, highly repetitive prompts.

    A small pool of ``vocab_size`` distinct prompts is sampled uniformly at
    random for each request, stressing the hit/miss ratio and eviction
    churn once the pool exceeds cache capacity.

    Args:
        n_requests: Number of requests to generate.
        vocab_size: Number of distinct prompts in the reuse pool.
        chunk_size: Tokens per chunk.
        min_chunks: Minimum chunks per prompt (inclusive).
        max_chunks: Maximum chunks per prompt (inclusive).
        seed: RNG seed for reproducibility.

    Returns:
        List of generated requests.

    Raises:
        ValueError: If any size parameter is non-positive or the chunk
            range is inverted.
    """
    if n_requests <= 0 or vocab_size <= 0 or chunk_size <= 0:
        raise ValueError("n_requests, vocab_size, and chunk_size must be positive")
    if min_chunks <= 0 or max_chunks < min_chunks:
        raise ValueError("min_chunks must be positive and <= max_chunks")

    rng = random.Random(seed)
    prompt_chunks = {i: rng.randint(min_chunks, max_chunks) for i in range(vocab_size)}

    requests: list[Request] = []
    for _ in range(n_requests):
        prompt_id = rng.randrange(vocab_size)
        num_chunks = prompt_chunks[prompt_id]
        requests.append(
            Request(
                total_tokens=num_chunks * chunk_size,
                chunk_hashes=_chunk_hashes(f"short-{prompt_id}", num_chunks),
                chunk_size=chunk_size,
            )
        )
    return requests


def novel_long(
    n_requests: int,
    min_tokens: int = 4096,
    max_tokens: int = 16384,
    chunk_size: int = 256,
    seed: int = 0,
) -> list[Request]:
    """Generate long prompts that are each unique (never reused).

    Every request gets a distinct prompt id, so no chunk is ever a cache
    hit. This isolates the pure bookkeeping overhead of a policy (insert +
    eviction-selection cost) from any benefit of caching.

    Args:
        n_requests: Number of requests to generate.
        min_tokens: Minimum request length in tokens (inclusive).
        max_tokens: Maximum request length in tokens (inclusive).
        chunk_size: Tokens per chunk.
        seed: RNG seed for reproducibility.

    Returns:
        List of generated requests.

    Raises:
        ValueError: If any size parameter is non-positive or the token
            range is inverted.
    """
    if n_requests <= 0 or chunk_size <= 0:
        raise ValueError("n_requests and chunk_size must be positive")
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("min_tokens must be positive and <= max_tokens")

    rng = random.Random(seed)
    requests: list[Request] = []
    for i in range(n_requests):
        total_tokens = rng.randint(min_tokens, max_tokens)
        num_chunks = total_tokens // chunk_size
        requests.append(
            Request(
                total_tokens=total_tokens,
                chunk_hashes=_chunk_hashes(f"novel-{i}", num_chunks),
                chunk_size=chunk_size,
            )
        )
    return requests


def _zipf_weights(n: int, s: float) -> list[float]:
    """Rank-based Zipf weights for ranks ``1..n`` with exponent ``s``."""
    return [1.0 / (rank**s) for rank in range(1, n + 1)]


def mixed_zipfian(
    n_requests: int,
    unique_prefixes: int = 200,
    chunk_size: int = 256,
    zipf_s: float = 1.2,
    min_chunks: int = 1,
    max_chunks: int = 8,
    seed: int = 0,
) -> list[Request]:
    """Generate requests whose prompt popularity follows a Zipf distribution.

    This is the primary cross-policy comparison workload: a small set of
    prompts is reused very frequently (production "hot" prefixes) while a
    long tail is touched rarely, similar to real chat/RAG traffic.

    Args:
        n_requests: Number of requests to generate.
        unique_prefixes: Number of distinct prompts in the popularity
            distribution.
        chunk_size: Tokens per chunk.
        zipf_s: Zipf exponent; larger values concentrate reuse on fewer
            prompts.
        min_chunks: Minimum chunks per prompt (inclusive).
        max_chunks: Maximum chunks per prompt (inclusive).
        seed: RNG seed for reproducibility.

    Returns:
        List of generated requests.

    Raises:
        ValueError: If any size parameter is non-positive, the chunk range
            is inverted, or ``zipf_s`` is not positive.
    """
    if n_requests <= 0 or unique_prefixes <= 0 or chunk_size <= 0:
        raise ValueError("n_requests, unique_prefixes, and chunk_size must be positive")
    if min_chunks <= 0 or max_chunks < min_chunks:
        raise ValueError("min_chunks must be positive and <= max_chunks")
    if zipf_s <= 0:
        raise ValueError("zipf_s must be positive")

    rng = random.Random(seed)
    weights = _zipf_weights(unique_prefixes, zipf_s)
    prompt_chunks = {
        i: rng.randint(min_chunks, max_chunks) for i in range(unique_prefixes)
    }
    population = list(range(unique_prefixes))

    requests: list[Request] = []
    for _ in range(n_requests):
        prompt_id = rng.choices(population, weights=weights, k=1)[0]
        num_chunks = prompt_chunks[prompt_id]
        requests.append(
            Request(
                total_tokens=num_chunks * chunk_size,
                chunk_hashes=_chunk_hashes(f"zipf-{prompt_id}", num_chunks),
                chunk_size=chunk_size,
            )
        )
    return requests


def multi_round_chat(
    n_sessions: int,
    rounds_per_session: int = 8,
    chunk_size: int = 256,
    tokens_per_round: int = 256,
    seed: int = 0,
) -> list[Request]:
    """Generate multi-round chat sessions with a monotonically growing prefix.

    Each round of a session reuses every chunk from all prior rounds of the
    same session (the shared conversation prefix) and appends one new
    chunk. Requests from different sessions are interleaved round-robin, as
    they would be in a live server. This workload specifically exercises
    the ``chunk_start`` / recompute-cost accounting that
    ``CostAwareEvictionPolicy`` uses but the simpler recency/frequency
    policies ignore.

    Args:
        n_sessions: Number of concurrent chat sessions.
        rounds_per_session: Number of rounds (turns) per session.
        chunk_size: Tokens per chunk; ``tokens_per_round`` must be a
            multiple of this.
        tokens_per_round: Tokens appended to the prefix per round.
        seed: Unused; present for signature symmetry with the other
            generators (this workload is deterministic).

    Returns:
        List of generated requests, interleaved round-robin across
        sessions in chronological order.

    Raises:
        ValueError: If any size parameter is non-positive or
            ``tokens_per_round`` is not a multiple of ``chunk_size``.
    """
    del seed  # deterministic generator, kept for a uniform call signature
    if n_sessions <= 0 or rounds_per_session <= 0 or chunk_size <= 0:
        raise ValueError(
            "n_sessions, rounds_per_session, and chunk_size must be positive"
        )
    if tokens_per_round <= 0 or tokens_per_round % chunk_size != 0:
        raise ValueError("tokens_per_round must be a positive multiple of chunk_size")

    chunks_per_round = tokens_per_round // chunk_size
    requests: list[Request] = []
    for round_idx in range(rounds_per_session):
        for session_idx in range(n_sessions):
            num_chunks = (round_idx + 1) * chunks_per_round
            requests.append(
                Request(
                    total_tokens=num_chunks * chunk_size,
                    chunk_hashes=_chunk_hashes(f"chat-{session_idx}", num_chunks),
                    chunk_size=chunk_size,
                )
            )
    return requests


WORKLOAD_REGISTRY = {
    "repetitive_short": repetitive_short,
    "novel_long": novel_long,
    "mixed_zipfian": mixed_zipfian,
    "multi_round_chat": multi_round_chat,
}
