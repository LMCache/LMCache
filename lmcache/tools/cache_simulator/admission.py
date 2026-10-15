# SPDX-License-Identifier: Apache-2.0
"""Bounded historical admission experiment on pinned Mooncake request traces.

This is a serialized, single-tier work model, not a serving benchmark. Only
full 512-token input blocks participate. Writes become available after each
request, and reads can reuse only a contiguous prefix. There is no GPU APC,
concurrency, transfer latency, lazy scheduler, or recompute-cost calibration.
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TypedDict
import argparse
import hashlib
import json
import math

# First Party
from lmcache.tools.cache_simulator.lru_cache import LRUCacheFast

BLOCK_SIZE = 512
MAX_REQUESTS = 100_000
MAX_BLOCKS_PER_REQUEST = 4096
MAX_DISTINCT_HASHES = 2_000_000


class ModeReport(TypedDict):
    """Measured work and insertion counts for one modeled strategy."""

    hit_tokens: int
    token_hit_rate: float
    written_blocks: int


class AdmissionReport(TypedDict):
    """Successful replay evidence; qualification explicitly excludes serving."""

    status: str
    qualification: str
    input_sha256: str
    block_tokens: int
    requests: int
    evaluation_start: int
    evaluation_requests: int
    input_tokens: int
    capacity_chunks: int
    history_keys: int
    max_request_gap: int
    tracked_keys_final: int
    modes: dict[str, ModeReport]
    unlimited_capacity_reuse_work_ceiling_tokens: int
    not_modeled: list[str]


def _integer(value: object, name: str, minimum: int = 0) -> int:
    """Validate a trace integer, rejecting bools and lossy conversions."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def read_requests(
    path: Path, expected_sha256: str
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Yield validated input lengths and full-block hashes in file order.

    Args:
        path: Mooncake FAST25 JSONL trace; timestamps must be nondecreasing.
        expected_sha256: Digest of the exact bytes consumed, checked at EOF.

    Yields:
        Input token count and ordered complete 512-token block identifiers.

    Raises:
        ValueError: Empty, malformed, incompatible, or over-limit input.
        OSError: Input cannot be read.

    Partial final blocks are excluded. A repeated identifier must have the
    same parent, including partial blocks, or this is not a prefix-hash trace.
    """
    parents: dict[int, int | None] = {}
    previous_timestamp = -math.inf
    requests = 0
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            digest.update(line)
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("request must be an object")
                timestamp = record["timestamp"]
                if (
                    type(timestamp) not in (int, float)
                    or (type(timestamp) is float and not math.isfinite(timestamp))
                    or timestamp < previous_timestamp
                ):
                    raise ValueError("timestamps must be finite and nondecreasing")
                previous_timestamp = timestamp
                input_length = _integer(record["input_length"], "input_length", 1)
                _integer(record["output_length"], "output_length")
                hashes = record["hash_ids"]
                if not isinstance(hashes, list):
                    raise ValueError("hash_ids must be a list")
                expected_blocks = (input_length + BLOCK_SIZE - 1) // BLOCK_SIZE
                if len(hashes) != expected_blocks:
                    raise ValueError("hash_ids must cover exactly the input blocks")
                if len(hashes) > MAX_BLOCKS_PER_REQUEST:
                    raise ValueError("request exceeds block limit")
                parent = None
                for block in hashes:
                    block = _integer(block, "hash_ids entry")
                    if block in parents and parents[block] != parent:
                        raise ValueError("hash identifier has inconsistent ancestry")
                    parents[block] = parent
                    if len(parents) > MAX_DISTINCT_HASHES:
                        raise ValueError("trace exceeds distinct-hash limit")
                    parent = block
                requests += 1
                if requests > MAX_REQUESTS:
                    raise ValueError("trace exceeds request limit")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield input_length, tuple(hashes[: input_length // BLOCK_SIZE])
    if not requests:
        raise ValueError("trace contains no requests")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("input SHA-256 does not match manifest")


@dataclass
class WorkTotals:
    """Evaluation-window totals; writes count newly inserted full blocks."""

    hit_tokens: int = 0
    written_blocks: int = 0


class HistoricalAdmission:
    """Bounded two-observation history, independent of cache residency.

    A block qualifies only when it was observed in an earlier request within
    ``max_request_gap`` requests. This is an inter-reference *request* gap,
    not distinct-key reuse distance or a calibrated expected-use probability.
    Forgotten blocks are cold. The current request is observed after choosing
    its admission prefix, so it cannot qualify itself.
    """

    def __init__(self, max_keys: int, max_request_gap: int) -> None:
        """Set positive history bounds; allocate no workers or I/O resources."""
        self._max_keys = _integer(max_keys, "max_keys", 1)
        self._max_request_gap = _integer(max_request_gap, "max_request_gap", 1)
        self._last_seen: OrderedDict[int, int] = OrderedDict()

    def select_prefix(self, blocks: tuple[int, ...], request_index: int) -> int:
        """Return the length of the eligible, prefix-closed write candidate.

        Args:
            blocks: Current request's ordered full blocks.
            request_index: Monotonic request ordinal in this replay.

        Returns:
            Number of leading blocks with sufficiently recent prior evidence.
        """
        for index, block in enumerate(blocks):
            previous = self._last_seen.get(block)
            if previous is None or request_index - previous > self._max_request_gap:
                return index
        return len(blocks)

    def observe(self, blocks: tuple[int, ...], request_index: int) -> None:
        """Record a completed observation, evicting oldest history at capacity."""
        for block in blocks:
            self._last_seen[block] = request_index
            self._last_seen.move_to_end(block)
            if len(self._last_seen) > self._max_keys:
                self._last_seen.popitem(last=False)

    @property
    def tracked_keys(self) -> int:
        """Number of retained history entries (never exceeds max_keys)."""
        return len(self._last_seen)


def _access_request(
    cache: LRUCacheFast, blocks: tuple[int, ...], write_prefix: int
) -> tuple[int, int]:
    """Read the pre-write prefix; then touch hits and insert selected misses."""
    prefix = 0
    for block in blocks:
        if not cache.contains(block):
            break
        prefix += 1
    writes = 0
    # Match the existing simulator's ordered touches and insertions. No newly
    # written block can improve this request's pre-write hit count.
    for index, block in enumerate(blocks):
        if cache.contains(block):
            cache.access(block)
        elif index < write_prefix:
            cache.insert(block)
            writes += 1
    return prefix * BLOCK_SIZE, writes


def evaluate(
    path: Path,
    expected_sha256: str,
    capacity_chunks: int,
    history_keys: int,
    max_request_gap: int,
    evaluation_start: int = 0,
) -> AdmissionReport:
    """Replay a byte-verified trace through baseline and historical admission.

    Args:
        path: Pinned Mooncake JSONL input.
        expected_sha256: SHA-256 from the independently acquired input manifest.
        capacity_chunks: Positive, identical full-block capacity of both caches.
        history_keys: Positive upper bound on online history entries.
        max_request_gap: Positive maximum prior-observation age in requests.
        evaluation_start: Number of training/warmup requests excluded from totals.

    Returns:
        JSON-safe report with explicit model restrictions and measured counts.

    Raises:
        ValueError: Digest mismatch, invalid settings/trace, or empty evaluation.
        OSError: Input cannot be read.

    The unlimited-capacity result is an optimistic reuse-work ceiling, NOT a
    finite-capacity optimum, a goodput bound, or evidence of serving benefit.
    """
    capacity_chunks = _integer(capacity_chunks, "capacity_chunks", 1)
    evaluation_start = _integer(evaluation_start, "evaluation_start")
    history = HistoricalAdmission(history_keys, max_request_gap)
    baseline = LRUCacheFast(capacity_chunks)
    historical = LRUCacheFast(capacity_chunks)
    unlimited: set[int] = set()
    totals = {name: WorkTotals() for name in ("no_offload", "lru", "historical")}
    ceiling_tokens = 0
    evaluation_requests = 0
    total_tokens = 0
    requests = 0
    for request_index, (input_length, blocks) in enumerate(
        read_requests(path, expected_sha256)
    ):
        prefix = history.select_prefix(blocks, request_index)
        baseline_hit, baseline_writes = _access_request(baseline, blocks, len(blocks))
        historical_hit, historical_writes = _access_request(historical, blocks, prefix)
        unlimited_hit = 0
        for block in blocks:
            if block not in unlimited:
                break
            unlimited_hit += BLOCK_SIZE
        if request_index >= evaluation_start:
            evaluation_requests += 1
            total_tokens += input_length
            ceiling_tokens += unlimited_hit
            for name, hits, writes in (
                ("lru", baseline_hit, baseline_writes),
                ("historical", historical_hit, historical_writes),
            ):
                totals[name].hit_tokens += hits
                totals[name].written_blocks += writes
        unlimited.update(blocks)
        history.observe(blocks, request_index)
        requests += 1
    if not evaluation_requests:
        raise ValueError("evaluation window contains no requests")
    return {
        "status": "PASS",
        "qualification": "serialized_work_model_only",
        "input_sha256": expected_sha256,
        "block_tokens": BLOCK_SIZE,
        "requests": requests,
        "evaluation_start": evaluation_start,
        "evaluation_requests": evaluation_requests,
        "input_tokens": total_tokens,
        "capacity_chunks": capacity_chunks,
        "history_keys": history_keys,
        "max_request_gap": max_request_gap,
        "tracked_keys_final": history.tracked_keys,
        "modes": {
            name: {
                "hit_tokens": result.hit_tokens,
                "token_hit_rate": result.hit_tokens / total_tokens,
                "written_blocks": result.written_blocks,
            }
            for name, result in totals.items()
        },
        "unlimited_capacity_reuse_work_ceiling_tokens": ceiling_tokens,
        "not_modeled": [
            "current_lazy_scheduler",
            "gpu_apc",
            "concurrency_and_write_completion_delay",
            "transfer_or_recompute_cost",
            "ttft_itl_goodput",
        ],
    }


def main() -> None:
    """Run the experiment; malformed or mismatched evidence exits nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--capacity-chunks", type=int, required=True)
    parser.add_argument("--history-keys", type=int, default=100_000)
    parser.add_argument("--max-request-gap", type=int, default=1000)
    parser.add_argument("--evaluation-start", type=int, default=0)
    args = parser.parse_args()
    try:
        report = evaluate(
            args.input,
            args.expected_sha256,
            args.capacity_chunks,
            args.history_keys,
            args.max_request_gap,
            args.evaluation_start,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
