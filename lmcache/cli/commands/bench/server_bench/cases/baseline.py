# SPDX-License-Identifier: Apache-2.0
"""Cold-store and warm-retrieve baseline case."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal
import itertools
import time

# First Party
from lmcache.cli.commands.bench.server_bench.cases.base import BenchResult
from lmcache.cli.commands.bench.server_bench.client import (
    LookupResult,
    ServerBenchClient,
    TransferResult,
)


def _lookup_succeeded(outcome: _RequestOutcome | None) -> bool:
    return outcome is not None and outcome.lookup.succeeded


def _transfer_succeeded(transfer: TransferResult | None) -> bool:
    return (
        transfer is not None
        and bool(transfer.attempted_worker_ranks)
        and transfer.succeeded
    )


@dataclass(frozen=True)
class _RequestOutcome:
    """Outcome of one cold or warm request."""

    lookup: LookupResult
    checksums: list[str] | None = None
    retrieve: TransferResult | None = None
    store: TransferResult | None = None


@dataclass(frozen=True)
class BaselineBenchCase:
    """Run the existing cold-store and warm-retrieve baseline."""

    sequence_count: int | None
    interval_seconds: float
    sequence_id_offset: int = 0
    name: str = "baseline"

    def __post_init__(self) -> None:
        if self.sequence_count is not None and self.sequence_count < 0:
            raise ValueError(
                f"sequence_count must be non-negative, got {self.sequence_count}"
            )
        if self.interval_seconds < 0:
            raise ValueError(
                f"interval_seconds must be non-negative, got {self.interval_seconds}"
            )

    def run(
        self,
        client: ServerBenchClient,
        log: Callable[[str], None],
    ) -> BenchResult:
        """Run the configured baseline sequence range.

        Args:
            client: Started server-bench client.
            log: Progress logger.

        Returns:
            Structured checks and operation latencies. An interrupted run
            contains results from completed sequences.
        """
        result = BenchResult(case_name=self.name)
        try:
            for sequence_id in self._sequence_ids():
                self._run_sequence(client, sequence_id, result, log)
        except KeyboardInterrupt:
            result.interrupted = True
        return result

    def _sequence_ids(self) -> Iterator[int]:
        if self.sequence_count is None:
            return itertools.count(self.sequence_id_offset)
        return iter(
            range(
                self.sequence_id_offset,
                self.sequence_id_offset + self.sequence_count,
            )
        )

    def _run_sequence(
        self,
        client: ServerBenchClient,
        sequence_id: int,
        result: BenchResult,
        log: Callable[[str], None],
    ) -> None:
        log("=== Request seq=%d ===" % sequence_id)

        cold_outcome = self._execute_request(client, sequence_id, "cold")
        self._record_request_latencies(result, "cold", cold_outcome)

        time.sleep(self.interval_seconds)

        warm_outcome = self._execute_request(client, sequence_id, "warm")
        self._record_request_latencies(result, "warm", warm_outcome)

        result.completed_runs += 1
        result.record_checks(self._evaluate_sequence(cold_outcome, warm_outcome))
        self._record_checksum_result(
            result,
            sequence_id,
            cold_outcome,
            warm_outcome,
            log,
        )

        time.sleep(self.interval_seconds)

    @staticmethod
    def _evaluate_sequence(
        cold: _RequestOutcome | None,
        warm: _RequestOutcome | None,
    ) -> dict[str, bool]:
        return {
            "cold_lookup_succeeded": _lookup_succeeded(cold),
            "cold_full_miss": bool(cold and cold.lookup.is_full_miss),
            "cold_store_succeeded": _transfer_succeeded(cold.store if cold else None),
            "warm_lookup_succeeded": _lookup_succeeded(warm),
            "warm_full_hit": bool(warm and warm.lookup.is_full_hit),
            "warm_retrieve_succeeded": _transfer_succeeded(
                warm.retrieve if warm else None
            ),
        }

    @staticmethod
    def _record_request_latencies(
        result: BenchResult,
        request_kind: Literal["cold", "warm"],
        outcome: _RequestOutcome | None,
    ) -> None:
        if outcome is None:
            return
        result.record_latency(f"{request_kind}.lookup", outcome.lookup.latency_ms)
        if outcome.store is not None:
            result.record_latency(f"{request_kind}.store", outcome.store.latency_ms)
        if outcome.retrieve is not None:
            result.record_latency(
                f"{request_kind}.retrieve",
                outcome.retrieve.latency_ms,
            )

    @staticmethod
    def _record_checksum_result(
        result: BenchResult,
        sequence_id: int,
        cold_outcome: _RequestOutcome | None,
        warm_outcome: _RequestOutcome | None,
        log: Callable[[str], None],
    ) -> None:
        cold_checksums = cold_outcome.checksums if cold_outcome else None
        warm_checksums = warm_outcome.checksums if warm_outcome else None
        if not cold_checksums or not warm_checksums:
            result.record_check("checksum_available", False)
            return

        result.record_check("checksum_available", True)
        matched = cold_checksums == warm_checksums
        result.record_check("checksum_match", matched)
        if matched:
            log("  [seq %d] CHECKSUM MATCH OK" % sequence_id)
            return

        log("  [seq %d] CHECKSUM MISMATCH!" % sequence_id)
        for index, (cold_checksum, warm_checksum) in enumerate(
            zip(cold_checksums, warm_checksums, strict=False)
        ):
            log(
                "    chunk %d: cold=%s warm=%s %s"
                % (
                    index,
                    cold_checksum[:12],
                    warm_checksum[:12],
                    "OK" if cold_checksum == warm_checksum else "FAIL",
                )
            )

    @staticmethod
    def _execute_request(
        client: ServerBenchClient,
        sequence_id: int,
        request_kind: Literal["cold", "warm"],
    ) -> _RequestOutcome | None:
        request = client.create_request(
            sequence_id,
            request_id="req-%d-%s" % (sequence_id, request_kind),
            request_kind=request_kind,
        )
        if request is None:
            return None

        lookup = client.lookup(request)
        if not lookup.succeeded:
            return _RequestOutcome(lookup=lookup)

        hit_tokens = lookup.hit_chunks * request.chunk_size
        miss_tokens = request.num_full_tokens - hit_tokens

        checksums: list[str] | None = None
        if request_kind == "cold" and miss_tokens > 0:
            checksums = client.compute_checksums(
                request,
                start_token=hit_tokens,
                token_count=miss_tokens,
            )
        if request_kind == "warm" and hit_tokens > 0:
            client.zero_destination(
                request,
                start_token=0,
                token_count=hit_tokens,
            )

        retrieve = client.retrieve(
            request,
            start_token=0,
            token_count=hit_tokens,
        )
        store = client.store(
            request,
            start_token=hit_tokens,
            token_count=miss_tokens,
        )

        if request_kind == "warm" and hit_tokens > 0:
            checksums = client.compute_checksums(
                request,
                start_token=0,
                token_count=hit_tokens,
            )

        client.end_session(request)
        return _RequestOutcome(
            lookup=lookup,
            checksums=checksums,
            retrieve=retrieve,
            store=store,
        )
