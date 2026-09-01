# SPDX-License-Identifier: Apache-2.0
"""Cold-store and warm-retrieve baseline case."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import itertools
import time

# First Party
from lmcache.cli.commands.bench.server_bench.case import BenchResult
from lmcache.cli.commands.bench.server_bench.client import (
    LookupResult,
    ServerBenchClient,
    TransferResult,
)


@dataclass(frozen=True)
class _RequestPassResult:
    """Results from one cold or warm pass."""

    lookup: LookupResult
    checksums: list[str] | None = None
    retrieve: TransferResult | None = None
    store: TransferResult | None = None


@dataclass(frozen=True)
class BaselineBenchCase:
    """Run the existing cold-store and warm-retrieve baseline."""

    start: int
    end: int | None
    interval: float
    name: str = "baseline"

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ValueError(f"interval must be non-negative, got {self.interval}")

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
            for seq_no in self._sequence_numbers():
                self._run_sequence(client, seq_no, result, log)
        except KeyboardInterrupt:
            result.interrupted = True
        return result

    def _sequence_numbers(self) -> Iterator[int]:
        if self.end is None:
            return itertools.count(self.start)
        return iter(range(self.start, self.end))

    def _run_sequence(
        self,
        client: ServerBenchClient,
        seq_no: int,
        result: BenchResult,
        log: Callable[[str], None],
    ) -> None:
        log("=== Request seq=%d ===" % seq_no)

        cold = self._run_pass(client, seq_no, "cold")
        self._record_pass(result, "cold", cold)

        time.sleep(self.interval)

        warm = self._run_pass(client, seq_no, "warm")
        self._record_pass(result, "warm", warm)

        result.completed_runs += 1
        result.record_check(
            "lookup_succeeded",
            cold is not None and cold.lookup.succeeded,
        )
        result.record_check(
            "lookup_succeeded",
            warm is not None and warm.lookup.succeeded,
        )
        result.record_check(
            "cold_full_miss",
            cold is not None and cold.lookup.is_full_miss,
        )
        result.record_check(
            "store_succeeded",
            cold is not None
            and cold.store is not None
            and bool(cold.store.attempted_worker_ranks)
            and cold.store.succeeded,
        )
        result.record_check(
            "warm_full_hit",
            warm is not None and warm.lookup.is_full_hit,
        )
        result.record_check(
            "retrieve_succeeded",
            warm is not None
            and warm.retrieve is not None
            and bool(warm.retrieve.attempted_worker_ranks)
            and warm.retrieve.succeeded,
        )
        self._record_checksum_result(result, seq_no, cold, warm, log)

        log("")
        time.sleep(self.interval)

    @staticmethod
    def _record_pass(
        result: BenchResult,
        label: str,
        request_pass: _RequestPassResult | None,
    ) -> None:
        if request_pass is None:
            return
        result.record_latency(f"{label}.lookup", request_pass.lookup.latency_ms)
        if request_pass.store is not None:
            result.record_latency(f"{label}.store", request_pass.store.latency_ms)
        if request_pass.retrieve is not None:
            result.record_latency(
                f"{label}.retrieve",
                request_pass.retrieve.latency_ms,
            )

    @staticmethod
    def _record_checksum_result(
        result: BenchResult,
        seq_no: int,
        cold: _RequestPassResult | None,
        warm: _RequestPassResult | None,
        log: Callable[[str], None],
    ) -> None:
        cold_checksums = cold.checksums if cold else None
        warm_checksums = warm.checksums if warm else None
        if not cold_checksums or not warm_checksums:
            result.record_check("checksum_available", False)
            return

        result.record_check("checksum_available", True)
        matched = cold_checksums == warm_checksums
        result.record_check("checksum_match", matched)
        if matched:
            log("  [seq %d] CHECKSUM MATCH OK" % seq_no)
            return

        log("  [seq %d] CHECKSUM MISMATCH!" % seq_no)
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
    def _run_pass(
        client: ServerBenchClient,
        seq_no: int,
        label: str,
    ) -> _RequestPassResult | None:
        request = client.create_request(
            seq_no,
            request_id="req-%d-%s" % (seq_no, label),
            label=label,
        )
        if request is None:
            return None

        lookup = client.lookup(request)
        if not lookup.succeeded:
            return _RequestPassResult(lookup=lookup)

        hit_tokens = lookup.hit_chunks * request.chunk_size
        miss_tokens = request.num_full_tokens - hit_tokens

        checksums: list[str] | None = None
        if label == "cold" and miss_tokens > 0:
            checksums = client.compute_checksums(
                request,
                start_token=hit_tokens,
                token_count=miss_tokens,
            )
        if label == "warm" and hit_tokens > 0:
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

        if label == "warm" and hit_tokens > 0:
            checksums = client.compute_checksums(
                request,
                start_token=0,
                token_count=hit_tokens,
            )

        client.end_session(request)
        return _RequestPassResult(
            lookup=lookup,
            checksums=checksums,
            retrieve=retrieve,
            store=store,
        )
