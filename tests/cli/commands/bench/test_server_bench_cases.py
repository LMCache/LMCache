# SPDX-License-Identifier: Apache-2.0
"""Contract tests for built-in server-bench cases."""

# Standard
from unittest.mock import MagicMock, call
import threading

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench.cases.base import BenchResult
from lmcache.cli.commands.bench.server_bench.cases.baseline import (
    BaselineBenchCase,
)
from lmcache.cli.commands.bench.server_bench.cases.concurrent import (
    run_concurrently,
)
from lmcache.cli.commands.bench.server_bench.client import (
    LookupResult,
    RequestContext,
    TransferResult,
)


def _request(request_kind: str) -> RequestContext:
    return RequestContext(
        sequence_id=0,
        request_id=f"req-0-{request_kind}",
        request_kind=request_kind,
        token_ids=(1, 2),
        num_full_tokens=2,
        total_chunks=1,
        chunk_size=2,
        block_offset=0,
        num_blocks=1,
    )


def _transfer(operation: str) -> TransferResult:
    return TransferResult(
        operation=operation,
        token_count=2,
        latency_ms=2.0,
        attempted_worker_ranks=(0,),
        successful_worker_ranks=(0,),
        failed_worker_ranks=(),
    )


def _client(cold_checksum: str, warm_checksum: str) -> MagicMock:
    cold = _request("cold")
    warm = _request("warm")
    client = MagicMock()
    client.create_request.side_effect = [cold, warm]
    client.lookup.side_effect = [
        LookupResult(0, 1, 1.0),
        LookupResult(1, 1, 1.5),
    ]
    client.compute_checksums.side_effect = [{0: [cold_checksum]}, {0: [warm_checksum]}]
    client.retrieve.side_effect = [None, _transfer("retrieve")]
    client.store.side_effect = [_transfer("store"), None]
    return client


def test_baseline_case_preserves_cold_warm_flow() -> None:
    client = _client("same", "same")
    messages: list[str] = []

    result = BaselineBenchCase(sequence_count=1, interval_seconds=0).run(
        client,
        messages.append,
    )

    cold = _request("cold")
    warm = _request("warm")
    no_op_calls = [
        call.retrieve(cold, start_token=0, token_count=0),
        call.store(warm, start_token=2, token_count=0),
    ]
    meaningful_calls = [
        method_call
        for method_call in client.method_calls
        if method_call not in no_op_calls
    ]
    assert meaningful_calls == [
        call.create_request(0, request_id="req-0-cold", request_kind="cold"),
        call.lookup(cold),
        call.compute_checksums(cold, start_token=0, token_count=2, operation="store"),
        call.store(cold, start_token=0, token_count=2),
        call.end_session(cold),
        call.create_request(0, request_id="req-0-warm", request_kind="warm"),
        call.lookup(warm),
        call.zero_destination(warm, start_token=0, token_count=2),
        call.retrieve(warm, start_token=0, token_count=2),
        call.compute_checksums(
            warm, start_token=0, token_count=2, operation="retrieve"
        ),
        call.end_session(warm),
    ]
    assert result.completed_runs == 1
    assert result.succeeded
    assert set(result.checks) == {
        "cold_lookup_succeeded",
        "cold_full_miss",
        "cold_store_succeeded",
        "warm_lookup_succeeded",
        "warm_full_hit",
        "warm_retrieve_succeeded",
        "checksum_available",
        "checksum_match",
    }
    assert all(values == [True] for values in result.checks.values())
    assert result.latencies_ms == {
        "cold.lookup": [1.0],
        "cold.store": [2.0],
        "warm.lookup": [1.5],
        "warm.retrieve": [2.0],
    }
    assert "  [seq 0] CHECKSUM MATCH OK" in messages
    assert messages[-1] == ""


def test_baseline_case_records_checksum_mismatch() -> None:
    messages: list[str] = []

    result = BaselineBenchCase(sequence_count=1, interval_seconds=0).run(
        _client("cold", "warm"),
        messages.append,
    )

    assert result.failed_count("checksum_match") == 1
    assert not result.succeeded
    assert "  [seq 0] CHECKSUM MISMATCH!" in messages


def test_baseline_case_preserves_results_when_interrupted() -> None:
    client = _client("same", "same")
    client.create_request.side_effect = [
        _request("cold"),
        _request("warm"),
        KeyboardInterrupt,
    ]

    result = BaselineBenchCase(sequence_count=None, interval_seconds=0).run(
        client,
        lambda _message: None,
    )

    assert result.interrupted
    assert result.completed_runs == 1
    assert result.checks["checksum_match"] == [True]


def test_baseline_case_steps_sequence_ids_by_stride() -> None:
    client = MagicMock()
    client.create_request.return_value = None

    BaselineBenchCase(
        sequence_count=3,
        interval_seconds=0,
        sequence_id_offset=1,
        sequence_id_stride=4,
    ).run(client, lambda _message: None)

    assert [c.args[0] for c in client.create_request.call_args_list] == [
        1,
        1,
        5,
        5,
        9,
        9,
    ]


def test_baseline_case_stops_before_next_sequence_when_stop_is_set() -> None:
    stop = threading.Event()
    stop.set()
    client = MagicMock()

    result = BaselineBenchCase(
        sequence_count=None,
        interval_seconds=0,
        stop=stop,
    ).run(client, lambda _message: None)

    assert result.interrupted
    assert result.completed_runs == 0
    client.create_request.assert_not_called()


class _FixedCase:
    name = "baseline"

    def __init__(self, result: BenchResult) -> None:
        self._result = result

    def run(self, client: object, log: object) -> BenchResult:
        return self._result


def test_run_concurrently_merges_every_case_result() -> None:
    results = [
        BenchResult(
            case_name="baseline",
            completed_runs=2,
            checks={"checksum_match": [True, True]},
            latencies_ms={"cold.lookup": [1.0, 2.0]},
        ),
        BenchResult(
            case_name="baseline",
            completed_runs=1,
            checks={"checksum_match": [False]},
            latencies_ms={"cold.lookup": [3.0]},
        ),
    ]

    merged = run_concurrently(
        [MagicMock(), MagicMock()],
        [_FixedCase(result) for result in results],
        threading.Event(),
        lambda _message: None,
    )

    assert merged.completed_runs == 3
    assert merged.passed_count("checksum_match") == 2
    assert merged.failed_count("checksum_match") == 1
    assert sorted(merged.latencies_ms["cold.lookup"]) == [1.0, 2.0, 3.0]
    assert not merged.interrupted


def test_run_concurrently_stops_other_cases_and_reraises_first_error() -> None:
    stop = threading.Event()
    blocked = threading.Event()

    class _FailingCase:
        name = "baseline"

        def run(self, client: object, log: object) -> BenchResult:
            raise RuntimeError("lookup timed out")

    class _WaitingCase:
        name = "baseline"

        def run(self, client: object, log: object) -> BenchResult:
            assert stop.wait(timeout=5)
            blocked.set()
            return BenchResult(case_name="baseline", interrupted=True)

    with pytest.raises(RuntimeError, match="lookup timed out"):
        run_concurrently(
            [MagicMock(), MagicMock()],
            [_WaitingCase(), _FailingCase()],
            stop,
            lambda _message: None,
        )

    assert blocked.is_set()


def test_run_concurrently_needs_one_client_per_case() -> None:
    with pytest.raises(ValueError, match="need one each"):
        run_concurrently(
            [MagicMock()],
            [_FixedCase(BenchResult(case_name="baseline"))] * 2,
            threading.Event(),
            lambda _message: None,
        )
