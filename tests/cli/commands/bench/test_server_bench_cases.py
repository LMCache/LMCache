# SPDX-License-Identifier: Apache-2.0
"""Contract tests for built-in server-bench cases."""

# Standard
from unittest.mock import MagicMock, call

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench.cases.base import BenchResult
from lmcache.cli.commands.bench.server_bench.cases.baseline import (
    BaselineBenchCase,
)
from lmcache.cli.commands.bench.server_bench.client import (
    LookupResult,
    RequestContext,
    TransferResult,
)


def _request(
    request_kind: str,
    num_full_tokens: int = 2,
    total_chunks: int = 1,
) -> RequestContext:
    return RequestContext(
        sequence_id=0,
        request_id=f"req-0-{request_kind}",
        request_kind=request_kind,
        token_ids=(1, 2),
        num_full_tokens=num_full_tokens,
        total_chunks=total_chunks,
        chunk_size=2,
        block_offset=0,
        num_blocks=num_full_tokens // 2,
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
    client.compute_checksums.side_effect = [[cold_checksum], [warm_checksum]]
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
    assert client.method_calls == [
        call.create_request(0, request_id="req-0-cold", request_kind="cold"),
        call.lookup(cold),
        call.compute_checksums(cold, start_token=0, token_count=2),
        call.retrieve(cold, start_token=0, token_count=0),
        call.store(cold, start_token=0, token_count=2),
        call.end_session(cold),
        call.create_request(0, request_id="req-0-warm", request_kind="warm"),
        call.lookup(warm),
        call.zero_destination(warm, start_token=0, token_count=2),
        call.retrieve(warm, start_token=0, token_count=2),
        call.store(warm, start_token=2, token_count=0),
        call.compute_checksums(warm, start_token=0, token_count=2),
        call.end_session(warm),
    ]
    assert result.completed_runs == 1
    assert result.succeeded
    assert result.checks == {
        "cold_lookup_succeeded": [True],
        "warm_lookup_succeeded": [True],
        "cold_full_miss": [True],
        "cold_store_succeeded": [True],
        "warm_full_hit": [True],
        "warm_retrieve_succeeded": [True],
        "checksum_available": [True],
        "checksum_match": [True],
    }
    assert result.latencies_ms == {
        "cold.lookup": [1.0],
        "cold.store": [2.0],
        "warm.lookup": [1.5],
        "warm.retrieve": [2.0],
    }
    assert "  [seq 0] CHECKSUM MATCH OK" in messages


def test_baseline_case_records_checksum_mismatch() -> None:
    messages: list[str] = []

    result = BaselineBenchCase(sequence_count=1, interval_seconds=0).run(
        _client("cold", "warm"),
        messages.append,
    )

    assert result.failed_count("checksum_match") == 1
    assert not result.succeeded
    assert "  [seq 0] CHECKSUM MISMATCH!" in messages


def test_baseline_case_preserves_partial_hit_ranges() -> None:
    cold = _request("cold", num_full_tokens=4, total_chunks=2)
    warm = _request("warm", num_full_tokens=4, total_chunks=2)
    client = MagicMock()
    client.create_request.side_effect = [cold, warm]
    client.lookup.side_effect = [
        LookupResult(1, 2, 1.0),
        LookupResult(1, 2, 1.5),
    ]
    client.compute_checksums.side_effect = [["same"], ["same"]]
    client.retrieve.side_effect = [_transfer("retrieve"), _transfer("retrieve")]
    client.store.side_effect = [_transfer("store"), _transfer("store")]

    result = BaselineBenchCase(sequence_count=1, interval_seconds=0).run(
        client,
        lambda _message: None,
    )

    assert client.compute_checksums.call_args_list == [
        call(cold, start_token=2, token_count=2),
        call(warm, start_token=0, token_count=2),
    ]
    assert client.retrieve.call_args_list == [
        call(cold, start_token=0, token_count=2),
        call(warm, start_token=0, token_count=2),
    ]
    assert client.store.call_args_list == [
        call(cold, start_token=2, token_count=2),
        call(warm, start_token=2, token_count=2),
    ]
    assert result.checks["cold_full_miss"] == [False]
    assert result.checks["warm_full_hit"] == [False]
    assert result.checks["cold_store_succeeded"] == [True]
    assert result.checks["warm_retrieve_succeeded"] == [True]


def test_baseline_case_preserves_results_when_interrupted() -> None:
    client = _client("same", "same")
    client.create_request.side_effect = [
        _request("cold"),
        _request("warm"),
        _request("cold"),
        KeyboardInterrupt,
    ]
    client.lookup.side_effect = [
        LookupResult(0, 1, 1.0),
        LookupResult(1, 1, 1.5),
        LookupResult(0, 1, 2.0),
    ]
    client.compute_checksums.side_effect = [["same"], ["same"], ["pending"]]
    client.retrieve.side_effect = [None, _transfer("retrieve"), None]
    client.store.side_effect = [_transfer("store"), None, _transfer("store")]

    result = BaselineBenchCase(sequence_count=2, interval_seconds=0).run(
        client,
        lambda _message: None,
    )

    assert result.interrupted
    assert result.completed_runs == 1
    assert result.checks["cold_lookup_succeeded"] == [True]
    assert result.checks["warm_lookup_succeeded"] == [True]
    assert result.checks["checksum_match"] == [True]
    assert result.latencies_ms["cold.lookup"] == [1.0, 2.0]
    assert result.latencies_ms["warm.lookup"] == [1.5]


def test_baseline_case_records_failed_lookup_and_latency() -> None:
    client = MagicMock()
    client.create_request.side_effect = [_request("cold"), _request("warm")]
    client.lookup.side_effect = [
        LookupResult(0, 1, 1.0, error="timeout"),
        LookupResult(1, 1, 1.5),
    ]
    client.compute_checksums.return_value = ["warm"]
    client.retrieve.side_effect = [None, _transfer("retrieve")]
    client.store.side_effect = [None, None]

    result = BaselineBenchCase(sequence_count=1, interval_seconds=0).run(
        client,
        lambda _message: None,
    )

    assert result.checks["cold_lookup_succeeded"] == [False]
    assert result.checks["warm_lookup_succeeded"] == [True]
    assert result.latencies_ms["cold.lookup"] == [1.0]
    assert result.latencies_ms["warm.lookup"] == [1.5]
    assert result.checks["cold_full_miss"] == [False]


def test_baseline_case_uses_sequence_offset_and_count() -> None:
    client = MagicMock()
    client.create_request.return_value = None

    result = BaselineBenchCase(
        sequence_count=2,
        sequence_id_offset=5,
        interval_seconds=0,
    ).run(client, lambda _message: None)

    assert client.create_request.call_args_list == [
        call(5, request_id="req-5-cold", request_kind="cold"),
        call(5, request_id="req-5-warm", request_kind="warm"),
        call(6, request_id="req-6-cold", request_kind="cold"),
        call(6, request_id="req-6-warm", request_kind="warm"),
    ]
    assert result.completed_runs == 2


def test_baseline_case_rejects_negative_sequence_count() -> None:
    with pytest.raises(ValueError, match="sequence_count must be non-negative"):
        BaselineBenchCase(sequence_count=-1, interval_seconds=0)


def test_baseline_case_rejects_negative_interval() -> None:
    with pytest.raises(ValueError, match="interval_seconds must be non-negative"):
        BaselineBenchCase(sequence_count=1, interval_seconds=-0.1)


def test_bench_result_records_checks_and_latencies() -> None:
    result = BenchResult(case_name="test")

    result.record_check("lookup", True)
    result.record_checks({"store": True, "retrieve": True})
    result.record_latency("lookup", 1.25)

    assert result.passed_count("lookup") == 1
    assert result.failed_count("lookup") == 0
    assert result.checks == {
        "lookup": [True],
        "store": [True],
        "retrieve": [True],
    }
    assert result.latencies_ms == {"lookup": [1.25]}
    assert result.succeeded
