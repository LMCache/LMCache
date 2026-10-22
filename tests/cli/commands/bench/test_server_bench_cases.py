# SPDX-License-Identifier: Apache-2.0
"""Contract tests for built-in server-bench cases."""

# Standard
from collections.abc import Callable
from unittest.mock import MagicMock, call
import threading

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.cli.commands.bench.server_bench import client as client_module
from lmcache.cli.commands.bench.server_bench import helpers
from lmcache.cli.commands.bench.server_bench.buffers import KVCacheBuffers
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
    ServerBenchClient,
    TransferResult,
)
from lmcache.cli.commands.bench.server_bench.config import BenchConfig
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
from lmcache.v1.multiprocess import transfer_context
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


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


@pytest.fixture
def run_bench(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[BenchResult, list[str], MagicMock]]:
    """Run the public baseline case while controlling transfer results."""

    def run(
        *,
        mla: bool = True,
        tp_size: int = 2,
        mode: str = "engine_driven",
        fault: str = "none",
    ) -> tuple[BenchResult, list[str], MagicMock]:
        allocated: list[KVCacheBuffers] = []
        stored: dict[int, dict[str, torch.Tensor]] = {}
        block_dim = 0 if mla else 1
        ready: MessagingFuture[bool] = MessagingFuture()
        ready.set_result(True)

        def allocate(
            *,
            groups: list[KVLayerGroupInfo],
            device: str | torch.device | None,
            block_size: int,
        ) -> KVCacheBuffers:
            buffers = KVCacheBuffers(groups, device=device, block_size=block_size)
            rank = len(allocated)
            for layer, tensor in enumerate(buffers.kv_caches.values()):
                tensor.fill_(10 * rank + layer + 1)
            allocated.append(buffers)
            return buffers

        class MemoryTransferContext:
            def __init__(self, instance_id: int, *_args: object) -> None:
                self.rank = instance_id - 1000
                self.register = MagicMock()
                self.unregister = MagicMock(return_value=None)
                self.create_recorded_event = MagicMock(return_value=None)
                self.flush_inflight_stores = MagicMock()
                self.close = MagicMock()

            def submit_store(
                self,
                request_id: str,
                key: IPCCacheServerKey,
                kv_caches: dict[str, torch.Tensor],
                block_ids: list[list[int]],
                event: object,
                blocks_in_chunk: int,
            ) -> MessagingFuture[bool]:
                assert key.worker_id is not None
                stored[key.worker_id] = {
                    name: tensor.index_select(block_dim, torch.tensor(block_ids[0]))
                    for name, tensor in kv_caches.items()
                }
                return ready

            def submit_retrieve(
                self,
                request_id: str,
                key: IPCCacheServerKey,
                kv_caches: dict[str, torch.Tensor],
                block_ids: list[list[int]],
                event: object,
                blocks_in_chunk: int,
                *,
                skip_first_n_tokens: int = 0,
            ) -> MessagingFuture[bool]:
                assert key.worker_id is not None
                if fault == "no_write" and self.rank == tp_size - 1:
                    return ready
                for name, tensor in kv_caches.items():
                    tensor.index_copy_(
                        block_dim,
                        torch.tensor(block_ids[0]),
                        stored[key.worker_id][name],
                    )
                if fault == "corrupt" and self.rank == tp_size - 1:
                    next(iter(kv_caches.values())).select(
                        block_dim, block_ids[0][-1]
                    ).fill_(-1)
                return ready

        def query_checksum(
            http_url: str,
            block_offset: int,
            num_blocks: int,
            block_size: int,
            chunk_size: int,
            instance_id: int,
        ) -> list[str] | None:
            if fault == "unavailable" and instance_id == 1000 + tp_size - 1:
                return None
            return allocated[instance_id - 1000].checksums(
                block_offset, num_blocks, chunk_size
            )

        checksum_query = MagicMock(side_effect=query_checksum)
        monkeypatch.setattr(zmq, "Context", MagicMock())
        monkeypatch.setattr(RequestClientFactory, "create", MagicMock())
        monkeypatch.setattr(helpers, "_get_chunk_size", lambda _client: 2)
        monkeypatch.setattr(client_module, "KVCacheBuffers", allocate)
        monkeypatch.setattr(helpers, "_send_lookup", MagicMock(return_value=True))
        monkeypatch.setattr(
            helpers, "_poll_prefetch_status", MagicMock(side_effect=[0, 2])
        )
        monkeypatch.setattr(helpers, "_send_end_session", MagicMock())
        monkeypatch.setattr(helpers, "_query_checksum", checksum_query)
        monkeypatch.setattr(
            transfer_context, "EngineDrivenTransferContext", MemoryTransferContext
        )
        monkeypatch.setattr(
            transfer_context,
            "create_transfer_context",
            lambda _caches, *, instance_id, **_kwargs: MemoryTransferContext(
                instance_id
            ),
        )
        messages: list[str] = []
        client = ServerBenchClient(
            BenchConfig(
                rpc_url="ipc:///tmp/test-receiver-checksums",
                http_url="http://localhost:8000",
                mode="cpu",
                transfer_mode=mode,
                tp_size=tp_size,
                use_mla=mla,
                num_tokens=3,
                kvcache_shape_spec=f"({1 if mla else 2},8,2,1,4):float16:2",
                num_blocks=8,
                block_size=2,
            ),
            messages.append,
        )
        try:
            client.start()
            result = BaselineBenchCase(
                sequence_count=1, interval_seconds=0, sequence_id_offset=1
            ).run(client, messages.append)
        finally:
            client.close()
        return result, messages, checksum_query

    return run


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


@pytest.mark.parametrize("mode", ["auto", "engine_driven", "lmcache_driven"])
@pytest.mark.parametrize(
    ("mla", "tp_size"), [(True, 1), (True, 2), (True, 4), (False, 2)]
)
def test_all_receivers_match_their_writer(
    run_bench: Callable[..., tuple[BenchResult, list[str], MagicMock]],
    mode: str,
    mla: bool,
    tp_size: int,
) -> None:
    """MLA receivers share rank 0's reference; ordinary TP keeps distinct shards."""
    result, _, query = run_bench(mla=mla, tp_size=tp_size, mode=mode)

    assert result.succeeded
    assert result.checks["checksum_match"] == [True]
    if mode == "lmcache_driven":
        writers = [1000] if mla else list(range(1000, 1000 + tp_size))
        receivers = list(range(1000, 1000 + tp_size))
        assert [c.kwargs["instance_id"] for c in query.call_args_list] == (
            writers + receivers
        )
    else:
        query.assert_not_called()


@pytest.mark.parametrize("mode", ["auto", "engine_driven", "lmcache_driven"])
@pytest.mark.parametrize(
    ("mla", "fault"), [(True, "corrupt"), (True, "no_write"), (False, "corrupt")]
)
def test_secondary_receiver_failure_fails_verification(
    run_bench: Callable[..., tuple[BenchResult, list[str], MagicMock]],
    mode: str,
    mla: bool,
    fault: str,
) -> None:
    """A successful transfer reply must not hide bad data on rank 1."""
    result, messages, _ = run_bench(mla=mla, mode=mode, fault=fault)

    assert result.checks["warm_retrieve_succeeded"] == [True]
    assert result.checks["checksum_match"] == [False]
    assert not result.succeeded
    assert any("rank 1" in message and "FAIL" in message for message in messages)


def test_missing_receiver_checksum_fails_verification(
    run_bench: Callable[..., tuple[BenchResult, list[str], MagicMock]],
) -> None:
    """An unavailable HTTP checksum on rank 1 cannot pass verification."""
    result, _, _ = run_bench(mode="lmcache_driven", fault="unavailable")

    assert result.checks["checksum_available"] == [False]
    assert "checksum_match" not in result.checks
    assert not result.succeeded
