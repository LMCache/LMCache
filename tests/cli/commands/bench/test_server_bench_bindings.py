# SPDX-License-Identifier: Apache-2.0
"""Request allocation and lifecycle tests."""

# Standard
from collections.abc import Iterator
from contextlib import nullcontext
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench import client as client_module
from lmcache.cli.commands.bench.server_bench.client import (
    RequestContext,
    ServerBenchClient,
)
from lmcache.cli.commands.bench.server_bench.config import BenchConfig
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

Bench = tuple[ServerBenchClient, list[MagicMock]]


def _request(
    client: ServerBenchClient, sequence: int = 0, tokens: int = 4
) -> RequestContext:
    request = client.create_request(
        sequence, str(sequence), "test", token_ids=range(tokens)
    )
    assert request is not None
    return request


@pytest.fixture
def bench(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[Bench]:
    """Real buffers with two mocked worker transports."""
    contexts = [MagicMock() for _ in range(2)]
    for context in contexts:
        ready: MessagingFuture[bool] = MessagingFuture()
        ready.set_result(True)
        context.create_recorded_event.return_value = None
        context.submit_store.return_value = ready
        context.submit_retrieve.return_value = ready
        context.unregister.return_value = None
    monkeypatch.setattr(RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(client_module, "_get_chunk_size", lambda client: 4)
    monkeypatch.setattr(
        client_module, "create_transfer_context", MagicMock(side_effect=contexts)
    )
    client = ServerBenchClient(
        BenchConfig(
            rpc_url="ipc:///tmp/bench-binding-test",
            http_url="",
            mode="cpu",
            transfer_mode="lmcache_driven",
            tp_size=2,
            use_mla=False,
            num_tokens=3,
            num_blocks=8,
            block_size=2,
            kvcache_shape_spec="(2,8,2,1,4):float32:1;(1,8,2,1,8):bfloat16:1",
            **getattr(request, "param", {}),
        ),
        lambda message: None,
    )
    client.start()
    try:
        yield client, contexts
    finally:
        client.close()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_live_requests_are_disjoint_and_released_blocks_can_be_reused(
    bench: Bench,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    client, _ = bench
    first, second = _request(client), _request(client, 1)
    assert first.block_allocation.as_block_ids() == [[0, 1], [0, 1]]
    assert second.block_allocation.as_block_ids() == [[2, 3], [2, 3]]
    cleanup = MagicMock(
        side_effect=RuntimeError("cleanup failed") if cleanup_fails else None
    )
    monkeypatch.setattr(client_module, "_send_end_session", cleanup)
    with (
        pytest.raises(RuntimeError, match="cleanup failed")
        if cleanup_fails
        else nullcontext()
    ):
        client.release_request(first)
    cleanup.assert_called_once()
    assert cleanup.call_args.args[1] == first.request_id
    assert _request(client, 2).block_allocation == first.block_allocation
    with pytest.raises(RuntimeError, match="not active"):
        client.lookup(first)


@pytest.mark.parametrize("bench", [{"block_allocation": "random"}], indirect=True)
def test_checksums_follow_request_blocks_and_receiver_ranks(bench: Bench) -> None:
    """Equal KV hashes equally across physical bindings, with independent TP ranks."""
    client, _ = bench
    first, second = _request(client, tokens=8), _request(client, 1, tokens=8)
    assert client.compute_checksums(first, 0, 0, operation="store") is None
    for start_token, token_count in ((2, 4), (0, 2)):
        with pytest.raises(ValueError, match="chunk-aligned"):
            client.compute_checksums(first, start_token, token_count, operation="store")
    client.fill_deterministic_kv(first)
    client.fill_deterministic_kv(second)
    expected = client.compute_checksums(first, 0, 8, operation="store")
    assert expected is not None and set(expected) == {0, 1}
    assert all(len(chunks) == 2 for chunks in expected.values())
    assert expected[0] != expected[1]
    assert client.compute_checksums(second, 0, 8, operation="retrieve") == expected
    assert client.compute_checksums(second, 4, 4, operation="retrieve") == {
        rank: chunks[1:] for rank, chunks in expected.items()
    }
    client.zero_destination(second, 4, 4)
    actual = client.compute_checksums(second, 0, 8, operation="retrieve")
    assert actual is not None
    for rank, chunks in expected.items():
        assert actual[rank][0] == chunks[0]
        assert actual[rank][1] != chunks[1]
    assert client.compute_checksums(first, 0, 8, operation="store") == expected


@pytest.mark.parametrize("bench", [{"block_allocation": "random"}], indirect=True)
@pytest.mark.parametrize("operation", ["store", "retrieve"])
def test_transfers_use_shared_tp_group_blocks(bench: Bench, operation: str) -> None:
    client, contexts = bench
    request = _request(client, tokens=8)
    result = getattr(client, operation)(request, 4, 4)
    assert result is not None and result.successful_worker_ranks == (0, 1)
    for context in contexts:
        submit = getattr(context, f"submit_{operation}")
        assert submit.call_args.args[3] == request.blocks(4, 4).as_block_ids()


@pytest.mark.parametrize("operation", ["store", "retrieve"])
@pytest.mark.parametrize(
    "failure,rank", [("timeout", 0), ("timeout", 1), ("submit", 1), ("wait", 1)]
)
def test_uncertain_transfer_retains_blocks_until_close(
    bench: Bench,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: str,
    rank: int,
) -> None:
    client, contexts = bench
    submit = getattr(contexts[rank], f"submit_{operation}")
    if failure == "submit":
        submit.side_effect = RuntimeError("transfer failed")
    elif failure == "wait":
        submit.return_value.set_exception(RuntimeError("transfer failed"))
    else:
        submit.return_value = MagicMock(spec=MessagingFuture)
        submit.return_value.result.side_effect = TimeoutError
    request = _request(client)
    cleanup = MagicMock()
    monkeypatch.setattr(client_module, "_send_end_session", cleanup)
    with (
        pytest.raises(RuntimeError, match="transfer failed")
        if failure in ("submit", "wait")
        else nullcontext()
    ):
        result = getattr(client, operation)(request, 0, 4)
        assert result is not None and result.failed_worker_ranks == (rank,)
        assert result.successful_worker_ranks == (1 - rank,)
    client.release_request(request)
    cleanup.assert_not_called()
    with pytest.raises(RuntimeError, match="completion is uncertain"):
        _request(client, 1)
    buffers = contexts[rank].register.call_args.args[0]
    assert buffers
    client.close()
    assert not buffers
