# SPDX-License-Identifier: Apache-2.0
"""Request allocation and lifecycle tests."""

# Standard
from collections.abc import Iterator
from contextlib import nullcontext
from queue import Queue
from threading import Barrier, Thread, current_thread
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench import helpers
from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient
from lmcache.cli.commands.bench.server_bench.config import BenchConfig
from lmcache.v1.multiprocess import transfer_context
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


@pytest.fixture
def bench(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[ServerBenchClient, list[MagicMock]]]:
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
    monkeypatch.setattr(helpers, "_get_chunk_size", lambda client: 4)
    monkeypatch.setattr(
        transfer_context, "create_transfer_context", MagicMock(side_effect=contexts)
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
    bench: tuple[ServerBenchClient, list[MagicMock]],
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    client, _ = bench
    first = client.create_request(0, "first", "test", token_ids=range(4))
    second = client.create_request(1, "second", "test", token_ids=range(4))
    assert first is not None and second is not None
    assert first.block_ids == (((0, 1), (0, 1)),) * 2
    assert second.block_ids == (((2, 3), (2, 3)),) * 2
    cleanup = MagicMock(
        side_effect=RuntimeError("cleanup failed") if cleanup_fails else None
    )
    monkeypatch.setattr(helpers, "_send_end_session", cleanup)
    with (
        pytest.raises(RuntimeError, match="cleanup failed")
        if cleanup_fails
        else nullcontext()
    ):
        client.release_request(first)
    cleanup.assert_called_once()
    assert cleanup.call_args.args[1] == first.request_id
    reused = client.create_request(2, "reused", "test", token_ids=range(4))
    assert reused is not None and reused.block_ids == first.block_ids
    with pytest.raises(RuntimeError, match="not active"):
        client.fill_deterministic_kv(first)


@pytest.mark.parametrize("operation", ["store", "retrieve"])
def test_concurrent_transfers_use_worker_group_blocks(
    bench: tuple[ServerBenchClient, list[MagicMock]],
    operation: str,
) -> None:
    client, contexts = bench
    barrier = Barrier(2, timeout=5)
    threads: Queue[Thread] = Queue()

    def submit(*args: object) -> MessagingFuture[bool]:
        threads.put(current_thread())
        barrier.wait()
        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(True)
        return future

    for context in contexts:
        getattr(context, f"submit_{operation}").side_effect = submit
    ids = [[[7, 2, 5, 0], [6, 1, 4, 3]], [[0, 3, 2, 7], [1, 4, 5, 6]]]
    request = client.create_request(
        0, "partial", "test", token_ids=range(8), block_ids=ids
    )
    assert request is not None
    for _ in range(2):
        result = getattr(client, operation)(request, 4, 4)
        assert result is not None and result.successful_worker_ranks == (0, 1)
    first_threads = {threads.get_nowait() for _ in range(2)}
    assert first_threads == {threads.get_nowait() for _ in range(2)}
    for context, groups in zip(contexts, ids, strict=True):
        submit_mock = getattr(context, f"submit_{operation}")
        assert submit_mock.call_count == 2
        assert submit_mock.call_args.args[3] == [blocks[2:] for blocks in groups]


@pytest.mark.parametrize("operation", ["store", "retrieve"])
@pytest.mark.parametrize(
    "failure,rank", [("timeout", 0), ("timeout", 1), ("submit", 1), ("wait", 1)]
)
def test_uncertain_transfer_prevents_block_reuse(
    bench: tuple[ServerBenchClient, list[MagicMock]],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: str,
    rank: int,
) -> None:
    client, contexts = bench
    submit = getattr(contexts[rank], f"submit_{operation}")
    if failure == "timeout":
        monkeypatch.setattr(
            helpers,
            "_wait_for_result",
            lambda future, **kwargs: (
                helpers._TIMEOUT if future is submit.return_value else True
            ),
        )
    elif failure == "submit":
        submit.side_effect = RuntimeError("transfer failed")
    else:
        submit.return_value.set_exception(RuntimeError("transfer failed"))
    request = client.create_request(0, "failed", "test", token_ids=range(4))
    assert request is not None
    with (
        pytest.raises(RuntimeError, match="transfer failed")
        if failure != "timeout"
        else nullcontext()
    ):
        result = getattr(client, operation)(request, 0, 4)
        assert result is not None and result.failed_worker_ranks == (rank,)
        assert result.successful_worker_ranks == (1 - rank,)
    cleanup = MagicMock()
    monkeypatch.setattr(helpers, "_send_end_session", cleanup)
    client.release_request(request)
    cleanup.assert_not_called()
    with pytest.raises(RuntimeError, match="completion is uncertain"):
        client.create_request(1, "reuse", "test", token_ids=range(4))
