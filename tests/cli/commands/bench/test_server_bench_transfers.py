# SPDX-License-Identifier: Apache-2.0
"""Concurrent transfer and cleanup tests."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import nullcontext
from queue import Queue
from threading import Barrier, Event, Thread, current_thread
from unittest.mock import MagicMock
import time

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


def _request(client: ServerBenchClient, sequence: int = 0) -> RequestContext:
    request = client.create_request(sequence, str(sequence), "test", token_ids=range(8))
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
            num_tokens=7,
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


@pytest.mark.parametrize(
    "bench",
    [{"block_allocation": order} for order in ("sequential", "random")],
    indirect=True,
)
@pytest.mark.parametrize("operation", ["store", "retrieve"])
def test_concurrent_transfers_reuse_threads(
    bench: Bench,
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
    request = _request(client)
    for _ in range(2):
        result = getattr(client, operation)(request, 4, 4)
        assert result is not None and result.successful_worker_ranks == (0, 1)
    first_threads = {threads.get_nowait() for _ in range(2)}
    assert first_threads == {threads.get_nowait() for _ in range(2)}
    for context in contexts:
        submit_mock = getattr(context, f"submit_{operation}")
        assert submit_mock.call_count == 2
        assert submit_mock.call_args.args[3] == request.blocks(4, 4).as_block_ids()


@pytest.mark.parametrize("bench", [{"transfer_timeout_s": 0.1}], indirect=True)
@pytest.mark.parametrize("operation", ["store", "retrieve"])
@pytest.mark.parametrize(
    "failure,rank",
    [
        ("timeout", 0),
        ("timeout", 1),
        ("submit", 1),
        ("wait", 1),
        ("blocked_submit", 0),
        ("blocked_wait", 0),
    ],
)
def test_uncertain_transfer_retains_buffers_until_close(
    bench: Bench,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: str,
    rank: int,
) -> None:
    client, contexts = bench
    submit = getattr(contexts[rank], f"submit_{operation}")
    unblock, entered = Event(), Event()
    blocked = failure.startswith("blocked_")
    ready = submit.return_value

    def stall(*args: object, **kwargs: object) -> MessagingFuture[bool] | bool:
        entered.set()
        assert unblock.wait(5)
        return ready if failure == "blocked_submit" else True

    if failure == "submit":
        submit.side_effect = RuntimeError("transfer failed")
    elif failure == "wait":
        ready.set_exception(RuntimeError("transfer failed"))
    elif failure == "blocked_submit":
        submit.side_effect = stall
    else:
        future = MagicMock(spec=MessagingFuture)
        future.result.side_effect = stall if blocked else TimeoutError
        submit.return_value = future
    request = _request(client)
    cleanup = MagicMock()
    monkeypatch.setattr(client_module, "_send_end_session", cleanup)
    try:
        started = time.monotonic()
        with (
            pytest.raises(RuntimeError, match="transfer failed")
            if failure in ("submit", "wait")
            else nullcontext()
        ):
            result = getattr(client, operation)(request, 0, 4)
            assert result is not None and result.failed_worker_ranks == (rank,)
            assert result.successful_worker_ranks == (1 - rank,)
        assert time.monotonic() - started < 1
        client.release_request(request)
        cleanup.assert_not_called()
        with pytest.raises(RuntimeError, match="completion is uncertain"):
            _request(client, 1)
        if blocked:
            assert entered.is_set()
            buffers = contexts[rank].register.call_args.args[0]
            with ThreadPoolExecutor(max_workers=1) as executor:
                closing = executor.submit(client.close)
                try:
                    with pytest.raises(FuturesTimeoutError):
                        closing.result(timeout=0.2)
                    assert buffers
                    for context in contexts:
                        context.unregister.assert_not_called()
                        context.close.assert_not_called()
                finally:
                    unblock.set()
                closing.result(timeout=5)
            assert not buffers
            for context in contexts:
                context.close.assert_called_once()
    finally:
        unblock.set()
