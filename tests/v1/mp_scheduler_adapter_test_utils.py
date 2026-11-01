# SPDX-License-Identifier: Apache-2.0
"""Build an MP scheduler adapter through its public constructor for unit tests.

The adapter talks to mocked request clients, and its heartbeat thread is
replaced by a no-op, so tests drive it only through its public API and observe
what each server receives.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import zmq

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_module
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient


class _NoopHeartbeatThread:
    """Heartbeat stand-in that never starts a thread."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def register_recover_callback(self, callback: Any) -> None:
        """Accept the recovery callback without running it."""

    def start(self) -> None:
        """Do not start a thread."""

    def stop(self) -> None:
        """Nothing to stop."""


def ready_future(value: Any) -> MessagingFuture[Any]:
    """Return a completed future carrying ``value``."""
    future: MessagingFuture[Any] = MessagingFuture()
    future.set_result(value)
    return future


def make_request_client(chunk_size: int) -> MagicMock:
    """Return a request client mock for one LMCache server.

    The server reports ``chunk_size``. LOOKUP and QUERY_PREFETCH_STATUS return
    pending futures (``client.lookup.return_value`` and
    ``client.query_prefetch_status.return_value``) that the test completes;
    FREE_LOOKUP_LOCKS, CLEAR and END_SESSION complete immediately.

    Args:
        chunk_size: LMCache chunk size the server reports.

    Returns:
        A ``RequestClient``-spec mock.
    """
    client = MagicMock(spec=RequestClient)
    client.get_chunk_size.return_value = ready_future(chunk_size)
    client.lookup.return_value = MessagingFuture()
    client.query_prefetch_status.return_value = MessagingFuture()
    client.free_lookup_locks.return_value = ready_future(None)
    client.clear.return_value = ready_future(None)
    client.end_session.return_value = ready_future(None)
    return client


def make_scheduler_adapter(
    monkeypatch: pytest.MonkeyPatch,
    clients: dict[str, MagicMock],
    vllm_block_size: int,
    model_name: str = "test_model",
) -> LMCacheMPSchedulerAdapter:
    """Construct a scheduler adapter backed by ``clients``.

    Args:
        monkeypatch: Used to route the adapter's client factory to ``clients``
            and to disable its heartbeat thread.
        clients: Request client of every server, keyed by server URL.
        vllm_block_size: vLLM block size; must divide the servers' chunk size.
        model_name: Model name carried in every key.

    Returns:
        The adapter, with one TP rank and one pipeline stage.
    """
    monkeypatch.setattr(
        adapter_module.RequestClientFactory, "create", lambda url, **_: clients[url]
    )
    monkeypatch.setattr(adapter_module, "HeartbeatThread", _NoopHeartbeatThread)
    return LMCacheMPSchedulerAdapter(
        list(clients),
        zmq.Context.instance(),
        model_name,
        vllm_block_size,
        ParallelStrategy(
            mla_only=False,
            vllm_world_size=1,
            vllm_worker_id=0,
            tp_size=1,
            pp_size=1,
            n_servers=len(clients),
        ),
    )
