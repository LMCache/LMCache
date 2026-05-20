# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPWorkerAdapter.register_kv_caches``.

Behavioural coverage of the heartbeat-driven recovery path
(``HeartbeatThread.register_recover_callback`` →
worker re-registration) lives in the buildkite end-to-end test
``.buildkite/k3_tests/multiprocess/scripts/run-restart-recovery.sh``.
That path requires driving the periodic-thread tick loop, which is
deliberately not reachable through any public interface.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.protocol import RequestType


@pytest.fixture
def fake_adapter(monkeypatch):
    """Build an adapter through its real ``__init__`` with the network
    boundary stubbed out. Returns ``(adapter, send_mock, future)`` where
    ``send_mock`` is the patched ``send_lmcache_request`` and ``future``
    is its return value (a ``MagicMock`` whose ``result()`` defaults to
    succeed; tests can attach ``side_effect`` to simulate failures).
    """
    # Stub the MQ boundary so __init__'s chunk-size query and any later
    # send_lmcache_request call don't touch a real socket.
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)

    future = MagicMock(name="future")
    future.result.return_value = None
    send_mock = MagicMock(name="send_lmcache_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    # KV-cache wrapping pulls in CUDA IPC; bypass for unit tests.
    monkeypatch.setattr(adapter_mod, "wrap_kv_caches", lambda kv: list(kv.values()))
    # ``vllm_layout_hints`` returns a ``LayoutHints`` (TypedDict / dict at
    # runtime); the production path performs item assignment on it
    # (``layout_hints["inference_engine_logical_block_size"] = ...``), so
    # the stub must also be a real dict — a string would raise
    # ``TypeError: 'str' object does not support item assignment``.
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
    )

    parallel_strategy = ParallelStrategy(
        use_mla=False,
        kv_world_size=1,
        kv_worker_id=0,
        actual_world_size=1,
        actual_worker_id=0,
        tp_size=1,
        pp_size=1,
    )
    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=parallel_strategy,
        mq_timeout=5.0,
    )
    # __init__ issues exactly one MQ call (the chunk-size query). Reset
    # so individual tests start with a clean call count.
    send_mock.reset_mock()
    return adapter, send_mock, future


def test_register_kv_caches_updates_kv_caches_and_submits(fake_adapter):
    """Public register_kv_caches stores the dict and submits one request."""
    adapter, send_mock, _ = fake_adapter
    new_caches = {"layer.0": object(), "layer.1": object()}

    adapter.register_kv_caches(new_caches)

    assert adapter.kv_caches is new_caches
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE


def test_register_kv_caches_raises_connection_error_on_timeout(fake_adapter):
    """Public register_kv_caches surfaces ConnectionError on MQ timeout."""
    adapter, _send_mock, future = fake_adapter
    future.result.side_effect = TimeoutError("server down")

    with pytest.raises(ConnectionError, match="did not respond"):
        adapter.register_kv_caches({"layer.0": object()})
