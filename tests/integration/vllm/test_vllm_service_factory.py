# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for vLLM offload endpoint identity.

Worker tests run the real factory, server constructor, and RPC path helper
with a mocked engine, ZMQ context, and background thread.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import zmq

# First Party
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache


@pytest.mark.parametrize("global_rank", [0, 2, 3])
def test_worker_offload_server_binds_global_rank_endpoint(
    global_rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each worker must bind the endpoint derived from its own global rank.

    In a PP=2, TP=2 layout, workers 0 and 2 share TP-local rank 0 but must
    bind different endpoints within the same engine's IPC namespace.
    """
    engine_id = "offload-rank-test"
    rpc_port = 731
    monkeypatch.setenv("LMCACHE_OFFLOAD_RPC_PORT", str(rpc_port))
    vllm_config = MagicMock()
    vllm_config.parallel_config.rank = global_rank
    engine = MagicMock(spec=LMCacheEngine)
    engine.metadata = SimpleNamespace(engine_id=engine_id, worker_id=global_rank)
    factory = VllmServiceFactory(
        LMCacheEngineConfig.from_defaults(), vllm_config, role="worker"
    )
    # Supply an existing engine so this test only exercises service creation.
    factory.lmcache_engine = engine
    expected_path = get_zmq_rpc_path_lmcache(
        engine_id, "offload", rpc_port, global_rank
    )

    with (
        patch(
            "lmcache.v1.offload_server.zmq_server.get_zmq_context"
        ) as context_factory,
        patch("lmcache.v1.offload_server.zmq_server.threading.Thread"),
    ):
        server = factory.maybe_create_offload_server()

    assert server is not None
    context = context_factory.return_value
    context.socket.assert_called_once_with(zmq.REP)
    context.socket.return_value.bind.assert_called_once_with(f"ipc://{expected_path}")


def test_scheduler_does_not_create_offload_server() -> None:
    """The scheduler must create neither an offload server nor a cache engine."""
    factory = VllmServiceFactory(
        LMCacheEngineConfig.from_defaults(), MagicMock(), role="scheduler"
    )

    with patch(
        "lmcache.integration.vllm.vllm_service_factory.ZMQOffloadServer"
    ) as offload_server_cls:
        offload_server = factory.maybe_create_offload_server()

    offload_server_cls.assert_not_called()
    assert offload_server is None
    assert factory.lmcache_engine is None
