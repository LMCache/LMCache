# SPDX-License-Identifier: Apache-2.0
"""Ready-boundary registration against a real MP server and CUDA IPC."""

# Standard
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import cast
import multiprocessing as mp
import socket

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorRole,
)

# First Party
from lmcache import torch_dev, torch_device_type  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.v1.distributed.config import (  # noqa: E402
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import (  # noqa: E402
    DEFAULT_OBSERVABILITY_CONFIG,
)
from lmcache.v1.multiprocess.config import MPServerConfig  # noqa: E402
from lmcache.v1.multiprocess.server import run_cache_server  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        torch_device_type != "cuda" or not torch_dev.is_available(),
        reason="requires CUDA IPC",
    ),
    pytest.mark.skipif(
        not hasattr(KVConnectorBase_V1, "on_kv_cache_ready"),
        reason="requires vllm-project/vllm#66574",
    ),
]


def _serve(connection: Connection, port: int) -> None:
    """Expose context snapshots over a test-only pipe, outside the RPC path."""
    result = run_cache_server(
        MPServerConfig(host="127.0.0.1", port=port),
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 << 20,
                    init_size_in_bytes=64 << 20,
                    use_lazy=True,
                    shm_name="",
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        DEFAULT_OBSERVABILITY_CONFIG,
        return_engine=True,
        start_prometheus_http_server=False,
    )
    assert result is not None
    server, engine = result
    connection.send("ready")
    try:
        while connection.recv() != "stop":
            contexts = engine.cache_contexts or {}
            connection.send(
                [
                    [tensor.flatten()[0].item() for tensor in context.kv_tensors]
                    for context in contexts.values()
                ]
            )
    finally:
        server.close()
        engine.close()
        connection.close()


def _snapshot(connection: Connection) -> list[list[float]]:
    """Read registered tensors after the preceding connector call returned."""
    connection.send("snapshot")
    assert connection.poll(30), "server snapshot timed out"
    return connection.recv()


def test_real_server_registration_only_after_ready() -> None:
    """No pre-ready import; final tensors are imported before the hook returns."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve, args=(child, port))
    process.start()
    child.close()
    worker = None
    try:
        assert parent.poll(120), "MP server startup timed out"
        assert parent.recv() == "ready"
        config = cast(
            VllmConfig,
            SimpleNamespace(
                model_config=SimpleNamespace(model="ready-test", use_mla=False),
                parallel_config=SimpleNamespace(
                    world_size=1,
                    rank=0,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                ),
                cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
                scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
                kv_transfer_config=KVTransferConfig(
                    kv_connector="LMCacheMPConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={"lmcache.mp.port": port},
                ),
            ),
        )
        worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
        initial = torch.full((2, 32, 16, 8, 128), 0.25, device=torch_device_type)
        final = torch.full_like(initial, 0.5)
        torch_dev.synchronize()
        worker.register_kv_caches({"layer": initial})
        assert _snapshot(parent) == []
        worker.register_kv_caches({"layer": final})
        worker.on_kv_cache_ready()
        assert _snapshot(parent) == [[0.5]]
        worker.on_kv_cache_ready()
        assert _snapshot(parent) == [[0.5]]
        worker.shutdown()
        worker = None
        assert _snapshot(parent) == []
    finally:
        if worker is not None:
            worker.shutdown()
        if process.is_alive():
            parent.send("stop")
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
