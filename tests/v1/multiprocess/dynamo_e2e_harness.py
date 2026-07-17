# SPDX-License-Identifier: Apache-2.0

"""Shared harness for the Dynamo KV-event end-to-end tests.

Houses the geometry constants, subprocess server runner, GPU KV-cache
builders, single-key store helper, and the ``server`` fixture reused by both
``test_dynamo_kv_events_e2e.py`` (Tier-1) and ``test_dynamo_indexer_e2e.py``
(Tier-2). Both test modules import what they need from here directly (the
``server`` fixture included), so this harness stays scoped to the Dynamo tests
rather than leaking into the rest of ``tests/v1/multiprocess`` via a conftest.

The non-``test_`` filename keeps pytest from collecting it as a test module;
it only holds shared fixtures and helpers.
"""

# Standard
from typing import Generator
import multiprocessing as mp
import socket
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

CHUNK_SIZE = 256
KV_BLOCK_SIZE = 16  # == page_size; BlockStored.block_size should equal this.
BLOCKS_PER_KEY = CHUNK_SIZE // KV_BLOCK_SIZE  # 16 blocks per stored chunk.
NUM_KEYS = 4
CPU_BUFFER_GB = 1.0
DEFAULT_TIMEOUT = 20.0
MODEL_NAME = "testmodel"
DP_RANK = 0

# Minimal client-side GPU KV cache (the only real VRAM consumer here).
NUM_PAGES = 256
NUM_LAYERS = 4
NUM_HEADS = 4
HEAD_SIZE = 64


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _server_runner(
    host: str,
    port: int,
    chunk_size: int,
    cpu_buffer_gb: float,
    zmq_bind: str,
    kv_block_size: int,
) -> None:
    """Subprocess entry point: run a cache server with Dynamo events on."""
    mp_config = MPServerConfig(host=host, port=port, chunk_size=chunk_size)
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=int(cpu_buffer_gb * 1024**3),
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    # EventBus must be enabled so the evict path (L1_KEYS_EVICTED ->
    # DynamoEvictSubscriber -> BlockRemoved) is drained. Metrics/logging are
    # off to keep the server light and avoid binding the Prometheus port.
    obs_config = ObservabilityConfig(
        enabled=True,
        metrics_enabled=False,
        logging_enabled=False,
        tracing_enabled=False,
        enable_dynamo_kv_events=True,
        dynamo_kv_block_size=kv_block_size,
        dynamo_zmq_bind=zmq_bind,
        dynamo_medium="GPU",
        dynamo_dp_rank=DP_RANK,
    )
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
        start_prometheus_http_server=False,
    )


def _make_kv_cache(device: torch.device) -> list[torch.Tensor]:
    torch.random.manual_seed(42)
    return [
        torch.rand(
            (2, NUM_PAGES, KV_BLOCK_SIZE, NUM_HEADS, HEAD_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(NUM_LAYERS)
    ]


def _wrap_kv_cache(tensors: list[torch.Tensor]) -> KVCache:
    return [CudaIPCWrapper(t) for t in tensors]


def _make_key(index: int, prefix: str = "e2e_request") -> IPCCacheServerKey:
    """A single-chunk key whose tokens are all ``index`` (CHUNK_SIZE long)."""
    token_ids = [index] * CHUNK_SIZE
    return IPCCacheServerKey.from_token_ids(
        MODEL_NAME,
        1,
        0,
        token_ids,
        start=0,
        end=CHUNK_SIZE,
        request_id=f"{prefix}_{index}",
    )


def _store_key(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    instance_id: int,
    block_ids: list[int],
    event: torch.cuda.Event,
) -> None:
    future = client.submit_request(
        RequestType.STORE,
        [key, instance_id, [block_ids], event.ipc_handle()],
        get_response_class(RequestType.STORE),
    )
    result = future.to_cuda_future().result(timeout=DEFAULT_TIMEOUT)
    assert result is True, "store should succeed"


@pytest.fixture(scope="module")
def server() -> Generator[tuple[str, str], None, None]:
    """Spawn the cache server subprocess; yield (mq_url, zmq_endpoint)."""
    mp.set_start_method("spawn", force=True)
    host = "127.0.0.1"
    mq_port = _free_port()
    zmq_port = _free_port()
    mq_url = f"tcp://{host}:{mq_port}"
    zmq_endpoint = f"tcp://{host}:{zmq_port}"

    process = mp.Process(
        target=_server_runner,
        args=(host, mq_port, CHUNK_SIZE, CPU_BUFFER_GB, zmq_endpoint, KV_BLOCK_SIZE),
        daemon=True,
    )
    process.start()
    time.sleep(3)  # let the server bind its MQ + PUB sockets

    try:
        yield mq_url, zmq_endpoint
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
