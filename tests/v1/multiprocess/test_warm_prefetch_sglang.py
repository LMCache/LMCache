# SPDX-License-Identifier: Apache-2.0
"""End-to-end warm-prefetch test through the SGLang wire shape.

Runs the real MP server with its HTTP frontend and a filesystem L2 adapter,
registers a KV cache exactly the way ``lmcache.integration.sglang.
multi_process_adapter.LMCacheMPConnector`` does (flat ``[K_layers... ,
V_layers...]`` 3-D pools, ``EngineType.SGLANG``, a ``tokens_per_block``
layout hint, empty engine group infos), stores KV over ZMQ, then drives the
coordinator-facing warm-prefetch surface:

1. ``STORE`` writes the chunks to L1 and (write-through) to the fs L2 adapter.
2. ``POST /cache/clear`` empties L1.
3. ``POST /cache/prefetches`` (token-addressed, the payload the coordinator
   forwards verbatim) reloads the chunks L2 -> L1; the status poll reports
   ``found_keys == total_keys``.
4. The L2 copies are deleted, and a normal SGLang-shaped ``LOOKUP`` +
   ``RETRIEVE`` round-trip returns byte-identical KV -- data that can only
   have come from the warm-prefetched L1 entries.
"""

# Standard
from typing import Generator
import multiprocessing as mp
import os
import tempfile
import time

# Third Party
import httpx
import pytest
import torch
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    HTTPFrontendConfig,
    MPServerConfig,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5661
HTTP_PORT = 8661
SERVER_URL = f"tcp://{SERVER_HOST}:{SERVER_PORT}"
HTTP_URL = f"http://{SERVER_HOST}:{HTTP_PORT}"

MODEL_NAME = "sglang-test-model"
WORLD_SIZE = 1
WORKER_ID = 0
CHUNK_SIZE = 256
PAGE_SIZE = 32
NUM_PAGES = 64
NUM_LAYERS = 4
NUM_HEADS = 4
HEAD_SIZE = 64
NUM_TOKENS = 2 * CHUNK_SIZE
NUM_CHUNKS = NUM_TOKENS // CHUNK_SIZE
NUM_BLOCKS = NUM_TOKENS // PAGE_SIZE

DEFAULT_TIMEOUT = 20.0
POLL_DEADLINE = 30.0


def _cuda_ipc_available() -> bool:
    """Report whether CUDA tensor IPC sharing works on this host."""
    if not torch.cuda.is_available():
        return False
    try:
        buf = torch.empty(1024, device="cuda")
        return buf.untyped_storage()._share_cuda_() is not None
    except Exception:
        return False


if not _cuda_ipc_available():
    pytest.skip(
        "CUDA tensor IPC is not available on this system",
        allow_module_level=True,
    )


def _server_process_runner(l2_path: str) -> None:
    """Run the MP server with HTTP frontend and a filesystem L2 adapter.

    Args:
        l2_path: Directory backing the fs L2 adapter.
    """
    # First Party
    from lmcache.v1.multiprocess.http_server import run_http_server

    run_http_server(
        http_config=HTTPFrontendConfig(http_host=SERVER_HOST, http_port=HTTP_PORT),
        mp_config=MPServerConfig(
            host=SERVER_HOST, port=SERVER_PORT, chunk_size=CHUNK_SIZE
        ),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=1 << 30,
                    use_lazy=True,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[FSL2AdapterConfig(base_path=l2_path)]
            ),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
        coordinator_config=CoordinatorConfig(),
    )


class _SGLangShapedClient:
    """ZMQ client that mirrors the SGLang MP connector's wire calls.

    Holds per-layer K and V pools shaped ``(num_pages * page_size, num_heads,
    head_size)`` and issues REGISTER / STORE / LOOKUP / RETRIEVE requests with
    the exact payloads ``LMCacheMPConnector`` sends.
    """

    def __init__(self, client: MessageQueueClient, device: torch.device) -> None:
        torch.random.manual_seed(42)
        self.client = client
        self.device = device
        self.instance_id = os.getpid()
        pool_shape = (NUM_PAGES * PAGE_SIZE, NUM_HEADS, HEAD_SIZE)
        self.k_pool = [
            torch.rand(pool_shape, dtype=torch.bfloat16, device=device)
            for _ in range(NUM_LAYERS)
        ]
        self.v_pool = [
            torch.rand(pool_shape, dtype=torch.bfloat16, device=device)
            for _ in range(NUM_LAYERS)
        ]

    def wrapped_pools(self) -> KVCache:
        """Return the flat ``[K_layers..., V_layers...]`` IPC-wrapped pools."""
        wrapped: KVCache = []
        wrapped.extend(CudaIPCWrapper(t) for t in self.k_pool)
        wrapped.extend(CudaIPCWrapper(t) for t in self.v_pool)
        return wrapped

    def register(self) -> None:
        """Register the pools with ``EngineType.SGLANG`` and a page-size hint."""
        result = self.client.submit_request(
            RequestType.REGISTER_KV_CACHE,
            [
                self.instance_id,
                self.wrapped_pools(),
                MODEL_NAME,
                WORLD_SIZE,
                EngineType.SGLANG,
                {"tokens_per_block": PAGE_SIZE},
                [],
            ],
            get_response_class(RequestType.REGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)
        assert result is None

    def unregister(self) -> None:
        """Unregister the instance from the server."""
        self.client.submit_request(
            RequestType.UNREGISTER_KV_CACHE,
            [self.instance_id],
            get_response_class(RequestType.UNREGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)

    def store(self, token_ids: list[int], block_ids: list[int]) -> None:
        """STORE a chunk-aligned token range from the given pool blocks."""
        key = IPCCacheServerKey.from_token_ids(
            MODEL_NAME,
            WORLD_SIZE,
            WORKER_ID,
            token_ids,
            start=0,
            end=len(token_ids),
            request_id="warm-sgl-store",
        )
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        result = (
            self.client.submit_request(
                RequestType.STORE,
                [key, self.instance_id, [block_ids], event.ipc_handle()],
                get_response_class(RequestType.STORE),
            )
            .to_cuda_future(device=self.device)
            .result(timeout=DEFAULT_TIMEOUT)
        )
        assert result is True

    def lookup(self, token_ids: list[int], request_id: str) -> int:
        """LOOKUP a token range and poll until the matched chunk count is
        ready."""
        key = IPCCacheServerKey.from_token_ids(
            MODEL_NAME,
            WORLD_SIZE,
            WORKER_ID,
            token_ids,
            start=0,
            end=len(token_ids),
            request_id=request_id,
        ).no_worker_id_version()
        self.client.submit_request(
            RequestType.LOOKUP,
            [key, WORLD_SIZE],
            get_response_class(RequestType.LOOKUP),
        ).result(timeout=DEFAULT_TIMEOUT)
        deadline = time.monotonic() + POLL_DEADLINE
        while time.monotonic() < deadline:
            matched = self.client.submit_request(
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
                get_response_class(RequestType.QUERY_PREFETCH_STATUS),
            ).result(timeout=DEFAULT_TIMEOUT)
            if matched is not None:
                return matched
            time.sleep(0.1)
        raise TimeoutError("LOOKUP result not published in time")

    def retrieve(
        self, token_ids: list[int], block_ids: list[int], request_id: str
    ) -> bool:
        """RETRIEVE a looked-up token range into the given pool blocks."""
        key = IPCCacheServerKey.from_token_ids(
            MODEL_NAME,
            WORLD_SIZE,
            WORKER_ID,
            token_ids,
            start=0,
            end=len(token_ids),
            request_id=request_id,
        )
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        return (
            self.client.submit_request(
                RequestType.RETRIEVE,
                [key, self.instance_id, [block_ids], event.ipc_handle(), 0],
                get_response_class(RequestType.RETRIEVE),
            )
            .to_cuda_future(device=self.device)
            .result(timeout=DEFAULT_TIMEOUT)
        )


@pytest.fixture(scope="module")
def l2_dir() -> Generator[str, None, None]:
    """Provide a temp directory backing the fs L2 adapter."""
    path = tempfile.mkdtemp(prefix="lmcache_sgl_warm_l2_")
    yield path


@pytest.fixture(scope="module")
def server_process(l2_dir: str) -> Generator[mp.Process, None, None]:
    """Start the MP server (ZMQ + HTTP) in a spawned subprocess."""
    mp.set_start_method("spawn", force=True)
    process = mp.Process(target=_server_process_runner, args=(l2_dir,), daemon=True)
    process.start()

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{HTTP_URL}/healthcheck", timeout=2.0).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("MP server HTTP frontend did not come up in 60s")

    yield process

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


@pytest.fixture(scope="function")
def sgl_client(
    server_process: mp.Process,
) -> Generator[_SGLangShapedClient, None, None]:
    """Provide a registered SGLang-shaped client; unregister on teardown."""
    client = MessageQueueClient(server_url=SERVER_URL, context=zmq.Context.instance())
    sgl = _SGLangShapedClient(client, torch.device("cuda:0"))
    sgl.register()
    yield sgl
    try:
        sgl.unregister()
    finally:
        client.close()
        del sgl.k_pool, sgl.v_pool
        torch.cuda.empty_cache()


def _poll_l2_files(l2_dir: str, min_count: int) -> list[str]:
    """Poll the fs L2 adapter's directory until ``min_count`` data files exist.

    Args:
        l2_dir: The fs adapter's base directory.
        min_count: Number of data files to wait for.

    Returns:
        Paths of the data files found.
    """
    deadline = time.monotonic() + POLL_DEADLINE
    while time.monotonic() < deadline:
        files = [
            os.path.join(l2_dir, name)
            for name in os.listdir(l2_dir)
            if name.endswith(".data")
        ]
        if len(files) >= min_count:
            return files
        time.sleep(0.2)
    raise TimeoutError(f"L2 never reached {min_count} data files")


def test_warm_prefetch_restores_sglang_kv(
    sgl_client: _SGLangShapedClient, l2_dir: str
) -> None:
    """Warm prefetch reloads SGLang-stored chunks L2 -> L1 and the data
    survives a lookup + retrieve round trip after the L2 copies are gone."""
    token_ids = list(range(NUM_TOKENS))
    src_blocks = list(range(NUM_BLOCKS))
    dst_blocks = list(range(NUM_BLOCKS, 2 * NUM_BLOCKS))

    sgl_client.store(token_ids, src_blocks)
    l2_files = _poll_l2_files(l2_dir, NUM_CHUNKS)

    resp = httpx.post(f"{HTTP_URL}/cache/clear", json={}, timeout=10.0)
    assert resp.status_code == 200

    resp = httpx.post(
        f"{HTTP_URL}/cache/prefetches",
        json={
            "model_name": MODEL_NAME,
            "world_size": WORLD_SIZE,
            "token_ids": token_ids,
        },
        timeout=10.0,
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "submitted"
    assert body["chunks"] == NUM_CHUNKS
    request_id = body["request_id"]

    deadline = time.monotonic() + POLL_DEADLINE
    status: dict = {}
    while time.monotonic() < deadline:
        status = httpx.get(
            f"{HTTP_URL}/cache/prefetches/{request_id}", timeout=5.0
        ).json()
        if status.get("status") == "completed":
            break
        time.sleep(0.2)
    assert status.get("status") == "completed", status
    assert status["total_keys"] == NUM_CHUNKS * WORLD_SIZE
    assert status["found_keys"] == status["total_keys"]

    # Remove the L2 copies so the retrieve below can only be served by the
    # warm-prefetched L1 entries.
    for path in l2_files:
        os.remove(path)

    matched_chunks = sgl_client.lookup(token_ids, request_id="warm-sgl-load")
    assert matched_chunks == NUM_CHUNKS

    retrieved = sgl_client.retrieve(token_ids, dst_blocks, "warm-sgl-load")
    assert retrieved is True

    torch.cuda.synchronize()
    rows = NUM_TOKENS
    dst_row0 = NUM_BLOCKS * PAGE_SIZE
    for layer in range(NUM_LAYERS):
        for pool in (sgl_client.k_pool, sgl_client.v_pool):
            assert torch.equal(
                pool[layer][:rows], pool[layer][dst_row0 : dst_row0 + rows]
            ), f"KV mismatch at layer {layer}"
