# SPDX-License-Identifier: Apache-2.0
"""Two MP servers share one Device-DAX region; KV stored through one is retrieved
bit-exact through the other. The region is a tmpfs file, or the device named by
``LMCACHE_TEST_SHARED_DEVDAX_PATH`` (its first ``REGION_BYTES`` are overwritten)."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import json
import multiprocessing as mp
import os
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.distributed.config import parse_args
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.transport.request_client import RequestClient
from tests.v1.distributed.shared_l1.utils import region_usage
from tests.v1.memory_orchestrator.harness import (
    REGION_ID,
    free_port,
    running_orchestrator,
)

pytestmark = pytest.mark.cuda

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires available CUDA runtime", allow_module_level=True)

# First Party
from tests.v1.multiprocess.test_cache_server import (  # noqa: E402
    BLOCKS_PER_KEY,
    CHUNK_SIZE,
    ClientContext,
    _recorded_event_handle,
    create_cache_key,
    lookup_all,
    retrieve_keys,
    store_keys,
)

NUM_KEYS = 4
# One chunk is 4 layers x 2 x 256 tokens x 8 heads x 128 x 2 bytes = 4 MiB.
REGION_BYTES = 64 << 20
TIMEOUT = 30.0


def _serve(port: int, region: str, endpoint: str, client_id: str) -> None:
    """Run one MP server whose only L1 is the shared region."""
    shared = dict(orchestrator=endpoint, region_id=REGION_ID, client_id=client_id)
    l1 = dict(type="DEVDAX", tag="cxl", size_gb=REGION_BYTES / 2**30, path=region)
    run_cache_server(
        MPServerConfig(port=port, chunk_size=CHUNK_SIZE),
        parse_args(["--l1-manager", json.dumps({**l1, "shared": shared})]),
        DEFAULT_OBSERVABILITY_CONFIG,
        start_prometheus_http_server=False,
    )


def _wait_until(check: Callable[[], bool]) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


@contextmanager
def _running_servers(region: str, endpoint: str) -> Iterator[list[RequestClient]]:
    """Run two MP servers on ``region`` and yield a client of each once both answer."""
    ports = [free_port(), free_port()]
    servers = [
        mp.get_context("spawn").Process(
            target=_serve, args=(port, region, endpoint, f"server-{i}"), daemon=True
        )
        for i, port in enumerate(ports)
    ]
    clients = [RequestClientFactory.create(f"tcp://localhost:{port}") for port in ports]
    try:
        for server in servers:
            server.start()
        # A request made before its server binds stays queued; one call waits it out.
        for client in clients:
            client.get_chunk_size().result(timeout=TIMEOUT)
        yield clients
    finally:
        for client in clients:
            client.close()
        for server in servers:
            server.terminate()
            server.join(timeout=10)
            if server.is_alive():
                server.kill()


@pytest.fixture
def region() -> Iterator[str]:
    if device := os.environ.get("LMCACHE_TEST_SHARED_DEVDAX_PATH"):
        yield device
        return
    with tempfile.NamedTemporaryFile(prefix="lmcache-shared-", dir="/dev/shm") as file:
        file.truncate(REGION_BYTES)
        yield file.name


def test_store_on_one_server_retrieve_on_the_other(tmp_path: Path, region: str) -> None:
    kv = ClientContext(torch.device(torch_device_type), num_pages=256, num_layers=4)
    instance_id = os.getpid()
    keys = [create_cache_key(i) for i in range(NUM_KEYS)]
    stored = list(range(NUM_KEYS * BLOCKS_PER_KEY))
    loaded = [block + 8 * BLOCKS_PER_KEY for block in stored]
    with (
        running_orchestrator(tmp_path, capacity_bytes=REGION_BYTES) as orchestrator,
        _running_servers(region, orchestrator.endpoint) as (writer, reader),
        region_usage(orchestrator.endpoint) as usage,
    ):
        for client in (writer, reader):
            client.register_kv_cache(
                instance_id, kv.get_kv_cache(), "testmodel", 1, EngineType.VLLM, {}, []
            ).result(timeout=TIMEOUT)
        store_keys(writer, keys, instance_id, stored, _recorded_event_handle(), TIMEOUT)
        _wait_until(lambda: usage().valid == NUM_KEYS and usage().writing == 0)
        # Every chunk is served from the region.
        assert lookup_all(reader, keys, TIMEOUT) == NUM_KEYS
        handle = _recorded_event_handle()
        assert all(retrieve_keys(reader, keys, instance_id, loaded, handle, TIMEOUT))
        torch_dev.synchronize()
        for layer in kv.gpu_kv_caches:
            assert torch.equal(layer[:, stored], layer[:, loaded])
        # Every grant committed and every lease released once the copies drained.
        _wait_until(lambda: usage().read_leases == 0)
        assert (usage().writing, usage().valid) == (0, NUM_KEYS)
