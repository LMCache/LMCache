# SPDX-License-Identifier: Apache-2.0
"""Scheduler steps an L2-backed lookup takes across the ZMQ transport.

Runs the real ``MPCacheServerContext``, ``StorageManager``,
``PrefetchController``, ``DramL1Manager``, bandwidth-limited
``MockL2Adapter``, ``LookupModule``, ZMQ request server/client, and vLLM's
``LookupState``. A scheduler loop calls ``on_scheduler_check`` every
``step`` seconds, as vLLM does for a waiting request.

Reported per cell (median over ``--runs`` lookups of 20 chunks from L2):
  steps     scheduler checks until the hit is seen, counting the first one
  ms_late   milliseconds from L2 publish to the check that observed the hit
  rpcs      status RPCs sent per lookup

Run the same script on ``dev`` and on this branch from the repository root::

    PYTHONPATH=. python benchmarks/microbenchmark/prefetch_status_benchmark.py
"""

# Standard
from collections.abc import Iterator
import argparse
import itertools
import logging
import socket
import statistics
import time

# Third Party
import torch

# First Party
from lmcache.integration.vllm.mp_lookup_state import LookupState
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    MemoryLayoutDesc,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    L2AdaptersConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import l1_owners_of
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.transport.zmq_impl.server import (
    build_zmq_request_server,
)

CHUNK_SIZE = 256
NUM_KEYS = 20
LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([100, 2, 512])],
    dtypes=[torch.bfloat16],
)
OBJ_MB = 100 * 2 * 512 * 2 / 1e6


def _free_port() -> int:
    """Return an unused localhost TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_cell(
    step_s: float,
    load_ms: float,
    runs: int,
    blocking_lookup: bool,
    counter: Iterator[int],
) -> tuple[float, float, float]:
    """Measure one (scheduler step, L2 load latency) cell."""
    bandwidth_gb = NUM_KEYS * OBJ_MB / 1e3 / (load_ms / 1e3)
    ctx = MPCacheServerContext(
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024 * 1024,
                    use_lazy=False,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[
                    MockL2AdapterConfig(
                        max_size_gb=0.2,
                        mock_bandwidth_gb=bandwidth_gb,
                    )
                ]
            ),
        ),
        chunk_size=CHUNK_SIZE,
    )
    ctx.layout_desc_registry.register(
        "m",
        1,
        LAYOUT,
        chunk_size=CHUNK_SIZE,
        hash_algorithm=ctx.hash_algorithm_name,
    )
    sm = ctx.storage_manager
    adapter = sm.l2_adapters()[0][1]
    if not isinstance(adapter, MockL2Adapter):
        raise TypeError(f"Expected MockL2Adapter, got {type(adapter).__name__}")

    last_load_done = 0.0
    orig_query_load = adapter.query_load_result

    def stamped_query_load(task_id: int):
        nonlocal last_load_done
        res = orig_query_load(task_id)
        if res is not None:
            last_load_done = time.monotonic()
        return res

    adapter.query_load_result = stamped_query_load  # type: ignore[method-assign]

    module = LookupModule(ctx)
    port = _free_port()
    server = build_zmq_request_server(
        [module],
        MPServerConfig(host="127.0.0.1", port=port, max_cpu_workers=1),
    )
    server.start()
    url = f"tcp://127.0.0.1:{port}"
    client = RequestClientFactory.create(url)

    status_rpcs = 0
    orig_query = client.query_prefetch_status
    orig_wait = client.wait_prefetch_status

    def counted_query(request_id: str):
        nonlocal status_rpcs
        status_rpcs += 1
        return orig_query(request_id)

    def counted_wait(request_id: str, timeout: float):
        nonlocal status_rpcs
        status_rpcs += 1
        return orig_wait(request_id, timeout)

    client.query_prefetch_status = counted_query  # type: ignore[method-assign]
    client.wait_prefetch_status = counted_wait  # type: ignore[method-assign]

    tokens = list(range(NUM_KEYS * CHUNK_SIZE))
    seed_key = IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=0,
        token_ids=tokens,
        start=0,
        end=len(tokens),
        request_id="seed",
    )
    hashes = ctx.token_hasher.compute_packed_chunk_hashes(
        seed_key.token_bytes, end=seed_key.end
    )
    obj_keys = ipc_key_to_object_keys(
        seed_key, hashes, [0], DEFAULT_SHARD_REPLICATION_DESC
    )[0]

    steps: list[int] = []
    late_ms: list[float] = []
    rpcs: list[int] = []
    try:
        reserved = sm.reserve_write(obj_keys, LAYOUT)
        sm.finish_write(list(reserved), l1_owners_of(reserved))
        l1 = sm.l1_managers()[0]
        while not all(
            adapter.debug_has_key(k) and l1.is_key_evictable(k) for k in obj_keys
        ):
            time.sleep(0.01)

        for _ in range(runs):
            sm.delete_l1_keys(obj_keys)
            request_id = f"r{next(counter)}"
            key = IPCCacheServerKey.from_token_ids(
                model_name="m",
                world_size=1,
                worker_id=None,
                token_ids=tokens,
                start=0,
                end=len(tokens),
                request_id=request_id,
            )
            state = LookupState(
                key,
                {"s": client},
                tp_size=1,
                tokens_per_chunk=CHUNK_SIZE,
                timeout=300.0,
                blocking_lookup=blocking_lookup,
            )
            before = status_rpcs
            state.submit()
            checks = 0
            while True:
                state.on_scheduler_check()
                checks += 1
                seen = time.monotonic()
                if state.get_result() is not None:
                    break
                time.sleep(step_s)
            if state.get_result() != len(tokens):
                raise RuntimeError(f"Unexpected hit tokens: {state.get_result()}")
            steps.append(checks)
            late_ms.append(max(0.0, (seen - last_load_done) * 1e3))
            rpcs.append(status_rpcs - before)
            module.free_lookup_locks(key, tp_size=1)
            module.end_session(request_id)
    finally:
        client.close()
        server.close()
        module.close()
        ctx.close()
    return (
        float(statistics.median(steps)),
        float(statistics.median(late_ms)),
        float(statistics.median(rpcs)),
    )


def main() -> None:
    """Run the scheduler-step / L2-load grid and print median numbers."""
    logging.getLogger("lmcache").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--blocking-status", action="store_true")
    args = parser.parse_args()

    counter = itertools.count()
    print(f"{'step ms':>7} {'L2 load ms':>10} {'steps':>6} {'ms late':>8} {'rpcs':>5}")
    for step_ms in (20, 100):
        for load_ms in (30, 150, 400):
            med_steps, med_late, med_rpcs = run_cell(
                step_ms / 1e3,
                load_ms,
                args.runs,
                args.blocking_status,
                counter,
            )
            print(
                f"{step_ms:7d} {load_ms:10d} {med_steps:6.1f} "
                f"{med_late:8.1f} {med_rpcs:5.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
