# SPDX-License-Identifier: Apache-2.0
"""GPU check for the preemption block-reuse race (T5 in the preemption TDD).

When vLLM preempts a request it frees the request's GPU blocks immediately
and can hand them to another request in the same scheduling step.  A STORE
submitted for those blocks in the previous step may still be reading them.
``LMCacheMPWorkerAdapter.handle_preemptions(True)`` must not return until
every submitted store has finished reading, so the next forward pass cannot
overwrite blocks under an in-flight copy.

This test drives the real worker adapter and scheduler adapter against a
real LMCache MP server process: store a pattern, call ``handle_preemptions``,
overwrite the blocks with a different pattern (the "next forward pass"),
then retrieve into fresh blocks and check the original pattern came back.
"""

# Standard
from collections.abc import Generator
import multiprocessing as mp
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type

pytestmark = pytest.mark.cuda

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires a CUDA device", allow_module_level=True)

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
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
from lmcache.v1.multiprocess.group_view import EngineGroupInfo  # noqa: E402
from lmcache.v1.multiprocess.server import run_cache_server  # noqa: E402

SERVER_HOST = "localhost"
# One server per transfer mode; the server's supported_transfer_mode must
# admit the worker's mode or engine-driven registration never completes.
SERVER_PORTS = {"lmcache_driven": 5611, "engine_driven": 5612}
CHUNK_TOKENS = 64
BLOCK_TOKENS = 16
BLOCKS_PER_CHUNK = CHUNK_TOKENS // BLOCK_TOKENS
NUM_LAYERS = 4
NUM_PAGES = 64
NUM_HEADS = 8
HEAD_SIZE = 128
TIMEOUT = 30.0
MODEL_NAME = "preemption-race-model"


def _server_runner(port: int, transfer_mode: str) -> None:
    """Entry point of the LMCache MP server process for one transfer mode."""
    run_cache_server(
        mp_config=MPServerConfig(
            transport="zmq",
            host=SERVER_HOST,
            port=port,
            chunk_size=CHUNK_TOKENS,
            supported_transfer_mode=transfer_mode,  # type: ignore[arg-type]
        ),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=1024**3,
                    use_lazy=True,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture(params=["lmcache_driven", "engine_driven"])
def transfer_mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def server_process(transfer_mode: str) -> Generator[mp.Process, None, None]:
    mp.set_start_method("spawn", force=True)
    process = mp.Process(
        target=_server_runner,
        args=(SERVER_PORTS[transfer_mode], transfer_mode),
        daemon=True,
    )
    process.start()
    time.sleep(2)
    yield process
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


def _strategy() -> ParallelStrategy:
    return ParallelStrategy(
        mla_only=False,
        vllm_world_size=1,
        vllm_worker_id=0,
        tp_size=1,
        pp_size=1,
        n_servers=1,
    )


def _kv_caches(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        f"model.layers.{i}.self_attn.attn": torch.zeros(
            (2, NUM_PAGES, BLOCK_TOKENS, NUM_HEADS, HEAD_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        for i in range(NUM_LAYERS)
    }


def _fill(kv_caches: dict[str, torch.Tensor], blocks: list[int], base: float) -> None:
    """Write a block-distinct pattern into ``blocks`` on the current stream."""
    for layer_idx, tensor in enumerate(kv_caches.values()):
        for i, block in enumerate(blocks):
            tensor[:, block] = base + layer_idx + i / 16.0


def _snapshot(
    kv_caches: dict[str, torch.Tensor], blocks: list[int]
) -> list[torch.Tensor]:
    torch_dev.synchronize()
    return [t[:, blocks].clone().cpu() for t in kv_caches.values()]


def _wait_finished(
    adapter: LMCacheMPWorkerAdapter, request_id: str, *, sending: bool
) -> None:
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        finished_sending, finished_recving = adapter.get_finished({request_id})
        done = finished_sending if sending else finished_recving
        if done and request_id in done:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"{request_id} never finished ({'send' if sending else 'recv'})"
    )


def _lookup(
    scheduler_adapter: LMCacheMPSchedulerAdapter, rid: str, tokens: list[int]
) -> int:
    deadline = time.monotonic() + TIMEOUT
    scheduler_adapter.maybe_submit_lookup_request(rid, tokens)
    while time.monotonic() < deadline:
        hit = scheduler_adapter.check_lookup_result(rid)
        if hit is not None:
            return hit
        time.sleep(0.01)
    raise AssertionError("lookup did not complete")


def _lookup_until_hit(
    scheduler_adapter: LMCacheMPSchedulerAdapter, tokens: list[int], want: int
) -> int:
    """Poll lookups until ``want`` tokens hit or the deadline passes.

    The server commits a store (``finish_write``) on its stream *after*
    recording the completion event the worker waits on, so a lookup issued
    immediately after the store's future resolves can still miss.  In vLLM
    the resume lookup happens at least one scheduler step later.
    """
    deadline = time.monotonic() + TIMEOUT
    attempt = 0
    hit = 0
    while time.monotonic() < deadline:
        rid = f"resume-{attempt}"
        hit = _lookup(scheduler_adapter, rid, tokens)
        if hit >= want:
            return hit
        scheduler_adapter.cleanup_lookup_result(rid)
        scheduler_adapter.end_session(rid)
        attempt += 1
        time.sleep(0.05)
    return hit


def test_store_survives_block_reuse_after_handle_preemptions(
    server_process: mp.Process, transfer_mode: str
) -> None:
    assert server_process.is_alive()
    url = f"tcp://{SERVER_HOST}:{SERVER_PORTS[transfer_mode]}"
    context = zmq.Context.instance()
    device = torch.device(torch_device_type)
    kv_caches = _kv_caches(device)

    worker = LMCacheMPWorkerAdapter(
        server_url=url,
        context=context,
        model_name=MODEL_NAME,
        vllm_block_size=BLOCK_TOKENS,
        parallel_strategy=_strategy(),
        mq_timeout=TIMEOUT,
        extra_config={"lmcache.mp.mp_transfer_mode": transfer_mode},
    )
    scheduler = LMCacheMPSchedulerAdapter(
        server_urls=[url],
        context=context,
        model_name=MODEL_NAME,
        vllm_block_size=BLOCK_TOKENS,
        parallel_strategy=_strategy(),
        mq_timeout=TIMEOUT,
    )
    try:
        worker.register_kv_caches(
            kv_caches,
            engine_group_infos=[
                EngineGroupInfo(
                    engine_group_id=0,
                    layer_indices=tuple(range(NUM_LAYERS)),
                    tokens_per_block=BLOCK_TOKENS,
                )
            ],
            layout_hints={},
        )

        # Distinct tokens per test so the cache is cold for this key.
        base_token = 10_000 + (hash(transfer_mode) % 1000) * 100
        tokens = [base_token + i for i in range(CHUNK_TOKENS)]
        victim_blocks = list(range(0, BLOCKS_PER_CHUNK))
        reuse_blocks = list(range(BLOCKS_PER_CHUNK, 2 * BLOCKS_PER_CHUNK))

        # Step N: the victim's forward pass wrote pattern A; wait_for_save
        # submits the store for its blocks.
        _fill(kv_caches, victim_blocks, base=1.0)
        expected = _snapshot(kv_caches, victim_blocks)
        event = worker.create_recorded_event()
        worker.submit_store_request(
            "victim", LoadStoreOp(tokens, [victim_blocks], 0, CHUNK_TOKENS), event
        )

        # Step N+1: the scheduler preempted the victim and gave its blocks to
        # another request; the worker must drain the store before the
        # forward pass overwrites them.
        worker.handle_preemptions(need_flush_before_forward=True)
        _fill(kv_caches, victim_blocks, base=50.0)  # the other request's KV
        torch_dev.synchronize()

        _wait_finished(worker, "victim", sending=True)

        # Resume: lookup over the victim's tokens must hit, and the retrieve
        # into fresh blocks must reproduce pattern A, not the overwrite.
        assert _lookup_until_hit(scheduler, tokens, CHUNK_TOKENS) == CHUNK_TOKENS
        event = worker.create_recorded_event()
        worker.submit_retrieve_request(
            "resume", LoadStoreOp(tokens, [reuse_blocks], 0, CHUNK_TOKENS), event
        )
        _wait_finished(worker, "resume", sending=False)
        assert worker.get_block_ids_with_load_errors() == set()

        got = _snapshot(kv_caches, reuse_blocks)
        for layer_idx, (exp, act) in enumerate(zip(expected, got, strict=True)):
            assert torch.equal(exp, act), (
                f"[{transfer_mode}] layer {layer_idx}: retrieved KV differs from "
                f"what the victim computed; the store read overwritten blocks"
            )
    finally:
        scheduler.shutdown()
        worker.shutdown()
