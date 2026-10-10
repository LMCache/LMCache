# SPDX-License-Identifier: Apache-2.0
"""Measure device staging and verify paged KV writes for first-rank retrieval.

Drives a real ``LMCacheEngine`` with the vLLM paged connector on CUDA. Run with
the tested checkout on ``PYTHONPATH``. Single process::

    python -m tests.v1.manual_bounded_retrieval

Two NCCL ranks, one visible GPU each, the same rendezvous on both::

    python -m tests.v1.manual_bounded_retrieval --world-size 2 --rank 0 \
        --rendezvous tcp://HOST:PORT

Results are JSON lines. ``fixed_reservation_bytes`` is the staging buffer the
engine holds for its lifetime; ``extra_peak_bytes`` is the device memory a
retrieval allocated beyond it and must stay within one buffer. One source chunk
is ``LAYERS * CHUNK_TOKENS * WIDTH`` elements, i.e. 1 MiB at two bytes each, so
every budget in the matrix is at least that. Recoverable failures are injected
after real CUDA writes; a failed window may therefore be written but is not
committed in the returned mask. No device fault is induced.
"""

# Standard
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock
import argparse
import gc
import json
import time

# Third Party
import torch
import torch.distributed as dist

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import TokenDatabase

LAYERS, WIDTH, CHUNK_TOKENS, PAGE_TOKENS = 4, 128, 1024, 16
VLLM_CACHED_TOKENS = 17
MIB = 1024**2
BUDGETS = (MIB, 4 * MIB, 16 * MIB)


class FailingConnector(VLLMPagedMemGPUConnectorV2):
    """The vLLM paged connector with one injected write failure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_on_write: int | None = None
        self.write_count = 0
        self.written_until = 0

    def batched_to_gpu(
        self,
        memory_objs: Any = None,
        starts: Any = None,
        ends: Any = None,
        **kwargs: Any,
    ) -> None:
        """Write the window, then raise if this is the injected failure."""
        self.write_count += 1
        super().batched_to_gpu(memory_objs, starts, ends, **kwargs)
        self.written_until = max(self.written_until, max(ends))
        if self.write_count == self.fail_on_write:
            raise torch.OutOfMemoryError("injected failure after native CUDA enqueue")


def make_pinned_object(tensor: torch.Tensor) -> MemoryObj:
    """A KV_MLA_FMT memory object over pinned host memory, as L1 would hold it."""
    raw = tensor.reshape(-1).view(torch.uint8).pin_memory()
    return TensorMemoryObj(
        raw,
        MemoryObjMetadata(
            shape=tensor.shape,
            dtype=tensor.dtype,
            address=0,
            phy_size=raw.numel(),
            ref_count=1,
            fmt=MemoryFormat.KV_MLA_FMT,
        ),
        None,
    )


def build_engine(
    rank: int,
    world_size: int,
    host_group: Any,
    budget: int,
    dtype: torch.dtype,
    connector: VLLMPagedMemGPUConnectorV2,
) -> LMCacheEngine:
    """Create a first-rank retrieval engine wired to NCCL/Gloo or to itself."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_TOKENS, retrieve_buffer_size=budget, py_enable_gc=True
    )
    metadata = LMCacheMetadata(
        model_name="manual",
        world_size=world_size,
        local_world_size=1,
        worker_id=rank,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=(LAYERS, 1, CHUNK_TOKENS, 1, WIDTH),
        use_mla=True,
    )

    def broadcast_object(obj: Any, src: int) -> Any:
        if world_size == 1:
            return obj
        objects = [obj]
        dist.broadcast_object_list(objects, src=src, group=host_group)
        return objects[0]

    def broadcast(tensor: torch.Tensor, src: int) -> None:
        if world_size > 1:
            dist.broadcast(tensor, src=src)

    def agree(ready: bool) -> bool:
        flag = torch.tensor(int(ready), dtype=torch.int32)
        if world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=host_group)
        return bool(flag.item())

    return LMCacheEngine(
        config,
        metadata,
        MagicMock(spec=TokenDatabase),
        connector,
        broadcast,
        broadcast_object,
        agree,
    )


def attach_sources(
    engine: LMCacheEngine, expected: torch.Tensor, missing_chunk: int | None
) -> list[MemoryObj]:
    """Serve ``expected`` from mocked storage in ``CHUNK_TOKENS`` chunks."""
    num_chunks = expected.shape[2] // CHUNK_TOKENS
    keys = [
        CacheEngineKey("manual", 1, 0, index, expected.dtype)
        for index in range(num_chunks)
    ]
    chunk_infos = [
        (key, index * CHUNK_TOKENS, (index + 1) * CHUNK_TOKENS)
        for index, key in enumerate(keys)
    ]
    sources = [
        make_pinned_object(expected[:, :, start:end]) for _, start, end in chunk_infos
    ]
    engine.token_database.process_tokens.side_effect = lambda **kwargs: iter(  # type: ignore[attr-defined]
        [(start, end, key) for key, start, end in chunk_infos]
    )
    storage = MagicMock(spec=StorageManager)
    storage.get_block_mapping.return_value = {"LocalCPUBackend": chunk_infos}
    storage.batched_get.side_effect = lambda keys, location: [
        None if key.chunk_hash == missing_chunk else sources[key.chunk_hash]
        for key in keys
    ]
    engine.storage_manager = storage
    return sources


def check_pages(
    pages: list[torch.Tensor],
    slots: torch.Tensor,
    expected: torch.Tensor,
    committed: int,
    written: int,
) -> None:
    """Verify exactly tokens ``[VLLM_CACHED_TOKENS, written)`` landed.

    ``committed`` is what the mask reports; a failed window may have been
    written without being committed, so ``committed <= written``.
    """
    assert VLLM_CACHED_TOKENS <= committed <= written, (committed, written)
    num_tokens = expected.shape[2]
    for layer, page in enumerate(pages):
        flattened = page.reshape(-1, WIDTH)
        actual = flattened[slots].cpu()
        assert (actual[:VLLM_CACHED_TOKENS] == -1).all()
        torch.testing.assert_close(
            actual[VLLM_CACHED_TOKENS:written],
            expected[0, layer, VLLM_CACHED_TOKENS:written],
            rtol=0,
            atol=0,
        )
        assert (actual[written:] == -1).all()
        assert (flattened[num_tokens:] == -1).all()


def run_case(
    budget: int,
    num_chunks: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    host_group: Any,
    failure: str = "none",
) -> dict[str, Any]:
    """Retrieve one hit through the engine, verify the pages, measure staging.

    Args:
        budget: ``retrieve_buffer_size`` in bytes; at least one chunk.
        num_chunks: Number of ``CHUNK_TOKENS``-token source chunks.
        dtype: KV element type.
        rank: This process's rank; rank zero owns the sources.
        world_size: Number of participating ranks.
        host_group: Gloo group for headers and votes, or None in one process.
        failure: ``"none"``, ``"read"`` (chunk 2 missing on the first rank) or
            ``"write"`` (the last rank fails its second window write).

    Returns:
        JSON-compatible timing, memory, and integrity results.

    Raises:
        AssertionError: If data, bounds, or failure recovery are incorrect.
    """
    device = torch.device("cuda", 0)
    num_tokens = num_chunks * CHUNK_TOKENS
    expected = (
        torch.arange(LAYERS * num_tokens * WIDTH, dtype=torch.float32)
        .remainder_(251)
        .reshape(1, LAYERS, num_tokens, WIDTH)
        .to(dtype)
    )
    pages = [
        torch.full(
            (num_tokens // PAGE_TOKENS + 2, PAGE_TOKENS, WIDTH),
            -1,
            dtype=dtype,
            device=device,
        )
        for _ in range(LAYERS)
    ]
    # Reverse placement catches token-index errors; unused slots stay poisoned.
    slots = torch.arange(num_tokens - 1, -1, -1, dtype=torch.int64, device=device)
    connector = FailingConnector(WIDTH, LAYERS, use_mla=True, device=device)
    torch.cuda.synchronize()
    before_engine = torch.cuda.memory_allocated()
    engine = build_engine(rank, world_size, host_group, budget, dtype, connector)
    torch.cuda.synchronize()
    fixed_reservation = torch.cuda.memory_allocated() - before_engine
    kwargs = {
        "kvcaches": pages,
        "slot_mapping": slots,
        "vllm_cached_tokens": VLLM_CACHED_TOKENS,
    }
    try:
        # Warm up native pointer tables and communicators outside the measurement.
        attach_sources(engine, expected[:, :, :CHUNK_TOKENS], None)
        engine.retrieve(list(range(CHUNK_TOKENS)), req_id="warmup", **kwargs)
        for page in pages:
            page.fill_(-1)
        torch.cuda.synchronize()

        fault_rank = world_size - 1
        sources = attach_sources(engine, expected, 2 if failure == "read" else None)
        connector.write_count = 0
        connector.written_until = 0
        connector.fail_on_write = (
            2 if failure == "write" and rank == fault_rank else None
        )
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        mask = engine.retrieve(list(range(num_tokens)), req_id="measured", **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        extra_peak = torch.cuda.max_memory_allocated() - before

        committed = int(mask.sum().item())
        if failure == "read":
            assert committed == 2 * CHUNK_TOKENS, committed
        elif failure == "write" and rank == fault_rank:
            assert 0 < committed < num_tokens, committed
        else:
            assert committed == num_tokens, committed
        assert mask.tolist() == [True] * committed + [False] * (num_tokens - committed)
        # A failed window is written but not committed, so written >= committed.
        written = max(connector.written_until, committed)
        check_pages(pages, slots, expected, committed, written)
        assert all(source.get_ref_count() == 0 for source in sources)
        assert extra_peak <= budget, (extra_peak, budget)

        if failure != "none":
            # The very next request must succeed on every rank.
            connector.fail_on_write = None
            attach_sources(engine, expected, None)
            for page in pages:
                page.fill_(-1)
            recovery = engine.retrieve(
                list(range(num_tokens)), req_id="recovery", **kwargs
            )
            assert recovery.all()
            check_pages(pages, slots, expected, num_tokens, num_tokens)
    finally:
        engine.close()
    return {
        "rank": rank,
        "dtype": str(dtype),
        "mode": failure,
        "hit_bytes": expected.numel() * expected.element_size(),
        "budget_bytes": budget,
        "fixed_reservation_bytes": fixed_reservation,
        "extra_peak_bytes": extra_peak,
        "seconds": elapsed,
        "committed_tokens": committed,
        "written_tokens": written,
        "exact_data": True,
    }


def main() -> None:
    """Run a matrix of budgets, hit sizes, dtypes, and injected failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rendezvous", default="tcp://127.0.0.1:29673")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    host_group = None
    if args.world_size > 1:
        dist.init_process_group(
            "nccl",
            init_method=args.rendezvous,
            rank=args.rank,
            world_size=args.world_size,
            timeout=timedelta(seconds=60),
            device_id=torch.device("cuda", 0),
        )
        host_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for count in (1, 16, 128):
                for budget in BUDGETS:
                    result = run_case(
                        budget, count, dtype, args.rank, args.world_size, host_group
                    )
                    print(json.dumps(result), flush=True)
                    gc.collect()
        for failure in ("read", "write"):
            result = run_case(
                4 * MIB,
                16,
                torch.bfloat16,
                args.rank,
                args.world_size,
                host_group,
                failure,
            )
            print(json.dumps(result), flush=True)
            gc.collect()
    finally:
        if args.world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
