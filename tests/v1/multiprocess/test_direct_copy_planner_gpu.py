# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU test of the direct copy path through the Python planner.

Unlike ``test_direct_copy_transfer_gpu.py`` (which drives the native executor
with hand-built descriptors), this test goes through the public
``transfer_kv_per_object_group`` entry point with a real ``GPUCacheContext``
and real ``LazyMemoryAllocator`` objects, once with the kernel policy and once
with the direct policy, and requires bit-identical results. It covers the
planner's use of the host pointer lists, the kernel-group byte offset inside
the object, the allocator's virtual offsets (objects straddling a 64 MB pin
chunk are split), ``skip_first_n_tokens`` with a batch of 4, and a Kimi-like
1024-token MLA geometry.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import random
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.key_codec import encode_fs_filename  # noqa: E402
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (  # noqa: E402
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_allocators.lazy_memory_allocator import (  # noqa: E402
    LazyMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryObj  # noqa: E402
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    direct_transfer_supported,
    downsample_and_stage_block_ids,
    slice_staged_block_ids_for_window,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.streaming import FSStreamingReader  # noqa: E402
from lmcache.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
)
from lmcache.v1.platform.stream import record_completion_event  # noqa: E402
import lmcache.cuda_ops as cuda_ops  # noqa: E402
import lmcache.lmcache_native as lmcache_native  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(
        not hasattr(cuda_ops, "execute_direct_copy_transfer")
        or not cuda_ops.batch_memcpy_supported(),
        reason=(
            "cudaMemcpyBatchAsync unavailable (needs CUDA >= 12.8 or ROCm/HIP >= 7.15)"
        ),
    ),
]

H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H
_DEVICE = torch.device("cuda:0")
_DTYPE = torch.bfloat16


class _FakeIPCWrapper:
    """Hands ``GPUCacheContext`` a local CUDA tensor (same-process CUDA IPC
    cannot reimport its own handle)."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


# name -> (per-layer paged shape, expected format, chunk tokens, num chunks)
_CASES = {
    # Kimi-Linear-like: vLLM inflates the MLA page to 1024 tokens, one block
    # per LMCache chunk, 1.1 MB per (layer, block) entry. 16 chunks of 4.7 MB
    # straddle the allocator's 64 MB pin chunk.
    "mla_1024": (
        [20, 1024, 576],
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        1024,
        16,
    ),
    # Ordinary vLLM NHD flash-attention layout, 16 chunks of 16-token blocks.
    "nhd_16": (
        [2, 96, 16, 8, 128],
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        256,
        6,
    ),
    # SGLang unified MP registers each K/V component independently as one
    # token-major [NB, BS, NH, HS] tensor.
    "sglang_component": (
        [96, 16, 8, 128],
        lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS,
        256,
        6,
    ),
}
_NUM_LAYERS = 4


def _make_context(name: str) -> tuple[GPUCacheContext, list[torch.Tensor]]:
    shape, expected_fmt, chunk_tokens, _ = _CASES[name]
    tensors = [
        torch.rand(shape, dtype=_DTYPE, device=_DEVICE) for _ in range(_NUM_LAYERS)
    ]
    is_sglang = name == "sglang_component"
    ctx = GPUCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore
        lmcache_tokens_per_chunk=chunk_tokens,
        engine_type=EngineType.SGLANG if is_sglang else EngineType.VLLM,
    )
    assert ctx.get_engine_kv_format(0) == expected_fmt
    return ctx, tensors


def _allocate_objects(
    allocator: LazyMemoryAllocator, ctx: GPUCacheContext, num_chunks: int, fill: bool
) -> list[MemoryObj]:
    shape, dtype = ctx.get_kernel_group_shape_dtype(ctx.lmcache_tokens_per_chunk, 0)
    objs: list[MemoryObj] = []
    for _ in range(num_chunks):
        obj = allocator.allocate(shape, dtype)
        if obj is None:
            raise RuntimeError("allocator out of space")
        raw = obj.raw_tensor
        assert raw is not None
        view = raw.view(torch.uint8)[: obj.get_size()]
        if fill:
            view.copy_(torch.randint(0, 256, (obj.get_size(),), dtype=torch.uint8))
        else:
            view.zero_()
        objs.append(obj)
    return objs


def _object_bytes(obj: MemoryObj) -> torch.Tensor:
    raw = obj.raw_tensor
    assert raw is not None
    return raw.view(torch.uint8)[: obj.get_size()].clone()


def _run(
    ctx: GPUCacheContext,
    block_ids: list[list[int]],
    objs: list[MemoryObj],
    direction: "lmcache_native.TransferDirection",
    mode: str,
    batch_size: int,
    skip_first_n_tokens: int,
) -> None:
    host_ids = [list(ids) for ids in block_ids]
    with torch_dev.device(ctx.device), torch_dev.stream(ctx.stream):
        block_ids_gpu = downsample_and_stage_block_ids(ctx, host_ids)
        assert direct_transfer_supported(ctx, 0, objs, host_ids) is True
        # Withholding the host block ids is what forces the kernel path now
        # that the direct path is taken automatically whenever it can be.
        transfer_kv_per_object_group(
            ctx,
            block_ids_gpu,
            objs,
            object_group_id=0,
            batch_size=batch_size,
            skip_first_n_tokens=skip_first_n_tokens,
            direction=direction,
            transfer_key=f"test-{mode}",
            block_ids_host=host_ids if mode == "direct" else [],
        )
    ctx.stream.synchronize()


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(7)
    torch.manual_seed(7)


@pytest.fixture
def allocator():
    alloc = LazyMemoryAllocator(init_size=256 << 20, final_size=256 << 20)
    yield alloc
    alloc.close()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_store_direct_matches_kernel(name: str, allocator: LazyMemoryAllocator):
    """D2H through the planner: identical object bytes on both paths."""
    ctx, _tensors = _make_context(name)
    _, _, _chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(ctx.lmcache_tokens_per_chunk, 0)
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    try:
        kernel_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        direct_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        _run(ctx, block_ids, kernel_objs, D2H, "kernel", 1, 0)
        _run(ctx, block_ids, direct_objs, D2H, "direct", 1, 0)
        for k, d in zip(kernel_objs, direct_objs, strict=True):
            assert torch.equal(_object_bytes(k), _object_bytes(d))
            assert int(_object_bytes(d).sum()) != 0  # something was copied
        for obj in kernel_objs + direct_objs:
            allocator.free(obj)
    finally:
        ctx.close()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_retrieve_direct_matches_kernel(name: str, allocator: LazyMemoryAllocator):
    """H2D through the planner with batch size 4 and a skipped token prefix:
    identical pages on both paths."""
    ctx, tensors = _make_context(name)
    _, _, chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(chunk, 0)
    block_size = chunk // blocks_per_chunk
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    # Skip one whole chunk plus one block of the next one.
    skip_first_n_tokens = chunk + block_size
    try:
        objs = _allocate_objects(allocator, ctx, num_chunks, fill=True)
        before = [t.clone() for t in tensors]

        _run(
            ctx, block_ids, objs, H2D, "kernel", ctx.max_batch_size, skip_first_n_tokens
        )
        after_kernel = [t.clone() for t in tensors]
        for t, snap in zip(tensors, before, strict=True):
            t.copy_(snap)

        _run(
            ctx, block_ids, objs, H2D, "direct", ctx.max_batch_size, skip_first_n_tokens
        )
        for k, d in zip(after_kernel, tensors, strict=True):
            assert _bitwise_equal(k, d)
        # The skipped prefix was left alone, the rest was written.
        assert any(
            not _bitwise_equal(k, snap)
            for k, snap in zip(after_kernel, before, strict=True)
        )
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()


@pytest.mark.parametrize("num_requests", [1, 8, 32])
def test_mixed_fs_windows_match_l1_gpu_bytes_concurrently(
    num_requests: int, allocator: LazyMemoryAllocator, tmp_path
):
    """Shared FS windows and resident L1 produce identical GPU KV bytes."""
    contexts = []
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    reader = None
    objs = []
    try:
        first_ctx, first_tensors = _make_context("nhd_16")
        contexts.append((first_ctx, first_tensors))
        _, _, chunk, num_chunks = _CASES["nhd_16"]
        objs = _allocate_objects(allocator, first_ctx, num_chunks, fill=True)
        shape, dtype = first_ctx.get_kernel_group_shape_dtype(chunk, 0)
        layout = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        reader = FSStreamingReader(adapter, 2 * objs[0].get_size())
        keys = [
            ObjectKey(chunk_hash=bytes([i]), model_name="model", kv_rank=0)
            for i in range(1, num_chunks)
        ]
        for key, obj in zip(keys, objs[1:], strict=True):
            (tmp_path / encode_fs_filename(key)).write_bytes(bytes(obj.byte_array))
        assert reader.lookup(keys, {0: layout}).popcount() == len(keys)

        for _ in range(num_requests - 1):
            contexts.append(_make_context("nhd_16"))
        prepared = []
        for ctx, tensors in contexts:
            blocks = ctx.calculate_num_blocks(chunk, 0) * num_chunks
            ids = [random.sample(range(ctx.num_blocks), blocks)]
            before = [tensor.clone() for tensor in tensors]
            _run(ctx, ids, objs, H2D, "direct", ctx.max_batch_size, 0)
            expected = [tensor.clone() for tensor in tensors]
            for tensor, original in zip(tensors, before, strict=True):
                tensor.copy_(original)
            prepared.append((ctx, tensors, ids, expected))
        torch_dev.synchronize(_DEVICE)
        barrier = threading.Barrier(num_requests)

        def retrieve(item):
            ctx, tensors, ids, expected = item
            barrier.wait(timeout=60)
            with torch_dev.device(ctx.device), torch_dev.stream(ctx.stream):
                staged_ids = downsample_and_stage_block_ids(ctx, ids)
                gpu_ids, host_ids = slice_staged_block_ids_for_window(
                    ctx, staged_ids, ids, 0, 1
                )
                ctx.transfer_object_group(
                    gpu_ids,
                    [objs[0]],
                    object_group_id=0,
                    batch_size=ctx.max_batch_size,
                    skip_first_n_tokens=0,
                    direction=H2D,
                    transfer_key="mixed-l1",
                    block_ids_host=host_ids,
                )
                for index, key in enumerate(keys, start=1):
                    window = reader.pool.acquire(layout, timeout=60)
                    transferred = False
                    try:
                        assert reader.load_many([key], [window])
                        gpu_ids, host_ids = slice_staged_block_ids_for_window(
                            ctx, staged_ids, ids, index, 1
                        )
                        ctx.transfer_object_group(
                            gpu_ids,
                            [window.memory_obj],
                            object_group_id=0,
                            batch_size=ctx.max_batch_size,
                            skip_first_n_tokens=0,
                            direction=H2D,
                            transfer_key="mixed-fs",
                            block_ids_host=host_ids,
                        )
                        fence = record_completion_event(ctx.device, ctx.stream)
                        reader.release_after_transfer([window], fence)
                        transferred = True
                    finally:
                        if not transferred:
                            ctx.stream.synchronize()
                            reader.pool.release(window)
                ctx.stream.synchronize()
            return all(
                _bitwise_equal(a, b) for a, b in zip(tensors, expected, strict=True)
            )

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            assert all(executor.map(retrieve, prepared))
    finally:
        if reader is not None:
            reader.close()
        adapter.close()
        for ctx, _ in contexts:
            ctx.close()
        for obj in objs:
            allocator.free(obj)
