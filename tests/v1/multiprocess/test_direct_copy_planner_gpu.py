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
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.memory_allocators.lazy_memory_allocator import (  # noqa: E402
    LazyMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryObj  # noqa: E402
from lmcache.v1.multiprocess.config import TransferCopyPolicy  # noqa: E402
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (  # noqa: E402
    TransferCopyPath,
    downsample_and_stage_block_ids,
    select_transfer_copy_path,
    transfer_kv_per_object_group,
)
from lmcache.v1.platform.cuda.cache_context import GPUCacheContext  # noqa: E402
import lmcache.cuda_ops as cuda_ops  # noqa: E402
import lmcache.lmcache_native as lmcache_native  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(
        not hasattr(cuda_ops, "execute_direct_copy_transfer")
        or not cuda_ops.batch_memcpy_supported(),
        reason="cudaMemcpyBatchAsync unavailable (needs CUDA >= 12.8 build/driver)",
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
}
_NUM_LAYERS = 4


def _make_context(name: str) -> tuple[GPUCacheContext, list[torch.Tensor]]:
    shape, expected_fmt, chunk_tokens, _ = _CASES[name]
    tensors = [
        torch.rand(shape, dtype=_DTYPE, device=_DEVICE) for _ in range(_NUM_LAYERS)
    ]
    ctx = GPUCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore
        lmcache_tokens_per_chunk=chunk_tokens,
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
    policy = TransferCopyPolicy(mode=mode)  # type: ignore[arg-type]
    host_ids = [list(ids) for ids in block_ids]
    with torch.cuda.device(ctx.device), torch.cuda.stream(ctx.stream):
        block_ids_gpu = downsample_and_stage_block_ids(ctx, host_ids)
        expected = (
            TransferCopyPath.DIRECT if mode == "direct" else TransferCopyPath.KERNEL
        )
        assert select_transfer_copy_path(ctx, 0, objs, host_ids, policy) is expected
        transfer_kv_per_object_group(
            ctx,
            block_ids_gpu,
            objs,
            object_group_id=0,
            batch_size=batch_size,
            skip_first_n_tokens=skip_first_n_tokens,
            direction=direction,
            block_ids_host=host_ids,
            copy_policy=policy,
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
