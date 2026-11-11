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
from unittest.mock import MagicMock
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.memory_allocators.lazy_memory_allocator import (  # noqa: E402
    LazyMemoryAllocator,
)
from lmcache.v1.memory_management import (  # noqa: E402
    BorrowedMemoryObj,
    MemoryObj,
    PendingRead,
)
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    direct_transfer_supported,
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer import engines  # noqa: E402
from lmcache.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
)
from lmcache.v1.platform.landing_region import (  # noqa: E402
    LandingRegionTarget,
    set_landing_region_target,
)
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
        # The restore runs on the default stream; finish it before the next
        # transfer starts on the context's stream.
        torch_dev.synchronize()

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


def _mix_engines(monkeypatch, objs: list[MemoryObj]) -> None:
    """Treat every other pair of chunks as storage-backed, so the group
    splits into several runs across both engines. Host objects are valid
    input to the staged loop the second engine uses."""
    staged = {id(obj) for index, obj in enumerate(objs) if (index // 2) % 2}
    monkeypatch.setattr(
        engines,
        "select_engine",
        lambda obj: (
            engines.TransferEngine.NVME_DMA
            if id(obj) in staged
            else engines.TransferEngine.GPU_DMA
        ),
    )


@pytest.mark.parametrize("name", sorted(_CASES))
def test_retrieve_mixed_engines_matches_single_engine(
    name: str, allocator: LazyMemoryAllocator, monkeypatch
):
    """H2D with the chunks split across engines, batch size 4 and a skipped
    token prefix: the same pages as when one engine moves the whole group."""
    ctx, tensors = _make_context(name)
    _, _, chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(chunk, 0)
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    skip_first_n_tokens = chunk + chunk // blocks_per_chunk
    try:
        objs = _allocate_objects(allocator, ctx, num_chunks, fill=True)
        before = [t.clone() for t in tensors]
        _run(
            ctx, block_ids, objs, H2D, "direct", ctx.max_batch_size, skip_first_n_tokens
        )
        single = [t.clone() for t in tensors]
        for t, snap in zip(tensors, before, strict=True):
            t.copy_(snap)
        # The restore runs on the default stream; finish it before the next
        # transfer starts on the context's stream.
        torch_dev.synchronize()

        _mix_engines(monkeypatch, objs)
        _run(
            ctx, block_ids, objs, H2D, "direct", ctx.max_batch_size, skip_first_n_tokens
        )
        for expected, got in zip(single, tensors, strict=True):
            assert _bitwise_equal(expected, got)
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_store_mixed_engines_matches_single_engine(
    name: str, allocator: LazyMemoryAllocator, monkeypatch
):
    """D2H with the chunks split across engines: the same object bytes as
    when one engine moves the whole group."""
    ctx, _tensors = _make_context(name)
    _, _, _chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(ctx.lmcache_tokens_per_chunk, 0)
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    try:
        single_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        mixed_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        _run(ctx, block_ids, single_objs, D2H, "direct", ctx.max_batch_size, 0)
        _mix_engines(monkeypatch, mixed_objs)
        _run(ctx, block_ids, mixed_objs, D2H, "direct", ctx.max_batch_size, 0)
        for expected, got in zip(single_objs, mixed_objs, strict=True):
            assert torch.equal(_object_bytes(expected), _object_bytes(got))
            assert int(_object_bytes(got).sum()) != 0  # something was copied
        for obj in single_objs + mixed_objs:
            allocator.free(obj)
    finally:
        ctx.close()


class _LandedRead(PendingRead):
    """A read that already landed its bytes."""

    def is_finished(self) -> bool:
        return True

    def succeeded(self) -> bool:
        return True


def _peer_owned(ctx: GPUCacheContext, obj: MemoryObj) -> BorrowedMemoryObj:
    """An object a peer owns holding ``obj``'s bytes: its read copies them
    into the landing slot at the requested offset, standing in for the
    transfer channel."""
    slots = {
        slot.offset: slot.buffer
        for bank in range(ctx.landing_banks)
        for chunk_idx in range(ctx.max_batch_size)
        for slot in [ctx.get_landing_slot(bank, chunk_idx, 0)]
    }
    payload = _object_bytes(obj)

    def read_into(region_id: int, offset: int) -> PendingRead:
        assert region_id == _LANDING_REGION
        slots[offset].copy_(payload)
        torch_dev.synchronize()
        return _LandedRead()

    borrowed = BorrowedMemoryObj(obj.meta)
    borrowed.bind(
        (lambda: True, lambda: None), read_into=read_into, address=0, source="peer"
    )
    return borrowed


_LANDING_REGION = 5


@pytest.fixture
def landing_target():
    """Give new cache contexts a landing buffer registered with a stub."""
    registrar = MagicMock()
    registrar.register_memory_region.return_value = _LANDING_REGION
    set_landing_region_target(LandingRegionTarget(registrar, 4096))
    yield
    set_landing_region_target(None)


@pytest.mark.parametrize("peer_chunks", ["all", "some"])
@pytest.mark.parametrize("name", sorted(_CASES))
def test_retrieve_from_a_peer_matches_retrieve_from_host_memory(
    name: str, peer_chunks: str, allocator: LazyMemoryAllocator, landing_target
):
    """H2D where chunks are read into the landing buffer instead of copied
    from host memory, batch size 4 and a skipped token prefix: the same pages
    either way. ``some`` leaves every third chunk in host memory, so the NIC
    and GPU engines alternate within the group."""
    ctx, tensors = _make_context(name)
    _, _, chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(chunk, 0)
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    skip_first_n_tokens = chunk + chunk // blocks_per_chunk
    try:
        objs = _allocate_objects(allocator, ctx, num_chunks, fill=True)
        before = [t.clone() for t in tensors]
        _run(
            ctx, block_ids, objs, H2D, "direct", ctx.max_batch_size, skip_first_n_tokens
        )
        from_host = [t.clone() for t in tensors]
        for t, snap in zip(tensors, before, strict=True):
            t.copy_(snap)
        torch_dev.synchronize()

        mixed = [
            obj if peer_chunks == "some" and index % 3 == 2 else _peer_owned(ctx, obj)
            for index, obj in enumerate(objs)
        ]
        host_ids = [list(ids) for ids in block_ids]
        with torch_dev.device(ctx.device), torch_dev.stream(ctx.stream):
            block_ids_gpu = downsample_and_stage_block_ids(ctx, host_ids)
            transfer_kv_per_object_group(
                ctx,
                block_ids_gpu,
                mixed,
                object_group_id=0,
                batch_size=ctx.max_batch_size,
                skip_first_n_tokens=skip_first_n_tokens,
                direction=H2D,
                transfer_key="test-peer",
                block_ids_host=host_ids,
            )
        ctx.stream.synchronize()
        for expected, got in zip(from_host, tensors, strict=True):
            assert _bitwise_equal(expected, got)
        assert any(
            not _bitwise_equal(got, snap)
            for got, snap in zip(tensors, before, strict=True)
        )
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()
